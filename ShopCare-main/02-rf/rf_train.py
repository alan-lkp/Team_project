"""
ShopCare 随机森林对照模型训练 (02-rf)

流程(和 04-bert 完全对齐, 便于横向比较):
    读数据 -> 文本清洗 -> TF-IDF 向量化 -> 逐标签训练 RandomForest(OvR)
           -> 在 dev 上算指标 -> 网格搜索双阈值 -> 在 test 上出最终报告 -> 存模型

多标签怎么落到传统模型上:
    9 个标签各训练**一个独立的二分类器**(One-vs-Rest)。
    第 i 个分类器只回答"这条工单是不是第 i 类", 输出 P(y_i=1|x) ∈ [0,1]。
    9 个概率拼起来就是 (9,) 的概率向量, 后面直接复用和 BERT 相同的双阈值拒识逻辑
    —— 注意这里**没有 softmax**, 因为多标签的标签之间不是互斥关系
    ("退款"和"物流"完全可能同时成立)。

关于"某标签在训练集里一个正例都没有"的边界情况:
    真跑大模型时经常遇到(比如某类样本被切分时全落到了 dev)。
    直接训会报 "only one class present"; 这里降级为**常数概率**(拉普拉斯平滑先验),
    保证流程不崩, 同时在日志里明确告警 —— 静默失败比崩溃更危险。

运行方式:
    python 02-rf/rf_train.py                    # 默认: 训练 + 标定阈值 + 测试 + 存盘
    python 02-rf/rf_train.py --grid-search      # 额外做一轮小规模超参搜索
    python 02-rf/rf_train.py --full-retrain     # 用 train+dev 重训(阈值已标定, 上线前跑)
    python 02-rf/rf_train.py --feature-mode word
"""

import argparse
import json
import os
import sys
import time

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_STAGE_DIR)
for _p in (_STAGE_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import RFConfig  # noqa: E402

from tools.ml_metrics import (  # noqa: E402
    compute_business_metrics,
    compute_metrics,
    grid_search_thresholds,
    print_metrics,
)
from tools.text_tokenize import clean_text, get_tokenizer  # noqa: E402
from tools.ticket_data import label_statistics, load_splits, to_xy  # noqa: E402


# ============================================================
# todo 1. 特征: TF-IDF
# ============================================================
def build_vectorizer(cfg, mode, tokenizer=None):
    """构造 TF-IDF 向量化器

    注意 tokenizer 必须是**模块级函数**(见 tools/text_tokenize.py 的说明),
    否则 joblib 存模型后, 换个脚本加载会报 "Can't get attribute"。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if tokenizer is None:
        _, tokenizer = get_tokenizer(mode)
    ngram = cfg.word_ngram_range if mode == "word" else cfg.char_ngram_range
    return TfidfVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,  # 用了自定义 tokenizer 就必须关掉默认的 token 正则
        ngram_range=ngram,
        max_features=cfg.tfidf_max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        lowercase=True,  # 英文品牌名统一小写, 减少稀疏度
        # 不指定 dtype: sklearn>=1.5 起只接受 np.float32/64 对象, 传字符串 'float32' 会被警告并
        # 自动转回 float64, 反而更容易造成误解; 用默认 float64 最省心
    )


def clean_texts(texts):
    """对文本列表做统一清洗(训练与推理必须用同一个函数, 否则特征分布会漂移)"""
    return [clean_text(t) for t in texts]


# ============================================================
# todo 2. 模型: 逐标签 One-vs-Rest 随机森林
# ============================================================
def train_ovr(X, Y, cfg, verbose=True):
    """为每个标签训练一个二分类随机森林

    返回: (estimators, constants)
          estimators[i] : 第 i 个标签的分类器(退化情形为 None)
          constants[i]  : 退化情形下的常数概率字典, 正常时为 None
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    Y = np.asarray(Y)
    estimators, constants, degenerate = [], [], []

    for i, name in enumerate(cfg.class_list):
        y = Y[:, i]
        n_pos = int(y.sum())
        n_all = len(y)
        if n_pos == 0 or n_pos == n_all:
            # 单一类别: 用拉普拉斯平滑先验当常数概率(不训模型, 但也不让流程崩)
            prior = (n_pos + 1) / (n_all + 2)
            constants.append(
                {
                    "type": "degenerate",
                    "prob": round(prior, 4),
                    "note": "训练集中该标签只有单一类别",
                }
            )
            estimators.append(None)
            degenerate.append(f"{name}(n_pos={n_pos}/{n_all})")
            continue

        clf = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features=cfg.rf_max_features,
            class_weight=cfg.class_weight,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            bootstrap=True,
        )
        clf.fit(X, y)
        estimators.append(clf)
        constants.append(None)
        if verbose:
            print(
                f"    [{i + 1}/{cfg.num_labels}] {name:<16} 正例 {n_pos:>5} 条  "
                f"({n_pos / n_all:.1%})  已训练"
            )

    if degenerate:
        print("\n  [警告] 以下标签在训练集中只有一个类别, 已退化为常数概率:")
        for d in degenerate:
            print("    - " + d)
    return estimators, constants


def predict_proba_matrix(vectorizer, estimators, constants, texts):
    """把文本批量转成 (N, C) 概率矩阵 —— 与 BERT 版 predict_proba 的口径完全一致"""
    import numpy as np

    X = vectorizer.transform(clean_texts(texts))
    probs = np.zeros((X.shape[0], len(estimators)), dtype=float)
    for i, (clf, const) in enumerate(zip(estimators, constants)):
        if clf is None:
            probs[:, i] = const["prob"]
        else:
            classes = list(clf.classes_)
            probs[:, i] = clf.predict_proba(X)[:, classes.index(1)]
    return probs


# ============================================================
# todo 3. 可解释性: 每个标签最"有信号"的词
# ============================================================
def top_features(vectorizer, estimators, cfg, top_n=15):
    """导出每个标签的特征重要性 Top-N

    这是 RF 相对 BERT 的**最大优势**: 业务方问"为什么判成物流问题",
    可以明确回答"因为出现了'未发货''揽收'这些词"。
    """
    import numpy as np

    names = vectorizer.get_feature_names_out()
    out = {}
    global_imp = np.zeros(len(names))
    for i, name in enumerate(cfg.class_list):
        clf = estimators[i]
        if clf is None:
            out[name] = []
            continue
        imp = clf.feature_importances_
        global_imp += imp
        idx = np.argsort(imp)[::-1][:top_n]
        out[name] = [
            {"feature": str(names[j]), "importance": round(float(imp[j]), 6)}
            for j in idx
            if imp[j] > 0
        ]
    idx = np.argsort(global_imp)[::-1][:top_n]
    out["_global"] = [
        {"feature": str(names[j]), "importance": round(float(global_imp[j]), 6)}
        for j in idx
        if global_imp[j] > 0
    ]
    return out


# ============================================================
# todo 4. 训练主流程
# ============================================================
def run(cfg, args):
    t_start = time.time()
    print("=" * 72)
    print("ShopCare 02-rf 随机森林对照模型 —— 训练")
    print("=" * 72)
    print(cfg.summary())
    print("-" * 72)

    problems = cfg.check_files(need_model=False)
    if problems:
        print("[错误] 文件自检未通过:")
        for p in problems:
            print("  - " + p)
        return None

    # ---------- 1. 读数据 ----------
    print("\n[1/6] 读取数据 ...")
    train_pairs, dev_pairs, test_pairs, class_list = load_splits()
    print(
        f"  训练集 {len(train_pairs)} 条 / 验证集 {len(dev_pairs)} 条 / 测试集 {len(test_pairs)} 条"
    )
    st = label_statistics(train_pairs, class_list)
    print(f"  平均标签数 {st['avg_labels']} (多标签任务的核心统计量)")
    long_tail = sorted(st["per_label"], key=lambda r: r["count"])[:3]
    print(
        "  最长尾的三个标签:",
        ", ".join(f"{r['label']}({r['count']})" for r in long_tail),
    )

    X_train_text, Y_train = to_xy(train_pairs, class_list)
    X_dev_text, Y_dev = to_xy(dev_pairs, class_list)
    X_test_text, Y_test = to_xy(test_pairs, class_list)
    if args.full_retrain:
        print("\n  [--full-retrain] 把验证集并入训练集(阈值沿用已标定结果)")
        X_train_text = X_train_text + X_dev_text
        Y_train = Y_train + Y_dev

    # ---------- 2. 向量化 ----------
    # 解析成实际模式并存进模型包, 避免推理端把 'auto' 重新解析成别的模式
    from tools.text_tokenize import get_tokenizer as _get_tok

    mode, _tok = _get_tok(args.feature_mode or cfg.feature_mode)
    print(f"\n[2/6] TF-IDF 向量化(feature_mode={mode}) ...")
    vectorizer = build_vectorizer(cfg, mode, _tok)
    X_train = vectorizer.fit_transform(clean_texts(X_train_text))
    print(f"  实际特征维度: {X_train.shape[1]} (max_features={cfg.tfidf_max_features})")
    print(f"  训练矩阵形状: {X_train.shape[0]} x {X_train.shape[1]}")

    # ---------- 3. 训练 ----------
    params_list = (
        cfg.grid_search_space
        if (args.grid_search or cfg.enable_grid_search)
        else [None]
    )
    best = None
    for k, params in enumerate(params_list, 1):
        if params:
            cfg.n_estimators = params.get("n_estimators", cfg.n_estimators)
            cfg.max_depth = params.get("max_depth", cfg.max_depth)
            print(f"\n[3/6] 训练 OvR 随机森林 —— 组合 {k}/{len(params_list)}: {params}")
        else:
            print("\n[3/6] 训练 OvR 随机森林(9 个标签各一个二分类器) ...")
        t0 = time.time()
        estimators, constants = train_ovr(X_train, Y_train, cfg, verbose=not params)
        print(f"  训练耗时 {time.time() - t0:.1f}s")

        probs_dev = predict_proba_matrix(vectorizer, estimators, constants, X_dev_text)
        m = compute_metrics(Y_dev, probs_dev, class_list, threshold=cfg.label_threshold)
        print(f"  dev Micro-F1 = {m['micro_f1']} / Macro-F1 = {m['macro_f1']}")
        if best is None or m["micro_f1"] > best["metrics"]["micro_f1"]:
            best = {
                "metrics": m,
                "estimators": estimators,
                "constants": constants,
                "params": params,
            }
    estimators, constants = best["estimators"], best["constants"]
    if len(params_list) > 1:
        print(
            f"\n  最优组合: {best['params']} -> dev Micro-F1 {best['metrics']['micro_f1']}"
        )

    # ---------- 4. 阈值标定 ----------
    probs_dev = predict_proba_matrix(vectorizer, estimators, constants, X_dev_text)
    if args.tune_threshold:
        print(
            f"\n[4/6] 在 dev 上网格搜索双阈值(目标拒识率 <= {cfg.target_reject_rate:.0%}) ..."
        )
        grid = grid_search_thresholds(
            probs_dev, Y_dev, class_list, target_reject_rate=cfg.target_reject_rate
        )
        if grid:
            top = grid[0]
            print(
                f"  候选组合 {len(grid)} 个, 最优: 单标签阈值 {top['label_threshold']} / "
                f"全局阈值 {top['global_threshold']}"
            )
            print(
                f"  -> 自动分流率 {top['auto_rate']:.2%}, 拒识率 {top['reject_rate']:.2%}, "
                f"自动分流 Micro-F1 {top['auto_micro_f1']}"
            )
            print("  Top-5:")
            for r in grid[:5]:
                print(
                    f"    label={r['label_threshold']} global={r['global_threshold']} "
                    f"auto={r['auto_rate']:.3f} reject={r['reject_rate']:.3f} "
                    f"micro_f1={r['auto_micro_f1']}"
                )
            cfg.label_threshold = top["label_threshold"]
            cfg.global_threshold = top["global_threshold"]
        else:
            print(
                "  [警告] 没有组合能满足拒识率上限, 保持默认阈值 "
                f"({cfg.label_threshold}/{cfg.global_threshold})"
            )
            grid = []
    else:
        print("\n[4/6] 跳过阈值标定(--tune-threshold 未开启)")
        grid = []

    # ---------- 5. 测试集评估 ----------
    print("\n[5/6] 测试集最终评估")
    probs_test = predict_proba_matrix(vectorizer, estimators, constants, X_test_text)
    metrics = compute_metrics(
        Y_test, probs_test, class_list, threshold=cfg.label_threshold
    )
    biz = compute_business_metrics(
        probs_test, Y_test, class_list, cfg.label_threshold, cfg.global_threshold
    )
    metrics.update(biz)
    print_metrics(metrics, title="02-rf 测试集结果(含业务指标)")
    print(
        '\n  说明: 拒识的工单不进自动分流, 所以 auto_micro_f1 才是"真正自动处理那部分"的准确度'
    )

    # ---------- 6. 存盘 ----------
    print("\n[6/6] 保存模型与产物 ...")
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    import joblib

    bundle = {
        "stage": "02-rf",
        "model_name": cfg.model_name,
        "vectorizer": vectorizer,
        "estimators": estimators,
        "constants": constants,
        "class_list": class_list,
        "feature_mode": mode,
        "label_threshold": cfg.label_threshold,
        "global_threshold": cfg.global_threshold,
        "params": {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "min_samples_leaf": cfg.min_samples_leaf,
            "rf_max_features": cfg.rf_max_features,
            "class_weight": cfg.class_weight,
            "ngram": cfg.word_ngram_range if mode == "word" else cfg.char_ngram_range,
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": len(X_train_text),
        "train_seconds": round(time.time() - t_start, 1),
    }
    joblib.dump(bundle, cfg.model_save_path, compress=3)
    print(
        f"  模型 -> {cfg.model_save_path} ({os.path.getsize(cfg.model_save_path) / 1024:.0f} KB)"
    )

    with open(cfg.metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": "02-rf",
                "model": cfg.model_name,
                "feature_mode": mode,
                "thresholds": {
                    "label": cfg.label_threshold,
                    "global": cfg.global_threshold,
                },
                "dev_micro_f1_before_tuning": best["metrics"]["micro_f1"],
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  指标 -> {cfg.metrics_save_path}")

    if grid:
        with open(cfg.threshold_search_path, "w", encoding="utf-8") as f:
            json.dump(grid, f, ensure_ascii=False, indent=2)
        print(f"  阈值搜索结果 -> {cfg.threshold_search_path}")

    feats = top_features(vectorizer, estimators, cfg)
    with open(cfg.importances_path, "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)
    print(f"  特征重要性 -> {cfg.importances_path}")
    print("\n  每个标签最具区分度的 5 个特征(业务可解释性的直接证据):")
    for name in class_list:
        items = feats.get(name, [])[:5]
        if items:
            print(f"    {name:<16}" + ", ".join(x["feature"] for x in items))

    print(f"\n[完成] 总耗时 {time.time() - t_start:.1f}s")
    print(
        f"       测试集 Micro-F1 = {metrics['micro_f1']}  Macro-F1 = {metrics['macro_f1']}"
    )
    print(f"       自动分流率 {biz['auto_rate']:.2%}, 拒识率 {biz['reject_rate']:.2%}")
    return metrics


def main():
    cfg = RFConfig()
    parser = argparse.ArgumentParser(description="ShopCare 02-rf 随机森林对照模型训练")
    parser.add_argument(
        "--feature-mode",
        choices=["word", "char", "auto"],
        default=None,
        help="覆盖特征模式(默认取 RF_FEATURE_MODE 环境变量, 未设置则 auto)",
    )
    parser.add_argument(
        "--grid-search", action="store_true", help="额外做一轮小规模超参搜索"
    )
    parser.add_argument(
        "--full-retrain",
        action="store_true",
        help="把验证集并入训练集后再训(上线前的最后一次训练)",
    )
    parser.add_argument(
        "--no-tune-threshold",
        dest="tune_threshold",
        action="store_false",
        help="跳过阈值标定, 直接用配置里的默认阈值",
    )
    args = parser.parse_args()
    if not args.tune_threshold:
        cfg.tune_threshold = False
    run(cfg, args)


if __name__ == "__main__":
    main()
