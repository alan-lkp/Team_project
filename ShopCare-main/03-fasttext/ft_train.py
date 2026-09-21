"""
ShopCare FastText 对照模型训练 (03-fasttext)

流程(与 02-rf / 04-bert 完全对齐):
    读数据 -> 文本清洗 + 分词 -> 转成 fasttext 语料(__label__xxx 前缀)
           -> train_supervised(loss='ova') -> dev 标定双阈值 -> test 出报告 -> 存模型 + 元数据

语料长什么样(03-fasttext/data/train.ft.txt):
    __label__logistics __label__service 快递 放 驿站 也 不 通知 我 付款 一直 失败
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 标签区(可多个)              ^^^^^^^^^^ 文本区(已分词)

为什么要额外存一份 model_meta.json:
    fasttext 的 .bin 里只装得下"词 -> 向量"和分类器权重, 装不下 Python 函数.
    但推理时必须用**和训练时完全一致**的分词方式, 否则输入分布直接漂移, 效果会莫名其妙变差.
    所以标签表、分词模式、阈值这些"模型的外部契约"单独存一份 json.
    这是所有"会把模型序列化出去"的项目都必须处理好的问题.

运行方式:
    python 03-fasttext/ft_train.py                 # 默认: 训练 + 标定阈值 + 测试 + 存盘
    python 03-fasttext/ft_train.py --grid-search   # 额外做一轮小规模超参搜索
    python 03-fasttext/ft_train.py --full-retrain  # 用 train+dev 重训
    python 03-fasttext/ft_train.py --feature-mode char
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

from config import FTConfig                                              # noqa: E402
from tools.ml_metrics import (compute_business_metrics, compute_metrics,  # noqa: E402
                              grid_search_thresholds, print_metrics)
from tools.prob_calibration import (apply_calibrators,                     # noqa: E402
                                    expected_calibration_error, fit_calibrators,
                                    summary as calibration_summary)
from tools.text_tokenize import clean_text, get_tokenizer                 # noqa: E402
from tools.ticket_data import label_statistics, load_splits, to_xy        # noqa: E402

# 空文本兜底: fasttext 遇到纯空串会给出无意义结果, 用一个占位 token 表示"这里什么都没识别到"
EMPTY_TOKEN = '[空]'


# ============================================================
# todo 1. 语料构造
# ============================================================
def tokenize_for_fasttext(text, tokenizer):
    """清洗 + 分词; 返回 token list"""
    cleaned = clean_text(text)
    tokens = tokenizer(cleaned) if cleaned else []
    return tokens or [EMPTY_TOKEN]


def to_fasttext_line(text, labels, cfg, tokenizer):
    """一行 fasttext 语料 = 若干 __label__xx + 空格 + 分词后的文本"""
    tags = ' '.join(cfg.label_prefix + lb for lb in labels)
    return f'{tags} {" ".join(tokenize_for_fasttext(text, tokenizer))}'


def xy_to_pairs(texts, Y, class_list):
    """把 multi-hot 的 Y 还原成 [(text, [标签名, ...]), ...]

    注意 to_xy 产出的 Y 是 0/1 矩阵, 而 fasttext 语料需要的是**标签名**.
    这两者搞混是很隐蔽的 bug: 代码不报错, 只是每行的标签全变成 "0" 和 "1",
    训练出来的模型标签表完全是错的.
    """
    return [(t, [c for c, v in zip(class_list, row) if v]) for t, row in zip(texts, Y)]


def write_corpus(path, pairs, cfg, tokenizer):
    """把 (text, labels) 列表写成 fasttext 语料文件, 返回行数"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for text, labels in pairs:
            f.write(to_fasttext_line(text, labels, cfg, tokenizer) + '\n')
    return len(pairs)


# ============================================================
# todo 2. 训练
# ============================================================
def train_fasttext(corpus_path, cfg, mode):
    """调 fasttext.train_supervised 训练多标签模型"""
    import fasttext

    minn, maxn = cfg.subword_range(mode)
    model = fasttext.train_supervised(
        input=corpus_path,
        lr=cfg.lr,
        dim=cfg.dim,
        epoch=cfg.epoch,
        wordNgrams=cfg.word_ngrams,
        loss=cfg.loss,              # ova -> 每个标签独立 sigmoid, 不是 softmax
        bucket=cfg.bucket,
        minCount=cfg.min_count,
        minCountLabel=cfg.min_count_label,
        minn=minn,
        maxn=maxn,
        thread=cfg.thread,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )
    return model


# ============================================================
# todo 3. 批量推理 -> (N, C) 概率矩阵
# ============================================================
def predict_proba_matrix(model, texts, class_list, tokenizer, cfg):
    """与 02-rf / 04-bert 同口径的概率矩阵

    fasttext.predict(text, k=C) 在 ova 模式下会返回**每个标签**的 sigmoid 概率;
    没被返回的标签(理论上不该出现)按 0 处理, 保证列顺序永远和 class_list 对齐.
    """
    import numpy as np

    k = len(class_list)
    probs = np.zeros((len(texts), k), dtype=float)
    index = {c: i for i, c in enumerate(class_list)}
    for row, text in enumerate(texts):
        tokens = tokenize_for_fasttext(text, tokenizer)
        labels, scores = model.predict(' '.join(tokens), k=k)
        for lb, sc in zip(labels, scores):
            name = lb[len(cfg.label_prefix):] if lb.startswith(cfg.label_prefix) else lb
            if name in index:
                probs[row, index[name]] = float(sc)
    return probs


# ============================================================
# todo 4. 训练主流程
# ============================================================
def run(cfg, args):
    t_start = time.time()
    print('=' * 72)
    print('ShopCare 03-fasttext 对照模型 —— 训练')
    print('=' * 72)
    print(cfg.summary())
    print('-' * 72)

    problems = cfg.check_files(need_model=False)
    if problems:
        print('[错误] 自检未通过:')
        for p in problems:
            print('  - ' + p)
        return None

    # ---------- 1. 读数据 ----------
    print('\n[1/7] 读取数据 ...')
    train_pairs, dev_pairs, test_pairs, class_list = load_splits()
    print(f'  训练集 {len(train_pairs)} 条 / 验证集 {len(dev_pairs)} 条 / 测试集 {len(test_pairs)} 条')
    st = label_statistics(train_pairs, class_list)
    print(f'  平均标签数 {st["avg_labels"]}')

    X_train_text, Y_train = to_xy(train_pairs, class_list)
    X_dev_text, Y_dev = to_xy(dev_pairs, class_list)
    X_test_text, Y_test = to_xy(test_pairs, class_list)
    if args.full_retrain:
        print('\n  [--full-retrain] 把验证集并入训练集')
        X_train_text = X_train_text + X_dev_text
        Y_train = Y_train + Y_dev

    # ---------- 2. 语料转换 ----------
    # get_tokenizer 会把 'auto' 解析成实际使用的模式; 存进元数据的必须是**已解析**的值,
    # 否则推理端拿到 'auto' 只能靠猜, 换台机器(装了/没装 jieba)就会静默地换分词方式
    mode, tokenizer = get_tokenizer(args.feature_mode or cfg.feature_mode)
    print(f'\n[2/7] 转换 fasttext 语料(feature_mode={mode}) ...')
    train_corpus = os.path.join(cfg.corpus_dir, 'train.ft.txt')
    dev_corpus = os.path.join(cfg.corpus_dir, 'dev.ft.txt')
    test_corpus = os.path.join(cfg.corpus_dir, 'test.ft.txt')
    n = write_corpus(train_corpus, xy_to_pairs(X_train_text, Y_train, class_list), cfg, tokenizer)
    write_corpus(dev_corpus, xy_to_pairs(X_dev_text, Y_dev, class_list), cfg, tokenizer)
    write_corpus(test_corpus, xy_to_pairs(X_test_text, Y_test, class_list), cfg, tokenizer)
    print(f'  已写出 {n} 行训练语料 -> {train_corpus}')
    first_labels = [c for c, v in zip(class_list, Y_train[0]) if v]
    print('  格式示例:', to_fasttext_line(X_train_text[0], first_labels, cfg, tokenizer)[:120])

    # ---------- 3. 训练(可选多组超参) ----------
    combos = cfg.grid_search_space if (args.grid_search or cfg.enable_grid_search) else [None]
    best = None
    for idx, combo in enumerate(combos, 1):
        if combo:
            cfg.dim = combo.get('dim', cfg.dim)
            cfg.epoch = combo.get('epoch', cfg.epoch)
            cfg.lr = combo.get('lr', cfg.lr)
            cfg.word_ngrams = combo.get('wordNgrams', cfg.word_ngrams)
            print(f'\n[3/7] 训练 FastText —— 组合 {idx}/{len(combos)}: {combo}')
        else:
            print('\n[3/7] 训练 FastText(loss=ova, 每标签独立 sigmoid) ...')
        t0 = time.time()
        model = train_fasttext(train_corpus, cfg, mode)
        print(f'  训练耗时 {time.time() - t0:.1f}s, 词表大小 {len(model.get_words())}')
        probs_dev = predict_proba_matrix(model, X_dev_text, class_list, tokenizer, cfg)
        m = compute_metrics(Y_dev, probs_dev, class_list, threshold=cfg.label_threshold)
        print(f'  dev Micro-F1 = {m["micro_f1"]} / Macro-F1 = {m["macro_f1"]}')
        if best is None or m['micro_f1'] > best['metrics']['micro_f1']:
            best = {'metrics': m, 'model': model, 'combo': combo}
    model = best['model']
    if len(combos) > 1:
        print(f'\n  最优组合: {best["combo"]} -> dev Micro-F1 {best["metrics"]["micro_f1"]}')

    # ---------- 4. 概率校准 ----------
    # FastText 用 ova(每标签一个独立 sigmoid), 在 10 万条语料上训练 25 轮后会饱和:
    # 5.9% 的标签概率正好等于 1.000, 而那一档的实际命中率只有 88%。不校准的话,
    # 前端会显示"100.0%", 既不可信、又和 RF 的概率尺度不可比。
    # 校准器**只在 dev 上拟合**, test 不参与, 所以下面的 test 指标依然干净。
    probs_dev_raw = predict_proba_matrix(model, X_dev_text, class_list, tokenizer, cfg)
    probs_test_raw = predict_proba_matrix(model, X_test_text, class_list, tokenizer, cfg)
    print('\n[4/7] 在 dev 上拟合概率校准器 ...')
    calibration = fit_calibrators(probs_dev_raw, Y_dev, class_list)
    probs_dev = apply_calibrators(probs_dev_raw, calibration)
    probs_test = apply_calibrators(probs_test_raw, calibration)
    print(calibration_summary(probs_test_raw, probs_test, Y_test, calibration,
                              '03-fasttext 概率校准效果(在 test 上评估)'))

    # ---------- 5. 阈值标定 ----------
    grid = []
    if args.tune_threshold:
        print(f'\n[5/7] 在 dev 上网格搜索双阈值(目标拒识率 <= {cfg.target_reject_rate:.0%}) ...')
        grid = grid_search_thresholds(probs_dev, Y_dev, class_list,
                                      target_reject_rate=cfg.target_reject_rate)
        if grid:
            top = grid[0]
            print(f'  候选组合 {len(grid)} 个, 最优: 单标签阈值 {top["label_threshold"]} / '
                  f'全局阈值 {top["global_threshold"]}')
            print(f'  -> 自动分流率 {top["auto_rate"]:.2%}, 拒识率 {top["reject_rate"]:.2%}, '
                  f'自动分流 Micro-F1 {top["auto_micro_f1"]}')
            cfg.label_threshold = top['label_threshold']
            cfg.global_threshold = top['global_threshold']
        else:
            print('  [警告] 没有组合满足拒识率上限, 保持默认阈值')
    else:
        print('\n[5/7] 跳过阈值标定')

    # ---------- 6. 测试集评估 ----------
    print('\n[6/7] 测试集最终评估')
    metrics = compute_metrics(Y_test, probs_test, class_list, threshold=cfg.label_threshold)
    biz = compute_business_metrics(probs_test, Y_test, class_list,
                                   cfg.label_threshold, cfg.global_threshold)
    metrics.update(biz)
    print_metrics(metrics, title='03-fasttext 测试集结果(含业务指标)')

    # ---------- 6. 存盘 ----------
    print('\n[7/7] 保存模型与产物 ...')
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    model.save_model(cfg.model_save_path)
    print(f'  模型 -> {cfg.model_save_path} ({os.path.getsize(cfg.model_save_path) / 1024:.0f} KB)')

    meta = {
        'stage': '03-fasttext',
        'model_name': cfg.model_name,
        'class_list': class_list,
        'feature_mode': mode,
        'label_prefix': cfg.label_prefix,
        'empty_token': EMPTY_TOKEN,
        'label_threshold': cfg.label_threshold,
        'global_threshold': cfg.global_threshold,
        'params': {'loss': cfg.loss, 'dim': cfg.dim, 'epoch': cfg.epoch, 'lr': cfg.lr,
                   'wordNgrams': cfg.word_ngrams, 'bucket': cfg.bucket,
                   'minCount': cfg.min_count, 'subword': cfg.subword_range(mode)},
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'train_size': len(X_train_text),
        'train_seconds': round(time.time() - t_start, 1),
        # 概率校准器(Platt 的 a/b 系数, 每标签一组)。推理端必须 apply 一下,
        # 否则线上给出的置信度和训练报告里的口径不一致。
        'calibration': calibration,
    }
    meta_path = os.path.join(cfg.result_dir, 'model_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'  推理契约(标签表/分词模式/阈值) -> {meta_path}')

    with open(cfg.metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump({'stage': '03-fasttext', 'feature_mode': mode,
                   'thresholds': {'label': cfg.label_threshold, 'global': cfg.global_threshold},
                   'calibration': {'method': calibration['method'],
                                   'ece_raw': expected_calibration_error(probs_test_raw, Y_test),
                                   'ece_calibrated': expected_calibration_error(probs_test, Y_test),
                                   'per_label': calibration['per_label']},
                   'metrics': metrics}, f, ensure_ascii=False, indent=2)
    print(f'  指标 -> {cfg.metrics_save_path}')

    if grid:
        with open(cfg.threshold_search_path, 'w', encoding='utf-8') as f:
            json.dump(grid, f, ensure_ascii=False, indent=2)
        print(f'  阈值搜索结果 -> {cfg.threshold_search_path}')

    # 打印每个标签的 top 词, 和 RF 的特征重要性对照着看很有意思
    print('\n  每个标签最相关的 5 个词(自己算余弦相似度, 避免不同 fasttext 版本返回结构不一致):')
    try:
        import numpy as _np
        vocab = model.get_words()
        word_vecs = _np.array([model.get_word_vector(w) for w in vocab], dtype=float)
        norms = _np.linalg.norm(word_vecs, axis=1)
        for name in class_list:
            lv = model.get_word_vector(cfg.label_prefix + name)
            sims = word_vecs @ lv / (norms * (_np.linalg.norm(lv) + 1e-9) + 1e-9)
            idx = _np.argsort(sims)[::-1][:5]
            print(f'    {name:<16}' + ', '.join(vocab[i] for i in idx))
    except Exception as exc:              # noqa: BLE001
        print(f'    [跳过] 词向量近邻打印失败: {type(exc).__name__}: {exc}')

    print(f'\n[完成] 总耗时 {time.time() - t_start:.1f}s')
    print(f'       测试集 Micro-F1 = {metrics["micro_f1"]}  Macro-F1 = {metrics["macro_f1"]}')
    print(f'       自动分流率 {biz["auto_rate"]:.2%}, 拒识率 {biz["reject_rate"]:.2%}')
    return metrics


def main():
    cfg = FTConfig()
    parser = argparse.ArgumentParser(description='ShopCare 03-fasttext 对照模型训练')
    parser.add_argument('--feature-mode', choices=['word', 'char', 'auto'], default=None)
    parser.add_argument('--grid-search', action='store_true')
    parser.add_argument('--full-retrain', action='store_true')
    parser.add_argument('--no-tune-threshold', dest='tune_threshold', action='store_false')
    args = parser.parse_args()
    if not args.tune_threshold:
        cfg.tune_threshold = False
    run(cfg, args)


if __name__ == '__main__':
    main()