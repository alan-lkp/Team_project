"""
ShopCare 随机森林对照模型配置 (02-rf)

定位:
    这是三套对照模型里**最便宜、训练最快、可解释性最好**的一套, 作用是给 BERT 立一个
    "传统方法能做到什么程度"的标尺。如果 RF 只差 BERT 两三个点, 那就该认真考虑是否值得
    上深度学习; 如果差十来个点, 说明语义泛化确实是这任务的核心难点 —— 这个结论本身就是
    项目要交付的价值之一。

技术选型与理由:
    * 特征  : TF-IDF(词级 ngram(1,2) 或字符级 ngram(2,4))
              —— 工单短文本 + 大量未登录词, 字符级 n-gram 常常比词级更稳(不需要词典)
    * 模型  : **One-vs-Rest + RandomForestClassifier**
              —— 多标签的标准做法: 9 个标签各训一个二分类器, 每个输出独立概率.
              没有用 sklearn 的 OneVsRestClassifier 包一层, 是为了:
                (1) 某个标签在训练集里只有单一类别时能优雅降级, 而不是直接抛异常;
                (2) 每个标签的概率列怎么算完全透明, 便于教学和排查.
    * 类别不平衡: class_weight='balanced_subsample' —— RF 的 bootstrap 子样本内自动加权,
              适合长尾标签(如 payment_account 只占 9.7%)

与其他阶段的接口约定:
    predict_fun(text, cfg=None, use_llm_fallback=True, top_k=3) 返回结构与 04-bert 完全一致
    (统一由 tools/ticket_predictor.BaseTicketPredictor 保证)
运行方式:
    python 02-rf/config.py       # 打印配置摘要 + 文件自检
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.ticket_data import (  # noqa: E402
    DATA_DIR,
    DEFAULT_CLASS_PATH,
    load_class_list,
)


class RFConfig:
    """02-rf 全局配置: 路径 / 特征 / 模型 / 拒识阈值"""

    def __init__(self):
        # ==================== todo 1. 基础路径 ====================
        self.root_path = PROJECT_ROOT.replace("\\", "/") + "/"
        self.model_name = "rf-tfidf"

        # 数据路径(与 01-data/data_format.md 一致)
        self.data_dir = DATA_DIR
        self.train_path = os.path.join(DATA_DIR, "train.txt")
        self.dev_path = os.path.join(DATA_DIR, "dev.txt")
        self.test_path = os.path.join(DATA_DIR, "test.txt")
        self.class_path = DEFAULT_CLASS_PATH

        # 模型与产物路径
        self.stage_dir = os.path.join(PROJECT_ROOT, "02-rf")
        self.save_dir = os.path.join(self.stage_dir, "save_models")
        self.result_dir = os.path.join(self.stage_dir, "result")
        self.model_save_path = os.path.join(self.save_dir, "rf_tfidf.joblib")
        self.metrics_save_path = os.path.join(self.result_dir, "metrics_rf.json")
        self.threshold_search_path = os.path.join(
            self.result_dir, "threshold_search.json"
        )
        self.importances_path = os.path.join(self.result_dir, "feature_importance.json")

        # ==================== todo 2. 类别体系 ====================
        self.class_list = load_class_list(self.class_path)
        self.num_labels = len(self.class_list)

        # ==================== todo 3. 特征工程 ====================
        # auto: 有 jieba 就词级, 否则字符级(默认, 保证任何环境都能跑)
        self.feature_mode = os.environ.get("RF_FEATURE_MODE", "auto")
        self.word_ngram_range = (1, 2)  # 词级: "退款" + "退款 未到账"
        self.char_ngram_range = (2, 4)  # 字符级: "快递" "递到" + 更长的片段
        self.tfidf_max_features = 200000  # 上限, 实际由 min_df/max_df 决定
        self.min_df = 2  # 至少在 2 篇里出现过(滤掉拼写噪声)
        self.max_df = 0.95  # 出现在 >95% 文档里的词无区分度
        self.sublinear_tf = True  # 用 1+log(tf) 压制长文本刷词频

        # ==================== todo 4. 随机森林超参数 ====================
        self.n_estimators = 300
        self.max_depth = None  # 不限制深度, 用 min_samples_leaf 控制复杂度
        self.min_samples_leaf = 1
        self.rf_max_features = "sqrt"  # 每次分裂只看 sqrt(特征数) 个特征
        self.class_weight = "balanced_subsample"  # 应对标签长尾
        self.n_jobs = -1  # 用满所有 CPU 核
        self.random_state = 42

        # ==================== todo 5. 拒识阈值(与 04-bert 同口径) ====================
        self.label_threshold = float(os.environ.get("LABEL_THRESHOLD", 0.5))
        self.global_threshold = float(os.environ.get("GLOBAL_THRESHOLD", 0.8))

        # ==================== todo 6. 阈值标定 ====================
        self.tune_threshold = True  # 在 dev 上网格搜索双阈值
        self.target_reject_rate = 0.15  # 拒识率上限: 超过这个值业务上不可接受

        # ==================== todo 7. 小规模超参搜索(可选, 很花时间) ====================
        self.enable_grid_search = False
        self.grid_search_space = [
            {"n_estimators": 300, "max_depth": None},
            {"n_estimators": 300, "max_depth": 40},
            {"n_estimators": 600, "max_depth": None},
        ]

    # ------------------------------------------------------------------
    def check_files(self, need_model=False):
        """启动前自检: 数据是否齐、模型是否已训练"""
        problems = []
        for name, path in [
            ("训练集", self.train_path),
            ("验证集", self.dev_path),
            ("测试集", self.test_path),
            ("类别文件", self.class_path),
        ]:
            if not os.path.exists(path):
                problems.append(f"{name}缺失: {path}")
        if need_model and not os.path.exists(self.model_save_path):
            problems.append(
                f"模型文件缺失: {self.model_save_path} (请先运行 rf_train.py)"
            )
        return problems

    def resolve_feature_mode(self):
        """把 'auto' 解析成实际使用的模式, 返回 (mode, tokenizer)"""
        from tools.text_tokenize import get_tokenizer

        return get_tokenizer(self.feature_mode)

    def summary(self):
        mode, _ = self.resolve_feature_mode()
        ngram = self.word_ngram_range if mode == "word" else self.char_ngram_range
        return "\n".join(
            [
                f"  项目根目录   : {self.root_path}",
                f"  标签体系     : {self.num_labels} 类 -> {', '.join(self.class_list)}",
                f"  特征         : TF-IDF, mode={mode}(config={self.feature_mode}), "
                f"ngram={ngram}, tfidf_max_features={self.tfidf_max_features}, min_df={self.min_df}",
                f"  模型         : OvR + RandomForest(n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, class_weight={self.class_weight}, "
                f"rf_max_features={self.rf_max_features})",
                f"  拒识阈值     : 单标签 {self.label_threshold} / 全局 {self.global_threshold}",
                f"  阈值标定     : {self.tune_threshold} (目标拒识率 <= {self.target_reject_rate:.0%})",
                f"  模型保存     : {self.model_save_path}",
            ]
        )


if __name__ == "__main__":
    cfg = RFConfig()
    print("=" * 72)
    print("ShopCare 02-rf 配置摘要")
    print("=" * 72)
    print(cfg.summary())
    problems = cfg.check_files(need_model=False)
    if problems:
        print("\n[警告] 文件自检发现问题:")
        for p in problems:
            print("  - " + p)
    else:
        print("\n[OK] 文件自检通过")
