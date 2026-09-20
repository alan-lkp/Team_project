"""
ShopCare FastText 对照模型配置 (03-fasttext)

定位:
    三套对照模型里的"中间档":
        RF/TF-IDF  : 纯字面 n-gram 匹配, 训练秒级, 完全可解释
        FastText   : 词向量 + n-gram 特征 + 线性分类器, 训练几秒~几十秒, 有稠密语义但很浅
        BERT+LoRA  : 深层上下文化语义, 训练分钟级, 需要 GPU 才舒服
    它存在的意义是回答一个很实际的问题: "我们真的需要上 BERT 吗?"
    如果 FastText 只比 BERT 低一两个点, 那上线的应该就是 FastText(成本差一个数量级).

技术选型与理由:
    * loss='ova' 是**多标签的关键**: 它给每个标签独立算 sigmoid, 而不是 softmax 互斥.
      用 softmax 会把"物流+售后"这种共现标签强行压成竞争关系, 是新手最常见的坑.
    * 字符级 vs 词级: 中文没有空格, 词级需要 jieba 先分词; 字符级则把每个汉字当一个"词",
      再靠 wordNgrams + subword(minn/maxn) 自动组合出"快递""退款"这类片段.
    * 本项目**不下载预训练词向量**: 小数据集上从零训的词向量已经够用,
      而且能避免"为了跑通一个 demo 还要先下 2GB 文件"的尴尬.

运行方式:
    python 03-fasttext/config.py       # 打印配置摘要 + 文件自检
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.ticket_data import DATA_DIR, DEFAULT_CLASS_PATH, load_class_list  # noqa: E402

# fasttext 规定标签必须以这个前缀开头, 自己随便写别的会直接报错
LABEL_PREFIX = '__label__'


class FTConfig:
    """03-fasttext 全局配置: 路径 / 语料格式 / 模型超参 / 拒识阈值"""

    def __init__(self):
        # ==================== todo 1. 基础路径 ====================
        self.root_path = PROJECT_ROOT.replace('\\', '/') + '/'
        self.model_name = 'fasttext'

        self.data_dir = DATA_DIR
        self.train_path = os.path.join(DATA_DIR, 'train.txt')
        self.dev_path = os.path.join(DATA_DIR, 'dev.txt')
        self.test_path = os.path.join(DATA_DIR, 'test.txt')
        self.class_path = DEFAULT_CLASS_PATH

        self.stage_dir = os.path.join(PROJECT_ROOT, '03-fasttext')
        self.save_dir = os.path.join(self.stage_dir, 'save_models')
        self.result_dir = os.path.join(self.stage_dir, 'result')
        # 转换后的 fasttext 语料(留档, 方便肉眼确认标签前缀有没有写对)
        self.corpus_dir = os.path.join(self.stage_dir, 'data')
        self.model_save_path = os.path.join(self.save_dir, 'fasttext_multilabel.bin')
        self.metrics_save_path = os.path.join(self.result_dir, 'metrics_fasttext.json')
        self.threshold_search_path = os.path.join(self.result_dir, 'threshold_search.json')

        # ==================== todo 2. 类别体系 ====================
        self.class_list = load_class_list(self.class_path)
        self.num_labels = len(self.class_list)
        self.label_prefix = LABEL_PREFIX

        # ==================== todo 3. 文本表示 ====================
        self.feature_mode = os.environ.get('FT_FEATURE_MODE', 'auto')   # auto / word / char
        self.lowercase = True

        # ==================== todo 4. 模型超参数 ====================
        self.dim = 100                 # 词向量维度(小数据 50~100 足够)
        self.epoch = 25                # 轮数; fasttext 迭代很快, 可以多跑几轮
        self.lr = 0.5                  # 学习率(ova 下的常用起点)
        self.word_ngrams = 2           # 用到 2-gram("快递 没" 这种搭配)
        self.bucket = 50000            # ngram/subword 哈希桶数; 模型体积 ≈ bucket × dim × 4 字节
        self.min_count = 1             # 词最少出现次数; 设 1 是因为工单里有大量低频专有词
        self.min_count_label = 1
        self.loss = 'ova'              # 多标签必选, 见 docstring
        self.thread = 8
        self.seed = 42
        self.verbose = 2

        # ==================== todo 5. 拒识阈值(与 02-rf / 04-bert 同口径) ====================
        self.label_threshold = float(os.environ.get('LABEL_THRESHOLD', 0.5))
        self.global_threshold = float(os.environ.get('GLOBAL_THRESHOLD', 0.8))

        # ==================== todo 6. 阈值标定 ====================
        self.tune_threshold = True
        self.target_reject_rate = 0.15

        # ==================== todo 7. 小规模超参搜索(可选) ====================
        self.enable_grid_search = False
        self.grid_search_space = [
            {'dim': 100, 'epoch': 25, 'lr': 0.5, 'wordNgrams': 2},
            {'dim': 200, 'epoch': 25, 'lr': 0.5, 'wordNgrams': 2},
            {'dim': 100, 'epoch': 40, 'lr': 0.3, 'wordNgrams': 3},
        ]

    # ------------------------------------------------------------------
    def resolve_feature_mode(self):
        """把 'auto' 解析成实际模式, 返回 (mode, tokenizer)"""
        from tools.text_tokenize import get_tokenizer
        return get_tokenizer(self.feature_mode)

    def subword_range(self, mode):
        """subword n-gram 范围

        词级: 关掉 subword(分词后词形已经很干净, 再切子串意义不大)
        字符级: 打开 2~4 元, 让模型自己学会"快递""退款"这类片段 —— 这是字符级能追上词级的关键
        """
        return (0, 0) if mode == 'word' else (2, 4)

    def check_files(self, need_model=False):
        problems = []
        for name, path in [('训练集', self.train_path), ('验证集', self.dev_path),
                           ('测试集', self.test_path), ('类别文件', self.class_path)]:
            if not os.path.exists(path):
                problems.append(f'{name}缺失: {path}')
        if need_model and not os.path.exists(self.model_save_path):
            problems.append(f'模型文件缺失: {self.model_save_path} (请先运行 ft_train.py)')
        try:
            import fasttext            # noqa: F401
        except ImportError:
            problems.append('未安装 fasttext, 请执行: pip install fasttext')
        return problems

    def summary(self):
        mode, _ = self.resolve_feature_mode()
        return chr(10).join([
            f'  项目根目录   : {self.root_path}',
            f'  标签体系     : {self.num_labels} 类 -> {", ".join(self.class_list)}',
            f'  文本表示     : mode={mode}(config={self.feature_mode}), subword={self.subword_range(mode)}, '
            f'lowercase={self.lowercase}',
            f'  模型         : FastText(loss={self.loss}, dim={self.dim}, epoch={self.epoch}, '
            f'lr={self.lr}, wordNgrams={self.word_ngrams}, bucket={self.bucket})',
            f'  拒识阈值     : 单标签 {self.label_threshold} / 全局 {self.global_threshold}',
            f'  阈值标定     : {self.tune_threshold} (目标拒识率 <= {self.target_reject_rate:.0%})',
            f'  模型保存     : {self.model_save_path}',
        ])


if __name__ == '__main__':
    cfg = FTConfig()
    print('=' * 72)
    print('ShopCare 03-fasttext 配置摘要')
    print('=' * 72)
    print(cfg.summary())
    problems = cfg.check_files(need_model=False)
    if problems:
        print(chr(10) + '[警告] 自检发现问题:')
        for p in problems:
            print('  - ' + p)
    else:
        print(chr(10) + '[OK] 文件自检通过')