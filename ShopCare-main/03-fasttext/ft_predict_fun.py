"""
对外唯一推理入口 (03-fasttext)

定位:
    backend 只通过本模块的 predict_fun() 调用 FastText 模型.
    返回结构与 02-rf / 04-bert **完全一致**(由 tools/ticket_predictor.BaseTicketPredictor 统一保证).

加载时的两个关键校验(都是踩过的坑):
    1. 标签表校验: 模型是拿旧版 class.txt 训的话, 概率列会整体错位, 而且**不会报错**,
       只会安静地判错。所以必须比对 class_list, 不一致就直接拒绝加载。
    2. 分词模式校验: 训练用词级、推理用字符级(或反过来), 输入分布直接漂移,
       指标会莫名其妙掉一大截。所以分词模式随模型一起存档(model_meta.json), 推理时照用。

运行方式:
    python 03-fasttext/ft_predict_fun.py     # 交互式输入工单文本, 查看判定结果
"""

import json
import os
import sys

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_STAGE_DIR)
for _p in (_STAGE_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import FTConfig                                              # noqa: E402
# 从训练脚本里复用分词函数: 训练/推理**必须**是同一份实现, 所以故意不做第二份拷贝
from ft_train import EMPTY_TOKEN, tokenize_for_fasttext                  # noqa: E402
from tools.ascii_path import readable                                    # noqa: E402
from tools.text_tokenize import get_tokenizer                            # noqa: E402
from tools.ticket_predictor import BaseTicketPredictor                   # noqa: E402
from tools.ticket_data import DEMO_SAMPLES, load_class_list               # noqa: E402


def default_config():
    return FTConfig()


class FTPredictor(BaseTicketPredictor):
    """FastText 预测器: 只负责"文本 -> 9 个概率", 其余业务逻辑全在基类"""

    model_name = 'fasttext'

    def __init__(self, cfg=None, enable_llm=True):
        cfg = cfg or FTConfig()
        super().__init__(cfg, enable_llm=enable_llm)
        self.model = None
        self.meta = {}
        self.tokenizer = None

    # ---------------- 模型加载 ----------------
    def load(self, force=False):
        if self._loaded and not force:
            return self
        import time
        start = time.time()
        try:
            import fasttext

            meta_path = os.path.join(self.cfg.result_dir, 'model_meta.json')
            if not os.path.exists(self.cfg.model_save_path):
                raise FileNotFoundError(
                    f'未找到模型文件 {self.cfg.model_save_path}, 请先运行: python 03-fasttext/ft_train.py')
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f'缺少推理契约文件 {meta_path}. '
                    'fasttext 的 .bin 里存不下分词模式和标签表, 必须和训练脚本一起产出, '
                    '请重新运行: python 03-fasttext/ft_train.py')

            with open(meta_path, encoding='utf-8') as f:
                self.meta = json.load(f)

            if list(self.meta.get('class_list', [])) != list(self.cfg.class_list):
                raise ValueError(
                    '模型元数据的标签表与 01-data/class.txt 不一致, 概率列会错位.\n'
                    f'  模型: {self.meta.get("class_list")}\n  当前: {self.cfg.class_list}\n'
                    '  请重新训练: python 03-fasttext/ft_train.py')

            # load_model 也是 C++ 开文件, 中文路径会报"cannot be opened for loading"
            # (而上一行的 os.path.exists 明明是 True) —— 桥接见 tools/ascii_path.py
            with readable(self.cfg.model_save_path) as model_path:
                self.model = fasttext.load_model(model_path)
            # 用训练时存档的分词模式, 而不是当前环境碰巧能用的模式
            _, self.tokenizer = get_tokenizer(self.meta.get('feature_mode', 'auto'))
            self.cfg.label_threshold = self.meta.get('label_threshold', self.cfg.label_threshold)
            self.cfg.global_threshold = self.meta.get('global_threshold', self.cfg.global_threshold)
            self._loaded = True
            self.load_error = None
        except Exception as exc:                            # noqa: BLE001
            self.load_error = f'{type(exc).__name__}: {exc}'
        self.load_seconds = round(time.time() - start, 3)
        return self

    @property
    def status(self):
        base = super().status
        base.update(self._status_extra())
        return base

    def _status_extra(self):
        return {
            'model_path': getattr(self.cfg, 'model_save_path', None),
            'model_exists': bool(getattr(self.cfg, 'model_save_path', None)
                                 and os.path.exists(self.cfg.model_save_path)),
            'feature_mode': self.meta.get('feature_mode'),
            'trained_at': self.meta.get('trained_at'),
            'num_labels': len(self.cfg.class_list) if self.cfg else None,
        }

    # ---------------- 核心推理 ----------------
    def _infer_proba(self, text):
        tokens = tokenize_for_fasttext(text, self.tokenizer)
        labels, scores = self.model.predict(' '.join(tokens), k=len(self.cfg.class_list))
        index = {c: i for i, c in enumerate(self.cfg.class_list)}
        probs = [0.0] * len(self.cfg.class_list)
        for lb, sc in zip(labels, scores):
            name = lb[len(self.cfg.label_prefix):] if lb.startswith(self.cfg.label_prefix) else lb
            if name in index:
                probs[index[name]] = float(sc)
        return probs

    def _infer_proba_batch(self, texts):
        return [self._infer_proba(t) for t in texts]


# ============================================================
# 模块级单例入口(backend 与脚本都用这两个)
# ============================================================
_predictor = None


def get_predictor(cfg=None, enable_llm=True):
    global _predictor
    if _predictor is None:
        _predictor = FTPredictor(cfg, enable_llm=enable_llm)
    return _predictor


def predict_fun(text, cfg=None, use_llm_fallback=True, top_k=3):
    """便捷函数: 单条工单预测(backend 调用这一行即可)"""
    return get_predictor(cfg).predict(text, use_llm_fallback=use_llm_fallback, top_k=top_k)


# ============================================================
# 交互式自测
# ============================================================
if __name__ == '__main__':
    cfg = FTConfig()
    print('=' * 72)
    print('ShopCare 03-fasttext —— 交互式推理')
    print('=' * 72)
    print(cfg.summary())

    if not os.path.exists(cfg.model_save_path):
        print('\n[警告] 未找到训练好的模型:', cfg.model_save_path)
        print('请先运行: python 03-fasttext/ft_train.py')
        print('\n可用性检查(仅查看状态):')
        print(json.dumps(get_predictor(cfg, enable_llm=False).status, ensure_ascii=False, indent=2))
        sys.exit(0)

    predictor = get_predictor(cfg, enable_llm=True)
    predictor.load()
    if not predictor._loaded:
        print('[错误] 模型加载失败:', predictor.load_error)
        sys.exit(1)
    print(f'\n模型加载完成, 耗时 {predictor.load_seconds}s')
    print('分词模式:', predictor.meta.get('feature_mode'),
          '| 训练时间:', predictor.meta.get('trained_at'))
    print(f'阈值: 单标签 {cfg.label_threshold} / 全局 {cfg.global_threshold}')

    print('\n---- 内置样例(前三条同分布, 第 4 条为语料外新说法) ----')
    for text, gold in DEMO_SAMPLES:
        r = predictor.predict(text)
        labels = ', '.join(f'{x["cn"]}({x["score"]:.2f})' for x in r['labels']) or '无'
        print(f'\n  文本: {text}')
        if gold:
            print(f'  真实: {gold}')
        print(f'  预测: {labels}')
        print(f'  平均置信度: {r["avg_confidence"]} | 判定: '
              f'{"拒识" if r["rejected"] else "自动分流"} | 处理方: {r["resolved_by"]} '
              f'| 耗时: {r["latency_ms"]}ms')

    print('\n---- 交互输入(直接回车退出) ----')
    while True:
        try:
            text = input('\n请输入工单文本: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        print(json.dumps(predictor.predict(text), ensure_ascii=False, indent=2))