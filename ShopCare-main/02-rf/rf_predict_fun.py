"""
对外唯一推理入口 (02-rf)

定位:
    backend 只通过本模块的 predict_fun() 调用随机森林模型, 不直接碰 joblib 文件.
    与 04-bert/bert_predict_fun.py 保持**完全相同的入参和返回结构**
    (由 tools/ticket_predictor.BaseTicketPredictor 统一保证), 所以 backend 可以
    无差别地把三套模型挂到同一个 /classify 接口上.

与 BERT 版的差异:
    * 不需要 GPU/分词器, 加载快(几十 MB, 毫秒级), 单条推理 1~5ms —— 这也是它能作为
      "兜底模型"的原因: 当 BERT 模型文件缺失或推理异常时, backend 可以自动降级到 RF;
    * 但它的泛化能力受限于 n-gram 字面匹配, 遇到"换一种说法"的同义表达就容易漏判,
      这正好是 BERT 要解决的问题, 也是两类模型对比的价值所在.

运行方式:
    python 02-rf/rf_predict_fun.py        # 交互式输入工单文本, 查看判定结果
"""

import json
import os
import sys

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_STAGE_DIR)
for _p in (_STAGE_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import RFConfig  # noqa: E402

from tools.ticket_data import DEMO_SAMPLES, load_class_list  # noqa: E402
from tools.ticket_predictor import BaseTicketPredictor  # noqa: E402


# ============================================================
# todo 1. 默认配置
# ============================================================
def default_config():
    """供 backend 在不想 import 本模块时拿到一份默认配置(标签表等)"""
    return RFConfig()


# ============================================================
# todo 2. 预测器
# ============================================================
class RFPredictor(BaseTicketPredictor):
    """随机森林预测器: 只负责"文本 -> 9 个概率", 其余业务逻辑全在基类"""

    model_name = "rf-tfidf"

    def __init__(self, cfg=None, enable_llm=True):
        cfg = cfg or RFConfig()
        super().__init__(cfg, enable_llm=enable_llm)
        self.vectorizer = None
        self.estimators = None
        self.constants = None
        self.bundle_meta = {}

    # ---------------- 模型加载 ----------------
    def load(self, force=False):
        """加载 joblib 模型包(懒加载; 文件缺失只记录原因, 不抛异常)"""
        if self._loaded and not force:
            return self
        import time

        start = time.time()
        try:
            import joblib

            if not os.path.exists(self.cfg.model_save_path):
                raise FileNotFoundError(
                    f"未找到模型文件 {self.cfg.model_save_path}, 请先运行: python 02-rf/rf_train.py"
                )
            bundle = joblib.load(self.cfg.model_save_path)

            # 校验标签顺序: 训练时的标签表与当前 class.txt 必须一致, 否则概率列会错位
            if list(bundle.get("class_list", [])) != list(self.cfg.class_list):
                raise ValueError(
                    "模型文件的标签表与 01-data/class.txt 不一致, 概率列会错位.\n"
                    f"  模型: {bundle.get('class_list')}\n  当前: {self.cfg.class_list}\n"
                    "  请重新训练: python 02-rf/rf_train.py"
                )

            self.vectorizer = bundle["vectorizer"]
            self.estimators = bundle["estimators"]
            self.constants = bundle["constants"]
            self.bundle_meta = {
                k: v
                for k, v in bundle.items()
                if k not in ("vectorizer", "estimators", "constants")
            }
            # 训练时标定出来的阈值优先于配置默认值(前端"模型选择"里展示的就是这个值)
            self.cfg.label_threshold = bundle.get(
                "label_threshold", self.cfg.label_threshold
            )
            self.cfg.global_threshold = bundle.get(
                "global_threshold", self.cfg.global_threshold
            )
            self._loaded = True
            self.load_error = None
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
        self.load_seconds = round(time.time() - start, 3)
        return self

    @property
    def status(self):
        base = super().status
        base.update(
            {
                "model_path": self.cfg.model_save_path,
                "model_exists": os.path.exists(self.cfg.model_save_path),
            }
        )
        return base

    def _status_extra(self):
        return {
            "model_path": getattr(self.cfg, "model_save_path", None),
            "model_exists": bool(
                getattr(self.cfg, "model_save_path", None)
                and os.path.exists(self.cfg.model_save_path)
            ),
            "feature_mode": self.bundle_meta.get("feature_mode"),
            "trained_at": self.bundle_meta.get("trained_at"),
            "num_labels": len(self.cfg.class_list) if self.cfg else None,
        }

    # ---------------- 核心推理 ----------------
    def _infer_proba(self, text):
        from tools.text_tokenize import clean_text

        X = self.vectorizer.transform([clean_text(text)])
        probs = []
        for clf, const in zip(self.estimators, self.constants):
            if clf is None:
                probs.append(float(const["prob"]))
            else:
                classes = list(clf.classes_)
                probs.append(float(clf.predict_proba(X)[0, classes.index(1)]))
        return probs

    def _infer_proba_batch(self, texts):
        """批量推理: 一次 transform 全部文本, 比逐条快得多(服务层批量接口用这个)"""
        import numpy as np

        from tools.text_tokenize import clean_text

        X = self.vectorizer.transform([clean_text(t) for t in texts])
        probs = np.zeros((len(texts), len(self.estimators)), dtype=float)
        for i, (clf, const) in enumerate(zip(self.estimators, self.constants)):
            if clf is None:
                probs[:, i] = const["prob"]
            else:
                classes = list(clf.classes_)
                probs[:, i] = clf.predict_proba(X)[:, classes.index(1)]
        return probs.tolist()


# ============================================================
# todo 3. 模块级单例入口(backend 与脚本都用这两个)
# ============================================================
_predictor = None


def get_predictor(cfg=None, enable_llm=True):
    """获取全局单例(多次调用只加载一次模型)"""
    global _predictor
    if _predictor is None:
        _predictor = RFPredictor(cfg, enable_llm=enable_llm)
    return _predictor


def predict_fun(text, cfg=None, use_llm_fallback=True, top_k=3):
    """便捷函数: 单条工单预测(backend 调用这一行即可)"""
    return get_predictor(cfg).predict(
        text, use_llm_fallback=use_llm_fallback, top_k=top_k
    )


# ============================================================
# todo 4. 交互式自测
# ============================================================
if __name__ == "__main__":
    cfg = RFConfig()
    print("=" * 72)
    print("ShopCare 02-rf 随机森林 —— 交互式推理")
    print("=" * 72)
    print(cfg.summary())

    if not os.path.exists(cfg.model_save_path):
        print(f"\n[警告] 未找到训练好的模型: {cfg.model_save_path}")
        print("请先运行: python 02-rf/rf_train.py")
        print("\n可用性检查(仅查看状态):")
        print(
            json.dumps(
                get_predictor(cfg, enable_llm=False).status,
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(0)

    predictor = get_predictor(cfg, enable_llm=True)
    predictor.load()
    if not predictor._loaded:
        print("[错误] 模型加载失败:", predictor.load_error)
        sys.exit(1)
    print(f"\n模型加载完成, 耗时 {predictor.load_seconds}s")
    print(
        "特征模式:",
        predictor.bundle_meta.get("feature_mode"),
        "| 训练时间:",
        predictor.bundle_meta.get("trained_at"),
    )
    print(f"阈值: 单标签 {cfg.label_threshold} / 全局 {cfg.global_threshold}")

    samples = DEMO_SAMPLES
    print("\n---- 内置样例(前三条同分布, 第 4 条为语料外新说法) ----")
    for text, gold in samples:
        r = predictor.predict(text)
        labels = ", ".join(f"{x['cn']}({x['score']:.2f})" for x in r["labels"]) or "无"
        print(f"\n  文本: {text}")
        if gold:
            print(f"  真实: {gold}")
        print(f"  预测: {labels}")
        print(
            f"  平均置信度: {r['avg_confidence']} | 判定: "
            f"{'拒识' if r['rejected'] else '自动分流'} | 处理方: {r['resolved_by']} "
            f"| 耗时: {r['latency_ms']}ms"
        )

    print("\n---- 交互输入(直接回车退出) ----")
    while True:
        try:
            text = input("\n请输入工单文本: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        print(json.dumps(predictor.predict(text), ensure_ascii=False, indent=2))
