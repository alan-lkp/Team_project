"""
ShopCare 工单预测器公共基类 (tools)

为什么需要这个文件:
    项目里有三套对照模型(02-rf / 03-fasttext / 04-bert), backend 要对它们**一视同仁**地
    调用: 同样的入参、同样的出参字典。如果三套各自拼一遍返回结构, 迟早会有人少返回一个
    `reject_reason` 或者把 `score` 写成 `confidence`, 前端就得写三套兼容逻辑。
    所以把"业务决策"这一层统一抽到这里, 子类**只需要实现一个方法**:

        _infer_proba(text) -> [p0, p1, ..., p8]

    剩下的双阈值拒识、LLM 兜底、标签中文名/部门补全、top-k 展示、耗时统计, 全部由基类完成。

一次预测的完整链路(三套模型完全一致):
    文本 -> 模型推理得到 9 个独立概率
         -> 双阈值拒识判定(tools/ml_metrics.decide)
             ├─ 通过 -> 输出多标签结果            resolved_by='model'
             └─ 拒识 -> LLM 兜底(开关 + 可用性都满足时)
                          ├─ 成功 -> 输出结果       resolved_by='llm'
                          └─ 失败 -> 转人工复核     resolved_by='human'

predict() 的返回结构(backend 的 /classify 接口直接透传):
    {
      text, model_used, labels:[{label, cn, dept, base_priority, score}],
      confidences:{label: score}, avg_confidence, rejected, reject_reason,
      reject_reason_cn, llm_fallback, llm_reason, need_human_review,
      resolved_by, top_k_scores:[{label, score}], latency_ms
    }

子类约定:
    * 模型**懒加载**: 只有第一次 predict 才真正读文件, 保证 backend 启动快、模型缺失也不崩;
    * 模型文件缺失时 load_error 记下原因, predict 抛 RuntimeError, 由 backend 决定是否降级到别的模型。
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 复用 04-bert 下的 LLM 兜底实现(纯标准库, 无第三方依赖), 避免同一份逻辑存两份
_BERT_DIR = os.path.join(PROJECT_ROOT, '04-bert')
if _BERT_DIR not in sys.path:
    sys.path.insert(0, _BERT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm_fallback import LLMFallback                      # noqa: E402
from tools.ml_metrics import decide, should_use_llm        # noqa: E402
from tools.ticket_data import load_label_meta              # noqa: E402

__all__ = ['BaseTicketPredictor']

# predict() 的返回字段(顺序即前端展示顺序); 用于 tools/verify_all_phases.py 做一致性校验
RESULT_KEYS = (
    'text', 'model_used', 'labels', 'confidences', 'avg_confidence', 'rejected',
    'reject_reason', 'reject_reason_cn', 'llm_fallback', 'llm_reason',
    'need_human_review', 'resolved_by', 'top_k_scores', 'latency_ms',
)


class BaseTicketPredictor:
    """三套对照模型共用的业务决策基类"""

    model_name = 'base'

    def __init__(self, cfg=None, enable_llm=True):
        self.cfg = cfg
        self.enable_llm = enable_llm
        # label_meta.json 是标签元数据的唯一事实来源(cn / dept / base_priority)
        self.label_meta = load_label_meta().get('labels', {})
        self.llm = None
        self._loaded = False
        self.load_error = None
        self.load_seconds = None

    # ============================================================
    # todo 1. 子类需要实现的接口
    # ============================================================
    def load(self, force=False):
        """加载模型(子类实现); 重复调用应当只加载一次"""
        raise NotImplementedError

    def _infer_proba(self, text):
        """单条文本 -> 长度 = 标签数的独立概率列表(子类实现)"""
        raise NotImplementedError

    def _infer_proba_batch(self, texts):
        """批量推理; 默认逐条循环, 有原生批量能力的子类可以覆盖"""
        return [self._infer_proba(t) for t in texts]

    def _status_extra(self):
        """子类补充自己的状态字段(模型路径、特征维度等)"""
        return {}

    # ============================================================
    # todo 2. 状态(供 /health 与 /models 接口使用)
    # ============================================================
    @property
    def status(self):
        base = {
            'model': self.model_name,
            'loaded': self._loaded,
            'load_error': self.load_error,
            'load_seconds': self.load_seconds,
            'label_threshold': self.cfg.label_threshold if self.cfg else None,
            'global_threshold': self.cfg.global_threshold if self.cfg else None,
            'llm_fallback_available': bool(self.llm and self.llm.available),
            'llm_disabled_reason': (self.llm.disable_reason if self.llm else 'LLM 兜底未初始化'),
        }
        base.update(self._status_extra())
        return base

    def _ensure_llm(self):
        """LLM 兜底客户端只在需要时建一次(没配 API Key 时它会自己标记为不可用)"""
        if self.enable_llm and self.llm is None:
            classes = self.cfg.class_list if self.cfg else None
            self.llm = LLMFallback(classes)

    def _llm_usable(self):
        return bool(self.enable_llm and self.llm and self.llm.available)

    # ============================================================
    # todo 3. 一次完整预测(基类统一实现, 子类不要覆盖)
    # ============================================================
    def predict(self, text, use_llm_fallback=True, top_k=3):
        """对一条工单做完整业务判定

        参数:
            text             : 工单文本
            use_llm_fallback : 本次调用是否允许 LLM 兜底(前端有开关, 关掉可省 API 成本)
            top_k            : 返回概率最高的前 k 个标签(含未激活的), 让前端能看到"模型在犹豫什么"
        返回: 见模块 docstring 的返回结构
        """
        text = (text or '').strip()
        start = time.time()

        if not text:
            return self._empty_result(text)

        if not self._loaded:
            self.load()
        if not self._loaded:
            # 模型不可用就明确报错, 不假装成功 —— backend 会据此降级到其它模型
            raise RuntimeError(f'{self.model_name} 模型不可用: {self.load_error}')

        probs = self._infer_proba(text)
        decision = decide(probs, self.cfg.class_list,
                          label_threshold=self.cfg.label_threshold,
                          global_threshold=self.cfg.global_threshold)

        result = {
            'text': text,
            'model_used': self.model_name,
            'labels': self._enrich(decision['labels'], decision['confidences']),
            'confidences': decision['confidences'],
            'avg_confidence': decision['avg_confidence'],
            'rejected': decision['rejected'],
            'reject_reason': decision['reject_reason'],
            'reject_reason_cn': decision['reject_reason_cn'],
            'llm_fallback': False,
            'llm_reason': None,
            'need_human_review': decision['rejected'],
            'resolved_by': 'human' if decision['rejected'] else 'model',
            'top_k_scores': self._top_k(decision['all_scores'], top_k),
            'latency_ms': 0.0,
        }

        self._ensure_llm()
        if should_use_llm(decision, bool(use_llm_fallback and self._llm_usable())):
            llm_result = self.llm.parse(text)
            result['llm_fallback'] = True
            if llm_result:
                result['labels'] = self._enrich(
                    llm_result['labels'],
                    {lb: llm_result['confidence'] for lb in llm_result['labels']})
                result['confidences'] = {lb: llm_result['confidence'] for lb in llm_result['labels']}
                result['avg_confidence'] = llm_result['confidence']
                result['llm_reason'] = llm_result['reason']
                result['need_human_review'] = False
                result['resolved_by'] = 'llm'
            else:
                result['llm_reason'] = 'LLM 兜底解析失败, 已转人工复核'
                result['need_human_review'] = True
                result['resolved_by'] = 'human'

        result['latency_ms'] = round((time.time() - start) * 1000, 2)
        return result

    def predict_batch(self, texts, use_llm_fallback=True, top_k=3):
        """批量预测(逐条; 便于 backend 的批量接口直接复用)"""
        return [self.predict(t, use_llm_fallback=use_llm_fallback, top_k=top_k) for t in texts]

    # ============================================================
    # todo 4. 内部工具
    # ============================================================
    def _empty_result(self, text):
        return {
            'text': text, 'model_used': self.model_name, 'labels': [],
            'confidences': {}, 'avg_confidence': 0.0,
            'rejected': True, 'reject_reason': 'empty_text', 'reject_reason_cn': '文本为空',
            'llm_fallback': False, 'llm_reason': None,
            'need_human_review': False, 'resolved_by': 'none',
            'top_k_scores': [], 'latency_ms': 0.0,
        }

    def _enrich(self, labels, confidences):
        """给标签补中文名与责任部门(前端展示和按部门分流都要用)"""
        enriched = []
        for name in labels:
            meta = self.label_meta.get(name, {})
            enriched.append({
                'label': name,
                'cn': meta.get('cn', name),
                'dept': meta.get('dept', '未分配'),
                'base_priority': meta.get('base_priority', 'P2'),
                'score': round(float(confidences.get(name, 0.0)), 4),
            })
        return sorted(enriched, key=lambda x: x['score'], reverse=True)

    @staticmethod
    def _top_k(all_scores, k):
        items = sorted(all_scores.items(), key=lambda t: t[1], reverse=True)
        return [{'label': name, 'score': score} for name, score in items[:k]]


if __name__ == '__main__':
    # 用一个"假模型"验证基类的完整链路(不需要任何模型文件)
    class FakePredictor(BaseTicketPredictor):
        model_name = 'fake'

        def load(self, force=False):
            self._loaded = True
            self.load_error = None
            self.load_seconds = 0.0

        def _infer_proba(self, text):
            # 文本里出现"快递"就给 logistics 高分, 否则全部低分(触发无标签拒识)
            if '快递' in text:
                return [0.93, 0.05, 0.04, 0.03, 0.02, 0.02, 0.10, 0.08, 0.01]
            return [0.20, 0.10, 0.05, 0.03, 0.02, 0.02, 0.11, 0.08, 0.01]

    class _Cfg:
        from tools.ticket_data import load_class_list as _lcl
        class_list = _lcl()
        label_threshold = 0.5
        global_threshold = 0.8
        model_save_path = '(fake)'

    p = FakePredictor(_Cfg(), enable_llm=False)
    print('=' * 72)
    print('ticket_predictor 基类自检(假模型)')
    print('=' * 72)

    r1 = p.predict('快递到广州十天了还没动静')
    print('\n[正常激活]', r1['text'])
    print('  labels      :', [(x['label'], x['cn'], x['dept'], x['score']) for x in r1['labels']])
    print('  平均置信度  :', r1['avg_confidence'], '| 拒识:', r1['rejected'], '| 处理方:', r1['resolved_by'])
    assert r1['labels'][0]['label'] == 'logistics' and r1['rejected'] is False
    assert r1['labels'][0]['cn'] == '物流配送' and r1['labels'][0]['dept'] == '物流仓储部'
    assert set(r1.keys()) == set(RESULT_KEYS), set(r1.keys()) ^ set(RESULT_KEYS)

    r2 = p.predict('aaaaaaaaaa')
    print('\n[无标签拒识]', r2['text'])
    print('  拒识原因    :', r2['reject_reason'], '->', r2['reject_reason_cn'])
    assert r2['rejected'] is True and r2['reject_reason'] == 'no_label_activated'

    r3 = p.predict('   ')
    print('\n[空文本]', repr(r3['text']), '->', r3['reject_reason_cn'])
    assert r3['resolved_by'] == 'none'

    r4 = p.predict('快递没收到')
    print('\n[top-k 展示]', r4['top_k_scores'])
    assert len(r4['top_k_scores']) == 3

    print(f'\n[耗时] latency_ms={r1["latency_ms"]}')
    print('\n[OK] ticket_predictor 自检通过')