"""
ShopCare 业务决策流水线 (backend/common/pipeline.py)

职责: 把"模型推理"与"业务规则"串成**一条**链路, 供所有入口复用.

    文本
      -> model_registry.predict()          9 个独立概率 -> 双阈值拒识 -> LLM 兜底(可选)
      -> 08-sentiment    情感极性(词典 + 规则, 支撑优先级)
      -> 09-priority     优先级 / SLA / 是否需要人工复核(规则加权, 不单独训模型)
      -> 10-reply-recommend  回复草稿 + 审批闸门
      -> 11-ticket-kg    标签关联提示 / 高风险组合
      -> 统一结果字典(接口直接透传, 落库也用这同一份数据)

为什么单独抽一层:
    /classify、/tickets、批量接口用的是同一条链. 如果每个入口各拼一遍, 迟早出现
    "页面上看到的"和"落库的"不一致 —— 这是最难查的一类 bug. 这里保证展示与落库
    来自**同一次计算**.

关于目录名:
    08-sentiment / 09-priority / 10-reply-recommend / 11-ticket-kg 带连字符, 不是合法的
    Python 包名, 只能按文件路径加载(不能用 import 语句). 这四个模块加载失败不会让接口
    500: 会退化成"中性情感 + 默认优先级 + 无话术", 并在结果的 reason 里注明 degraded.
    宁可少给信息, 也不要因为一个可选模块就让主链路挂掉.
"""

import importlib.util
import os
import sys
import time

from backend.common.config import get_config
from backend.common.model_registry import MODEL_DISPLAY, get_registry
from backend.common.shop_utils import dept_of, load_label_meta, mask_pii, new_ticket_id

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE_FILES = {
    'sentiment': ('08-sentiment', 'sentiment_classifier.py'),
    'priority': ('09-priority', 'priority_engine.py'),
    'reply': ('10-reply-recommend', 'reply_recommender.py'),
    'kg': ('11-ticket-kg', 'ticket_kg.py'),
    'llm': ('04-bert', 'llm_fallback.py'),
}

# 这几个阶段的模块只依赖标准库, 不需要把自己的目录加进 sys.path.
# 04-bert 尤其不能常驻 sys.path: 它下面有 config.py, 会和 02-rf / 03-fasttext 的同名模块串味.
NO_SYSPATH_STAGES = ('llm',)

# 结果字典的固定字段(便于 tools/verify_all_phases.py 做一致性校验, 也方便前端写死解析逻辑)
RESULT_KEYS = (
    'text', 'text_masked', 'model_key', 'model_used', 'model_cn', 'model_display',
    'fallback_from', 'labels', 'confidences', 'top_k_scores', 'avg_confidence',
    'rejected', 'reject_reason', 'reject_reason_cn', 'llm_fallback', 'llm_reason',
    'need_human_review', 'resolved_by', 'sentiment', 'priority', 'dept', 'sla_hours',
    'suggested_reply', 'kg', 'model_latency_ms', 'latency_ms', 'trace',
)


class ModelUnavailableError(RuntimeError):
    """三套模型全部不可用时抛出; 接口层转成 HTTP 503 + 业务码 50002"""


def load_stage_module(stage):
    """按文件路径加载业务模块; 目录或文件不存在时返回 None, 由调用方降级.

    这里把阶段目录 append 进 sys.path 并保留: 这几个目录里的模块名互不冲突,
    留着可以让模块内部再 import 自己的兄弟文件(比如知识图谱模块读取 .json 配置).
    """
    if stage not in STAGE_FILES:
        return None
    rel_dir, filename = STAGE_FILES[stage]
    abs_dir = os.path.join(PROJECT_ROOT, rel_dir)
    path = os.path.join(abs_dir, filename)
    if not os.path.exists(path):
        return None
    module_name = '_shopcare_stage_' + stage
    if stage not in NO_SYSPATH_STAGES and abs_dir not in sys.path:
        sys.path.append(abs_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:                       # noqa: BLE001  可选模块, 失败即降级
        sys.modules.pop(module_name, None)
        return None


class TicketPipeline:
    """一次工单分析的完整链路(进程内共用一个实例)"""

    def __init__(self, cfg=None):
        self.cfg = cfg or get_config()
        self.registry = get_registry(self.cfg)
        meta = load_label_meta()
        self.label_meta = meta.get('labels', {})
        self.priority_table = meta.get('priority', {})
        self._modules = {}

    # ============================================================
    # todo 1. 业务模块(懒加载 + 降级)
    # ============================================================
    def _stage(self, stage):
        if stage not in self._modules:
            self._modules[stage] = load_stage_module(stage)
        return self._modules[stage]

    def modules_status(self):
        """给 /health 用: 四个业务模块是否加载成功"""
        return {stage: self._stage(stage) is not None for stage in STAGE_FILES}

    # ---- 情感 ----
    def sentiment_of(self, text):
        module = self._stage('sentiment')
        if module is not None:
            try:
                return module.predict_sentiment(text)
            except Exception as exc:            # noqa: BLE001
                return self._sentiment_fallback('情感模块执行异常: %s' % exc)
        return self._sentiment_fallback('情感模块缺失(08-sentiment), 已按中性处理')

    @staticmethod
    def _sentiment_fallback(reason):
        return {'label': 'neutral', 'cn': '中性', 'score': 0.0, 'confidence': 0.0,
                'positive_hits': [], 'negative_hits': [], 'modifiers': [],
                'exclamations': 0, 'reason': reason, 'degraded': True}

    # ---- 优先级 ----
    def priority_of(self, labels, sentiment='neutral', text=''):
        module = self._stage('priority')
        if module is not None:
            try:
                return module.evaluate_priority(labels, sentiment, text)
            except Exception as exc:            # noqa: BLE001
                return self._priority_fallback(labels, '优先级模块执行异常: %s' % exc)
        return self._priority_fallback(labels, '优先级模块缺失(09-priority), 已按标签基础档处理')

    def _priority_fallback(self, labels, reason):
        """规则引擎缺失时的最小兜底: 取最急标签的 base_priority"""
        rank = {'P0': 0, 'P1': 1, 'P2': 2}
        best = 'P2'
        for name in labels or []:
            prio = (self.label_meta.get(name) or {}).get('base_priority', 'P2')
            if rank.get(prio, 3) < rank.get(best, 3):
                best = prio
        table = self.priority_table.get(best) or {}
        sla = table.get('sla_hours', 24)
        return {'priority': best, 'cn': table.get('cn', best), 'score': 0.0, 'sla_hours': sla,
                'due_hint': '建议 %s 小时内首次响应' % sla, 'base_score': 0.0,
                'base_priority_from': list(labels or []), 'sentiment': sentiment,
                'sentiment_weight': 1.0, 'escalations': [], 'label_count_bonus': 0.0,
                'need_manager': best == 'P0', 'reasons': ['降级: ' + reason], 'degraded': True}

    # ---- 话术 ----
    def reply_of(self, labels, priority='P2', sentiment='neutral', text='', model_used=None):
        module = self._stage('reply')
        if module is None:
            return None
        try:
            return module.suggest_reply(labels, priority, sentiment, text, model_used)
        except Exception:                       # noqa: BLE001
            return None

    # ---- 标签知识图谱 ----
    def kg_of(self, labels, text=''):
        module = self._stage('kg')
        if module is None:
            return None
        try:
            return module.analyze_ticket(labels, text)
        except Exception:                       # noqa: BLE001
            return None

    # ---- LLM 直连(第 4 档对照) ----
    def _enrich_labels(self, labels, confidences):
        """补中文名/部门/基础优先级 - 与 tools/ticket_predictor.py 的返回契约保持一致"""
        enriched = []
        for name in labels or []:
            meta = self.label_meta.get(name) or {}
            enriched.append({
                'label': name,
                'cn': meta.get('cn', name),
                'dept': meta.get('dept', '未分配'),
                'base_priority': meta.get('base_priority', 'P2'),
                'score': round(float(confidences.get(name, 0.0)), 4),
            })
        return sorted(enriched, key=lambda item: item['score'], reverse=True)

    def _llm_decision(self, text, top_k=3):
        """构造一份与本地模型同结构的决策字典, 后面的情感/优先级/话术链路因此可以完全复用"""
        started = time.time()
        module = self._stage('llm')
        if module is None:
            raise ModelUnavailableError('LLM 模块缺失(04-bert/llm_fallback.py)')
        parser = module.LLMFallback(sorted(self.label_meta.keys()))
        if not parser.available:
            raise ModelUnavailableError('LLM 直连不可用: %s' % (parser.disable_reason or '未配置'))
        parsed = parser.parse(text)
        latency = round((time.time() - started) * 1000, 2)
        if not parsed:
            # LLM 也没解析出合法标签 -> 直接落人工复核, 不拿脏数据冒充结果
            return {'text': text, 'model_used': 'llm', 'labels': [], 'confidences': {},
                    'avg_confidence': 0.0, 'rejected': True, 'reject_reason': 'llm_parse_failed',
                    'reject_reason_cn': 'LLM 未返回合法标签, 已转人工复核', 'llm_fallback': True,
                    'llm_reason': 'LLM 直连结果无法解析或标签非法', 'need_human_review': True,
                    'resolved_by': 'human', 'top_k_scores': [], 'latency_ms': latency}
        confidences = {name: parsed['confidence'] for name in parsed['labels']}
        enriched = self._enrich_labels(parsed['labels'], confidences)
        return {'text': text, 'model_used': 'llm', 'labels': enriched, 'confidences': confidences,
                'avg_confidence': round(float(parsed['confidence']), 4), 'rejected': False,
                'reject_reason': None, 'reject_reason_cn': None, 'llm_fallback': True,
                'llm_reason': parsed.get('reason'), 'need_human_review': False, 'resolved_by': 'llm',
                'top_k_scores': [{'label': item['label'], 'score': item['score']}
                                 for item in enriched[:top_k]],
                'latency_ms': latency}

    # ============================================================
    # todo 2. 主入口
    # ============================================================
    def analyze(self, text, model=None, use_llm_fallback=None, top_k=3, want_reply=True):
        """单条工单 -> 完整业务决策字典

        参数:
            text             : 工单原文
            model            : rf / fasttext / bert / None(用配置里的默认模型)
            use_llm_fallback : 覆盖配置里的 LLM 兜底开关(None = 跟随配置)
            top_k            : top-k 候选标签个数
            want_reply       : 是否生成回复草稿(批量接口可以关掉省时间)
        异常:
            ModelUnavailableError: 三套模型全部不可用
        """
        started = time.time()
        raw = text or ''
        stripped = raw.strip()
        trace = []

        if not stripped:
            result = self._empty_result(raw)
            result['trace'] = [{'step': '前置校验', 'detail': '文本为空, 未调用模型'}]
            result['latency_ms'] = round((time.time() - started) * 1000, 2)
            return result

        model_arg = (model or '').strip().lower() or None
        if model_arg == 'auto':
            # auto = 默认模型 + 打开 LLM 兜底. "自动"不是玄学调度:
            # 本地模型给得出结果就用, 拒识了才让 LLM 兜底, LLM 也失败就转人工.
            model_arg = None
            use_llm_fallback = True

        if model_arg == 'llm':
            # LLM 直连 = 第 4 档对照: 完全不使用本地模型
            decision = self._llm_decision(stripped, top_k)
            model_key, fallback_from = 'llm', None
        else:
            try:
                decision, model_key, fallback_from = self.registry.predict(
                    stripped, model=model_arg, use_llm_fallback=use_llm_fallback, top_k=top_k)
            except RuntimeError as exc:
                raise ModelUnavailableError(str(exc)) from exc

        model_latency = decision.get('latency_ms')
        labels = [item['label'] for item in (decision.get('labels') or [])]
        display = MODEL_DISPLAY.get(model_key) or {'name': 'llm-direct', 'cn': 'LLM 直连',
                                                 'desc': '不走本地模型, 直接调用大模型'}
        trace.append({'step': '模型推理',
                      'detail': '%s 激活 %d 个标签, 平均置信度 %s' % (
                          display.get('cn', model_key), len(labels), decision.get('avg_confidence'))})
        if fallback_from:
            trace.append({'step': '模型降级',
                          'detail': '请求的 %s 不可用, 实际使用 %s' % (fallback_from, model_key)})
        if decision.get('llm_fallback'):
            trace.append({'step': 'LLM 兜底', 'detail': decision.get('llm_reason') or '已触发兜底'})

        sentiment = self.sentiment_of(stripped)
        trace.append({'step': '情感分析',
                      'detail': '%s | %s' % (sentiment.get('cn'), sentiment.get('reason'))})

        priority = self.priority_of(labels, sentiment.get('label', 'neutral'), stripped)
        trace.append({'step': '优先级判定',
                      'detail': '%s / SLA %sh / 分值 %s' % (
                          priority.get('priority'), priority.get('sla_hours'), priority.get('score'))})

        dept = dept_of(labels)
        reply = None
        if want_reply:
            reply = self.reply_of(labels, priority.get('priority', 'P2'),
                                  sentiment.get('label', 'neutral'), stripped, model_key)
            if reply is not None and (decision.get('rejected') or decision.get('need_human_review')):
                # 模型自己都没把握的工单, 模板话术绝不能标成"可直接发送"
                if not reply.get('needs_approval'):
                    reply['needs_approval'] = True
                    reply['approval_reason'] = '模型结果被拒识或需人工复核, 回复草稿仅供客服参考'
            if reply is not None:
                trace.append({'step': '话术推荐',
                              'detail': '模板 %s | 需审批: %s' % (
                                  '/'.join(reply.get('template_ids') or []), reply.get('needs_approval'))})

        kg = self.kg_of(labels, stripped)
        if kg:
            trace.append({'step': '标签关联', 'detail': kg.get('summary', '')})

        result = dict(decision)
        result.update({
            'text': stripped,
            'text_masked': mask_pii(stripped),
            'model_key': model_key,
            'model_used': model_key,
            'model_cn': display.get('cn', model_key),
            'model_display': display.get('name', model_key),
            'fallback_from': fallback_from,
            'labels': decision.get('labels') or [],
            'sentiment': sentiment,
            'priority': priority,
            'dept': dept,
            'sla_hours': priority.get('sla_hours'),
            'suggested_reply': reply,
            'kg': kg,
            'model_latency_ms': model_latency,
            'latency_ms': round((time.time() - started) * 1000, 2),
            'trace': trace,
        })
        return result

    def analyze_batch(self, texts, model=None, use_llm_fallback=None, top_k=3, want_reply=False):
        """批量: 工单之间互相独立, 逐条走同一条链(保证与单条结果完全一致)"""
        return [self.analyze(t, model=model, use_llm_fallback=use_llm_fallback,
                             top_k=top_k, want_reply=want_reply) for t in (texts or [])]

    # ============================================================
    # todo 3. 落库 / 缓存辅助
    # ============================================================
    @staticmethod
    def to_ticket_record(result, ticket_id=None):
        """把分析结果压成 tickets 表的一行(labels/confidences 交给 store 序列化)"""
        reply = result.get('suggested_reply') or {}
        return {
            'ticket_id': ticket_id or new_ticket_id(),
            'text': result.get('text'),
            'text_masked': result.get('text_masked'),
            'model_used': result.get('model_used'),
            'labels': [item['label'] for item in (result.get('labels') or [])],
            'confidences': result.get('confidences') or {},
            'avg_confidence': result.get('avg_confidence'),
            'rejected': 1 if result.get('rejected') else 0,
            'reject_reason': result.get('reject_reason'),
            'sentiment': (result.get('sentiment') or {}).get('label'),
            'priority': (result.get('priority') or {}).get('priority'),
            'dept': result.get('dept'),
            'sla_hours': result.get('sla_hours'),
            'suggested_reply': reply.get('reply'),
            'needs_approval': 1 if reply.get('needs_approval') else 0,
            'resolved_by': result.get('resolved_by'),
            'latency_ms': result.get('latency_ms'),
            'status': 'pending',
        }

    @staticmethod
    def cache_key(text, model=None, label_threshold=None, global_threshold=None, top_k=3):
        """分类结果的缓存键: 文本 + 影响结果的全部参数都要进 key, 否则会串味"""
        import hashlib
        raw = '|'.join([str(text), str(model), str(label_threshold),
                        str(global_threshold), str(top_k)])
        return 'shopcare:cache:classify:' + hashlib.md5(raw.encode('utf-8')).hexdigest()

    @staticmethod
    def _empty_result(text):
        """空文本的完整结构(前端不用为空输入写特殊分支)"""
        return {
            'text': text, 'text_masked': text, 'model_key': None, 'model_used': None,
            'model_cn': None, 'model_display': None, 'fallback_from': None,
            'labels': [], 'confidences': {}, 'top_k_scores': [], 'avg_confidence': 0.0,
            'rejected': True, 'reject_reason': 'empty_text', 'reject_reason_cn': '文本为空',
            'llm_fallback': False, 'llm_reason': None, 'need_human_review': False,
            'resolved_by': 'none',
            'sentiment': {'label': 'neutral', 'cn': '中性', 'score': 0.0, 'confidence': 0.0,
                          'reason': '文本为空', 'positive_hits': [], 'negative_hits': []},
            'priority': {'priority': 'P2', 'cn': '常规', 'score': 0.0, 'sla_hours': 24,
                         'due_hint': '建议 24 小时内首次响应', 'reasons': ['文本为空']},
            'dept': '未分配', 'sla_hours': 24, 'suggested_reply': None, 'kg': None,
            'model_latency_ms': 0.0, 'latency_ms': 0.0, 'trace': [],
        }


_pipeline = None


def get_pipeline(cfg=None):
    global _pipeline
    if _pipeline is None:
        _pipeline = TicketPipeline(cfg)
    return _pipeline


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare pipeline 自测')
    print('=' * 72)
    pipe = get_pipeline()
    print('\n业务模块加载情况:', pipe.modules_status())

    print('\n[用例 1] 正常工单(fasttext, 最快且本地已训练)')
    r = pipe.analyze('快递到广州十天了还没动静, 客服也不回复, 我要退款', model='fasttext')
    print('  模型      :', r['model_key'], '/', r['model_cn'])
    print('  标签      :', [(x['label'], x['cn'], x['dept'], x['score']) for x in r['labels']])
    print('  拒识      :', r['rejected'], '|', r['reject_reason'])
    print('  情感      :', r['sentiment']['label'], r['sentiment']['score'])
    print('  优先级    :', r['priority']['priority'], r['priority']['cn'], 'SLA', r['sla_hours'])
    print('  部门      :', r['dept'])
    print('  话术审批  :', (r['suggested_reply'] or {}).get('needs_approval'))
    print('  耗时      : 模型 %.2fms / 全链路 %.2fms' % (r['model_latency_ms'], r['latency_ms']))
    print('  决策链路  :')
    for item in r['trace']:
        print('    -', item['step'], ':', item['detail'])
    missing = set(RESULT_KEYS) - set(r.keys())
    assert not missing, '结果缺字段: %s' % missing
    assert r['model_used'] == 'fasttext'
    assert r['sla_hours'] and r['dept'] != '未分配'
    assert len(r['trace']) >= 4

    print('\n[用例 2] 空文本(结构必须完整, 不抛异常)')
    r2 = pipe.analyze('   ')
    print('  ', r2['reject_reason_cn'], '| resolved_by =', r2['resolved_by'])
    assert r2['rejected'] is True and r2['resolved_by'] == 'none'
    assert not (set(RESULT_KEYS) - set(r2.keys()))

    print('\n[用例 3] 落库记录(压平成一行)')
    rec = pipe.to_ticket_record(r)
    print('  ', {k: rec[k] for k in ('ticket_id', 'model_used', 'labels', 'priority', 'dept',
                                      'sla_hours', 'sentiment', 'needs_approval')})
    assert isinstance(rec['labels'], list) and rec['labels']
    assert rec['ticket_id'].startswith('SC')

    print('\n[用例 4] 缓存键: 文本或参数变了就必须变')
    k1 = pipe.cache_key('abc', 'rf')
    k2 = pipe.cache_key('abc', 'fasttext')
    k3 = pipe.cache_key('abc', 'rf')
    print('  ', k1[-12:], k2[-12:], k3[-12:])
    assert k1 == k3 and k1 != k2

    print('\n[用例 5] 批量接口(关闭话术以省时间)')
    batch = pipe.analyze_batch(['收到货就是坏的', '发票开错了能重开吗'], model='fasttext')
    print('  ', [[x['label'] for x in item['labels']] for item in batch])
    assert len(batch) == 2

    print('\n[OK] pipeline 自测通过')