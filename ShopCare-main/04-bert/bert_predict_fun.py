"""
对外唯一推理入口 (04-bert)

定位:
    backend(FastAPI)只通过本模块的 predict_fun() 调用 BERT 模型, 不直接碰模型权重.
    好处: 模型层可以被单独训练/评估/替换, 不会和 Web 框架耦合
    (这也是参考项目 MindCare 里 predict_fun 被 backend 复用的做法).

完整推理链路(一条工单进, 一个业务决策出):
    文本 -> tokenizer -> BERT+LoRA 前向 -> 9 个标签概率(独立 sigmoid)
         -> 双阈值拒识判定(reject_utils)
             ├─ 通过     -> 直接输出多标签结果(标 resolved_by='model')
             └─ 拒识     -> 若开启 LLM 兜底且可用 -> llm_fallback 解析
                              ├─ 成功 -> 输出结果(标 resolved_by='llm')
                              └─ 失败 -> 转人工复核(标 resolved_by='human')

返回结构里 labels 的每一项都带 (label, cn, score, dept), 前端可以直接按部门分组展示.

运行方式:
    python 04-bert/bert_predict_fun.py        # 交互式输入工单文本, 查看判定结果
"""

import json
import os
import sys
import time

# 让脚本无论从哪个目录启动都能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from reject_utils import decide, should_use_llm
from llm_fallback import LLMFallback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(PROJECT_ROOT, '01-data', 'label_meta.json')

# the three inference scripts share one demo set (see tools/ticket_data.py)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from tools.ticket_data import DEMO_SAMPLES        # noqa: E402


def load_label_meta(path=None):
    """读取标签元数据(中文名 / 责任部门 / 基础优先级), 缺失时降级为空表"""
    path = path or META_PATH
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f).get('labels', {})


# ============================================================
# todo 1. 预测器(模型懒加载 + 单例复用)
# ============================================================
class TicketPredictor:
    """工单分类预测器: 封装 tokenizer / 模型 / 拒识 / LLM 兜底

    参数:
        cfg         : Config 实例(不传则新建)
        enable_llm  : 是否允许 LLM 兜底(逐次调用还可用 use_llm_fallback 覆盖)
    说明:
        模型采用**懒加载**: 只有第一次调用 predict() 时才真正加载权重,
        这样 backend 启动就很快, 且模型文件缺失也不会导致服务起不来(只影响该模型可用性).
    """

    def __init__(self, cfg=None, enable_llm=True):
        self.cfg = cfg or Config()
        self.enable_llm = enable_llm
        self.label_meta = load_label_meta()

        self.tokenizer = None
        self.model = None
        self.llm = None
        self._loaded = False
        self.load_error = None
        self.load_seconds = None

    # ---------------- 模型加载 ----------------
    def load(self, force=False):
        """加载 tokenizer + 模型 + LLM 兜底客户端(重复调用只加载一次)"""
        if self._loaded and not force:
            return self
        start = time.time()
        try:
            import torch                                   # noqa: F401  (确认 torch 可用)
            from dataloader_utils import load_tokenizer
            from multilabel_model import load_trained_model

            self.tokenizer = load_tokenizer(self.cfg)
            self.model, self.meta = load_trained_model(self.cfg)
            self._loaded = True
            self.load_error = None
        except SystemExit as exc:                          # 模型缺失时给出明确原因
            self.load_error = str(exc)
        except Exception as exc:                           # noqa: BLE001
            self.load_error = f'{type(exc).__name__}: {exc}'

        if self.enable_llm and self.llm is None:
            self.llm = LLMFallback(self.cfg.class_list)
        self.load_seconds = round(time.time() - start, 2)
        return self

    @property
    def status(self):
        """返回模型/兜底可用状态(供 /health 与 /models 接口使用)"""
        return {
            'model': 'bert',
            'loaded': self._loaded,
            'load_error': self.load_error,
            'load_seconds': self.load_seconds,
            'model_path': self.cfg.model_save_path,
            'model_exists': os.path.exists(self.cfg.model_save_path),
            'llm_fallback_available': bool(self.llm and self.llm.available),
            'llm_disabled_reason': (self.llm.disable_reason if self.llm else 'LLM 兜底未初始化'),
            'label_threshold': self.cfg.label_threshold,
            'global_threshold': self.cfg.global_threshold,
        }

    # ---------------- 核心预测 ----------------
    def predict(self, text, use_llm_fallback=True, top_k=3):
        """对一条工单做完整判定

        参数:
            text             : 工单文本
            use_llm_fallback : 本次调用是否允许 LLM 兜底(前端有开关)
            top_k            : 返回概率最高的前 k 个标签(便于前端展示"模型在犹豫什么")
        返回: 业务决策字典(见模块 docstring)
        """
        text = (text or '').strip()
        start = time.time()
        if not text:
            return {'text': text, 'model_used': 'bert', 'labels': [], 'rejected': True,
                    'reject_reason': 'empty_text', 'reject_reason_cn': '文本为空',
                    'llm_fallback': False, 'need_human_review': False,
                    'resolved_by': 'none', 'avg_confidence': 0.0,
                    'confidences': {}, 'all_scores': {}, 'latency_ms': 0.0}

        if not self._loaded:
            self.load()
        if not self._loaded:
            # 模型不可用: 明确报错, 不假装成功(由 backend 决定是否降级到其它模型)
            raise RuntimeError(f'BERT 模型不可用: {self.load_error}')

        import torch
        # 1. 分词 -> 张量 -> 设备
        encoded = self.tokenizer(text, max_length=self.cfg.max_len, padding='max_length',
                                 truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.cfg.device)
        attention_mask = encoded['attention_mask'].to(self.cfg.device)
        token_type_ids = encoded.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.cfg.device)

        # 2. 逐标签独立 sigmoid 概率
        with torch.no_grad():
            probs = self.model.predict_proba(input_ids, attention_mask, token_type_ids)[0].cpu()

        # 3. 双阈值拒识判定
        decision = decide(probs, self.cfg.class_list,
                          label_threshold=self.cfg.label_threshold,
                          global_threshold=self.cfg.global_threshold)

        result = {
            'text': text,
            'model_used': 'bert',
            'labels': self._enrich(decision['labels'], decision['confidences']),
            'confidences': decision['confidences'],
            'avg_confidence': decision['avg_confidence'],
            'rejected': decision['rejected'],
            'reject_reason': decision['reject_reason'],
            'reject_reason_cn': decision['reject_reason_cn'],
            'llm_fallback': False,
            'llm_reason': None,
            'need_human_review': decision['rejected'],
            'resolved_by': 'model' if not decision['rejected'] else 'human',
            'top_k_scores': self._top_k(decision['all_scores'], top_k),
            'latency_ms': 0.0,
        }

        # 4. 拒识 + 兜底开关 -> LLM 兜底(失败则保持人工复核)
        llm_enabled = bool(use_llm_fallback and self.enable_llm and self.llm and self.llm.available)
        if should_use_llm(decision, llm_enabled):
            llm_result = self.llm.parse(text)
            if llm_result:
                result['labels'] = self._enrich(llm_result['labels'],
                                                {lb: llm_result['confidence'] for lb in llm_result['labels']})
                result['confidences'] = {lb: llm_result['confidence'] for lb in llm_result['labels']}
                result['avg_confidence'] = llm_result['confidence']
                result['llm_fallback'] = True
                result['llm_reason'] = llm_result['reason']
                result['need_human_review'] = False
                result['resolved_by'] = 'llm'
            else:
                result['llm_fallback'] = True          # 尝试过兜底但失败
                result['llm_reason'] = 'LLM 兜底解析失败, 已转人工复核'
                result['need_human_review'] = True
                result['resolved_by'] = 'human'

        result['latency_ms'] = round((time.time() - start) * 1000, 2)
        return result

    def predict_batch(self, texts, use_llm_fallback=True, top_k=3):
        """批量预测(逐条循环; 批量推理可后续按需优化)"""
        return [self.predict(t, use_llm_fallback=use_llm_fallback, top_k=top_k) for t in texts]

    # ---------------- 内部工具 ----------------
    def _enrich(self, labels, confidences):
        """给标签补上中文名与责任部门(前端展示与业务分流都要用)"""
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
        """取概率最高的 k 个标签(含未激活的), 让前端能展示"模型在犹豫什么" """
        items = sorted(all_scores.items(), key=lambda t: t[1], reverse=True)
        return [{'label': name, 'score': score} for name, score in items[:k]]


# ============================================================
# todo 2. 模块级单例入口(backend 与脚本都用这两个)
# ============================================================
_predictor = None


def get_predictor(cfg=None, enable_llm=True):
    """获取全局预测器单例(避免每次请求都重新加载模型)"""
    global _predictor
    if _predictor is None:
        _predictor = TicketPredictor(cfg, enable_llm=enable_llm)
    return _predictor


def predict_fun(text, cfg=None, use_llm_fallback=True, top_k=3):
    """便捷函数: 单条工单预测(backend 调用这一行即可)"""
    return get_predictor(cfg).predict(text, use_llm_fallback=use_llm_fallback, top_k=top_k)


# ============================================================
# todo 3. 交互式自测
# ============================================================
if __name__ == '__main__':
    cfg = Config()
    problems = cfg.check_files(need_model=True)
    if not os.path.exists(cfg.model_save_path):
        problems.append(f'未找到训练好的模型: {cfg.model_save_path} (请先运行 train_bert.py)')

    print('=' * 72)
    print('ShopCare BERT+LoRA 工单分类 —— 交互式推理')
    print('=' * 72)
    print(cfg.summary())

    if problems:
        print('\n[警告] 以下问题会导致无法推理:')
        for p in problems:
            print('  - ' + p)
        print('\n可用性检查(仅查看状态):')
        print(get_predictor(cfg, enable_llm=False).status)
        sys.exit(0)

    print('\n正在加载模型...')
    predictor = get_predictor(cfg, enable_llm=True)
    print(f'加载完成, 耗时 {predictor.load_seconds}s')

    print('\n---- 内置样例(前三条同分布, 第 4 条为语料外新说法) ----')
    for text, gold in DEMO_SAMPLES:
        r = predictor.predict(text)
        labels = ', '.join(f'{x["cn"]}({x["score"]:.2f})' for x in r['labels']) or '无'
        print(f'\n  文本: {text}')
        if gold:
            print(f'  真实: {gold}')
        print(f'  预测: {labels}')
        print(f'  平均置信度: {r["avg_confidence"]} | 判定: {"拒识" if r["rejected"] else "自动分流"}'
              f' | 处理方: {r["resolved_by"]} | 耗时: {r["latency_ms"]}ms')

    print('\n---- 交互输入(直接回车退出) ----')
    while True:
        try:
            text = input('\n请输入工单文本: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        r = predictor.predict(text)
        print(json.dumps(r, ensure_ascii=False, indent=2))