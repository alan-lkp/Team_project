"""
ShopCare 分类主接口 (backend/api/classify_api.py)

路由:
    POST /classify         核心: 一条工单 -> 标签 + 情感 + 优先级 + 话术 + 是否复核
    POST /classify/batch   批量(上限 50 条), 默认不生成话术以省时间
    POST /sentiment        单独的情感极性分析
    POST /priority         单独的优先级判定
    POST /reply            单独的话术推荐

三个设计决定:

1) 为什么是同步 def 而不是 async def:
   模型推理是 CPU 密集的阻塞调用(torch / sklearn 推理期间不会让出 GIL)。
   写成 async def 会把事件循环**整个卡住**, 让所有并发请求一起变慢;
   写成普通 def, FastAPI 会把它丢进线程池, 天然隔离.

2) 缓存(设计文档 10.2 的 shopcare:cache:classify:*):
   key 里带上**所有影响结果的参数**(模型 / 两个阈值 / top_k), 只漏一个就会出现
   "改了阈值还读到旧结果" —— 这是缓存最经典的串味 bug.
   命中缓存时响应里带 cached=true, 前端可以明确告诉用户这是复用结果.

3) save=true 时落库为工单: 展示的数据与落库的数据来自同一次计算(pipeline 保证),
   不会出现"页面显示的标签"和"数据库里的标签"对不上的情况.
"""

import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.api.deps import api_error, current_user, get_pipeline
from backend.common.config import get_config
from backend.common.pipeline import ModelUnavailableError, TicketPipeline
from backend.common.shop_utils import (CODE_BAD_REQUEST, CODE_MODEL_UNAVAILABLE, load_label_meta, ok)
from backend.common.store import get_store

router = APIRouter(tags=['classify'])

# 分类结果缓存时长(秒). 演示场景下 60s 足够挡住"来回切模型反复点"造成的重复推理.
CLASSIFY_CACHE_TTL = 60


class ClassifyBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000, description='工单原文')
    model: str | None = Field(default=None,
                              description='rf / fasttext / bert / auto / llm; 留空用配置里的默认模型')
    use_llm_fallback: bool | None = Field(default=None, description='留空跟随 ENABLE_LLM_FALLBACK')
    top_k: int = Field(default=3, ge=1, le=9, description='top-k 候选标签个数')
    save: bool = Field(default=False, description='是否把这次结果落库为工单')


class BatchClassifyBody(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=50, description='一次最多 50 条')
    model: str | None = None
    use_llm_fallback: bool | None = None
    top_k: int = Field(default=3, ge=1, le=9)


class SentimentBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class PriorityBody(BaseModel):
    labels: list[str] = Field(default_factory=list, description='模型给出的标签')
    sentiment: str = Field(default='neutral', pattern='^(positive|neutral|negative)$')
    text: str = Field(default='', max_length=2000, description='原文, 用于提取升级信号')


class ReplyBody(BaseModel):
    labels: list[str] = Field(default_factory=list)
    priority: str = Field(default='P2', pattern='^P[012]$')
    sentiment: str = Field(default='neutral', pattern='^(positive|neutral|negative)$')
    text: str = Field(default='', max_length=2000)
    model: str | None = None


def _valid_labels(labels):
    """过滤掉体系外的标签: 脏输入不该把规则引擎带偏"""
    known = set(load_label_meta().get('labels', {}))
    return [name for name in (labels or []) if name in known]


@router.post('/classify', summary='工单分类(核心接口)')
def classify(body: ClassifyBody, user=Depends(current_user)):
    cfg = get_config()
    store = get_store()
    pipe = get_pipeline()

    # 只对"预览"(不落库)走缓存: 落库要生成新的工单号与时间戳, 命中缓存反而不合理
    model_arg = body.model or cfg.default_model
    cache_key = TicketPipeline.cache_key(body.text, model_arg,
                                         cfg.label_threshold, cfg.global_threshold, body.top_k,
                                         model_version=pipe.cache_version(model_arg))
    kv = store.kv()
    if not body.save:
        cached = kv.get_json(cache_key)
        if cached:
            cached = dict(cached)
            cached['cached'] = True
            return ok(cached, message='命中缓存(60 秒内同参数同文本)')

    try:
        result = pipe.analyze(body.text, model=body.model,
                              use_llm_fallback=body.use_llm_fallback, top_k=body.top_k)
    except ModelUnavailableError as exc:
        api_error(503, CODE_MODEL_UNAVAILABLE, str(exc))

    result['cached'] = False
    if body.save:
        record = pipe.to_ticket_record(result)
        store.create_ticket(record)
        result['ticket_id'] = record['ticket_id']
    else:
        kv.set_json(cache_key, result, ttl=CLASSIFY_CACHE_TTL)

    store.log_action(user.username, 'classify', target=result.get('ticket_id') or 'preview',
                     detail={'model': result.get('model_key'),
                             'labels': [item['label'] for item in result['labels']],
                             'rejected': result.get('rejected')})
    return ok(result, message='建单并分类完成' if body.save else '分类完成')


@router.post('/classify/batch', summary='批量分类(最多 50 条)')
def classify_batch(body: BatchClassifyBody, user=Depends(current_user)):
    pipe = get_pipeline()
    started = time.time()
    try:
        results = pipe.analyze_batch(body.texts, model=body.model,
                                     use_llm_fallback=body.use_llm_fallback,
                                     top_k=body.top_k, want_reply=False)
    except ModelUnavailableError as exc:
        api_error(503, CODE_MODEL_UNAVAILABLE, str(exc))

    get_store().log_action(user.username, 'classify_batch', target='batch',
                           detail={'count': len(results)})
    return ok({'count': len(results), 'items': results,
               'total_latency_ms': round((time.time() - started) * 1000, 2),
               'note': '批量默认不生成回复话术(单条可用 /reply 生成)'})


@router.post('/sentiment', summary='情感极性分析')
def sentiment(body: SentimentBody, user=Depends(current_user)):
    return ok(get_pipeline().sentiment_of(body.text))


@router.post('/priority', summary='优先级判定')
def priority(body: PriorityBody, user=Depends(current_user)):
    labels = _valid_labels(body.labels)
    result = get_pipeline().priority_of(labels, body.sentiment, body.text)
    result['labels_used'] = labels
    return ok(result)


@router.post('/reply', summary='回复话术推荐')
def reply(body: ReplyBody, user=Depends(current_user)):
    result = get_pipeline().reply_of(_valid_labels(body.labels), body.priority,
                                     body.sentiment, body.text, body.model)
    if result is None:
        api_error(503, CODE_MODEL_UNAVAILABLE, '话术模块(10-reply-recommend)不可用或执行失败')
    return ok(result)