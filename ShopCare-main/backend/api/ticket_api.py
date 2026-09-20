"""
ShopCare 工单与复核接口 (backend/api/ticket_api.py)

路由:
    GET   /tickets                工单分页查询(状态/优先级/部门/模型/是否拒识/关键词)
    GET   /tickets/{ticket_id}    工单详情
    POST  /tickets                建单: 文本 -> 分类 -> 落库(与 /classify?save=true 等价)
    PATCH /tickets/{ticket_id}    更新工单(只允许改白名单字段)
    POST  /feedback               人工复核回写(认可 / 修正标签 / 确认无效)
    GET   /reviews                复核记录

关于"人工复核回写"的意义:
    它不只是把工单从队列里移走. 修正后的标签是**唯一可信的监督信号**来源 ——
    真实场景下"模型判错但人工改对"的样本, 正是下一轮难例采样与阈值调整的输入.
    所以这里把修正标签原样存进 reviews 表, 而不是只改 tickets 的状态.
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from backend.api.deps import api_error, current_user, get_pipeline
from backend.common.pipeline import ModelUnavailableError
from backend.common.shop_utils import (CODE_BAD_REQUEST, CODE_MODEL_UNAVAILABLE,
                                       CODE_NOT_FOUND, load_label_meta, ok)
from backend.common.store import get_store

router = APIRouter(tags=['ticket'])

ALLOWED_STATUS = ('pending', 'reviewed', 'closed')
ALLOWED_PRIORITY = ('P0', 'P1', 'P2')
ALLOWED_ACTIONS = ('approve', 'correct', 'reject')


class TicketCreateBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    model: str | None = None
    use_llm_fallback: bool | None = None


class TicketPatchBody(BaseModel):
    status: str | None = Field(default=None, description='pending / reviewed / closed')
    priority: str | None = Field(default=None, description='P0 / P1 / P2')
    dept: str | None = Field(default=None, max_length=32)


class FeedbackBody(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=32)
    action: str = Field(default='correct', description='approve 认可 / correct 修正 / reject 确认无效')
    corrected_labels: list[str] = Field(default_factory=list)
    note: str | None = Field(default=None, max_length=500)


@router.get('/tickets', summary='工单列表')
def list_tickets(page: int = Query(1, ge=1),
                 size: int = Query(20, ge=1, le=100),
                 status: str | None = None,
                 priority: str | None = None,
                 dept: str | None = None,
                 model_used: str | None = None,
                 rejected: bool | None = None,
                 keyword: str | None = None,
                 user=Depends(current_user)):
    data = get_store().list_tickets(page=page, size=size, status=status, priority=priority,
                                    dept=dept, model_used=model_used, rejected=rejected,
                                    keyword=keyword)
    return ok(data)


@router.get('/tickets/{ticket_id}', summary='工单详情')
def get_ticket(ticket_id: str, user=Depends(current_user)):
    row = get_store().get_ticket(ticket_id)
    if not row:
        api_error(404, CODE_NOT_FOUND, '工单不存在: %s' % ticket_id)
    return ok(row)


@router.post('/tickets', summary='建单(文本 -> 分类 -> 落库)')
def create_ticket(body: TicketCreateBody, user=Depends(current_user)):
    store = get_store()
    pipe = get_pipeline()
    try:
        result = pipe.analyze(body.text, model=body.model,
                              use_llm_fallback=body.use_llm_fallback)
    except ModelUnavailableError as exc:
        api_error(503, CODE_MODEL_UNAVAILABLE, str(exc))

    record = pipe.to_ticket_record(result)
    store.create_ticket(record)
    result['ticket_id'] = record['ticket_id']
    store.log_action(user.username, 'ticket_create', target=record['ticket_id'],
                     detail={'model': record['model_used'], 'priority': record['priority']})
    return ok(result, message='建单成功')


@router.patch('/tickets/{ticket_id}', summary='更新工单')
def patch_ticket(ticket_id: str, body: TicketPatchBody, user=Depends(current_user)):
    store = get_store()
    if not store.get_ticket(ticket_id):
        api_error(404, CODE_NOT_FOUND, '工单不存在: %s' % ticket_id)

    fields = {}
    if body.status is not None:
        if body.status not in ALLOWED_STATUS:
            api_error(400, CODE_BAD_REQUEST, 'status 只能是 %s' % ' / '.join(ALLOWED_STATUS))
        fields['status'] = body.status
    if body.priority is not None:
        if body.priority not in ALLOWED_PRIORITY:
            api_error(400, CODE_BAD_REQUEST, 'priority 只能是 %s' % ' / '.join(ALLOWED_PRIORITY))
        fields['priority'] = body.priority
    if body.dept is not None:
        fields['dept'] = body.dept
    if not fields:
        api_error(400, CODE_BAD_REQUEST, '没有需要更新的字段')

    changed = store.update_ticket(ticket_id, fields)
    store.log_action(user.username, 'ticket_update', target=ticket_id, detail=fields)
    return ok({'ticket_id': ticket_id, 'changed': changed, 'fields': fields}, message='已更新')


@router.post('/feedback', summary='人工复核回写')
def feedback(body: FeedbackBody, user=Depends(current_user)):
    store = get_store()
    if body.action not in ALLOWED_ACTIONS:
        api_error(400, CODE_BAD_REQUEST, 'action 只能是 %s' % ' / '.join(ALLOWED_ACTIONS))
    if not store.get_ticket(body.ticket_id):
        api_error(404, CODE_NOT_FOUND, '工单不存在: %s' % body.ticket_id)

    # 修正标签必须落在体系内的 9 类里, 否则统计口径会被污染
    bad = [name for name in body.corrected_labels
           if name not in set(load_label_meta().get('labels', {}))]
    if bad:
        api_error(400, CODE_BAD_REQUEST, '非法标签: %s' % ', '.join(bad))
    if body.action == 'correct' and not body.corrected_labels:
        api_error(400, CODE_BAD_REQUEST, 'action=correct 时必须给出 corrected_labels')

    review_id = store.create_review(body.ticket_id, body.action, body.corrected_labels,
                                    body.note, user.username)
    store.log_action(user.username, 'ticket_feedback', target=body.ticket_id,
                     detail={'action': body.action, 'labels': body.corrected_labels})
    return ok({'review_id': review_id, 'ticket_id': body.ticket_id, 'action': body.action,
               'corrected_labels': body.corrected_labels}, message='复核已记录')


@router.get('/reviews', summary='复核记录')
def list_reviews(page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100),
                 user=Depends(current_user)):
    return ok(get_store().list_reviews(page, size))