"""
ShopCare 看板与审计接口 (backend/api/dashboard_api.py)

路由:
    GET /dashboard/overview      看板总览(标签分布 / 模型调用 / 拒识率 / 趋势 / 平均耗时)
    GET /dashboard/review_queue  待人工复核队列
    GET /dashboard/audit         审计日志(仅管理员)

统计口径说明(重要, 答辩时会被问到):
    所有指标都从 tickets 表**现算**(store.stats_summary), 而不是依赖 Redis 里的累加计数器.
    原因是: 累加计数器一旦和落库不同步(比如重试、并发), 就再也对不上账, 而现算永远自洽.
    数据量真上来了再换成"定时物化 + 计数器"的混合方案, 那是以后的事.
"""

from fastapi import APIRouter, Depends, Query

from backend.api.deps import admin_user, current_user, get_pipeline
from backend.common.shop_utils import all_labels, ok
from backend.common.store import get_store

router = APIRouter(tags=['dashboard'])


@router.get('/dashboard/overview', summary='看板总览')
def overview(days: int = Query(7, ge=1, le=30, description='趋势图天数'),
             user=Depends(current_user)):
    store = get_store()
    summary = store.stats_summary()
    summary['labels'] = all_labels()                       # 前端图例的中文名
    summary['model_catalog'] = get_pipeline().registry.models()
    summary['storage_mode'] = store.health()['storage_mode']
    summary['days'] = days
    return ok(summary)


@router.get('/dashboard/review_queue', summary='待人工复核队列')
def review_queue(page: int = Query(1, ge=1),
                 size: int = Query(20, ge=1, le=100),
                 status: str = Query('pending', description='默认只看 pending; 传 all 看全部'),
                 user=Depends(current_user)):
    store = get_store()
    data = store.list_tickets(page=page, size=size,
                              status=None if status == 'all' else status)
    for item in data['items']:
        # 队列里"谁更该先看"要一眼可见: 拒识的、回复需要人工确认的, 都标出来
        item['needs_attention'] = bool(item.get('rejected')) or bool(item.get('needs_approval'))
    data['attention_count'] = sum(1 for item in data['items'] if item['needs_attention'])
    return ok(data)


@router.get('/dashboard/audit', summary='审计日志(管理员)')
def audit(page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=200),
          user=Depends(admin_user)):
    return ok(get_store().list_audit(page, size))