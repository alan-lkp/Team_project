"""
ShopCare 模型与元数据接口 (backend/api/model_api.py)

路由:
    GET /models   页面"模型选择"下拉框的数据源: 有哪些模型 / 是否已训练 / 不可用的原因
    GET /labels   9 类标签的中文名·责任部门·基础优先级·SLA·描述(前端图例与筛选下拉)
    GET /health   启动自检(免登录): MySQL/Redis 降级情况 + 模型加载状态 + 业务模块就绪情况

/health 为什么必须免登录:
    容器编排、负载均衡、监控探针都拿不到业务令牌, 探针必须能直接访问;
    它同时也是"降级必须如实暴露"这条原则的落点 —— 存储降级、模型缺失、配置告警
    在这里一眼可见, 而不是等业务出错才发现.
"""

from fastapi import APIRouter, Depends

from backend.api.deps import current_user, get_pipeline
from backend.common.config import get_config
from backend.common.shop_utils import all_labels, load_label_meta, ok
from backend.common.store import get_store

router = APIRouter(tags=["model"])


@router.get("/health", summary="健康检查(免登录)")
def health():
    """任何降级都在这里如实报告: storage_mode / degraded / model.load_error / warnings"""
    cfg = get_config()
    store = get_store()
    pipe = get_pipeline()
    registry = pipe.registry

    storage = store.health()
    catalog = registry.models()
    usable = [item["key"] for item in catalog if item["available"]]
    warnings = list(cfg.validate()) + list(storage.get("warnings") or [])

    return ok(
        {
            "status": "ok" if usable else "degraded",
            "healthy": bool(usable),
            "storage": storage,
            "models": registry.status(),
            "model_catalog": catalog,
            "business_modules": pipe.modules_status(),
            "config": cfg.as_dict(),
            "warnings": warnings,
        }
    )


@router.get("/models", summary="可用模型清单")
def list_models(user=Depends(current_user)):
    registry = get_pipeline().registry
    items = registry.models()
    return ok(
        {
            "default": get_config().default_model,
            "available": [item["key"] for item in items if item["available"]],
            "items": items,
            "virtual_choices": [
                {
                    "key": "auto",
                    "cn": "自动",
                    "desc": "先用默认模型, 拒识时自动升级 LLM 兜底",
                },
                {
                    "key": "llm",
                    "cn": "LLM 直连",
                    "desc": "不走本地模型, 直接调用大模型(需配置 API Key)",
                },
            ],
            "hint": "切换模型后会用同一条工单重新推理, 方便直接对比三套对照组的效果差异",
        }
    )


@router.get("/labels", summary="标签元数据")
def list_labels(user=Depends(current_user)):
    meta = load_label_meta()
    labels = all_labels()
    return ok(
        {
            "count": len(labels),
            "labels": labels,
            "priority": meta.get("priority", {}),
            "sentiment": meta.get("sentiment", {}),
            "exclusive": "invalid",
            "hint": "invalid(无效/骚扰) 独占, 不与其它标签共现; 其余 8 类可多标签共现",
        }
    )
