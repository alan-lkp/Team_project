"""
ShopCare 后端应用入口 (backend/main.py)

组装顺序: 中间件 -> 路由 -> 异常处理器 -> 静态前端

启动:
    cd ShopCare-main
    python backend/main.py                  # 默认 127.0.0.1:8000
    uvicorn backend.main:app --reload       # 开发模式(改代码自动重启)
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4   # 多进程

    页面: http://127.0.0.1:8000/app
    文档: http://127.0.0.1:8000/docs
    探针: http://127.0.0.1:8000/health

关于文件名:
    设计文档里写的是 backend/app.py; 这里用 main.py, 因为 uvicorn 的写法
    "backend.main:app" 更符合"backend 是一个包"的直觉. 两种叫法都能用.

关于 .env:
    自己实现了 40 行的加载器, 不引入 python-dotenv. 真实环境变量优先于 .env(setdefault),
    这样"服务器上 export 的值"不会被仓库里的示例文件悄悄覆盖.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api import classify_api, dashboard_api, model_api, ticket_api, user_api
from backend.api.deps import get_pipeline
from backend.api.model_api import health as health_endpoint
from backend.common.config import get_config
from backend.common.shop_utils import CODE_BAD_REQUEST, CODE_INTERNAL, fail
from backend.common.store import get_store
from backend.middleware.context import RequestContextMiddleware, setup_logging

API_PREFIX = "/api/v1"
FRONTEND_DIR = os.path.join(_PROJECT_ROOT, "frontend", "web")
ENV_PATH = os.path.join(_PROJECT_ROOT, "backend", ".env")


def _load_dotenv(path=ENV_PATH):
    """极简 .env 加载: 支持 KEY=VALUE 与 # 注释; 已存在的真实环境变量优先"""
    if not os.path.exists(path):
        return 0
    loaded = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded += 1
            # key 已存在时保持不动: 服务器上 export 的值优先于 .env
    return loaded


_load_dotenv()


# ============================================================
# 启动/关闭
# ============================================================
def _boot():
    """启动自检: 探测存储(MySQL/Redis)、打印降级告警. 任何失败都不阻止进程启动."""
    cfg = get_config()
    logger = logging.getLogger("shopcare.boot")
    store = get_store()
    health = store.init()

    logger.info(
        "存储模式: %s (mysql=%s, redis=%s)",
        health["storage_mode"],
        health["mysql"]["available"],
        health["redis"]["available"],
    )
    for warning in health.get("warnings") or []:
        logger.warning("启动告警: %s", warning)
    for warning in cfg.validate():
        logger.warning("配置告警: %s", warning)
    if os.path.isdir(FRONTEND_DIR):
        logger.info("前端目录就绪: %s", FRONTEND_DIR)
    else:
        logger.warning("前端目录缺失: %s (页面打不开, 但接口不受影响)", FRONTEND_DIR)
    logger.info(
        "可用模型: %s | 默认模型: %s",
        get_pipeline().registry.configured(),
        cfg.default_model,
    )
    logger.info("启动完成: 页面 /app | 文档 /docs | 探针 /health")


@asynccontextmanager
async def lifespan(app):
    _boot()
    yield
    logging.getLogger("shopcare.boot").info("ShopCare 已停止")


# ============================================================
# 异常处理: 让所有错误都是同一种响应结构
# ============================================================
def _install_exception_handlers(app):
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        """deps.api_error 抛出的 detail 本身就是响应体, 这里原样透传"""
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = detail
        else:
            # HTTP 状态码 * 100 + 1 是刻意的: 401 -> 40101, 404 -> 40401, 与 shop_utils 的业务码对齐
            body = fail(exc.status_code * 100 + 1, str(detail))
        return JSONResponse(
            status_code=exc.status_code,
            content=body,
            headers=getattr(exc, "headers", None) or None,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        """把 pydantic 的报错压成一句人话.

        默认 FastAPI 返回 422 与一长串嵌套结构, 前端还得再写一遍解析;
        这里统一成 400 + 业务码 40001, 并在 data 里保留原始错误便于排查.
        """
        first = (exc.errors() or [{}])[0]
        location = ".".join(
            str(part) for part in first.get("loc", []) if part != "body"
        )
        message = "参数不合法: %s %s" % (location or "body", first.get("msg", ""))
        return JSONResponse(
            status_code=400,
            content=fail(CODE_BAD_REQUEST, message, data={"errors": exc.errors()}),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        logging.getLogger("shopcare.error").exception(
            "未捕获异常 [%s] %s %s", request_id, request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content=fail(
                CODE_INTERNAL,
                "服务内部错误: %s: %s" % (type(exc).__name__, exc),
                request_id=request_id,
            ),
        )


# ============================================================
# 前端静态资源
# ============================================================
def _mount_frontend(app):
    if not os.path.isdir(FRONTEND_DIR):
        logging.getLogger("shopcare.boot").warning(
            "跳过前端挂载(目录不存在): %s", FRONTEND_DIR
        )
        return

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/app", include_in_schema=False)
    def _app_page():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/", include_in_schema=False)
    def _root():
        # 直接访问根路径时跳到工作台, 省得用户还要手动敲 /app
        return RedirectResponse("/app")


# ============================================================
# 应用
# ============================================================
def create_app():
    cfg = get_config()
    setup_logging(cfg)

    app = FastAPI(
        title="ShopCare 电商客服工单智能处理平台",
        description=(
            "多标签工单分类 + 双阈值拒识 + 情感/优先级/话术推荐. "
            "三套本地对照模型(随机森林 / FastText / BERT+LoRA)可自由切换, "
            "并额外提供 LLM 直连与自动升级两档."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(RequestContextMiddleware)
    # CORS 最后添加 => 位于最外层, 这样被限流拦掉的 429 响应也带 CORS 头, 前端才能读到错误信息
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for module in (user_api, model_api, classify_api, ticket_api, dashboard_api):
        app.include_router(module.router, prefix=API_PREFIX)

    # 根级 /health 别名: 容器/负载均衡的探针习惯直接打 /health
    app.add_api_route(
        "/health",
        health_endpoint,
        methods=["GET"],
        summary="健康检查(根级别名)",
        tags=["model"],
    )

    _install_exception_handlers(app)
    _mount_frontend(app)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print("=" * 72)
    print("ShopCare 电商客服工单智能处理平台")
    print("  页面    : http://%s:%s/app" % (host, port))
    print("  接口文档: http://%s:%s/docs" % (host, port))
    print("  健康检查: http://%s:%s/health" % (host, port))
    print("  日志级别: %s (Ctrl+C 停止)" % get_config().log_level)
    print("=" * 72)
    uvicorn.run(app, host=host, port=port)
