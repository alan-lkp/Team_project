"""
ShopCare 请求上下文中间件 (backend/middleware/context.py)

一个中间件干三件"每个请求都要做一遍"的事:
    1) 生成/透传 X-Request-ID: 出问题时用户报一个 id, 就能在日志里精确定位那一次请求;
    2) 限流: Redis 优先, 不可用时自动退化为进程内计数; 超限直接返回 429, 不进业务逻辑;
    3) 访问日志 + 耗时, 并把耗时回写到 X-Process-Time-Ms 响应头.

为什么限流放在中间件而不是每个接口的依赖里:
    中间件是"所有请求的必经之路", 忘不掉; 而且它在路由匹配之前就返回 429,
    连参数解析、模型加载都不会发生, 对"防刷"这个目标更彻底.

为什么按"登录用户 > 客户端 IP"限流:
    同一个 IP 后面可能是整个办公室的客服(共享出口), 按 IP 限会把无辜的人一起限掉;
    登录后按用户名计数更准确. 未登录才退回按 IP.
"""

import logging
import logging.handlers
import os
import time
import uuid

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.deps import client_key, get_limiter, optional_user
from backend.common.config import get_config
from backend.common.shop_utils import CODE_RATE_LIMITED, fail

logger = logging.getLogger('shopcare.access')

# 免限流路径: 探针要能随时探测, 文档页与静态资源不该被业务限流波及
SKIP_EXACT = ('/health', '/api/v1/health', '/docs', '/redoc', '/openapi.json', '/favicon.ico')
SKIP_PREFIX = ('/static', '/app')


class RequestContextMiddleware(BaseHTTPMiddleware):
    """请求 ID + 限流 + 访问日志"""

    async def dispatch(self, request, call_next):
        request_id = request.headers.get('x-request-id') or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        path = request.url.path

        if not self._skipped(path):
            limited = self._rate_limit_response(request, request_id)
            if limited is not None:
                return limited

        started = time.perf_counter()
        response = await call_next(request)
        cost_ms = (time.perf_counter() - started) * 1000

        response.headers['X-Request-ID'] = request_id
        response.headers['X-Process-Time-Ms'] = '%.2f' % cost_ms
        logger.info('[%s] %s %s -> %s (%.2f ms)', request_id, request.method, path,
                    response.status_code, cost_ms)
        return response

    # ---------------- 内部 ----------------
    @staticmethod
    def _skipped(path):
        if path in SKIP_EXACT:
            return True
        return any(path.startswith(prefix) for prefix in SKIP_PREFIX)

    @staticmethod
    def _rate_limit_response(request, request_id):
        """返回 429 响应, 或 None 表示放行"""
        # 这里单独解析一次令牌只是为了拿到"限流身份"; 真正的鉴权仍在接口依赖里做.
        # 结果是"登录用户按用户名限流", 未登录的按 IP.
        user = None
        try:
            user = optional_user(request.headers.get('authorization'))
        except Exception:                    # noqa: BLE001 限流不该因为解析失败而中断请求
            user = None
        key = client_key(request, user)

        try:
            allowed, remaining, retry_after = get_limiter().check(key)
        except Exception:                    # noqa: BLE001 限流器故障时放行, 宁可放过不可错杀
            return None
        if allowed:
            return None

        body = fail(CODE_RATE_LIMITED,
                    '请求过于频繁, 请 %d 秒后重试(上限 %d 次/分钟, 身份 %s)'
                    % (retry_after, get_limiter().limit, key))
        logger.warning('[%s] %s %s 触发限流, 身份 %s', request_id, request.method,
                       request.url.path, key)
        return JSONResponse(status_code=429, content=body,
                            headers={'Retry-After': str(retry_after),
                                     'X-Request-ID': request_id})


def setup_logging(cfg=None):
    """控制台 + 滚动文件日志; 日志目录不可写时只保留控制台(不让服务起不来)"""
    cfg = cfg or get_config()
    handlers = [logging.StreamHandler()]
    log_path = None
    try:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_path = os.path.join(cfg.log_dir, 'shopcare.log')
        handlers.append(logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'))
    except OSError as exc:                    # noqa: BLE001
        log_path = None
        handlers[0].stream.write('[警告] 日志目录不可写(%s), 仅输出到控制台\n' % exc)

    logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO),
                        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
                        handlers=handlers, force=True)
    logging.getLogger('shopcare.boot').info('日志初始化完成: 文件=%s 级别=%s',
                                            log_path or '(仅控制台)', cfg.log_level)
    return log_path


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare middleware.context 自测')
    print('=' * 72)

    path = setup_logging()
    assert path is None or os.path.exists(path)

    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except ImportError as exc:
        print('\n[跳过 HTTP 自测] 缺少依赖: %s' % exc)
        print('  提示: pip install fastapi "httpx" 后重跑本脚本')
        raise SystemExit(0)

    from backend.api import deps
    from backend.common.security import RateLimiter

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get('/health')
    def _health():
        return {'ok': True, 'skipped_by_limiter': True}

    @app.get('/api/v1/echo')
    def _echo():
        return {'ok': True}

    # 把限流阈值临时降到 3 次/分钟, 才能在一次自测里看到 429
    deps._limiter = RateLimiter(3, 60, None)
    client = TestClient(app)

    print('\n[1] 请求 ID 与耗时响应头')
    resp = client.get('/api/v1/echo', headers={'X-Request-ID': 'selftest-0001'})
    print('   X-Request-ID =', resp.headers.get('X-Request-ID'),
          '| X-Process-Time-Ms =', resp.headers.get('X-Process-Time-Ms'))
    assert resp.headers.get('X-Request-ID') == 'selftest-0001'
    assert float(resp.headers['X-Process-Time-Ms']) >= 0
    print('   [OK] 透传客户端传来的 request id; 没传时会自动生成')

    print('\n[2] 未传 request id 时自动生成(每次不同)')
    a = client.get('/api/v1/echo').headers['X-Request-ID']
    b = client.get('/api/v1/echo').headers['X-Request-ID']
    print('   ', a, b)
    assert a != b and len(a) == 16

    print('\n[3] 限流: 第 4 次应当被拒(阈值临时设为 3/分钟)')
    # 重新计数: 前面两个用例已经用掉了同一身份的配额, 不重置的话这里一开始就是 429
    deps._limiter = RateLimiter(3, 60, None)
    codes = [client.get('/api/v1/echo').status_code for _ in range(4)]
    print('   连续 4 次状态码:', codes)
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    blocked = client.get('/api/v1/echo')
    print('   限流响应体:', blocked.json()['code'], blocked.json()['message'][:34] + '...')
    print('   Retry-After =', blocked.headers.get('Retry-After'))
    assert blocked.json()['code'] == 42901 and blocked.headers.get('Retry-After')

    print('\n[4] /health 免限流(探针随时可用)')
    health_codes = [client.get('/health').status_code for _ in range(6)]
    print('   连续 6 次 /health 状态码:', health_codes)
    assert set(health_codes) == {200}

    print('\n[OK] middleware.context 自测通过')