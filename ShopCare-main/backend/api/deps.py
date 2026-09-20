"""
ShopCare 接口层公共依赖 (backend/api/deps.py)

FastAPI 的依赖注入把"每个接口都要做一遍的事"收到一处:
    * 进程内单例: 配置 / 存储 / 业务流水线(避免每个请求都重新加载模型)
    * 当前用户: 解析 Bearer JWT, 失败时给出统一格式的错误
    * 分页参数规整

统一错误格式:
    抛出的 HTTPException 的 detail 就是最终响应体, 由 main.py 的异常处理器原样透传:
        {"code": 40101, "message": "登录已过期, 请重新登录", "data": null}
    HTTP 状态码与业务码互补: 前者给网关/浏览器, 后者给前端业务逻辑.

关于"是否每次请求都查库确认用户存在":
    不查. JWT 是签过名的, 直接信任其中的 sub/role, 换来的是 MySQL 抖动时已登录用户不会
    集体掉线. 代价是"删号后令牌在有效期内仍可用" —— 这个阶段可以接受, 生产环境应配合
    Redis 令牌黑名单(见设计文档 10.2 的 key 设计).
"""

import os
import sys
import threading

from fastapi import Depends, Header, HTTPException, Query

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.common.config import get_config
from backend.common.security import RateLimiter, TokenError, decode_token
from backend.common.shop_utils import CODE_FORBIDDEN, CODE_UNAUTHORIZED, fail
from backend.common.store import get_store

_lock = threading.Lock()
_pipeline = None
_limiter = None


def get_pipeline():
    """业务流水线单例(内部持有模型注册表; 模型只在第一次真正用到时加载)"""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                from backend.common.pipeline import TicketPipeline
                _pipeline = TicketPipeline(get_config())
    return _pipeline


def get_limiter():
    """限流器单例; kv 取"Redis 优先, 不可用时退化为进程内字典" """
    global _limiter
    if _limiter is None:
        with _lock:
            if _limiter is None:
                cfg = get_config()
                _limiter = RateLimiter(cfg.rate_limit_per_minute, 60, get_store().kv())
    return _limiter


def api_error(status_code, code, message):
    """抛出统一格式的业务错误(由 main.py 的异常处理器转成响应体)"""
    raise HTTPException(status_code=status_code, detail=fail(code, message))


class CurrentUser:
    """当前登录用户(轻量对象, 不依赖数据库行)"""

    def __init__(self, username, role='user', extra=None):
        self.username = username
        self.role = role or 'user'
        self.extra = extra or {}

    @property
    def is_admin(self):
        return self.role == 'admin'

    def to_dict(self):
        return {'username': self.username, 'role': self.role, 'is_admin': self.is_admin}

    def __repr__(self):
        return 'CurrentUser(%s, %s)' % (self.username, self.role)


def _token_from_header(authorization):
    """支持 'Bearer xxx' 与裸令牌两种写法, 前端调试时少踩一个坑"""
    if not authorization:
        return None
    parts = str(authorization).split()
    if len(parts) == 2 and parts[0].lower() == 'bearer':
        return parts[1]
    return str(authorization).strip() or None


def _is_revoked(jti):
    """查令牌黑名单(登出时写入). 任何异常都当作"未撤销", 绝不因黑名单故障误伤合法请求"""
    try:
        kv = get_store().kv()
        return kv.get_json('shopcare:jwt:blacklist:' + str(jti)) is not None
    except Exception:                      # noqa: BLE001
        return False


def revoke_token(jti, ttl_seconds):
    """把 jti 写入黑名单, TTL = 令牌剩余有效期; 返回是否真的写成功(Redis 不可用时为 False)"""
    if not jti:
        return False
    ttl = int(ttl_seconds or 0)
    if ttl <= 0:
        ttl = 60                            # 已经过期的令牌不必长期占位
    try:
        return bool(get_store().kv().set_json('shopcare:jwt:blacklist:' + str(jti),
                                              {'revoked': True}, ttl))
    except Exception:                      # noqa: BLE001
        return False


def optional_user(authorization: str = Header(default=None)):
    """可选登录态: 无令牌/令牌无效都返回 None, 由接口自己决定是否强制"""
    token = _token_from_header(authorization)
    if not token:
        return None
    cfg = get_config()
    try:
        payload = decode_token(token, cfg.jwt_secret, cfg.jwt_algorithm)
    except TokenError:
        return None
    except Exception:                     # noqa: BLE001  PyJWT 未安装等
        return None
    jti = payload.get('jti')
    if jti and _is_revoked(jti):
        # 已登出的令牌: 即使签名还没过期也不再放行
        return None
    return CurrentUser(payload.get('sub'), payload.get('role'),
                       {'jti': jti, 'exp': payload.get('exp')})


def current_user(user=Depends(optional_user)):
    """强制登录"""
    if user is None:
        api_error(401, CODE_UNAUTHORIZED, '未登录或登录已过期, 请重新登录')
    return user


def admin_user(user=Depends(current_user)):
    """管理员专属操作"""
    if not user.is_admin:
        api_error(403, CODE_FORBIDDEN, '该操作需要管理员权限')
    return user


def pagination(page: int = Query(1, ge=1, description='页码, 从 1 开始'),
               size: int = Query(20, ge=1, le=100, description='每页条数, 上限 100')):
    return {'page': int(page), 'size': int(size)}


def client_key(request, user=None):
    """限流身份标识: 登录用户按用户名, 未登录按真实 IP(考虑反向代理的 X-Forwarded-For)"""
    if user is not None:
        return 'u:' + str(user.username)
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded:
        return 'ip:' + forwarded.split(',')[0].strip()
    host = getattr(getattr(request, 'client', None), 'host', None) or 'unknown'
    return 'ip:' + str(host)


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare api.deps 自测')
    print('=' * 72)


    class _FakeClient:
        host = '127.0.0.1'


    class _FakeRequest:
        def __init__(self, headers=None):
            self.headers = headers or {}
            self.client = _FakeClient()


    print('\n[1] token 解析')
    assert _token_from_header(None) is None
    assert _token_from_header('Bearer abc.def.ghi') == 'abc.def.ghi'
    assert _token_from_header('abc.def.ghi') == 'abc.def.ghi'
    assert _token_from_header('   ') is None
    print('   Bearer / 裸令牌 / 空 三种写法都正确')

    print('\n[2] 可选登录: 无令牌与坏令牌都返回 None')
    assert optional_user(None) is None
    assert optional_user('Bearer not-a-jwt') is None
    print('   [OK] 不会因为坏令牌抛异常打断请求')

    print('\n[3] 有效令牌能解析出用户')
    from backend.common.security import create_access_token
    cfg = get_config()
    token, exp = create_access_token(cfg.admin_username, 'admin', secret=cfg.jwt_secret,
                                     algorithm=cfg.jwt_algorithm,
                                     expires_minutes=cfg.jwt_expire_minutes)
    user = optional_user('Bearer ' + token)
    assert user is not None and user.username == cfg.admin_username and user.is_admin
    print('   ', user.to_dict(), 'exp =', exp)

    print('\n[3b] 登出后令牌立即失效(黑名单)')
    assert revoke_token(user.extra['jti'], 60) is True
    assert optional_user('Bearer ' + token) is None
    print('   [OK] 已撤销的 jti 会被拒绝(即使签名仍在有效期内)')

    print('\n[4] 限流身份标识')
    assert client_key(_FakeRequest(), user) == 'u:' + cfg.admin_username
    assert client_key(_FakeRequest()) == 'ip:127.0.0.1'
    assert client_key(_FakeRequest({'x-forwarded-for': '1.2.3.4, 5.6.7.8'})) == 'ip:1.2.3.4'
    print('   [OK] 登录用户按用户名, 游客按真实 IP')

    print('\n[5] 限流器(Redis 不可用时自动退化为进程内计数)')
    limiter = get_limiter()
    allowed, remaining, retry_after = limiter.check('selftest:deps')
    print('   第 1 次 -> allowed=%s remaining=%s retry_after=%s' % (allowed, remaining, retry_after))
    assert allowed is True
    kv = get_store().kv()
    print('   kv 实现:', type(kv).__name__)

    print('\n[6] 分页参数与流水线单例')
    assert pagination(2, 50) == {'page': 2, 'size': 50}
    pipe = get_pipeline()
    assert pipe is get_pipeline()
    print('   流水线单例 ok, 业务模块:', pipe.modules_status())

    print('\n[OK] api.deps 自测通过')