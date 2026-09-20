"""
ShopCare 用户接口 (backend/api/user_api.py)

路由(统一前缀 /api/v1):
    POST /user/register   注册(可用 ALLOW_REGISTER=false 关闭)
    POST /user/login      登录, 返回 JWT
    POST /user/logout     登出: 把 jti 写进 Redis 黑名单; Redis 不可用时退化为"仅前端清除令牌"
    GET  /user/profile    当前用户信息

安全细节:
    * 密码只存 pbkdf2_sha256 哈希(见 common/security.py), 库里看不到明文;
    * 登录失败**不区分**"用户不存在"与"密码错误", 避免攻击者拿接口枚举账号;
    * 成功与失败都写审计日志 —— 登录失败属于安全事件, 必须留痕.
"""

import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.api.deps import api_error, current_user, revoke_token
from backend.common.config import get_config
from backend.common.security import create_access_token, verify_password
from backend.common.shop_utils import (CODE_BAD_REQUEST, CODE_FORBIDDEN, CODE_INTERNAL,
                                       CODE_UNAUTHORIZED, ok)
from backend.common.store import get_store

router = APIRouter(prefix='/user', tags=['user'])


class RegisterBody(BaseModel):
    username: str = Field(min_length=3, max_length=32, description='登录名, 3-32 个字符')
    password: str = Field(min_length=6, max_length=64, description='密码, 至少 6 位')


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=64)


@router.post('/register', summary='注册')
def register(body: RegisterBody):
    cfg = get_config()
    if not cfg.allow_register:
        api_error(403, CODE_FORBIDDEN, '管理员已关闭注册(ALLOW_REGISTER=false)')
    store = get_store()
    new_id, error = store.create_user(body.username, body.password, 'user')
    if error:
        api_error(400, CODE_BAD_REQUEST, error)
    store.log_action(body.username, 'user_register', target=str(new_id))
    return ok({'username': body.username, 'id': new_id}, message='注册成功, 请登录')


@router.post('/login', summary='登录')
def login(body: LoginBody):
    cfg = get_config()
    store = get_store()
    row = store.get_user(body.username)
    if not row or not verify_password(body.password, row.get('password_hash')):
        store.log_action(body.username, 'login_failed')
        api_error(401, CODE_UNAUTHORIZED, '用户名或密码错误')
    try:
        token, expires_at = create_access_token(
            row['username'], row.get('role', 'user'), secret=cfg.jwt_secret,
            algorithm=cfg.jwt_algorithm, expires_minutes=cfg.jwt_expire_minutes)
    except ImportError:
        api_error(500, CODE_INTERNAL, 'PyJWT 未安装, 请先执行 pip install PyJWT')
    store.log_action(row['username'], 'login')
    return ok({
        'token': token,
        'token_type': 'Bearer',
        'expires_at': expires_at,
        'expires_in': cfg.jwt_expire_minutes * 60,
        'user': {'username': row['username'], 'role': row.get('role', 'user')},
    }, message='登录成功')


@router.post('/logout', summary='登出')
def logout(user=Depends(current_user)):
    ttl = int((user.extra.get('exp') or 0) - time.time())
    revoked = revoke_token(user.extra.get('jti'), ttl)
    get_store().log_action(user.username, 'logout')
    return ok({'revoked': revoked, 'ttl_seconds': max(ttl, 0)},
              message='已登出' if revoked else
                      '已登出(令牌黑名单不可用, 请前端一并清除本地令牌)')


@router.get('/profile', summary='当前用户信息')
def profile(user=Depends(current_user)):
    store = get_store()
    row = store.get_user(user.username) or {}
    health = store.health()
    return ok({
        'username': user.username,
        'role': user.role,
        'is_admin': user.is_admin,
        'created_at': row.get('created_at'),
        'storage_mode': health['storage_mode'],
        'degraded': health['degraded'],
    })