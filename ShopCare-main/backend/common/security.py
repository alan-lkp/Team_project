"""
ShopCare 认证与限流 (backend/common/security.py)

本模块**刻意不依赖 fastapi**: 里面全是纯函数和纯 Python 类,
这样不装 fastapi 也能单独跑自测, 逻辑正确性不必等整个 Web 框架就位才能验证.
FastAPI 的 Depends 包装放在 backend/api/deps.py.

为什么不用 passlib:
    passlib 在新版 Python 上经常和 bcrypt 版本打架(典型的
    `AttributeError: module 'bcrypt' has no attribute '__about__'`),
    而我们只需要"安全地存密码"这一个功能. 标准库 hmac + hashlib.pbkdf2_hmac
    完全够用, 还少一个依赖 —— 少一个依赖就少一类"装不上"的问题.

密码存储格式(自描述, 便于以后换算法):
    pbkdf2_sha256$<迭代次数>$<salt_b64>$<hash_b64>
"""

import base64
import hashlib
import hmac
import os
import secrets
import time

# ---------------- 密码哈希 ----------------
PBKDF2_ITERATIONS = 200_000          # OWASP 2023 对 pbkdf2-sha256 的建议下限
_HASH_PREFIX = 'pbkdf2_sha256'


def hash_password(password, iterations=PBKDF2_ITERATIONS):
    """生成密码哈希; 每次调用都用新的随机 salt"""
    if not password:
        raise ValueError('密码不能为空')
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    salt_b64 = base64.b64encode(salt).decode('ascii')
    hash_b64 = base64.b64encode(dk).decode('ascii')
    return f'{_HASH_PREFIX}${iterations}${salt_b64}${hash_b64}'


def verify_password(password, stored):
    """校验密码; 用 hmac.compare_digest 做**常数时间**比较, 防时序侧信道"""
    if not password or not stored:
        return False
    try:
        prefix, iters, salt_b64, hash_b64 = stored.split('$')
        if prefix != _HASH_PREFIX:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, int(iters))
        return hmac.compare_digest(dk, expected)
    except (ValueError, TypeError):
        return False


# ---------------- JWT ----------------
class TokenError(Exception):
    """令牌无效/过期"""


def create_access_token(subject, role='user', extra=None, secret=None,
                        algorithm='HS256', expires_minutes=720):
    """签发 JWT

    返回: (token, expires_at_epoch_seconds)
    依赖 PyJWT; 未安装时抛 ImportError, 由上层给出明确的安装提示.
    """
    import jwt

    now = int(time.time())
    payload = {
        'sub': str(subject),
        'role': role,
        'iat': now,
        'exp': now + int(expires_minutes * 60),
        'jti': secrets.token_hex(8),      # 唯一 id, 便于以后做黑名单/撤销
    }
    if extra:
        payload.update(extra)
    token = jwt.encode(payload, secret, algorithm=algorithm)
    if isinstance(token, bytes):          # PyJWT 1.x 返回 bytes, 2.x 返回 str
        token = token.decode('utf-8')
    return token, payload['exp']


def decode_token(token, secret=None, algorithm='HS256'):
    """解析并校验 JWT; 失败抛 TokenError"""
    import jwt

    try:
        return jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError('令牌已过期, 请重新登录') from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError(f'令牌无效: {exc}') from exc


# ---------------- 限流 ----------------
class RateLimiter:
    """固定窗口限流

    优先用 Redis(多进程/多实例共享计数), Redis 不可用时自动退化为进程内字典,
    单机演示够用. 窗口实现故意选最简单的"固定窗口"而不是滑动窗口:
    实现短、行为可预测, 对"防止有人刷接口"这个目标已经足够.
    """

    def __init__(self, limit=120, window_seconds=60, kv=None):
        self.limit = int(limit)
        self.window = int(window_seconds)
        self.kv = kv                      # 有 incr_with_ttl 方法的对象(RedisStore / MemoryStore)
        self._local = {}

    def check(self, key):
        """返回 (allowed, remaining, retry_after_seconds)"""
        bucket = int(time.time() // self.window)
        skey = f'ratelimit:{key}:{bucket}'

        count = None
        if self.kv is not None:
            try:
                count = self.kv.incr_with_ttl(skey, self.window)
            except Exception:             # noqa: BLE001  限流不该把主流程拖垮
                count = None

        if count is None:                 # 内存兜底
            # 顺手清理过期桶, 否则字典会一直涨
            if len(self._local) > 10000:
                self._local = {k: v for k, v in self._local.items()
                               if v[0] >= time.time() - self.window}
            now = time.time()
            entry = self._local.get(skey)
            if entry is None or entry[0] < now:
                self._local[skey] = [now + self.window, 1]
                count = 1
            else:
                entry[1] += 1
                count = entry[1]

        remaining = max(0, self.limit - count)
        retry_after = 0 if count <= self.limit else self.window - (int(time.time()) % self.window)
        return count <= self.limit, remaining, retry_after


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare security 自测')
    print('=' * 72)

    # ---- 密码哈希 ----
    pwd = 'admin123'
    h1 = hash_password(pwd)
    h2 = hash_password(pwd)
    print('哈希示例:', h1[:40] + '...')
    # 显式校验结构: 曾经在写文件时把 f-string 里的 $ 段吃掉, 只剩 'pbkdf2_sha256',
    # 结果"哈希"生成正常、校验永远失败 —— 这类 bug 必须靠格式断言挡住
    parts = h1.split('$')
    assert len(parts) == 4, '哈希必须是 4 段(pbkdf2_sha256$迭代$盐$哈希), 实际: %r' % h1
    assert parts[0] == _HASH_PREFIX, parts[0]
    assert int(parts[1]) == PBKDF2_ITERATIONS, parts[1]
    assert len(base64.b64decode(parts[2])) == 16, '盐应为 16 字节'
    assert len(base64.b64decode(parts[3])) == 32, 'sha256 摘要应为 32 字节'
    assert pwd not in h1, '哈希里绝不能出现明文密码'
    assert h1 != h2, '两次哈希必须不同(盐不同)'
    assert verify_password(pwd, h1) and verify_password(pwd, h2)
    assert not verify_password('wrong', h1)
    assert not verify_password('', h1)
    assert not verify_password(pwd, 'garbage')
    print('[OK] 密码哈希: 相同密码 -> 不同哈希, 校验正确, 脏数据不崩')

    # ---- JWT(没装 PyJWT 就跳过) ----
    try:
        import jwt  # noqa: F401
        token, exp = create_access_token('1', 'admin', {'username': 'admin'},
                                        secret='test-secret', expires_minutes=5)
        print('JWT 前缀:', token[:30] + '...')
        claims = decode_token(token, secret='test-secret')
        assert claims['sub'] == '1' and claims['role'] == 'admin' and claims['username'] == 'admin'
        # 用错密钥必须失败
        try:
            decode_token(token, secret='wrong-secret')
            raise AssertionError('错误密钥不该通过')
        except TokenError:
            pass
        # 过期必须失败
        expired, _ = create_access_token('1', secret='test-secret', expires_minutes=-1)
        try:
            decode_token(expired, secret='test-secret')
            raise AssertionError('过期令牌不该通过')
        except TokenError as exc:
            print('[OK] JWT: 正常解析 / 错密钥拒绝 / 过期拒绝 ->', exc)
    except ImportError:
        print('[跳过] 未安装 PyJWT, 执行 pip install PyJWT 后可测试 JWT')

    # ---- 限流 ----
    rl = RateLimiter(limit=3, window_seconds=60, kv=None)
    results = [rl.check('1.2.3.4')[0] for _ in range(5)]
    print('限流前 5 次(限 3):', results)
    assert results == [True, True, True, False, False]
    allowed, remaining, retry = rl.check('1.2.3.4')
    print(f'[OK] 限流: allowed={allowed} remaining={remaining} retry_after={retry}s')
    assert rl.check('5.6.7.8')[0] is True, '不同 IP 应各自独立计数'

    print('\n[OK] security 自测通过')