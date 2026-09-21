"""
ShopCare 后端配置 (backend/common/config.py)

设计原则: **一切连接信息都走环境变量**.
    数据库密码、JWT 密钥这类东西绝不能出现在代码或 git 里(这是最常见的翻车点),
    所以这里只提供默认值和校验, 真实值请放到 backend/.env 里(参考 .env.example).

降级策略(很重要, 决定了"能不能一键跑起来"):
    * MySQL 连不上 -> 自动降级到**内存存储**, 接口照常可用, 只是 /health 会标 degraded,
      数据重启即丢. 这样用户没配数据库也能先把页面和模型跑通, 不会一开始就卡住.
    * Redis 连不上 -> 限流/复核队列退化为进程内实现, 单机演示完全够用.
    明确把"降级"暴露在 /health 里, 而不是假装正常 —— 静默降级是运维事故的源头.

环境变量清单(全部可选, 括号内是默认值):
    DB_HOST(127.0.0.1) DB_PORT(3306) DB_USER(root) DB_PASSWORD('') DB_NAME(shopcare)
    DB_CHARSET(utf8mb4) DB_POOL_SIZE(5)
    REDIS_HOST(127.0.0.1) REDIS_PORT(6379) REDIS_DB(0) REDIS_PASSWORD('')
    JWT_SECRET(dev-secret-change-me) JWT_ALGORITHM(HS256) JWT_EXPIRE_MINUTES(720)
    ADMIN_USERNAME(admin) ADMIN_PASSWORD(admin123)
    MODEL_AVAILABLE(rf,fasttext,bert)      # 逗号分隔, 决定 /models 里哪些可选
    DEFAULT_MODEL(bert)
    ENABLE_LLM_FALLBACK(false)             # 默认关: 避免不知情地产生 API 费用
    LABEL_THRESHOLD(0.5) GLOBAL_THRESHOLD(0.8)
    RATE_LIMIT_PER_MINUTE(120)
    LOG_DIR(backend/logs) LOG_LEVEL(INFO)
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')


def _env(key, default=None):
    val = os.environ.get(key)
    return default if val is None or val == '' else val


def _env_int(key, default):
    try:
        return int(_env(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key, default):
    try:
        return float(_env(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key, default=False):
    val = _env(key)
    if val is None:
        return default
    return str(val).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def _env_list(key, default):
    val = _env(key)
    if val is None:
        return list(default)
    return [x.strip() for x in str(val).split(',') if x.strip()]


class Config:
    """全局配置(进程启动时读一次环境变量)"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.backend_dir = BACKEND_DIR

        # ---------------- 数据库(MySQL) ----------------
        self.db_host = _env('DB_HOST', '127.0.0.1')
        self.db_port = _env_int('DB_PORT', 3306)
        self.db_user = _env('DB_USER', 'root')
        self.db_password = _env('DB_PASSWORD', '')
        self.db_name = _env('DB_NAME', 'shopcare')
        self.db_charset = _env('DB_CHARSET', 'utf8mb4')
        self.db_pool_size = _env_int('DB_POOL_SIZE', 5)
        # 连不上时不要卡住启动: 3 秒足够判定, 也别让页面转圈
        self.db_connect_timeout = _env_int('DB_CONNECT_TIMEOUT', 3)
        self.db_auto_create = _env_bool('DB_AUTO_CREATE', True)

        # ---------------- 缓存(Redis) ----------------
        self.redis_host = _env('REDIS_HOST', '127.0.0.1')
        self.redis_port = _env_int('REDIS_PORT', 6379)
        self.redis_db = _env_int('REDIS_DB', 0)
        self.redis_password = _env('REDIS_PASSWORD', '')
        self.redis_timeout = _env_int('REDIS_TIMEOUT', 2)

        # ---------------- 认证 ----------------
        self.jwt_secret = _env('JWT_SECRET', 'dev-secret-change-me')
        self.jwt_algorithm = _env('JWT_ALGORITHM', 'HS256')
        self.jwt_expire_minutes = _env_int('JWT_EXPIRE_MINUTES', 720)
        self.admin_username = _env('ADMIN_USERNAME', 'admin')
        self.admin_password = _env('ADMIN_PASSWORD', 'admin123')
        self.allow_register = _env_bool('ALLOW_REGISTER', True)

        # ---------------- 模型 ----------------
        self.model_available = [m.lower() for m in _env_list('MODEL_AVAILABLE', ['rf', 'fasttext', 'bert'])]
        self.default_model = _env('DEFAULT_MODEL', 'bert').lower()
        self.enable_llm_fallback = _env_bool('ENABLE_LLM_FALLBACK', False)
        self.label_threshold = _env_float('LABEL_THRESHOLD', 0.5)
        self.global_threshold = _env_float('GLOBAL_THRESHOLD', 0.8)

        # ---------------- 服务 ----------------
        self.rate_limit_per_minute = _env_int('RATE_LIMIT_PER_MINUTE', 120)
        self.log_dir = _env('LOG_DIR', os.path.join(BACKEND_DIR, 'logs'))
        self.log_level = _env('LOG_LEVEL', 'INFO').upper()
        self.cors_origins = _env_list('CORS_ORIGINS', ['*'])
        self.mysql_unavailable_hint = (
            'MySQL 不可用, 已降级到内存存储(数据重启后丢失). '
            '要启用持久化, 请检查 backend/.env 里的 DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME, '
            '并确认 MySQL 已启动、shopcare 库已创建(见 backend/schema.sql).'
        )

    # ------------------------------------------------------------------
    def as_dict(self):
        """给 /health 用的脱敏配置摘要(绝不含密码)"""
        return {
            'mysql': {'host': self.db_host, 'port': self.db_port, 'user': self.db_user,
                      'database': self.db_name, 'password_set': bool(self.db_password)},
            'redis': {'host': self.redis_host, 'port': self.redis_port, 'db': self.redis_db,
                      'password_set': bool(self.redis_password)},
            'models': {'available': self.model_available, 'default': self.default_model,
                       'llm_fallback_enabled': self.enable_llm_fallback},
            'thresholds': {'label': self.label_threshold, 'global': self.global_threshold},
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'log_level': self.log_level,
        }

    def validate(self):
        """返回配置层面的告警列表(不阻断启动, 只提醒)"""
        warnings = []
        if self.jwt_secret == 'dev-secret-change-me':
            warnings.append('JWT_SECRET 仍是默认值, 生产环境必须改成随机长字符串')
        if self.admin_password == 'admin123':
            warnings.append('ADMIN_PASSWORD 仍是默认值 admin123, 请修改')
        if self.default_model not in self.model_available:
            warnings.append(f'DEFAULT_MODEL={self.default_model} 不在 MODEL_AVAILABLE={self.model_available} 里, '
                            f'已自动回退到 {self.model_available[0] if self.model_available else "rf"}')
            self.default_model = self.model_available[0] if self.model_available else 'rf'
        if not self.model_available:
            warnings.append('MODEL_AVAILABLE 为空, 至少需要一个模型')
        return warnings


# 进程级单例
_config = None


def get_config():
    global _config
    if _config is None:
        _config = Config()
    return _config