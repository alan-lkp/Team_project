"""
ShopCare 存储层 (backend/common/store.py)

三个存储实现 + 一个门面:
    MySQLStore  : 主存储. 工单、用户、复核记录、审计日志. 用 PyMySQL + 每次请求独立连接.
    RedisStore  : 缓存/限流/队列. 连不上就降级.
    MemoryStore : MySQL/Redis 都不可用时的兜底, 让**接口依然能跑通**.
    Store       : 对上层暴露统一的业务方法(create_ticket / list_tickets / ...),
                  内部自动选择走 MySQL 还是内存.

为什么不做成"只有 MySQL 一种实现":
    用户第一次跑这个项目时, MySQL/Redis 大概率还没配好. 如果这时候接口直接 500,
    整个项目就"跑不起来"了, 体验极差. 所以宁可多写一层兜底:
    服务**永远能启动**, /health 如实报告当前用的是哪种存储, 数据丢了也不假装没丢.

关于连接池:
    PyMySQL 没有内置池. 这里用"每请求一个连接"(connect -> use -> close),
    配合 connect_timeout=3. 对演示/中小流量足够, 也避免了自己写池子引入的泄漏风险.
    真要上量, 换成 DBUtils.PooledDB 或 asyncmy 即可 —— 上层接口不用改.
"""

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime

# 让脚本既能被 "python backend/common/store.py" 直接跑, 也能被 uvicorn 以包的形式导入
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.common.config import get_config
from backend.common.security import hash_password

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema.sql"
)

# 内存兜底时自动建的表结构(只需列表字段, 不需要类型)
MEMORY_TABLES = ("users", "tickets", "reviews", "audit_logs")


# ============================================================
# todo 1. MySQL
# ============================================================
class MySQLStore:
    """PyMySQL 封装; 所有异常都被吞掉并记在 last_error 里, 由 available() 反映状态"""

    def __init__(self, cfg=None):
        self.cfg = cfg or get_config()
        self._available = None
        self.last_error = None
        self._lock = threading.Lock()

    def connect(self, retries=1):
        """探测连接; 结果缓存在 _available 上(成功则永久缓存, 失败允许后续重试)"""
        for _ in range(max(1, retries)):
            try:
                import pymysql  # noqa: F401
            except ImportError as exc:
                self._available = False
                self.last_error = f"未安装 PyMySQL: {exc}"
                return False
            try:
                conn = self._new_conn()
                conn.close()
                self._available = True
                self.last_error = None
                return True
            except Exception as exc:  # noqa: BLE001
                self._available = False
                self.last_error = f"{type(exc).__name__}: {exc}"
        return bool(self._available)

    def _new_conn(self):
        import pymysql

        return pymysql.connect(
            host=self.cfg.db_host,
            port=self.cfg.db_port,
            user=self.cfg.db_user,
            password=self.cfg.db_password,
            database=self.cfg.db_name,
            charset=self.cfg.db_charset,
            autocommit=False,
            connect_timeout=self.cfg.db_connect_timeout,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def available(self):
        if self._available is None:
            self.connect()
        return bool(self._available)

    @contextmanager
    def cursor(self, commit=False):
        """with store.cursor(commit=True) as cur: ... —— 自动 commit/rollback + 关闭"""
        conn = self._new_conn()
        try:
            with conn.cursor() as cur:
                yield cur
            if commit:
                conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def execute(self, sql, args=None):
        with self.cursor(commit=True) as cur:
            cur.execute(sql, args or ())
            return cur.rowcount, cur.lastrowid

    def query(self, sql, args=None, one=False):
        with self.cursor() as cur:
            cur.execute(sql, args or ())
            rows = cur.fetchall() or []
        return (rows[0] if rows else None) if one else rows

    def init_schema(self, schema_path=None):
        """执行 schema.sql 建库建表(语句按分号切分, 逐条执行)"""
        path = schema_path or SCHEMA_PATH
        if not os.path.exists(path):
            return False, f"找不到建表脚本: {path}"
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            # 先按行剔除 -- 注释, 再按分号切分.
            # 不能简单地"跳过以 -- 开头的整块": 注释行通常和 CREATE 语句在同一块里,
            # 那样会把建库语句一并跳掉 —— 这个坑只会在第一次建库时暴露.
            body = "\n".join(
                ln for ln in raw.splitlines() if not ln.strip().startswith("--")
            )
            statements = [s.strip() for s in body.split(";") if s.strip()]
            # 建库语句要单独用不带 database 的连接执行
            created_db = False
            for stmt in statements:
                if stmt.strip().upper().startswith(("CREATE DATABASE", "USE")):
                    self._execute_plain(stmt)
                    created_db = True
                else:
                    self.execute(stmt)
            if created_db:
                # 建完库再连一次, 确保后续语句作用在新库上
                self.connect()
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    def _execute_plain(self, sql):
        """不带 database 的连接执行(用于 CREATE DATABASE / USE)"""
        import pymysql

        conn = pymysql.connect(
            host=self.cfg.db_host,
            port=self.cfg.db_port,
            user=self.cfg.db_user,
            password=self.cfg.db_password,
            charset=self.cfg.db_charset,
            autocommit=True,
            connect_timeout=self.cfg.db_connect_timeout,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        finally:
            conn.close()


# ============================================================
# todo 2. Redis
# ============================================================
class RedisStore:
    """Redis 封装; 每个方法在不可用时返回 None/False, 由调用方决定降级行为"""

    def __init__(self, cfg=None):
        self.cfg = cfg or get_config()
        self._available = None
        self.last_error = None
        self._client = None

    def connect(self):
        try:
            import redis  # noqa: F401
        except ImportError as exc:
            self._available = False
            self.last_error = f"未安装 redis: {exc}"
            return False
        try:
            import redis

            client = redis.Redis(
                host=self.cfg.redis_host,
                port=self.cfg.redis_port,
                #protocol=2,
                db=self.cfg.redis_db,
                password=self.cfg.redis_password or None,
                socket_timeout=self.cfg.redis_timeout,
                socket_connect_timeout=self.cfg.redis_timeout,
                decode_responses=True,
            )
            client.ping()
            self._client = client
            self._available = True
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self._client = None
            self._available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
        return bool(self._available)

    def available(self):
        if self._available is None:
            self.connect()
        return bool(self._available)

    # ---- 原子计数 + 过期(限流用) ----
    def incr_with_ttl(self, key, ttl):
        if not self.available():
            return None
        try:
            pipe = self._client.pipeline()
            pipe.incr(key, 1)
            pipe.expire(key, int(ttl) + 1)
            return int(pipe.execute()[0])
        except Exception:  # noqa: BLE001
            self._client = None
            self._available = False
            return None

    # ---- JSON 存取 ----
    def get_json(self, key):
        if not self.available():
            return None
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001
            return None

    def set_json(self, key, value, ttl=None):
        if not self.available():
            return False
        try:
            self._client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
            return True
        except Exception:  # noqa: BLE001
            return False

    def delete(self, key):
        if not self.available():
            return False
        try:
            self._client.delete(key)
            return True
        except Exception:  # noqa: BLE001
            return False


# ============================================================
# todo 3. 内存兜底
# ============================================================
class MemoryStore:
    """进程内的极简存储: kv + 几张"表"

    只用于"MySQL/Redis 没配好"的场景. 数据随进程消失, 且没有并发控制,
    但足以让接口和页面完整跑通 —— 这正是它的全部目的.
    """

    def __init__(self):
        self._kv = {}
        self._tables = {t: {} for t in MEMORY_TABLES}
        self._autoinc = {t: 0 for t in MEMORY_TABLES}
        self._lock = threading.Lock()

    # ---- kv ----
    def incr_with_ttl(self, key, ttl):
        with self._lock:
            now = time.time()
            item = self._kv.get(key)
            if item is None or item[0] < now:
                self._kv[key] = [now + ttl, 1]
                return 1
            item[1] += 1
            return item[1]

    def get_json(self, key):
        with self._lock:
            item = self._kv.get(key)
            if item is None or (isinstance(item, list) and item[0] < time.time()):
                return None
            return item[1] if len(item) > 1 else None

    def set_json(self, key, value, ttl=None):
        with self._lock:
            self._kv[key] = [time.time() + (ttl or 86400), value]
        return True

    def delete(self, key):
        with self._lock:
            self._kv.pop(key, None)
        return True

    # ---- 表 ----
    def insert(self, table, record):
        with self._lock:
            self._autoinc[table] += 1
            rid = self._autoinc[table]
            row = dict(record)
            row["id"] = rid
            row.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
            self._tables[table][rid] = row
            return rid

    def select(self, table, where=None, order_by=None, desc=True, limit=None, offset=0):
        with self._lock:
            rows = list(self._tables[table].values())
        if where:
            rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        if order_by:
            rows.sort(
                key=lambda r: (r.get(order_by) is None, r.get(order_by)), reverse=desc
            )
        if offset:
            rows = rows[offset:]
        if limit:
            rows = rows[:limit]
        return [dict(r) for r in rows]

    def count(self, table, where=None):
        return len(self.select(table, where=where))

    def update(self, table, rid, fields):
        with self._lock:
            row = self._tables[table].get(int(rid))
            if row is None:
                return 0
            row.update(fields)
            return 1

    def get(self, table, rid):
        with self._lock:
            row = self._tables[table].get(int(rid))
            return dict(row) if row else None


# ============================================================
# todo 4. 门面: 上层只用这一层
# ============================================================
class Store:
    """业务级存储接口; 内部自动在 MySQL / 内存之间切换"""

    def __init__(self, cfg=None):
        self.cfg = cfg or get_config()
        self.mysql = MySQLStore(self.cfg)
        self.redis = RedisStore(self.cfg)
        self.memory = MemoryStore()
        self.boot_warnings = []

    # ---------------- 启动 ----------------
    def init(self):
        """启动时调用: 探测 MySQL/Redis, 建表, 创建默认管理员"""
        if self.mysql.connect():
            if self.cfg.db_auto_create:
                ok, err = self.mysql.init_schema()
                if not ok:
                    self.boot_warnings.append(f"建表失败: {err}")
            try:
                self._ensure_admin_mysql()
            except Exception as exc:  # noqa: BLE001
                self.boot_warnings.append(f"创建默认管理员失败: {exc}")
        else:
            self.boot_warnings.append(self.cfg.mysql_unavailable_hint)

        if not self.redis.connect():
            self.boot_warnings.append(
                f"Redis 不可用({self.redis.last_error}), 限流与缓存已降级为进程内实现"
            )
        else:
            try:
                self._ensure_admin_mysql()  # 若走内存则不涉及
            except Exception:  # noqa: BLE001
                pass

        # 内存模式下也要有管理员, 否则登录不了
        if not self.mysql.available():
            self._ensure_admin_memory()
        return self.health()

    def _ensure_admin_mysql(self):
        rows = self.mysql.query(
            "SELECT id FROM users WHERE username=%s", (self.cfg.admin_username,)
        )
        if rows:
            return
        self.mysql.execute(
            "INSERT INTO users(username, password_hash, role) VALUES(%s, %s, %s)",
            (self.cfg.admin_username, hash_password(self.cfg.admin_password), "admin"),
        )

    def _ensure_admin_memory(self):
        existing = self.memory.select(
            "users", where={"username": self.cfg.admin_username}
        )
        if existing:
            return
        self.memory.insert(
            "users",
            {
                "username": self.cfg.admin_username,
                "password_hash": hash_password(self.cfg.admin_password),
                "role": "admin",
            },
        )

    def health(self):
        return {
            "mysql": {
                "available": self.mysql.available(),
                "error": self.mysql.last_error,
                "host": self.cfg.db_host,
                "database": self.cfg.db_name,
            },
            "redis": {
                "available": self.redis.available(),
                "error": self.redis.last_error,
                "host": self.cfg.redis_host,
            },
            "storage_mode": "mysql" if self.mysql.available() else "memory",
            "degraded": not self.mysql.available(),
            "warnings": list(self.boot_warnings),
        }

    # ---------------- 限流用的 kv ----------------
    def kv(self):
        """返回可用的 kv 实现(Redis 优先)"""
        return self.redis if self.redis.available() else self.memory

    # ---------------- 用户 ----------------
    def get_user(self, username):
        if self.mysql.available():
            return self.mysql.query(
                "SELECT * FROM users WHERE username=%s", (username,), one=True
            )
        rows = self.memory.select("users", where={"username": username}, limit=1)
        return rows[0] if rows else None

    def create_user(self, username, password, role="user"):
        if self.get_user(username):
            return None, "用户名已存在"
        if self.mysql.available():
            _, rid = self.mysql.execute(
                "INSERT INTO users(username, password_hash, role) VALUES(%s,%s,%s)",
                (username, hash_password(password), role),
            )
            return rid, None
        rid = self.memory.insert(
            "users",
            {
                "username": username,
                "password_hash": hash_password(password),
                "role": role,
            },
        )
        return rid, None

    # ---------------- 工单 ----------------
    TICKET_FIELDS = (
        "ticket_id",
        "text",
        "text_masked",
        "model_used",
        "labels",
        "confidences",
        "avg_confidence",
        "rejected",
        "reject_reason",
        "sentiment",
        "priority",
        "dept",
        "sla_hours",
        "suggested_reply",
        "needs_approval",
        "resolved_by",
        "latency_ms",
        "status",
        "created_at",
    )

    def create_ticket(self, record):
        """落库; list/dict 字段自动 JSON 序列化"""
        rec = dict(record)
        for key in ("labels", "confidences"):
            if isinstance(rec.get(key), (list, dict)):
                rec[key] = json.dumps(rec[key], ensure_ascii=False)
        rec.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        rec.setdefault("status", "pending")

        if self.mysql.available():
            cols = [k for k in self.TICKET_FIELDS if k in rec]
            sql = "INSERT INTO tickets({}) VALUES({})".format(
                ",".join(cols), ",".join(["%s"] * len(cols))
            )
            _, rid = self.mysql.execute(sql, tuple(rec[c] for c in cols))
            return rid
        return self.memory.insert("tickets", rec)

    @staticmethod
    def _decode_ticket(row):
        if not row:
            return row
        out = dict(row)
        if isinstance(out.get("labels"), str):
            try:
                out["labels"] = json.loads(out["labels"])
            except ValueError:
                out["labels"] = []
        if isinstance(out.get("confidences"), str):
            try:
                out["confidences"] = json.loads(out["confidences"])
            except ValueError:
                out["confidences"] = {}
        if "rejected" in out and out["rejected"] is not None:
            out["rejected"] = bool(out["rejected"])
        return out

    def get_ticket(self, ticket_id):
        if self.mysql.available():
            row = self.mysql.query(
                "SELECT * FROM tickets WHERE ticket_id=%s", (ticket_id,), one=True
            )
        else:
            rows = self.memory.select(
                "tickets", where={"ticket_id": ticket_id}, limit=1
            )
            row = rows[0] if rows else None
        return self._decode_ticket(row)

    def list_tickets(
        self,
        page=1,
        size=20,
        status=None,
        priority=None,
        dept=None,
        model_used=None,
        rejected=None,
        keyword=None,
    ):
        """分页查询(返回 items/total/page/size/pages)"""
        page = max(1, int(page or 1))
        size = max(1, min(int(size or 20), 100))
        offset = (page - 1) * size

        if self.mysql.available():
            where, args = [], []
            for col, val in (
                ("status", status),
                ("priority", priority),
                ("dept", dept),
                ("model_used", model_used),
            ):
                if val:
                    where.append(f"{col}=%s")
                    args.append(val)
            if rejected is not None:
                where.append("rejected=%s")
                args.append(1 if rejected else 0)
            if keyword:
                where.append("text LIKE %s")
                args.append(f"%{keyword}%")
            clause = ("WHERE " + " AND ".join(where)) if where else ""
            total_row = self.mysql.query(
                f"SELECT COUNT(*) AS n FROM tickets {clause}", tuple(args), one=True
            )
            total = int(total_row["n"]) if total_row else 0
            rows = self.mysql.query(
                f"SELECT * FROM tickets {clause} ORDER BY id DESC LIMIT %s OFFSET %s",
                tuple(args) + (size, offset),
            )
            items = [self._decode_ticket(r) for r in rows]
        else:
            rows = self.memory.select("tickets", order_by="id", desc=True)
            if status:
                rows = [r for r in rows if r.get("status") == status]
            if priority:
                rows = [r for r in rows if r.get("priority") == priority]
            if dept:
                rows = [r for r in rows if r.get("dept") == dept]
            if model_used:
                rows = [r for r in rows if r.get("model_used") == model_used]
            if rejected is not None:
                rows = [r for r in rows if bool(r.get("rejected")) == bool(rejected)]
            if keyword:
                rows = [r for r in rows if keyword in (r.get("text") or "")]
            total = len(rows)
            items = [self._decode_ticket(r) for r in rows[offset : offset + size]]

        return {
            "items": items,
            "total": total,
            "page": page,
            "size": size,
            "pages": (total + size - 1) // size if size else 0,
        }

    def update_ticket(self, ticket_id, fields):
        if self.mysql.available():
            cols = [k for k in fields if k in self.TICKET_FIELDS]
            if not cols:
                return 0
            sql = (
                "UPDATE tickets SET "
                + ",".join(f"{c}=%s" for c in cols)
                + " WHERE ticket_id=%s"
            )
            rowcount, _ = self.mysql.execute(
                sql, tuple(fields[c] for c in cols) + (ticket_id,)
            )
            return rowcount
        rows = self.memory.select("tickets", where={"ticket_id": ticket_id}, limit=1)
        return self.memory.update("tickets", rows[0]["id"], fields) if rows else 0

    # ---------------- 人工复核 ----------------
    def create_review(
        self, ticket_id, action, corrected_labels=None, note=None, reviewer=None
    ):
        rec = {
            "ticket_id": ticket_id,
            "action": action,  # approve / correct / reject
            "corrected_labels": json.dumps(corrected_labels or [], ensure_ascii=False),
            "note": note,
            "reviewer": reviewer,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.mysql.available():
            _, rid = self.mysql.execute(
                "INSERT INTO reviews(ticket_id, action, corrected_labels, note, reviewer) "
                "VALUES(%s,%s,%s,%s,%s)",
                (
                    rec["ticket_id"],
                    rec["action"],
                    rec["corrected_labels"],
                    rec["note"],
                    rec["reviewer"],
                ),
            )
        else:
            rid = self.memory.insert("reviews", rec)
        # 复核完成 -> 工单状态更新; 若给了修正标签, 同时写回工单
        fields = {"status": "reviewed"}
        if corrected_labels:
            fields["labels"] = json.dumps(corrected_labels, ensure_ascii=False)
            fields["resolved_by"] = "human"
        self.update_ticket(ticket_id, fields)
        return rid

    def list_reviews(self, page=1, size=20):
        page = max(1, int(page or 1))
        size = max(1, min(int(size or 20), 100))
        offset = (page - 1) * size
        if self.mysql.available():
            total_row = self.mysql.query(
                "SELECT COUNT(*) AS n FROM reviews", (), one=True
            )
            total = int(total_row["n"]) if total_row else 0
            rows = self.mysql.query(
                "SELECT * FROM reviews ORDER BY id DESC LIMIT %s OFFSET %s",
                (size, offset),
            )
        else:
            total = self.memory.count("reviews")
            rows = self.memory.select(
                "reviews", order_by="id", desc=True, limit=size, offset=offset
            )
        return {
            "items": rows,
            "total": total,
            "page": page,
            "size": size,
            "pages": (total + size - 1) // size if size else 0,
        }

    # ---------------- 审计 ----------------
    def log_action(self, username, action, target=None, detail=None):
        rec = {
            "username": username,
            "action": action,
            "target": target,
            "detail": json.dumps(detail, ensure_ascii=False)
            if detail is not None
            else None,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            if self.mysql.available():
                self.mysql.execute(
                    "INSERT INTO audit_logs(username, action, target, detail) VALUES(%s,%s,%s,%s)",
                    (rec["username"], rec["action"], rec["target"], rec["detail"]),
                )
            else:
                self.memory.insert("audit_logs", rec)
        except Exception:  # noqa: BLE001  审计失败不该影响主流程
            pass
        return True

    def list_audit(self, page=1, size=50):
        size = max(1, min(int(size or 50), 200))
        offset = (max(1, int(page or 1)) - 1) * size
        if self.mysql.available():
            rows = self.mysql.query(
                "SELECT * FROM audit_logs ORDER BY id DESC LIMIT %s OFFSET %s",
                (size, offset),
            )
        else:
            rows = self.memory.select(
                "audit_logs", order_by="id", desc=True, limit=size, offset=offset
            )
        return {"items": rows, "size": size, "page": page}

    # ---------------- 统计 ----------------
    def _all_tickets(self):
        if self.mysql.available():
            return [
                self._decode_ticket(r)
                for r in self.mysql.query("SELECT * FROM tickets ORDER BY id DESC")
            ]
        return [
            self._decode_ticket(r)
            for r in self.memory.select("tickets", order_by="id", desc=True)
        ]

    def stats_summary(self):
        """汇总指标: 总量 / 自动分流率 / 拒识率 / 平均耗时 / 优先级与部门分布"""
        from backend.common.shop_utils import day_list

        tickets = self._all_tickets()
        total = len(tickets)
        if total == 0:
            return {
                "total": 0,
                "auto_rate": 0.0,
                "reject_rate": 0.0,
                "avg_latency_ms": 0.0,
                "priority_dist": {},
                "dept_dist": {},
                "model_dist": {},
                "sentiment_dist": {},
                "trend": [{"date": d, "count": 0} for d in day_list(7)],
                "top_labels": [],
                "pending_review": 0,
            }

        auto = [t for t in tickets if not t.get("rejected")]
        by_reason = {}
        for t in tickets:
            if t.get("rejected"):
                key = t.get("reject_reason") or "unknown"
                by_reason[key] = by_reason.get(key, 0) + 1

        def counter(field):
            out = {}
            for t in tickets:
                key = t.get(field) or "未知"
                out[key] = out.get(key, 0) + 1
            return dict(sorted(out.items(), key=lambda kv: -kv[1]))

        label_counter = {}
        for t in tickets:
            for lb in t.get("labels") or []:
                name = lb.get("label") if isinstance(lb, dict) else lb
                label_counter[name] = label_counter.get(name, 0) + 1

        dates = day_list(7)
        trend_map = {d: 0 for d in dates}
        for t in tickets:
            created_at = t.get("created_at")

            if isinstance(created_at, datetime):
                d = created_at.strftime("%Y-%m-%d")
            elif isinstance(created_at, str):
                d = created_at[:10]
            else:
                d = ""

            # d = (t.get('created_at') or '')[:10]
            if d in trend_map:
                trend_map[d] += 1

        latencies = [t["latency_ms"] for t in tickets if t.get("latency_ms")]
        return {
            "total": total,
            "auto_rate": round(len(auto) / total, 4),
            "reject_rate": round((total - len(auto)) / total, 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2)
            if latencies
            else 0.0,
            "avg_confidence": round(
                sum(t.get("avg_confidence") or 0 for t in tickets) / total, 4
            ),
            "priority_dist": counter("priority"),
            "dept_dist": counter("dept"),
            "model_dist": counter("model_used"),
            "sentiment_dist": counter("sentiment"),
            "reject_reason_dist": by_reason,
            "top_labels": [
                {"label": k, "count": v}
                for k, v in sorted(label_counter.items(), key=lambda kv: -kv[1])[:9]
            ],
            "trend": [{"date": d, "count": trend_map[d]} for d in dates],
            "pending_review": sum(1 for t in tickets if t.get("status") == "pending"),
        }


# ============================================================
# 进程级单例
# ============================================================
_store = None


def get_store():
    global _store
    if _store is None:
        _store = Store()
    return _store


if __name__ == "__main__":
    print("=" * 72)
    print("ShopCare store 自测(未装 MySQL/Redis 时应自动降级到内存)")
    print("=" * 72)
    st = Store()
    health = st.init()
    print("存储模式:", health["storage_mode"], "| 降级:", health["degraded"])
    for w in health["warnings"]:
        print("  [提示]", w)

    # 内存路径必须完整可用, 这是"没配数据库也能跑通"的底线
    st.memory.insert(
        "tickets",
        {
            "ticket_id": "SC-TEST-1",
            "text": "快递一直没到",
            "labels": json.dumps([{"label": "logistics", "score": 0.9}]),
            "confidences": "{}",
            "rejected": False,
            "priority": "P1",
            "dept": "物流仓储部",
            "model_used": "rf",
            "sentiment": "negative",
            "avg_confidence": 0.9,
            "latency_ms": 12.3,
            "status": "pending",
        },
    )
    page = st.list_tickets()
    print("内存工单:", page["total"], "条")
    got = st.get_ticket("SC-TEST-1")
    assert got and got["labels"][0]["label"] == "logistics", got
    summary = st.stats_summary()
    print(
        "统计:", {k: summary[k] for k in ("total", "auto_rate", "reject_rate", "trend")}
    )
    assert summary["total"] >= 1
    assert len(summary["trend"]) == 7
    st.create_review("SC-TEST-1", "correct", ["logistics"], "测试", "tester")
    assert st.get_ticket("SC-TEST-1")["status"] == "reviewed"
    st.log_action("tester", "review", "SC-TEST-1", {"ok": True})
    assert st.list_reviews()["total"] >= 1
    print("\n[OK] store 自测通过")
