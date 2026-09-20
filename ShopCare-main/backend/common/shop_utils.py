"""
ShopCare 业务公共工具 (backend/common/shop_utils.py)

这里放"所有接口都会用到的小东西", 避免每个路由各写一份:
    * 标签元数据读取(唯一来源: 01-data/label_meta.json)
    * 统一响应结构(成功/失败一个格式, 前端只写一套解析)
    * 工单号生成 / 分页 / 文本脱敏

统一响应约定:
    成功: {"code": 0, "message": "ok", "data": {...}}
    失败: {"code": 40001, "message": "参数不合法", "data": null}
    HTTP 状态码仍然照常用(401/403/429/500), 响应体里的 code 是**业务码**,
    两者互补: 前端可以只判断业务码, 也可以捕捉 HTTP 状态.
"""

import json
import os
import random
import re
import time
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META_PATH = os.path.join(PROJECT_ROOT, '01-data', 'label_meta.json')

CODE_OK = 0
CODE_BAD_REQUEST = 40001
CODE_UNAUTHORIZED = 40101
CODE_FORBIDDEN = 40301
CODE_NOT_FOUND = 40401
CODE_RATE_LIMITED = 42901
CODE_INTERNAL = 50001
CODE_MODEL_UNAVAILABLE = 50002

# 简单的 24 小时缓存: label_meta.json 是配置, 不需要每次请求都读盘
_meta_cache = {'value': None, 'ts': 0.0}
_META_TTL = 60.0


def load_label_meta(force=False):
    """读取标签元数据(带缓存); 缺失时返回空壳而不是抛异常"""
    now = time.time()
    if not force and _meta_cache['value'] is not None and now - _meta_cache['ts'] < _META_TTL:
        return _meta_cache['value']
    try:
        with open(META_PATH, encoding='utf-8') as f:
            meta = json.load(f)
    except (OSError, ValueError):
        meta = {'labels': {}, 'sentiment': {}, 'priority': {}}
    _meta_cache['value'] = meta
    _meta_cache['ts'] = now
    return meta


def label_info(name):
    """单个标签的元数据(缺失时给出安全的兜底值)"""
    meta = load_label_meta().get('labels', {})
    info = meta.get(name) or {}
    return {
        'label': name,
        'cn': info.get('cn', name),
        'dept': info.get('dept', '未分配'),
        'base_priority': info.get('base_priority', 'P2'),
        'sla_hours': info.get('sla_hours', 24),
        'desc': info.get('desc', ''),
    }


def dept_of(labels):
    """按"最急标签优先"给出建议处理部门"""
    rank = {'P0': 0, 'P1': 1, 'P2': 2}
    best, best_dept = 99, '未分配'
    for lb in labels or []:
        info = label_info(lb)
        r = rank.get(info['base_priority'], 3)
        if r < best:
            best, best_dept = r, info['dept']
    return best_dept


def sla_of(labels, priority='P2'):
    """SLA = min(优先级 SLA, 各标签 SLA), 取更紧的那个"""
    table = load_label_meta().get('priority', {})
    sla = table.get(priority, {}).get('sla_hours', 24)
    label_slas = [label_info(lb)['sla_hours'] for lb in (labels or [])]
    label_slas = [h for h in label_slas if h]
    return min([sla] + label_slas) if label_slas else sla


def priority_cn(priority):
    return load_label_meta().get('priority', {}).get(priority, {}).get('cn', priority)


def all_labels():
    """返回标签清单(前端下拉框/图例用), 顺序与 01-data/class.txt 一致"""
    meta = load_label_meta().get('labels', {})
    class_path = os.path.join(PROJECT_ROOT, '01-data', 'class.txt')
    try:
        with open(class_path, encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
    except OSError:
        names = list(meta.keys())
    return [label_info(n) for n in names]


# ---------------- 工单号 ----------------
def new_ticket_id(prefix='SC'):
    """生成工单号: SC + 日期 + 4 位随机, 例如 SC20260920-4821"""
    return '{}{}-{:04d}'.format(prefix, datetime.now().strftime('%Y%m%d'), random.randint(0, 9999))


# ---------------- 统一响应 ----------------
def ok(data=None, message='ok', **extra):
    body = {'code': CODE_OK, 'message': message, 'data': data}
    body.update(extra)
    return body


def fail(code=CODE_INTERNAL, message='服务内部错误', data=None, **extra):
    body = {'code': code, 'message': message, 'data': data}
    body.update(extra)
    return body


# ---------------- 文本处理 ----------------
_RE_PHONE = re.compile(r'(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)')
_RE_ORDER = re.compile(r'(?<![A-Za-z0-9])(\d{4})\d{6,}(\d{2})(?![A-Za-z0-9])')


def mask_pii(text):
    """打码手机号/长单号 —— 落库和展示前都该做, 避免把用户隐私原样存进日志和页面"""
    if not text:
        return text
    text = _RE_PHONE.sub(r'\1****\2', text)
    text = _RE_ORDER.sub(r'\1******\2', text)
    return text


def truncate(text, limit=200):
    text = text or ''
    return text if len(text) <= limit else text[:limit] + '...'


def now_str(fmt='%Y-%m-%d %H:%M:%S'):
    return datetime.now().strftime(fmt)


def paginate(items, page=1, size=20):
    """内存分页(数据库分页在 store 里做)"""
    page = max(1, int(page or 1))
    size = max(1, min(int(size or 20), 200))
    total = len(items)
    start = (page - 1) * size
    return {
        'items': items[start:start + size],
        'total': total,
        'page': page,
        'size': size,
        'pages': (total + size - 1) // size if size else 0,
    }


def day_list(days=7):
    """最近 N 天的日期字符串列表(含今天), 用于趋势图补零"""
    today = datetime.now().date()
    return [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days - 1, -1, -1)]


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare shop_utils 自测')
    print('=' * 72)
    labels = all_labels()
    print('标签数:', len(labels))
    for info in labels:
        print('  {:<16}{:<10}{:<12}{}'.format(info['label'], info['cn'], info['dept'], info['base_priority']))
    assert len(labels) == 9
    assert label_info('logistics')['dept'] == '物流仓储部'
    assert label_info('不存在的标签')['cn'] == '不存在的标签'

    print('\n部门判定(取最急标签的部门):')
    print('  logistics+quality ->', dept_of(['logistics', 'quality']))
    print('  invoice+consult   ->', dept_of(['invoice', 'consult']))
    assert dept_of(['logistics', 'quality']) == '品控/供应商'

    print('\nSLA:', sla_of(['logistics'], 'P2'), sla_of(['payment_account'], 'P1'))
    assert sla_of(['payment_account'], 'P1') == 4

    print('\n工单号:', new_ticket_id(), new_ticket_id())
    print('脱敏:', mask_pii('手机13812345678 订单2024091500123'))
    # 手机号: 保留前 3 位 + 后 4 位(11 位 -> 3+4+4), 中段打码
    assert mask_pii('手机13812345678') == '手机138****5678'
    assert mask_pii('13812345678') == '138****5678'
    # 长单号: 保留前 4 位 + 后 2 位
    assert mask_pii('订单2024091500123') == '订单2024******23'
    # 短数字串不是手机号/单号, 必须原样保留(避免误伤订单号后四位之类的短编号)
    assert mask_pii('编号 1234567') == '编号 1234567'
    assert len(truncate('x' * 300)) == 203

    print('\n响应结构:', json.dumps(ok({'a': 1}), ensure_ascii=False))
    print('             ', json.dumps(fail(CODE_BAD_REQUEST, '参数不合法'), ensure_ascii=False))
    print('\n分页:', json.dumps(paginate(list(range(25)), page=2, size=10), ensure_ascii=False)[:90])
    print('日期序列:', day_list(3))
    print('\n[OK] shop_utils 自测通过')