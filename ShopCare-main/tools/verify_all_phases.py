"""
ShopCare 一键自检 (tools/verify_all_phases.py)

用法:
    python tools/verify_all_phases.py            全量: 各阶段自测 + 端到端接口自测
    python tools/verify_all_phases.py --api      只跑端到端接口自测(最快)
    python tools/verify_all_phases.py --fast     跳过"要加载模型"的慢阶段
    python tools/verify_all_phases.py --list     只列出有哪些检查项
    python tools/verify_all_phases.py --phase 08 只跑名字里含 "08" 的阶段

两个刻意的设计:

1) 阶段自测一律**用子进程**跑, 而不是 import 后调用函数.
   因为 02-rf / 03-fasttext / 04-bert 三个目录里都有 config.py, 在同一个 Python 进程里
   必然串味(这正是 backend/common/model_registry.py 要用隔离加载的原因).
   子进程是最干净的隔离方式, 也让"每个阶段都能独立运行"这件事保持可验证.

2) 不依赖 MySQL / Redis. 后端探测失败会自动降级到内存存储, 接口照样跑得通,
   端到端自检因此可以在"什么都没配"的机器上直接跑 —— 这是降低上手门槛的关键.
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# (名称, 相对路径, 是否需要加载模型(慢), 失败时是否可接受(给原因))
PHASES = (
    ('依赖与模型产物自检', 'tools/check_env.py', False, None),
    ('数据格式体检', 'tools/check_dataset.py', False, None),
    ('共享: 指标与拒识', 'tools/ml_metrics.py', False, None),
    ('共享: 分词与清洗', 'tools/text_tokenize.py', False, None),
    ('共享: 预测器基类', 'tools/ticket_predictor.py', False, None),
    ('共享: 工单数据读取', 'tools/ticket_data.py', False, None),
    ('02 随机森林推理', '02-rf/rf_predict_fun.py', True,
     '需要先训练: python 02-rf/rf_train.py'),
    ('03 FastText 推理', '03-fasttext/ft_predict_fun.py', True,
     '需要先训练: python 03-fasttext/ft_train.py'),
    ('04 BERT 推理', '04-bert/bert_predict_fun.py', True,
     '需要本地预训练模型与已训练权重(本项目不联网下载模型)'),
    ('08 情感极性', '08-sentiment/sentiment_classifier.py', False, None),
    ('09 优先级引擎', '09-priority/priority_engine.py', False, None),
    ('10 话术推荐', '10-reply-recommend/reply_recommender.py', False, None),
    ('11 工单知识图谱', '11-ticket-kg/ticket_kg.py', False, None),
    ('后端: 安全(pbkdf2/JWT/限流)', 'backend/common/security.py', False, None),
    ('后端: 标签工具与响应结构', 'backend/common/shop_utils.py', False, None),
    ('后端: 存储层(三层降级)', 'backend/common/store.py', False, None),
    ('后端: 模型注册表(隔离加载)', 'backend/common/model_registry.py', True, None),
    ('后端: 业务流水线', 'backend/common/pipeline.py', True, None),
    ('后端: 接口依赖与鉴权', 'backend/api/deps.py', True, None),
    ('后端: 中间件(限流/日志)', 'backend/middleware/context.py', False, None),
)

TAIL_LINES = 12


def run_phase(name, script, timeout=420):
    """跑一个阶段脚本, 返回 (状态, 摘要输出, 耗时秒)"""
    path = os.path.join(PROJECT_ROOT, script)
    if not os.path.exists(path):
        return 'MISSING', '文件不存在: %s' % script, 0.0

    env = dict(os.environ)
    env['PYTHONPATH'] = PROJECT_ROOT
    env['PYTHONIOENCODING'] = 'utf-8'
    started = time.time()
    try:
        # stdin=DEVNULL 不能省: 02-rf / 03-fasttext / 04-bert 三个阶段的脚本末尾都带
        # "交互式输入"自测循环(input()). 从终端直接跑本脚本时, 子进程会继承终端并
        # **永久阻塞在 input() 上**, 一直耗到 420 秒超时才被判定为失败. 接 /dev/null
        # 后 input() 立刻拿到 EOF 跳出循环 —— 自检结果与"从管道运行"时完全一致.
        proc = subprocess.run([sys.executable, path], cwd=PROJECT_ROOT, env=env,
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', timeout=timeout)
        output, code = proc.stdout or '', proc.returncode
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', '超过 %d 秒未结束' % timeout, time.time() - started
    cost = time.time() - started
    status = 'PASS' if code == 0 else 'FAIL'
    return status, output, cost


def tail(output, lines=TAIL_LINES):
    rows = [row for row in (output or '').splitlines() if row.strip()]
    return rows[-lines:]


# ============================================================
# 端到端接口自检
# ============================================================
def run_api_check(pause=0.0):
    """用 FastAPI TestClient 在进程内跑一遍真实接口(不启动端口, 不依赖外部服务)"""
    from fastapi.testclient import TestClient
    from backend.main import app

    steps = []

    def step(name, func, expect_status=200):
        try:
            detail = func()
            steps.append((name, True, detail or ''))
        except AssertionError as exc:
            steps.append((name, False, str(exc)))
        except Exception as exc:                       # noqa: BLE001
            steps.append((name, False, '%s: %s' % (type(exc).__name__, exc)))
        if pause:
            time.sleep(pause)

    def body_of(response, expect=200):
        data = response.json()
        assert response.status_code == expect, \
            '期望 HTTP %s, 实际 %s, 响应=%s' % (expect, response.status_code, str(data)[:200])
        return data

    with TestClient(app) as client:
        state = {}

        def _health():
            data = body_of(client.get('/health'))
            assert data['code'] == 0
            info = data['data']
            state['models'] = info['model_catalog']
            return '存储=%s 可用模型=%s 业务模块=%s' % (
                info['storage']['storage_mode'], info['models']['available'],
                {k: v for k, v in info['business_modules'].items()})

        def _register():
            data = body_of(client.post('/api/v1/user/register',
                                       json={'username': 'verify_user', 'password': 'verify123456'}))
            return '注册: %s' % data['message']

        def _login():
            data = body_of(client.post('/api/v1/user/login',
                                       json={'username': 'verify_user', 'password': 'verify123456'}))
            state['token'] = data['data']['token']
            state['auth'] = {'Authorization': 'Bearer ' + state['token']}
            return '登录成功, 令牌长度 %d' % len(state['token'])

        def _unauthorized():
            data = body_of(client.post('/api/v1/classify', json={'text': 'x'}), expect=401)
            assert data['code'] == 40101, data
            return '未带令牌 -> HTTP 401 / 业务码 %s' % data['code']

        def _validation():
            data = body_of(client.post('/api/v1/classify', json={'text': ''},
                                       headers=state['auth']), expect=400)
            assert data['code'] == 40001, data
            return '空文本 -> HTTP 400 / %s' % data['message']

        def _models():
            data = body_of(client.get('/api/v1/models', headers=state['auth']))
            items = data['data']['items']
            assert len(items) >= 3
            return '模型: %s' % ['%s(%s)' % (i['key'], '已训练' if i['available'] else '未就绪')
                                for i in items]

        def _labels():
            data = body_of(client.get('/api/v1/labels', headers=state['auth']))
            assert data['data']['count'] == 9, data['data']['count']
            return '标签数=%d 独占标签=%s' % (data['data']['count'], data['data']['exclusive'])

        def _classify_save():
            data = body_of(client.post('/api/v1/classify',
                                       json={'text': '快递到广州十天了还没动静, 客服也不回复, 我要退款',
                                             'model': 'fasttext', 'save': True},
                                       headers=state['auth']))
            result = data['data']
            state['ticket_id'] = result['ticket_id']
            state['classify_text'] = result['text']
            assert result['labels'], '应当至少激活一个标签'
            assert result['priority']['priority'] in ('P0', 'P1', 'P2')
            return '工单 %s | 标签=%s | 优先级=%s | 情感=%s | 耗时=%.1fms' % (
                result['ticket_id'], [x['label'] for x in result['labels']],
                result['priority']['priority'], result['sentiment']['label'], result['latency_ms'])

        def _classify_cached():
            payload = {'text': state['classify_text'], 'model': 'fasttext'}
            body_of(client.post('/api/v1/classify', json=payload, headers=state['auth']))
            data = body_of(client.post('/api/v1/classify', json=payload, headers=state['auth']))
            assert data['data'].get('cached') is True, data['data'].get('cached')
            return '第二次同参数请求命中缓存(cached=true)'

        def _batch():
            data = body_of(client.post('/api/v1/classify/batch',
                                       json={'texts': ['收到货就是坏的', '发票开错了能重开吗'],
                                             'model': 'fasttext'}, headers=state['auth']))
            assert data['data']['count'] == 2
            return '批量 %d 条, 总耗时 %.1fms' % (data['data']['count'],
                                                 data['data']['total_latency_ms'])

        def _sentiment():
            data = body_of(client.post('/api/v1/sentiment', json={'text': '太差了, 再也不买了'},
                                       headers=state['auth']))
            assert data['data']['label'] == 'negative', data['data']
            return '情感=%s 分值=%s' % (data['data']['label'], data['data']['score'])

        def _priority():
            data = body_of(client.post('/api/v1/priority',
                                       json={'labels': ['payment_account'], 'sentiment': 'negative',
                                             'text': '重复扣款了两次, 立刻给我退回来'},
                                       headers=state['auth']))
            assert data['data']['priority'] in ('P0', 'P1'), data['data']
            return '优先级=%s SLA=%sh' % (data['data']['priority'], data['data']['sla_hours'])

        def _reply():
            data = body_of(client.post('/api/v1/reply',
                                       json={'labels': ['logistics'], 'priority': 'P1',
                                             'sentiment': 'negative', 'text': '快递一直没动'},
                                       headers=state['auth']))
            assert data['data']['reply']
            return '话术 %d 字, 需审批=%s' % (len(data['data']['reply']),
                                             data['data']['needs_approval'])

        def _tickets():
            data = body_of(client.get('/api/v1/tickets', headers=state['auth']))
            assert data['data']['total'] >= 1, data['data']
            return '工单总数=%d' % data['data']['total']

        def _ticket_detail():
            data = body_of(client.get('/api/v1/tickets/%s' % state['ticket_id'],
                                      headers=state['auth']))
            assert data['data']['ticket_id'] == state['ticket_id']
            return '详情: 标签=%s 部门=%s' % (data['data']['labels'], data['data']['dept'])

        def _ticket_patch():
            data = body_of(client.patch('/api/v1/tickets/%s' % state['ticket_id'],
                                        json={'priority': 'P1'}, headers=state['auth']))
            assert data['data']['changed'] == 1, data['data']
            return '更新优先级 -> P1'

        def _feedback():
            data = body_of(client.post('/api/v1/feedback',
                                       json={'ticket_id': state['ticket_id'], 'action': 'correct',
                                             'corrected_labels': ['logistics', 'service'],
                                             'note': '自检用例'}, headers=state['auth']))
            assert data['data']['review_id']
            return '复核记录 id=%s' % data['data']['review_id']

        def _reviews():
            data = body_of(client.get('/api/v1/reviews', headers=state['auth']))
            assert data['data']['total'] >= 1
            return '复核记录数=%d' % data['data']['total']

        def _overview():
            data = body_of(client.get('/api/v1/dashboard/overview', headers=state['auth']))
            info = data['data']
            assert info['total'] >= 1
            return '总量=%d 自动分流率=%.2f 拒识率=%.2f 平均耗时=%.1fms' % (
                info['total'], info['auto_rate'], info['reject_rate'], info['avg_latency_ms'])

        def _queue():
            data = body_of(client.get('/api/v1/dashboard/review_queue', headers=state['auth']))
            assert 'items' in data['data']
            return '待复核 %d 条' % data['data']['total']

        def _profile():
            data = body_of(client.get('/api/v1/user/profile', headers=state['auth']))
            assert data['data']['username'] == 'verify_user'
            return '当前用户=%s 角色=%s 存储=%s' % (data['data']['username'], data['data']['role'],
                                                   data['data']['storage_mode'])

        def _frontend():
            resp = client.get('/app')
            assert resp.status_code == 200, resp.status_code
            assert '<html' in resp.text.lower(), resp.text[:120]
            return '页面 /app 可访问, %d 字节' % len(resp.text)

        def _logout():
            body_of(client.post('/api/v1/user/logout', headers=state['auth']))
            data = body_of(client.get('/api/v1/user/profile', headers=state['auth']), expect=401)
            assert data['code'] == 40101, data
            return '登出后旧令牌被拒(黑名单生效)'

        step('健康检查 /health', _health)
        step('注册', _register)
        step('登录并拿到 JWT', _login)
        step('未带令牌访问受保护接口 -> 401', _unauthorized)
        step('参数校验 -> 400 + 业务码', _validation)
        step('模型清单 /models', _models)
        step('标签元数据 /labels', _labels)
        step('工单分类(落库) /classify', _classify_save)
        step('分类结果缓存命中', _classify_cached)
        step('批量分类 /classify/batch', _batch)
        step('情感 /sentiment', _sentiment)
        step('优先级 /priority', _priority)
        step('话术 /reply', _reply)
        step('工单列表 /tickets', _tickets)
        step('工单详情', _ticket_detail)
        step('工单更新 PATCH', _ticket_patch)
        step('人工复核 /feedback', _feedback)
        step('复核记录 /reviews', _reviews)
        step('看板 /dashboard/overview', _overview)
        step('待复核队列', _queue)
        step('当前用户 /user/profile', _profile)
        step('前端页面 /app', _frontend)
        step('登出 + 令牌黑名单', _logout)

    return steps


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='ShopCare 一键自检')
    parser.add_argument('--api', action='store_true', help='只跑端到端接口自测')
    parser.add_argument('--fast', action='store_true', help='跳过需要加载模型的慢阶段')
    parser.add_argument('--phase', default=None, help='只跑名字里含该字符串的阶段')
    parser.add_argument('--list', action='store_true', help='只列出检查项')
    args = parser.parse_args()

    if args.list:
        print('%-34s %-40s %s' % ('名称', '脚本', '需要模型'))
        for name, script, slow, _ in PHASES:
            print('%-34s %-40s %s' % (name, script, '是' if slow else '否'))
        return 0

    print('=' * 78)
    print('ShopCare 一键自检')
    print('  项目根目录:', PROJECT_ROOT)
    print('  解释器    :', sys.executable)
    print('=' * 78)

    failures = []

    if not args.api:
        print('\n### 阶段自测(每个阶段独立子进程)')
        for name, script, slow, optional in PHASES:
            if args.phase and args.phase not in name and args.phase not in script:
                continue
            if args.fast and slow:
                print('\n[跳过] %s (--fast 且需要加载模型)' % name)
                continue
            status, output, cost = run_phase(name, script)
            if status == 'FAIL' and optional:
                status = 'SKIP'
            mark = {'PASS': '[通过]', 'FAIL': '[失败]', 'SKIP': '[跳过]',
                    'TIMEOUT': '[超时]', 'MISSING': '[缺失]'}[status]
            print('\n%s %s  (%s, %.1fs)' % (mark, name, script, cost))
            for row in tail(output):
                print('    | ' + row)
            if status == 'FAIL':
                failures.append('%s (%s)' % (name, script))
            elif status == 'SKIP':
                print('    | 说明: 该阶段可跳过 —— %s' % optional)
            elif status in ('TIMEOUT', 'MISSING'):
                failures.append('%s (%s)' % (name, script))

    if not args.phase:
        print('\n### 端到端接口自测(进程内 TestClient, 不需要 MySQL/Redis)')
        try:
            steps = run_api_check()
        except Exception as exc:                       # noqa: BLE001
            print('  [失败] 自检未能启动: %s: %s' % (type(exc).__name__, exc))
            failures.append('端到端接口自测启动失败')
            steps = []
        for name, ok, detail in steps:
            print('  %s %-32s %s' % ('[通过]' if ok else '[失败]', name, detail))
            if not ok:
                failures.append('接口: %s' % name)

    print('\n' + '=' * 78)
    if failures:
        print('自检未全部通过, 失败项 %d 个:' % len(failures))
        for item in failures:
            print('  -', item)
        print('=' * 78)
        return 1
    print('全部自检通过 [OK]')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())