"""
ShopCare 工单根因知识图谱 (11-ticket-kg/ticket_kg.py)

定位(第二轮模块, 这里是可直接运行的最小可用版本):
    把 9 类标签之间的**共现关系**建成一张小图, 在分类结果上再往前推一步 ——
    回答三个很实际的问题:

    1) 召回补全: "物流 + 售后"共现很强, 那即使本次没激活某个标签, 也提示客服顺便问一句;
    2) 风险预警: 某些组合(资金类叠退款)天然容易升级成投诉, 需要主管介入;
    3) 根因定位: 从表面标签指向更靠近根因的排查方向, 避免"头痛医头".

为什么不做成图数据库:
    9 个节点的图, 用 dict + 共现计数就够了 —— 引入 Neo4j 只会让"能不能一键跑起来"变难.
    等标签数量上到几百、需要多跳推理与路径检索时再换, 那时接口形状不用变.

数据来源:
    01-data/train.txt 的标签共现统计(纯标准库, 无外部服务).
    文件缺失时自动退化为纯规则(label_meta.json + RISK_COMBOS), 接口照常可用 ——
    与项目其它地方"可选模块失败就降级"的原则一致.

自测: python 11-ticket-kg/ticket_kg.py
"""

import json
import os
from collections import defaultdict
from itertools import combinations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT_ROOT, '01-data', 'train.txt')
META_PATH = os.path.join(PROJECT_ROOT, '01-data', 'label_meta.json')

# 高风险组合: (标签集合, 等级, 说明). 命中多条时按等级取最严重的那条作为首要提示.
RISK_COMBOS = (
    (('payment_account', 'after_sale'), 'P0',
     '资金类问题叠加退款诉求: 极易升级为投诉或资金争议, 需主管介入并留痕'),
    (('payment_account',), 'P1',
     '涉及资金: 需支付/财务侧核对流水后再回复, 不能让客服口头承诺到账时间'),
    (('quality', 'after_sale', 'service'), 'P1',
     '质量 + 售后 + 服务三连: 典型的差评/曝光前兆, 建议主动补偿并回访'),
    (('invoice', 'after_sale'), 'P2',
     '发票与退款交叉: 注意冲红/重开流程, 避免退款金额与开票金额对不上'),
    (('invalid',), 'P2',
     '无效/骚扰工单: 建议直接归档, 不占用人工复核工时'),
)

# 根因提示: 表面标签 -> 更靠近根因的排查方向
ROOT_CAUSE_HINTS = {
    'logistics': '先查承运商轨迹与仓库出库记录, 区分"根本没发货"还是"卡在途中"',
    'quality': '先要商品批次号与实物照片, 区分个别瑕疵还是批次性问题',
    'after_sale': '先确认是否符合退款/换货条件, 再谈时效, 避免承诺后无法兑现',
    'invoice': '先核对抬头/税号与开票状态; 已开票的需走冲红重开',
    'price_promo': '先核对活动规则与下单时间戳, 价保须在有效期内',
    'payment_account': '先比对支付流水与订单状态是否一致; 重复扣款要原路退回',
    'consult': '先给结论再给细节; 售前咨询最忌答非所问',
    'service': '先致歉并定位到具体坐席与时间点; 服务问题必须闭环回访',
    'invalid': '内容与业务无关, 直接归档',
}

_CACHE = {'edges': None, 'counts': None, 'total': 0}


def load_label_meta():
    try:
        with open(META_PATH, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {'labels': {}}


def _cn(name):
    return (load_label_meta().get('labels', {}).get(name) or {}).get('cn', name)


def build_graph(force=False):
    """统计标签共现, 返回 (edges, counts, total)

    edges  : {(a, b): 共现次数}, 其中 a < b(去重, 不区分方向)
    counts : {label: 出现次数}
    total  : 参与统计的工单条数

    注意: invalid 不参与共现 —— 它在数据规范里是"独占标签", 和别的标签共现没有业务含义,
    把它算进去只会让"无效工单"污染正常的关联强度.
    """
    if _CACHE['edges'] is not None and not force:
        return _CACHE['edges'], _CACHE['counts'], _CACHE['total']

    edges, counts, total = defaultdict(int), defaultdict(int), 0
    try:
        with open(TRAIN_PATH, encoding='utf-8') as fh:
            for line in fh:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                labels = [name for name in parts[1].split(',') if name]
                total += 1
                for name in labels:
                    counts[name] += 1
                if 'invalid' in labels or len(labels) < 2:
                    continue
                for pair in combinations(sorted(set(labels)), 2):
                    edges[pair] += 1
    except OSError:
        pass                                        # 数据缺失 -> 空图, 其余功能仍可用

    _CACHE['edges'] = dict(edges)
    _CACHE['counts'] = dict(counts)
    _CACHE['total'] = total
    return _CACHE['edges'], _CACHE['counts'], _CACHE['total']


def related_labels(labels, top=3):
    """与给定标签共现最强、但本次**未激活**的标签(用于"顺便问一句"的召回补全)"""
    edges, _counts, _total = build_graph()
    given = set(labels or [])
    score = defaultdict(int)
    for (left, right), weight in edges.items():
        if left in given and right not in given:
            score[right] += weight
        elif right in given and left not in given:
            score[left] += weight
    ranked = sorted(score.items(), key=lambda item: -item[1])[:top]
    return [{'label': name, 'cn': _cn(name), 'weight': weight} for name, weight in ranked]


def risk_of(labels):
    """高风险组合命中情况(按严重程度排序)"""
    given = set(labels or [])
    order = {'P0': 0, 'P1': 1, 'P2': 2}
    hits = [{'combo': list(combo), 'level': level, 'desc': desc}
            for combo, level, desc in RISK_COMBOS if set(combo).issubset(given)]
    hits.sort(key=lambda item: order.get(item['level'], 3))
    return hits


def analyze_ticket(labels, text=''):
    """pipeline 调用的主入口: 标签 -> 关联提示 + 风险 + 根因方向 + 一句话摘要"""
    labels = [name for name in (labels or []) if name]
    edges, counts, total = build_graph()
    related = related_labels(labels)
    risks = risk_of(labels)
    hints = [{'label': name, 'hint': ROOT_CAUSE_HINTS[name]}
             for name in labels if name in ROOT_CAUSE_HINTS]

    parts = []
    if labels:
        parts.append('已识别 ' + '、'.join(_cn(name) for name in labels))
    if related:
        parts.append('共现提示 ' + '、'.join('%s(%d)' % (item['cn'], item['weight'])
                                             for item in related))
    if risks:
        parts.append('风险 ' + risks[0]['level'] + ': ' + risks[0]['desc'])
    return {
        'labels': labels,
        'related': related,
        'risks': risks,
        'root_cause_hints': hints,
        'graph': {'nodes': len(counts), 'edges': len(edges), 'tickets': total},
        'summary': '; '.join(parts),
    }


def graph_summary(top=8):
    """给报告/自检用的图概览: 节点数、边数、最强的若干条边"""
    edges, counts, total = build_graph()
    strongest = sorted(edges.items(), key=lambda item: -item[1])[:top]
    return {
        'tickets': total,
        'nodes': [{'label': name, 'cn': _cn(name), 'count': count}
                  for name, count in sorted(counts.items(), key=lambda item: -item[1])],
        'edge_count': len(edges),
        'strongest_edges': [{'a': pair[0], 'b': pair[1], 'cn': '%s + %s' % (_cn(pair[0]), _cn(pair[1])),
                             'weight': weight} for pair, weight in strongest],
    }


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare 工单知识图谱 自测')
    print('=' * 72)

    info = graph_summary()
    print('\n[1] 图的规模')
    print('   参与统计工单: %d 条 | 节点(标签): %d 个 | 边(共现对): %d 条'
          % (info['tickets'], len(info['nodes']), info['edge_count']))
    assert info['tickets'] > 0, '训练数据为空, 请先运行 tools/generate_ticket_data.py'
    assert len(info['nodes']) == 9, '应当是 9 类标签, 实际 %d' % len(info['nodes'])

    print('\n[2] 最强的共现关系')
    for edge in info['strongest_edges'][:5]:
        print('   %-22s %d 次' % (edge['cn'], edge['weight']))
    assert info['strongest_edges'], '至少应有一对共现标签'

    print('\n[3] 关联补全: 只看得到 logistics 时, 还能提示什么')
    result = analyze_ticket(['logistics'])
    for item in result['related']:
        print('   %-12s 权重 %d' % (item['cn'], item['weight']))
    assert result['related'], 'logistics 应当有共现标签'
    assert all(item['label'] != 'logistics' for item in result['related'])

    print('\n[4] invalid 不参与共现(独占标签)')
    invalid_result = analyze_ticket(['invalid'])
    print('   invalid 的关联标签:', invalid_result['related'])
    assert not invalid_result['related'], 'invalid 不应参与共现'

    print('\n[5] 风险组合: 资金 + 退款 应当触发最高等级')
    risky = analyze_ticket(['payment_account', 'after_sale', 'logistics'])
    print('   摘要:', risky['summary'])
    print('   风险:', [item['level'] for item in risky['risks']])
    assert risky['risks'] and risky['risks'][0]['level'] == 'P0'
    assert len(risky['root_cause_hints']) == 3

    print('\n[6] 空标签 / 未知标签都不崩')
    empty = analyze_ticket([])
    assert empty['related'] == [] and empty['summary'] == ''
    unknown = analyze_ticket(['不存在的标签'])
    print('   未知标签的 cn 回退为原名:', unknown['summary'])
    assert '不存在的标签' in unknown['summary']

    print('\n[OK] ticket_kg 自测通过')