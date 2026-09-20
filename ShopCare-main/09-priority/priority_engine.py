"""
ShopCare 工单优先级引擎 (09-priority)

为什么单独做一个优先级引擎, 而不是让模型直接输出 P0/P1/P2:
    1. **优先级是业务规则, 不是语义**. 同样的"快递慢", 普通用户是 P2, 但
       "已经投诉到消协" 就必须是 P0 —— 这个判断依据是文本里的**信号词**, 不是语义相似度.
       让 BERT 去学这种规则, 既费标注又不可控.
    2. **可解释、可审计**. 客服主管一定会问"凭什么这单是 P0"。
       规则引擎能逐条列出"因为命中了'投诉'(+2.5)、情感为负面(×2.0)", 模型给不出这个.
    3. **可运营**. 阈值和权重都是 meta 里的数字, 业务方自己就能调, 不用重训模型.

设计(加权公式):
    基础分 = max(所有激活标签的 base_priority 权重)      P0=3 / P1=2 / P2=1
    情感系数 = meta['sentiment'][情感].weight            正面0.5 / 中性1.0 / 负面2.0
    升级加分 = Σ 命中的升级信号权重                      投诉维权 / 重复追问 / 长期未处理 ...
    多标签加分 = 0.3 × (标签数 - 1)                      一单里问题越多越该优先看

    总分 = 基础分 × 情感系数 + 升级加分 + 多标签加分

    总分 >= 6.0 -> P0 紧急(1h)    >= 3.0 -> P1 高(4h)    否则 P2 常规(24h)
    注: 阈值和权重都可以按业务实际调, 这里给的是"电商客诉"的常见配比.

对外接口:
    PriorityEngine().evaluate(labels, sentiment, text) -> dict
    evaluate_priority(labels, sentiment, text)         -> 便捷函数

运行方式:
    python 09-priority/priority_engine.py     # 内置样例 + 自测断言
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.ticket_data import load_label_meta            # noqa: E402

# 基础优先级 -> 数值权重(越大越急)
BASE_WEIGHT = {'P0': 3.0, 'P1': 2.0, 'P2': 1.0}
# 数值总分 -> 最终优先级
SCORE_THRESHOLDS = [(6.0, 'P0'), (3.0, 'P1')]
DEFAULT_PRIORITY = 'P2'
PRIORITY_CN = {'P0': '紧急', 'P1': '高', 'P2': '常规'}

# 升级信号: (关键词组, 权重, 说明)
# 每个组内部只要命中一次就计一次权, 不叠加 —— 否则用户把"投诉"写五遍就会虚高.
ESCALATION_SIGNALS = [
    (['投诉', '315', '消协', '工商局', '举报', '曝光', '起诉', '律师', '媒体', '监管部门'],
     2.5, '维权/升级投诉倾向'),
    (['第二次', '第三次', '又一次', '再次反馈', '反复', '已经很多次', '第4次'],
     1.5, '重复追问'),
    (['一直没', '还没', '仍未', '未处理', '没人管', '没人理', '不回复', '没回复', '没人处理'],
     1.5, '长期未处理'),
    (['退款没到', '退款未到', '没到账', '钱没到', '一直没退'],
     1.5, '资金未到账'),
    (['着急', '急用', '尽快', '马上', '今天必须', '急'],
     1.0, '用户明确催促'),
    (['老人', '孩子', '婴儿', '孕妇', '病人', '过敏', '受伤', '漏电', '起火', '安全隐患'],
     2.0, '涉及人身/安全隐患'),
]


class PriorityEngine:
    """基于业务规则的工单优先级判定"""

    def __init__(self, meta_path=None):
        meta = load_label_meta(meta_path)
        self.label_meta = meta.get('labels', {})
        self.sentiment_weight = {k: v.get('weight', 1.0)
                                 for k, v in meta.get('sentiment', {}).items()}
        self.priority_table = meta.get('priority', {})
        self.weights = {'base': BASE_WEIGHT, 'thresholds': SCORE_THRESHOLDS,
                        'signals': ESCALATION_SIGNALS}

    # ---------------- 基础分 ----------------
    def _base_score(self, labels):
        """所有激活标签里最急的那个作为基础分"""
        best, best_from = 0.0, []
        for lb in labels:
            prio = self.label_meta.get(lb, {}).get('base_priority', DEFAULT_PRIORITY)
            w = BASE_WEIGHT.get(prio, 1.0)
            if w > best:
                best, best_from = w, [lb]
            elif w == best:
                best_from.append(lb)
        if not labels:
            # 一个标签都没有(拒识)的情况: 按最低档走, 由人工复核决定
            return BASE_WEIGHT[DEFAULT_PRIORITY], []
        return best, best_from

    # ---------------- 升级信号 ----------------
    def _escalations(self, text):
        text = text or ''
        hits = []
        for keywords, weight, desc in ESCALATION_SIGNALS:
            matched = [k for k in keywords if k in text]
            if matched:
                hits.append({'group': desc, 'weight': weight, 'hits': matched[:5]})
        return hits

    # ---------------- 主入口 ----------------
    def evaluate(self, labels, sentiment='neutral', text=''):
        """判定工单优先级

        参数:
            labels    : 模型激活的标签列表(可空, 空表示拒识/无标签)
            sentiment : 'positive' / 'neutral' / 'negative'
            text      : 原始工单文本(用于提取升级信号)
        返回: dict(priority, cn, score, sla_hours, base..., reasons, need_manager ...)
        """
        labels = list(labels or [])
        base, base_from = self._base_score(labels)
        sent_w = self.sentiment_weight.get(sentiment, 1.0)
        escalations = self._escalations(text)
        esc_score = sum(e['weight'] for e in escalations)
        label_bonus = 0.3 * max(0, len(labels) - 1)

        score = base * sent_w + esc_score + label_bonus

        priority = DEFAULT_PRIORITY
        for threshold, prio in SCORE_THRESHOLDS:
            if score >= threshold:
                priority = prio
                break

        # SLA 取"优先级 SLA"与"标签 SLA"里更紧的那个 —— 宁可早处理, 不要晚处理
        sla = self.priority_table.get(priority, {}).get('sla_hours', 24)
        label_slas = [self.label_meta.get(lb, {}).get('sla_hours') for lb in labels]
        label_slas = [h for h in label_slas if h]
        if label_slas:
            sla = min(sla, min(label_slas))

        reasons = [
            f'基础分 {base:.1f} = 最急标签的 base_priority'
            + (f'({", ".join(base_from)})' if base_from else '(无标签, 按默认档)'),
            f'情感系数 ×{sent_w} (情感判定: {sentiment})',
        ]
        for e in escalations:
            reasons.append(f'升级信号 +{e["weight"]}: {e["group"]} -> 命中 {"/".join(e["hits"])}')
        if label_bonus:
            reasons.append(f'多标签加分 +{label_bonus:.1f} ({len(labels)} 个标签共现)')

        return {
            'priority': priority,
            'cn': PRIORITY_CN.get(priority, priority),
            'score': round(score, 3),
            'sla_hours': sla,
            'due_hint': f'建议 {sla} 小时内首次响应',
            'base_score': base,
            'base_priority_from': base_from,
            'sentiment': sentiment,
            'sentiment_weight': sent_w,
            'escalations': escalations,
            'label_count_bonus': round(label_bonus, 2),
            'need_manager': priority == 'P0',
            'reasons': reasons,
        }

    def dept_of(self, labels):
        """按"最急标签优先"给出建议处理部门(前端按部门分组展示要用)"""
        best_prio, best_dept = 99, '未分配'
        for lb in labels or []:
            meta = self.label_meta.get(lb, {})
            prio = meta.get('base_priority', DEFAULT_PRIORITY)
            rank = {'P0': 0, 'P1': 1, 'P2': 2}.get(prio, 3)
            if rank < best_prio:
                best_prio, best_dept = rank, meta.get('dept', '未分配')
        return best_dept


# ============================================================
# 模块级单例入口
# ============================================================
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = PriorityEngine()
    return _engine


def evaluate_priority(labels, sentiment='neutral', text=''):
    return get_engine().evaluate(labels, sentiment, text)


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare 09-priority 优先级引擎自测')
    print('=' * 72)
    cases = [
        # (标签, 情感, 文本, 期望优先级)
        (['consult'], 'neutral', '请问这个机械键盘的尺寸是多少', 'P2'),
        (['logistics'], 'negative', '快递三天了还没到', 'P1'),
        (['logistics'], 'negative', '快递一直没到, 我已经去315投诉了', 'P0'),
        (['payment_account'], 'negative', '重复扣款了两次, 钱一直没到账, 急用', 'P0'),
        (['invalid'], 'negative', '哈哈哈哈哈哈', 'P2'),
        (['quality', 'after_sale', 'invoice'], 'negative', '东西是坏的, 我要退款, 发票也没开, 已经投诉过了', 'P0'),
        ([], 'neutral', '这个和那个比哪个好一点呢', 'P2'),
    ]
    ok = 0
    for labels, senti, text, expect in cases:
        r = evaluate_priority(labels, senti, text)
        flag = 'OK ' if r['priority'] == expect else 'DIFF'
        if r['priority'] == expect:
            ok += 1
        print(f'\n[{flag}] {text}')
        print(f'      标签={labels} 情感={senti}')
        print(f'      -> {r["priority"]}({r["cn"]}) 总分={r["score"]} SLA={r["sla_hours"]}h '
              f'部门={get_engine().dept_of(labels)}')
        for reason in r['reasons']:
            print(f'         · {reason}')
    print(f'\n命中 {ok}/{len(cases)}')

    # 结构性断言
    r = evaluate_priority(['quality'], 'negative', '东西是坏的')
    assert r['priority'] in ('P0', 'P1', 'P2')
    assert set(r.keys()) >= {'priority', 'cn', 'score', 'sla_hours', 'reasons', 'need_manager'}
    assert evaluate_priority([], 'neutral', '')['priority'] == 'P2'
    # 投诉信号必须真的抬高优先级
    low = evaluate_priority(['logistics'], 'negative', '快递慢了')
    high = evaluate_priority(['logistics'], 'negative', '快递慢了, 我要投诉到消协')
    assert high['score'] > low['score'], (low['score'], high['score'])
    # 物流负面 1.0x2.0=2.0, 加'投诉'信号 +2.5 -> 4.5, 落在 P1 区间
    assert high['priority'] == 'P1', high
    # 再叠一个'长期未处理'信号: 2.0 + 2.5 + 1.5 = 6.0 -> 刚好够 P0
    worst = evaluate_priority(['logistics'], 'negative', '快递一直没到, 我要投诉到消协')
    assert worst['priority'] == 'P0', worst['score']
    print('\n[OK] priority_engine 自测通过')