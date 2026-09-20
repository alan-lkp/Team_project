"""
ShopCare 客服回复推荐 (10-reply-recommend)

定位:
    工单分好类之后, 客服最费时间的其实是"怎么回"。这个模块按**标签 + 优先级 + 情感**
    拼出一份可直接编辑的回复草稿, 让客服从"从零写"变成"审一遍改几个字".

为什么用模板而不是直接让 LLM 生成:
    1. **可控**: 涉及退款/赔付的措辞必须合规(不能随口承诺"全额退"), 模板里的承诺是法务审过的;
    2. **不会编**: LLM 容易编出"已为您优先派送"这类并不存在的操作;
    3. **快且免费**: 0ms, 无 API 成本, 也不会因为 API 挂了就不可用.
    需要更自然的文风时, 可以把这个草稿喂给 LLM 做润色 —— 输入输出都被模板框住了,
    属于"LLM 负责措辞, 规则负责事实"的分工, 风险比让 LLM 自由发挥低得多.

安全设计(needs_approval):
    下列情况**必须人工确认后才能发送**, 系统只给草稿, 不给"一键发送":
      * 优先级 P0(紧急工单, 说错话代价最大)
      * 涉及资金/权益的标签(退款退货、支付账号、价格优惠)
      * 已经出现维权/投诉信号
    这是"人在环上"的关键一环 —— 自动化率再高, 也得留一个闸门.

对外接口:
    ReplyRecommender().suggest(labels, priority, sentiment, text) -> dict
    suggest_reply(...)                                            -> 便捷函数

运行方式:
    python 10-reply-recommend/reply_recommender.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.ticket_data import load_label_meta            # noqa: E402

# 涉及资金的标签: 只要命中就必须人工审批
MONEY_LABELS = {'after_sale', 'payment_account', 'price_promo'}
# 出现这些词说明用户已经在升级诉求, 回复必须更谨慎
CLAIM_KEYWORDS = ['投诉', '315', '消协', '工商', '起诉', '举报', '曝光', '律师']

# 每个标签: 共情句 + 处理动作 + 需要用户补充的信息
REPLY_LIBRARY = {
    'logistics': {
        'empathy': '物流迟迟没有更新确实很让人着急',
        'actions': ['已为您拉取最新物流轨迹, 并联系承运商核实异常原因',
                    '若确认包裹停滞或丢失, 我们会直接为您补发或退款, 不需要您反复催促'],
        'need_info': ['订单号', '收件手机号后四位'],
    },
    'quality': {
        'empathy': '收到有问题的商品, 换谁都会不满意',
        'actions': ['已记录商品问题并同步品控与供应商核查同批次质量',
                    '请您提供问题部位的清晰照片或视频, 我们会第一时间判定并按售后政策处理'],
        'need_info': ['订单号', '商品问题照片/视频'],
    },
    'after_sale': {
        'empathy': '退款/退货流程没走完, 给您添麻烦了',
        'actions': ['已为您查询售后单当前节点, 并向售后专员加急催办',
                    '退款到账时间以支付渠道为准, 我们会持续跟进直到您收到款项'],
        'need_info': ['订单号', '售后单号(如有)'],
    },
    'invoice': {
        'empathy': '发票问题会影响您报销, 我们优先处理',
        'actions': ['已为您登记开票需求, 电子发票通常在 1 个工作日内发送到您的邮箱',
                    '若发票信息填写有误, 我们可为您作废重开'],
        'need_info': ['订单号', '开票抬头/税号', '接收邮箱'],
    },
    'price_promo': {
        'empathy': '优惠没生效或被多扣钱, 确实让人不痛快',
        'actions': ['已为您核对活动规则与订单实付金额',
                    '若确认属于系统原因导致的差价, 我们会按价保规则为您补差'],
        'need_info': ['订单号', '活动名称或优惠券截图'],
    },
    'payment_account': {
        'empathy': '支付/账号问题直接影响使用, 我们马上处理',
        'actions': ['已记录您的异常现象并同步技术侧排查支付通道与账号状态',
                    '涉及重复扣款的, 核实后会原路退回, 请放心'],
        'need_info': ['订单号', '支付流水号', '异常发生时间'],
    },
    'consult': {
        'empathy': '感谢您的咨询',
        'actions': ['已根据您的问题整理说明, 详见下方答复',
                    '如果还有不清楚的地方, 随时补充提问, 我们会继续为您解答'],
        'need_info': [],
    },
    'service': {
        'empathy': '客服服务体验不好, 是我们的问题, 向您道歉',
        'actions': ['已记录相关对话并进行质检复盘, 会对责任人进行辅导',
                    '我们已安排专人继续跟进您的问题, 不会再让您重复描述'],
        'need_info': ['订单号(便于定位之前的对话)'],
    },
    'invalid': {
        'empathy': '您好',
        'actions': ['您的消息中暂未识别到具体的售后诉求',
                    '如果是商品或订单相关问题, 麻烦补充订单号与具体描述, 我们会立即为您处理'],
        'need_info': ['具体问题描述', '订单号(如有)'],
    },
}

DEFAULT_TEMPLATE = {
    'empathy': '感谢您的反馈',
    'actions': ['我们已收到您的问题并开始处理'],
    'need_info': ['订单号'],
}

# 按优先级给出不同的响应时效承诺
SLA_LINE = {
    'P0': '我们已将该问题标记为**紧急**, 会在 1 小时内安排专人联系您; 给您带来的不便深表歉意。',
    'P1': '我们会在 4 小时内跟进处理, 请保持电话畅通。',
    'P2': '我们会在 24 小时内处理完毕, 感谢您的耐心等待。',
}

CLOSING = '再次为给您带来的不便致歉, 感谢您的理解与支持。'


class ReplyRecommender:
    """基于模板的回复草稿生成器"""

    def __init__(self, meta_path=None):
        meta = load_label_meta(meta_path)
        self.label_meta = meta.get('labels', {})
        self.priority_meta = meta.get('priority', {})

    # ---------------- 主入口 ----------------
    def suggest(self, labels, priority='P2', sentiment='neutral', text='', model_used=None):
        """生成回复草稿

        返回:
            reply           最终文本(客服可直接编辑后发送)
            sections        分段结构(前端可以分块高亮展示)
            template_ids    用到了哪些标签模板(便于统计)
            need_info       需要用户补充的信息
            needs_approval  是否必须人工确认
            approval_reason 需要确认的原因
            disclaimer      免责说明
        """
        labels = [lb for lb in (labels or []) if lb in REPLY_LIBRARY]
        template_ids = labels or ['invalid']
        need_info, actions, empathies = [], [], []

        for lb in template_ids:
            tpl = REPLY_LIBRARY.get(lb, DEFAULT_TEMPLATE)
            empathies.append(tpl['empathy'])
            actions.extend(tpl['actions'])
            for item in tpl.get('need_info', []):
                if item not in need_info:
                    need_info.append(item)

        # 去重并保持顺序(多标签时不同模板可能有重复动作)
        actions = list(dict.fromkeys(actions))

        labels_cn = '、'.join(self._cn(lb) for lb in template_ids)
        lines = [
            '尊敬的客户, 您好:',
            '',
            f'{ empathies[0] if empathies else DEFAULT_TEMPLATE["empathy"] }。',
            f'我们已识别到您反馈的问题类型: {labels_cn}。',
            '',
            '处理方案:',
        ]
        for i, act in enumerate(actions, 1):
            lines.append(f'  {i}. {act}')
        if need_info:
            lines.append('')
            lines.append('为了更快定位问题, 麻烦您补充以下信息: ' + '、'.join(need_info) + '。')
        if priority in SLA_LINE:
            lines.append('')
            lines.append(SLA_LINE[priority])
        lines.append('')
        lines.append(CLOSING)
        lines.append('—— ShopCare 客服中心')
        reply = '\n'.join(lines)

        needs_approval, reason = self._approval(priority, template_ids, sentiment, text)
        return {
            'reply': reply,
            'sections': [
                {'title': '共情', 'content': empathies[0] if empathies else DEFAULT_TEMPLATE['empathy']},
                {'title': '处理动作', 'content': actions},
                {'title': '待补充信息', 'content': need_info},
                {'title': '时效承诺', 'content': SLA_LINE.get(priority, '')},
            ],
            'template_ids': template_ids,
            'labels_cn': labels_cn,
            'need_info': need_info,
            'needs_approval': needs_approval,
            'approval_reason': reason,
            'model_used': model_used,
            'disclaimer': '本回复由模板自动生成, 承诺内容需客服确认后方可发送',
        }

    # ---------------- 内部 ----------------
    def _cn(self, label):
        return self.label_meta.get(label, {}).get('cn', label)

    @staticmethod
    def _approval(priority, labels, sentiment, text):
        """判断是否需要人工审批, 并给出原因"""
        reasons = []
        if priority == 'P0':
            reasons.append('优先级为 P0(紧急工单)')
        hit_money = sorted(set(labels) & MONEY_LABELS)
        if hit_money:
            reasons.append('涉及资金/权益: ' + '、'.join(hit_money))
        if any(k in (text or '') for k in CLAIM_KEYWORDS):
            reasons.append('用户已出现维权/投诉倾向')
        if sentiment == 'negative' and priority in ('P0', 'P1'):
            reasons.append('负面情绪 + 高优先级, 措辞风险高')
        return bool(reasons), ('; '.join(reasons) if reasons else None)


# ============================================================
# 模块级单例入口
# ============================================================
_recommender = None


def get_recommender():
    global _recommender
    if _recommender is None:
        _recommender = ReplyRecommender()
    return _recommender


def suggest_reply(labels, priority='P2', sentiment='neutral', text='', model_used=None):
    return get_recommender().suggest(labels, priority, sentiment, text, model_used)


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare 10-reply-recommend 回复推荐自测')
    print('=' * 72)

    r1 = suggest_reply(['logistics'], 'P1', 'negative', '快递一直没到, 已经投诉了')
    print('\n---- 案例 1: 物流 + 负面 + 已投诉 ----')
    print(r1['reply'])
    print('\n需要审批:', r1['needs_approval'], '| 原因:', r1['approval_reason'])
    assert r1['needs_approval'] is True, '出现投诉关键词必须人工审批'
    assert 'logistics' in r1['template_ids']

    r2 = suggest_reply(['consult'], 'P2', 'neutral', '请问这个键盘的尺寸')
    print('\n---- 案例 2: 售前咨询(无需审批) ----')
    print(r2['reply'])
    print('\n需要审批:', r2['needs_approval'])
    assert r2['needs_approval'] is False, '普通咨询不该要求审批'
    assert r2['need_info'] == []

    r3 = suggest_reply(['quality', 'after_sale', 'invoice'], 'P0', 'negative', '东西坏了要退款, 发票也没开')
    print('\n---- 案例 3: 多标签 + P0 ----')
    print(r3['reply'])
    print('\n需要审批:', r3['needs_approval'], '| 原因:', r3['approval_reason'])
    assert r3['needs_approval'] is True
    assert '涉及资金/权益' in (r3['approval_reason'] or '')
    assert len(r3['template_ids']) == 3
    # 多标签时待补充信息应该合并去重
    assert len(r3['need_info']) == len(set(r3['need_info']))

    r4 = suggest_reply([], 'P2', 'neutral', 'aaa')
    print('\n---- 案例 4: 无标签(拒识) ----')
    assert r4['template_ids'] == ['invalid']

    assert set(r1.keys()) >= {'reply', 'sections', 'needs_approval', 'disclaimer', 'need_info'}
    print('\n[OK] reply_recommender 自测通过')