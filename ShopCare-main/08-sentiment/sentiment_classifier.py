"""
ShopCare 情感倾向识别 (08-sentiment)

为什么不用 BERT 做情感:
    情感识别在这套系统里是**辅助信号**, 只为 09-priority 提供"要不要加急"的一个乘数.
    为它再训一个深度学习模型, 收益远小于成本. 所以这里用**词典 + 规则**的做法:
      * 零依赖(纯标准库), 任何环境都能跑;
      * 完全可解释 —— 能明确告诉客服"因为出现了'恶臭''推诿'这两个词, 所以判为负面";
      * 单条 0.1ms 级, 不会成为接口的瓶颈.

规则里处理的几个中文难点:
    1. **否定翻转**: "不是不好" 不能简单当成两个负面词. 做法是: 负面词前面 3 个字内出现
       "不/没/别/无/未/免/甭" 时, 把该词的极性强度打折(而不是直接反转成正面 ——
       客服场景里"没有解决问题"整体仍然是负面的, 直接反转会错得很离谱).
    2. **程度副词加权**: "非常差" 比 "差" 重; "有点慢" 比 "慢" 轻.
    3. **标点强调**: "！！！" "???" 密集出现通常意味着情绪激动, 给一个整体加成.
    4. **电商专有负面词**: "恶臭""推诿""踢皮球""三无""假货" 这类词通用情感词典往往没有,
       但对工单场景判断极准, 所以这里专门收录.

对外接口:
    SentimentClassifier().score_text(text) -> dict(label, cn, score, confidence, hits...)
    predict_sentiment(text)                -> 单条便捷函数

运行方式:
    python 08-sentiment/sentiment_classifier.py     # 跑内置样例 + 自测断言
"""

import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# todo 1. 情感词典(权重范围 -3 ~ +3, 绝对值表示强度)
# ------------------------------------------------------------
# 负面: 按"问题严重程度"分级, 而不是一刀切 1.0 —— 这样分数才有区分度
NEGATIVE_LEXICON = {
    # 极端负面(涉及维权/欺诈/人身攻击): 3
    '恶臭': 3.0, '假货': 3.0, '三无': 3.0, '欺诈': 3.0, '诈骗': 3.0, '骗人': 3.0,
    '垃圾东西': 3.0, '投诉': 2.6, '起诉': 3.0, '消协': 3.0, '工商': 2.8, '315': 3.0,
    '举报': 2.8, '曝光': 2.6, '差评': 2.4, '退一赔三': 3.0, '恶心': 2.8, '恶劣': 2.8,
    '辱骂': 3.0, '威胁': 3.0, '拉黑': 2.6, '再也不买': 2.6, '卸载': 2.2,
    # 强负面(核心诉求未被满足): 2
    '坏了': 2.0, '坏的': 2.0, '破损': 2.0, '损坏': 2.0, '少件': 2.0, '漏发': 2.0,
    '过期': 2.2, '变质': 2.2, '发霉': 2.4, '瑕疵': 1.8, '划痕': 1.8, '故障': 2.0,
    '不能用': 2.2, '用不了': 2.2, '打不开': 2.0, '退货': 1.6, '退款': 1.6, '换货': 1.4,
    '没收到': 2.0, '未收到': 2.0, '丢件': 2.6, '丢失': 2.4, '迟迟': 2.0, '停滞': 2.0,
    '推诿': 2.6, '踢皮球': 2.6, '敷衍': 2.4, '态度差': 2.6, '态度不好': 2.6, '态度恶劣': 3.0,
    '没人管': 2.6, '没人理': 2.6, '不理人': 2.4, '爱答不理': 2.6, '不回复': 2.2, '不回消息': 2.2,
    '推脱': 2.4, '扯皮': 2.4, '忽悠': 2.6, '欺骗': 2.8, '虚假': 2.6, '承诺未兑现': 2.4,
    '缺货': 1.6, '错发': 2.0, '发错': 2.0, '多扣': 2.4, '重复扣款': 2.6, '乱扣': 2.4,
    # 直接收录完整短语: 纯靠'否定词+正面词'的规则推导很容易漏,
    # 而'问题没被解决'恰恰是客服场景里最该判成负面的表达.
    '没解决': 2.2, '没有解决': 2.2, '未解决': 2.2, '没处理好': 2.0, '不处理': 2.2, '不给解决': 2.4,
    '扣款失败': 1.6, '支付失败': 1.8, '登录异常': 2.0, '被冻结': 2.4, '封禁': 2.4,
    '优惠券失效': 1.6, '没生效': 1.8, '不给补': 2.2, '不补差价': 2.2, '拒绝': 2.2,
    '不理': 2.0, '拖延': 2.2, '超时': 2.0, '等了很久': 2.0, '很久': 1.4, '太久': 1.8,
    # 一般负面(抱怨语气): 1
    '慢': 1.2, '太慢': 1.8, '很慢': 1.6, '烦': 1.4, '着急': 1.2, '急': 1.0, '着急用': 1.4,
    '差': 1.6, '很差': 2.2, '太差': 2.2, '不好': 1.6, '不行': 1.6, '不满意': 1.8,
    '失望': 2.0, '生气': 2.2, '气愤': 2.4, '无语': 1.6, '崩溃': 2.4, '难受': 1.6,
    # '麻烦' 故意不收录: '麻烦问下''麻烦帮我看看' 在电商语境里是礼貌开场白而不是抱怨,
    # 把它当负面词是情感词典最典型的坑.
    '问题': 0.8, '延误': 1.8, '延迟': 1.4, '慢了': 1.4, '卡住': 1.6,
    '还没': 1.2, '一直没': 1.8, '怎么还': 1.6, '到底': 1.2, '凭什么': 2.0,
    '为什么还': 1.6, '搞什么': 1.8, '什么情况': 1.2, '无语了': 2.0, '服了': 1.8,
}

POSITIVE_LEXICON = {
    '满意': 2.0, '非常满意': 2.6, '谢谢': 1.2, '感谢': 1.4, '辛苦了': 1.4,
    '很好': 2.0, '不错': 1.6, '挺好': 1.6, '喜欢': 1.6, '赞': 1.6, '给力': 1.8,
    '解决了': 2.0, '已解决': 2.0, '处理好了': 2.0, '处理很快': 2.2, '及时': 1.6,
    '速度很快': 2.0, '态度好': 2.0, '服务好': 2.0, '耐心': 1.8, '专业': 1.6,
    '好评': 2.0, '五星': 2.0, '值得': 1.4, '推荐': 1.6, '棒': 1.6, '完美': 2.0,
    '开心': 1.8, '谢谢你们': 2.0, '麻烦你了': 1.2, '好的呢': 1.0,
}

# 否定词: 出现在情感词前面时削弱其强度
# 长词排前面: 匹配时按长度降序, '不是很' 必须先于 '不' 命中
NEGATION_WORDS = ['不是很', '不是', '没什么', '一点也不', '不', '没', '没有', '未', '别', '无', '非', '免', '甭', '不用', '不要']
# 程度副词: 出现在情感词前面时放大强度
INTENSIFIERS = {
    '非常': 1.6, '特别': 1.5, '太': 1.5, '超': 1.5, '极其': 1.8, '极': 1.6,
    '十分': 1.5, '相当': 1.4, '真的': 1.3, '真是': 1.3, '巨': 1.5, '贼': 1.4,
    '有点': 0.6, '稍微': 0.5, '稍稍': 0.5, '略微': 0.5, '还算': 0.6, '比较': 0.8, '挺': 1.1,
}

# 礼貌开场白: 打分前先屏蔽掉.
# 为什么需要这一步: '麻烦问下' 里的 '烦' 恰好是词典里的负面词,
# 这类'单字恰好是某个负面词的一部分'的问题靠调权重解决不了, 只能先把噪声段摘掉.
POLITE_OPENERS = ['麻烦问一下', '麻烦帮我看看', '麻烦帮我', '麻烦问下', '麻烦看下', '麻烦看看',
                  '麻烦您', '麻烦你', '麻烦给我', '麻烦了',
                  '请问一下', '请问下', '请问', '想问一下', '想问下', '咨询一下', '咨询下']

# 判定阈值: |score| 超过它才算正/负面, 之间算中性
POSITIVE_THRESHOLD = 0.20
NEGATIVE_THRESHOLD = -0.20


class SentimentClassifier:
    """词典 + 规则的中文工单情感分类器(纯标准库, 无需分词器)"""

    def __init__(self, lexicon=None):
        self.neg = dict(NEGATIVE_LEXICON)
        self.pos = dict(POSITIVE_LEXICON)
        if lexicon:                      # 支持外部覆盖/扩充词典
            self.neg.update(lexicon.get('negative', {}))
            self.pos.update(lexicon.get('positive', {}))
        # 长词优先匹配: "非常满意" 必须比 "满意" 先命中, 否则强度会被算低
        self._neg_sorted = sorted(self.neg, key=len, reverse=True)
        self._pos_sorted = sorted(self.pos, key=len, reverse=True)

    # ---------------- 内部工具 ----------------
    @staticmethod
    def _scan(text, words, window=3):
        """在文本里找情感词, 返回 [(词, 基础权重, 前置修饰)] ; 长词命中后占用区间避免重复计分"""
        hits, taken = [], []
        for w in words:
            start = 0
            while True:
                i = text.find(w, start)
                if i < 0:
                    break
                span = range(i, i + len(w))
                # 已被更长的词覆盖(例如"非常满意"命中后,"满意"不再单独计分)
                if not any(j in taken for j in span):
                    prefix = text[max(0, i - window):i]
                    hits.append((w, i, prefix))
                    taken.extend(span)
                start = i + 1
        return hits

    @staticmethod
    def _modifier(prefix):
        """根据前置文本判断放大/削弱系数, 返回 (系数, 说明)"""
        factor, negated, note = 1.0, False, []
        for word, mult in INTENSIFIERS.items():
            if word in prefix:
                factor *= mult
                note.append(f'程度[{word}]×{mult}')
                break
        # 否定词可能被程度副词隔开: '不是很满意' 里否定词在 '很' 前面,
        # 所以先剥掉尾部的程度副词, 再看是否以否定词结尾.
        tail = prefix
        for word in sorted(INTENSIFIERS, key=len, reverse=True):
            if tail.endswith(word):
                tail = tail[:-len(word)]
                break
        for word in sorted(NEGATION_WORDS, key=len, reverse=True):
            if tail.endswith(word):
                negated = True
                note.append(f'否定[{word}]')
                break
        return factor, negated, note

    # ---------------- 核心 ----------------
    def score_text(self, text):
        """返回情感判定结果

        分数 score ∈ [-1, 1], 由原始加权分经 tanh 压缩得到.
        """
        text = (text or '').strip()
        for opener in POLITE_OPENERS:            # 见 POLITE_OPENERS 的注释
            text = text.replace(opener, ' ')
        text = text.strip()
        if not text:
            return {'label': 'neutral', 'cn': '中性', 'score': 0.0, 'confidence': 0.0,
                    'positive_hits': [], 'negative_hits': [], 'modifiers': [],
                    'exclamations': 0, 'reason': '文本为空'}

        raw = 0.0
        pos_hits, neg_hits, modifiers = [], [], []

        for w, idx, prefix in self._scan(text, self._pos_sorted):
            factor, negated, note = self._modifier(prefix)
            # 否定一个正面词 -> 温和的**负面**("不是很满意" 是抱怨, 不是轻度满意).
            # 如果只是把正面强度打折, 结果仍然是正的, 方向就错了.
            contrib = -0.6 * self.pos[w] if negated else self.pos[w] * factor
            raw += contrib
            pos_hits.append({'word': w, 'base': self.pos[w], 'factor': round(factor, 2),
                             'negated': negated, 'score': round(contrib, 2)})
            modifiers.extend(note)

        for w, idx, prefix in self._scan(text, self._neg_sorted):
            factor, negated, note = self._modifier(prefix)
            # 否定一个负面词 -> 温和的**正面**("没坏" 说明东西是好的).
            # 注意只给 0.3 倍力度: 中文里"没坏"远没有"很好"那么正面.
            contrib = 0.3 * self.neg[w] if negated else -self.neg[w] * factor
            raw += contrib
            neg_hits.append({'word': w, 'base': self.neg[w], 'factor': round(factor, 2),
                             'negated': negated, 'score': round(contrib, 2)})
            modifiers.extend(note)

        # 标点强调: 连续感叹号/问号是情绪激动的强信号
        exclam = len(re.findall(r'[!！]{2,}', text)) + len(re.findall(r'[?？]{2,}', text))
        if exclam and neg_hits:
            raw -= 0.8 * min(exclam, 3)

        # 全大写英文或大量重复字符(啊啊啊)也算情绪信号
        if re.search(r'(.)\1{3,}', text) and neg_hits:
            raw -= 0.5

        import math
        score = math.tanh(raw / 4.0)          # 除以 4 让"两三个强负面词"就能接近饱和
        if score >= POSITIVE_THRESHOLD:
            label, cn = 'positive', '正面'
        elif score <= NEGATIVE_THRESHOLD:
            label, cn = 'negative', '负面'
        else:
            label, cn = 'neutral', '中性'

        # 置信度: 分数越接近阈值边界越不自信; 无任何命中时给中性一个中等置信度
        if not pos_hits and not neg_hits:
            confidence = 0.55
        else:
            confidence = min(0.99, 0.5 + abs(score))

        return {
            'label': label,
            'cn': cn,
            'score': round(score, 4),
            'confidence': round(confidence, 4),
            'positive_hits': pos_hits,
            'negative_hits': neg_hits,
            'modifiers': modifiers,
            'exclamations': exclam,
            'reason': self._explain(label, pos_hits, neg_hits, exclam),
        }

    @staticmethod
    def _explain(label, pos_hits, neg_hits, exclam):
        # 注意: 被否定的正面词('不是很满意')贡献是负的, 属于负面证据;
        # 被否定的负面词('没坏')贡献是正的, 属于正面证据.
        # 所以不能只看'命中了哪本词典', 要看每个命中对总分的实际方向.
        neg_evidence = [h for h in neg_hits + pos_hits if h['score'] < 0]
        pos_evidence = [h for h in neg_hits + pos_hits if h['score'] > 0]
        if label == 'negative':
            top = '、'.join(h['word'] for h in sorted(neg_evidence, key=lambda x: x['score'])[:4])
            return f'命中负面词: {top}' + (f'; 含 {exclam} 处连续感叹/问号' if exclam else '')
        if label == 'positive':
            top = '、'.join(h['word'] for h in sorted(pos_evidence, key=lambda x: -x['score'])[:4])
            return f'命中正面词: {top}'
        return '未命中足够强的情感词, 判为中性'


# ============================================================
# 模块级单例入口
# ============================================================
_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = SentimentClassifier()
    return _classifier


def predict_sentiment(text):
    """单条便捷函数(backend 直接调用)"""
    return get_classifier().score_text(text)


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare 08-sentiment 情感识别自测')
    print('=' * 72)
    cases = [
        ('收到的电饭锅是坏的，根本用不了，太失望了！', 'negative'),
        ('快递到广州十天了还没动静，客服也不回复，没人管吗', 'negative'),
        ('态度特别恶劣，还威胁我，我要去315投诉你们！！！', 'negative'),
        ('这个挺好的，谢谢客服耐心解答，很满意', 'positive'),
        ('麻烦问下这个机械键盘的尺寸是多少', 'neutral'),
        ('请问什么时候发货', 'neutral'),
        # 否定与程度副词: 这两条最容易出错, 专门测
        ('没有解决我的问题', 'negative'),
        ('有点慢，但也能接受吧', 'neutral'),
        ('不是很满意', 'negative'),
    ]
    ok = 0
    for text, expect in cases:
        r = predict_sentiment(text)
        flag = 'OK ' if r['label'] == expect else 'DIFF'
        if r['label'] == expect:
            ok += 1
        print(f'\n[{flag}] {text}')
        print(f'      -> {r["cn"]}({r["label"]}) score={r["score"]} conf={r["confidence"]}')
        print(f'      依据: {r["reason"]}')
    print(f'\n命中 {ok}/{len(cases)}')
    # 结构性断言(不依赖具体分数, 免得改词典就崩)
    assert predict_sentiment('')['label'] == 'neutral'
    assert predict_sentiment('态度恶劣，投诉！')['label'] == 'negative'
    assert predict_sentiment('非常满意，谢谢')['label'] == 'positive'
    assert set(predict_sentiment('坏').keys()) >= {'label', 'cn', 'score', 'confidence', 'reason'}
    print('\n[OK] sentiment_classifier 自测通过')