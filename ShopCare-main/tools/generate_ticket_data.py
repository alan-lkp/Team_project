"""
ShopCare 电商客服工单语料生成器 (P1: 把链路先跑通)

功能说明:
    按电商客服真实场景生成**多标签**工单数据: 一条工单可同时命中 1~3 个标签,
    用于在真实业务数据到位之前, 跑通 "数据 -> 基线 -> BERT+LoRA -> 拒识 -> 接口 -> 页面" 全链路.

    生成语料的特点:
      1. 口语化: 含寒暄前缀、语气词、连续标点、残缺短句;
      2. 多标签: 45% 单标签 / 40% 双标签 / 15% 三标签, 模拟"一口气吐槽好几件事"的混合诉求工单;
      3. 长尾分布: consult / logistics / invalid 偏多, invoice / payment_account 偏少;
      4. 天然难例: 多标签重叠 + 表述模糊 —— 正是难例动态加权采样要重点学习的样本.

输出:
    01-data/train.txt, 01-data/dev.txt, 01-data/test.txt
    每行格式: 文本<TAB>标签1,标签2   (标签名与 01-data/class.txt 严格一致)

运行方式:
    python tools/generate_ticket_data.py                             # 默认 4000/600/600
    python tools/generate_ticket_data.py --n_train 20000 --n_dev 2000 --n_test 2000
    python tools/generate_ticket_data.py --preview 10                # 生成后预览 10 条

[!] 不要再用本脚本出指标 —— 它有模板级泄漏
=========================================
本脚本对 train / dev / test **各自独立随机生成**。整句去重看着没问题(交集为 0),
但所有 split 都从同一批子句模板池里采样, 所以 test 里 96% 的子句都能在 train 中原样
找到 —— 模型学到的是「子句 -> 标签」的查表规则, 于是 RF / BERT 都会拿到 Micro-F1=1.000,
这个指标没有任何区分度.

需要重新生成语料时请改用:

    python tools/build_dataset_v2.py

它做了三层修复(扩池到 540 条模板 / 家族分组切分 / 槽位隔离), test 子句在 train 中
原样出现的比例是 0.0000%. 详见该脚本的 docstring 与 README「三套对照模型」一节.

本脚本保留仅作对照与历史参考.

注意:
    本生成器只用于跑通工程链路与演示, 指标仅供参照, 不代表真实业务效果;
    真实数据请按 01-data/data_format.md 的格式替换 (可用 tools/check_dataset.py 做单标签升维).
"""

import argparse
import os
import random
import sys
from collections import Counter

# 项目根目录: 本文件位于 tools/ 下, 向上取一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, '01-data')

# ============================================================
# todo 1. 标签体系 (必须与 01-data/class.txt 严格一致, 顺序即索引)
# ============================================================
CLASSES = [
    'logistics',        # 物流配送
    'quality',          # 商品质量
    'after_sale',       # 退款退货
    'invoice',          # 发票问题
    'price_promo',      # 价格与优惠
    'payment_account',  # 支付与账号
    'consult',          # 售前与使用咨询
    'service',          # 服务态度
    'invalid',          # 无效与恶意
]

# 标签采样权重: 贴近真实业务长尾分布(咨询/物流/垃圾反馈多, 发票/支付账号少)
LABEL_WEIGHT = {
    'consult': 1.40,
    'logistics': 1.25,
    'invalid': 1.05,
    'after_sale': 1.10,
    'quality': 1.00,
    'service': 0.85,
    'price_promo': 0.75,
    'invoice': 0.60,
    'payment_account': 0.50,
}

# 标签个数分布: 1/2/3 个标签. 注意 invalid 独占(见 _make_ticket),
# 所以实际平均标签数会低于这里的期望值, 调参目标是落在 1.6 ~ 2.2
LABEL_CNT_DIST = [(1, 0.38), (2, 0.45), (3, 0.17)]

# ============================================================
# todo 2. 槽位词表 (模板中可替换的变量)
# ============================================================
GOODS = ['手机', '蓝牙耳机', '羽绒服', '运动鞋', '连衣裙', '面膜', '奶粉', '电饭锅',
         '保温杯', '充电宝', '笔记本电脑', '机械键盘', '台灯', '行李箱', '猫粮',
         '洗发水', '床垫', '扫地机器人', '电动牙刷', '空气炸锅']

PLACES = ['北京', '广州', '武汉', '郑州', '西安', '杭州', '成都', '沈阳']

DAYS = ['3', '5', '7', '10', '15', '20']

# 工单前缀(寒暄) 与 后缀(诉求), 让文本更像真人发的消息
PREFIXES = ['你好，', '在吗，', '客服你好，', '亲，', '请问一下，', '麻烦问下，', '']
SUFFIXES = ['麻烦尽快处理', '给个说法吧', '谢谢', '急', '希望能解决', '请回复我',
            '一直没人管', '实在是没办法了', '']
TONE_WORDS = ['啊', '呀', '吧', '呢', '了', '哦']
PUNCT_NOISE = ['！！！', '。。。', '!!', '??', '……']

# ============================================================
# todo 3. 各标签的句式模板 (每条模板 = 用户会怎么吐槽/怎么问)
# ============================================================
LABEL_TEMPLATES = {
    'logistics': [
        '买的{goods}都{day}天了还没发货，到底什么时候能发',
        '快递在{place}停了一周了，一直没有物流更新',
        '{goods}显示已发出，但是一直没揽收',
        '物流信息三天没动过了，是丢件了吗',
        '说好次日达的，结果{day}天还没送到',
        '快递员说放门口就行，东西到现在都没看到',
        '地址填错了想改一下，快递还没派送',
        '包裹一直卡在{place}中转站出不来',
        '东西到了但是没送到我家，显示已签收',
        '快递放驿站也不通知我，自己去翻才找到',
        '发货太慢了，等得我都没耐心了',
        '物流轨迹一直不更新，问客服也查不到',
        '同城发了五天，走路都送到了',
        '外包装烂得不成样子，里面的{goods}不知道有没有事',
        '快递员不肯送上楼，让我自己下来搬',
        '想问问我的{goods}什么时候能到，急用',
    ],
    'quality': [
        '{goods}收到就是坏的，根本用不了',
        '面料和图片完全不一样，做工也很粗糙',
        '{goods}用了两天就出故障了，质量太差',
        '收到的{goods}有明显划痕，像是别人退回来的',
        '说是正品，收到的包装和专柜完全不一样',
        '{goods}少发了一个配件，包装里就一个主体',
        '买的{goods}已经过期了，生产日期是去年的',
        '{goods}味道特别刺鼻，闻着头疼',
        '尺码严重偏小，标注的和实际差太多',
        '{goods}用一次就坏了，这质量也太差了',
        '收到的{goods}上有污渍，看着很不舒服',
        '实物和详情页描述不一致，颜色差很多',
        '{goods}电池不耐用，充一次电用不到两小时',
        '包装里少了一件，订单上明明写的是两件',
        '{goods}有异味，洗了好几次都散不掉',
        '刚拆开{goods}就发现屏幕有裂纹',
    ],
    'after_sale': [
        '我要退款，{goods}我不想再要了',
        '申请退货好几天了，一直没人处理',
        '退款都一个星期了还没到账，钱去哪了',
        '想换成大一码的，怎么走换货流程',
        '申请退款被拒绝了，说我超时，可我一直没收到货',
        '退货的快递已经签收了，退款什么时候能退',
        '{goods}坏了想维修，保修期内怎么处理',
        '我要退货，运费谁承担',
        '退款金额不对，少退了运费',
        '售后一直卡在审核中，能不能快点',
        '退货地址给一下，我自己寄回去',
        '换货申请通过了但是一直没发货',
        '已经寄回一周了，退款还是没有动静',
        '想取消订单，还没发货应该可以退吧',
        '售后电话打不通，只能来这里申请退款',
        '{goods}寄回去了，什么时候能给我退钱',
    ],
    'invoice': [
        '买完想开个发票，在哪里申请',
        '发票抬头写错了，能重开一张吗',
        '订单已经完成了，但是发票一直没收到',
        '要开公司抬头的专票，需要提供什么资料',
        '电子发票下载不了，链接点开是空的',
        '发票金额和订单金额对不上',
        '能不能把发票寄过来，我需要纸质版',
        '上个月开的发票现在还能补开吗',
        '报销要用发票，麻烦尽快开一下',
        '发票税号写错了，要作废重开',
        '同一个订单开两张发票可以吗',
        '发票内容想开成办公用品，可以改吗',
    ],
    'price_promo': [
        '刚买完就降价了，能补差价吗',
        '优惠券显示可以使用，结算的时候又用不了',
        '说好的满减活动，结算没给我减',
        '我领的券怎么突然失效了',
        '为什么我买的价格比别人的贵',
        '活动页面写的是买一送一，实际只发了一件',
        '赠品没有发，订单上明明写着有赠品',
        '多扣了我一笔钱，订单金额对不上',
        '秒杀价买的东西，付款的时候变回原价了',
        '会员折扣没有生效，白开会员了',
        '跨店满减没算进去，少减了三十块',
        '价格保护怎么申请，规则我没看懂',
    ],
    'payment_account': [
        '付款的时候一直失败，换了好几张卡都不行',
        '钱扣了但是订单显示未支付，麻烦查一下',
        '莫名其妙被扣了两次款，重复支付了',
        '我的账号登不上去了，提示被限制',
        '账号被冻结了，里面的优惠券还能用吗',
        '支付完订单消失了，钱也没退回来',
        '想改绑手机号，收不到验证码',
        '订单一直显示待支付，但是我明明付过了',
        '支付密码忘了，怎么重置',
        '账号被盗了，有一笔不是我下的单',
    ],
    'consult': [
        '这款{goods}支持无线充电吗',
        '想问下{goods}的尺寸是多少，适合多大的房间用',
        '这个{goods}现在有货吗，多久能发',
        '怎么安装{goods}，有教程吗',
        '这个{goods}适合三岁小孩用吗',
        '{goods}的保修期是多久',
        '想问问这个型号和另一个型号有什么区别',
        '{goods}怎么连接手机蓝牙，说明书看不懂',
        '这个{goods}能带上飞机吗',
        '买两件有没有优惠，想先咨询一下',
        '参数里的这个功能是什么意思，没看懂',
        '收到{goods}以后怎么激活，需要注册吗',
        '想问下退货规则，先了解一下再决定买不买',
        '{goods}和赠品的配件是一起发吗',
    ],
    'service': [
        '客服半天不回消息，问了三遍都没人理',
        '客服态度特别差，说话很不客气',
        '转接了四个客服，问题还是没解决',
        '客服一直敷衍我，让我自己看规则',
        '之前承诺24小时回复，到现在也没人联系我',
        '客服把责任推给快递，让我自己去找',
        '打客服电话一直占线，在线客服也排不上队',
        '客服答应给我补发，结果一直没动静',
        '问个问题被客服怼了，态度非常不好',
        '客服说帮我登记了让我等，等了三天没人管',
        '每个客服说的都不一样，到底听谁的',
        '机器人客服绕来绕去，根本转不到人工',
    ],
    'invalid': [
        '哈哈哈哈哈哈哈哈',
        '在吗',
        '测试测试测试',
        '这家店的东西都是垃圾，大家千万别买，差评差评差评',
        '加我微信可以领内部优惠',
        '???',
        '1234567890',
        '别买，都是骗人的，我已经举报了',
        '你们这个系统真烂，我要投诉到底，天天投诉',
        'asdfghjkl',
        '刷单了解一下，需要的私聊',
        '客服在吗在吗在吗在吗在吗',
        '随便点几下看看',
        '买它买它买它，这个价格太划算了，五星好评好评好评',
        '发错了吧，我没买过这个东西啊',
    ],
}

# 无效/恶意 工单的短句补充前缀, 用于制造足够多的"重复灌水"变体
INVALID_PREFIXES = ['', '在吗 ', '。。。', '??? ', ' ', '急 ', '测试 ', '啊啊啊 ', '！！！ ']


def _choice_by_weight(weights):
    """按权重字典随机取一个 key (不依赖第三方库的轻量实现)"""
    keys = list(weights.keys())
    values = [weights[k] for k in keys]
    return random.choices(keys, weights=values, k=1)[0]


def _fill(template):
    """把模板中的槽位替换成随机词"""
    return template.format(goods=random.choice(GOODS),
                           place=random.choice(PLACES),
                           day=random.choice(DAYS))


def _make_clause(label):
    """针对某个标签生成一句用户原话"""
    clause = _fill(random.choice(LABEL_TEMPLATES[label]))
    if label == 'invalid' and random.random() < 0.6:
        clause = (random.choice(INVALID_PREFIXES) + clause).strip()
    # 语气词: 让口语化程度更高
    if random.random() < 0.12 and clause[-1] not in '？?！!。吗呢吧啊呀了的':
        clause = clause + random.choice(TONE_WORDS)
    return clause


def _make_ticket():
    """生成一条工单: 先定标签组合, 再为每个标签生成一句话拼接起来"""
    n_labels = random.choices([c for c, _ in LABEL_CNT_DIST],
                              weights=[w for _, w in LABEL_CNT_DIST], k=1)[0]
    labels = []
    while len(labels) < n_labels:
        label = _choice_by_weight(LABEL_WEIGHT)
        if label not in labels:
            labels.append(label)
    # 无效/恶意反馈是整条工单无价值, 按 docs/标签规范.md 的边界规则不与其它标签共存:
    # 一旦命中 invalid 就只保留该标签, 避免出现 退款+恶意 这种自相矛盾的标注
    if 'invalid' in labels:
        labels = ['invalid']
    # 标签按 class.txt 索引升序排列, 与数据规范保持一致
    labels = sorted(labels, key=lambda x: CLASSES.index(x))

    parts = [_make_clause(label) for label in labels]
    # 多标签工单用连接词拼接, 模拟"一口气吐槽好几件事"
    joiner = random.choice(['，', '。', '，还有', '，另外', '，而且', '，顺便说下'])
    text = joiner.join(parts)

    if random.random() < 0.35:
        text = random.choice(PREFIXES) + text
    if random.random() < 0.30:
        text = text + '，' + random.choice(SUFFIXES)
    if random.random() < 0.15:
        text = text + random.choice(PUNCT_NOISE)

    return text, labels


def generate(n, max_dup=3):
    """生成 n 条工单; 同一文本最多重复 max_dup 次(灌水类反馈现实中确实会重复)"""
    rows = []
    seen = Counter()
    attempts = 0
    while len(rows) < n and attempts < n * 100:
        attempts += 1
        text, labels = _make_ticket()
        if seen[text] >= max_dup:
            continue
        seen[text] += 1
        rows.append((text, labels))
    if len(rows) < n:
        print(f'[警告] 仅生成 {len(rows)}/{n} 条(模板多样性不足), 建议增加模板或放宽重复上限',
              file=sys.stderr)
    return rows


def write_file(path, rows):
    """写出数据文件: UTF-8 编码 + \n 换行(避免 Windows 下写出 \r\n 污染最后一个标签)"""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for text, labels in rows:
            f.write(text + '\t' + ','.join(labels) + '\n')


def print_stats(name, rows, preview=0):
    """打印数据集统计: 规模 / 平均标签数 / 各标签分布 / 高频标签组合"""
    total = len(rows)
    label_cnt = Counter()
    combo_cnt = Counter()
    for _, labels in rows:
        label_cnt.update(labels)
        combo_cnt[','.join(labels)] += 1
    avg_labels = sum(len(l) for _, l in rows) / max(total, 1)

    print('\n' + '-' * 72)
    print(f'[{name}] 共 {total} 条, 平均标签数 {avg_labels:.3f}, '
          f'单标签占比 {sum(1 for _, l in rows if len(l) == 1) / max(total, 1) * 100:.1f}%')
    print('各标签分布:')
    for label in CLASSES:
        cnt = label_cnt.get(label, 0)
        print(f'  {label:<16} {cnt:>6}  {cnt / max(total, 1) * 100:>5.1f}%')
    print('高频标签组合 Top8:')
    for combo, cnt in combo_cnt.most_common(8):
        print(f'  {combo:<44} {cnt:>6}  {cnt / max(total, 1) * 100:>5.1f}%')
    if preview:
        print(f'样本预览(前 {preview} 条):')
        for text, labels in rows[:preview]:
            print(f'  [{",".join(labels):<28}] {text[:52]}')


def main():
    parser = argparse.ArgumentParser(description='ShopCare 电商客服工单语料生成器')
    parser.add_argument('--n_train', type=int, default=4000, help='训练集条数(默认 4000)')
    parser.add_argument('--n_dev', type=int, default=600, help='验证集条数(默认 600)')
    parser.add_argument('--n_test', type=int, default=600, help='测试集条数(默认 600)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子(默认 42, 保证可复现)')
    parser.add_argument('--preview', type=int, default=0, help='生成后预览条数(仅训练集)')
    parser.add_argument('--out_dir', type=str, default=DATA_DIR, help='输出目录')
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 72)
    print('ShopCare 工单语料生成器')
    print(f'  标签体系 : {len(CLASSES)} 类多标签 -> {", ".join(CLASSES)}')
    print(f'  生成规模 : train={args.n_train}  dev={args.n_dev}  test={args.n_test}  (seed={args.seed})')
    print('=' * 72)

    for name, n in [('train', args.n_train), ('dev', args.n_dev), ('test', args.n_test)]:
        rows = generate(n)
        path = os.path.join(args.out_dir, f'{name}.txt')
        write_file(path, rows)
        print_stats(name, rows, preview=args.preview if name == 'train' else 0)
        print(f'  -> 已写入 {path}')

    print(f'\n[OK] 生成完成, 数据目录: {args.out_dir}')
    print('     下一步: python tools/check_dataset.py    # 数据体检')
    print('             python 01-data/data_eda.py       # 分布与共现分析')


if __name__ == '__main__':
    main()
