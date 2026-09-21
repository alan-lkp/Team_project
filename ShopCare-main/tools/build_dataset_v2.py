# -*- coding: utf-8 -*-
"""
ShopCare 语料重建 v2:模板池扩充 + 家族分组切分 + 槽位隔离

问题背景
========
v1 语料(train 196856 / dev 24607 / test 24607)存在**模板级泄漏**:

    整句查重: train ∩ test = 0        (看起来没问题)
    子句查重: test 的子句 96.57% 能在 train 中原样找到
              test 中 99.22% 的文本,其所有子句都被 train 覆盖

每条 test 文本只是几千条模板子句的重新排列组合。整句查重查不出重复,但模型学到的
其实是「子句 → 标签」的查表规则 —— test 里每个子句它都见过、都背过。所以 BERT
一个 epoch 就 train_loss=0.0788、dev/test 全 1.0。Micro-F1=1.000 会被直接质疑泄漏。

根因在数据构造方式
==================
`tools/generate_ticket_data.py` 只有 123 条原始模板,其中 61 条完全没有槽位
(invoice / price_promo / payment_account / service / invalid 这 5 类的模板全是
无槽位的)。从 123 条模板生成 19 万行,只能靠子句反复重排 —— 这是规模倒逼出来的
泄漏,不是切分不小心。

本脚本的三层修复
================
1. **扩池**: 用 `tools/_tpl_part{1,2,3}.py` 的 540 条模板(每类 60 条)替换原来的
   123 条;模板正文一律不含标点,保证「按标点切子句」的口径精确。
2. **家族分组切分**: 一条原始模板 + 它的所有同义变体 = 一个 family,family 是切分的
   最小单位,整体只进一个 split —— 避免「同一模板的不同变体」分落 train/test 造成
   变相模板泄漏。
3. **槽位隔离**: `tools/ticket_vocab_v2.py` 把商品/城市/天数按 split 切开,train 只用
   TRAIN 段,dev/test 只用 HELDOUT 段。于是 test 即使命中与 train 相同的句式,槽位取值
   也一定没出现过 —— 从「句式不同」加固到「句式 + 槽位取值都不同」。

   注:dev 与 test 共用同一段 heldout 槽位是安全的 —— 它们的 family 本身互斥,不会产生
   相同子句;脚本末尾会用「从磁盘重读」的方式实测验证这一点。

生成过程中持续保证的四条不变量
==============================
  A. 同一个子句字符串绝不跨 split 出现(`ClauseCursor` + clause_owner 强制);
  B. dev / test 内部不出现完全相同的整句文本;
  C. 单个子句在同一 split 内出现次数有上限(dev/test 更严,避免灌水);
  D. invalid 标签独占,不与其它标签共现(与 v1 一致)。

运行方式
========
    python tools/build_dataset_v2.py                       # 默认 100000/8000/8000
    python tools/build_dataset_v2.py --n_train 100000 --n_dev 8000 --n_test 8000
    python tools/build_dataset_v2.py --preview 10
    python tools/build_dataset_v2.py --no_backup

注意:本脚本会**直接覆盖** 01-data/train.txt、dev.txt、test.txt。
      默认先把旧文件备份到 01-data/backup/ 目录。
"""

import argparse
import itertools
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, '01-data')
for _p in (_HERE, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import generate_ticket_data as gen
except Exception:  # pragma: no cover - 兼容包内导入
    from tools import generate_ticket_data as gen

import ticket_vocab_v2 as vocab2  # noqa: E402


CLASSES = gen.CLASSES
LABEL_WEIGHT = gen.LABEL_WEIGHT
LABEL_CNT_DIST = gen.LABEL_CNT_DIST

# 模板正文来自三个分片文件(每个标签 60 条)
TEMPLATE_PARTS = ('_tpl_part1', '_tpl_part2', '_tpl_part3')

# 短句模板(每个标签 30 条, 3~12 字)单独一个池子。
# 为什么需要它: 只由长句拼出来的语料, 9 个标签里只有 invalid 含超短样本
# ("在吗"/"哈哈哈哈"), 模型会学到一个错误的长度先验 ——「短文本 => 无效与恶意」。
# 实测: 输入「物流」判成 invalid 0.95; 输入「快递一直没到」(6 字)判成 invalid 1.00;
# 而「我的快递一直没到」(8 字)就正确判成 logistics 1.00。
# 给每个标签都补上短句模板后, "短"就不再是 invalid 独有的特征。
SHORT_TEMPLATE_PARTS = ('_tpl_short1', '_tpl_short2', '_tpl_short3')

# 多大比例的样本是「单句短文本」。太低修不掉长度先验, 太高会让语料偏离真实工单
# (真实工单大多数是多诉求的长文本)。
SHORT_RATIO = 0.15

# family 在 split 之间的分配比例。train 占大头;dev/test 各 20% 是有意的 ——
# 它们只能用 heldout 段槽位(取值更少),需要更多句式才能撑起目标条数。
FAMILY_RATIO = {'train': 0.60, 'dev': 0.20, 'test': 0.20}

# 每个 family 最多展开出多少个子句(防止 {goods}/{place}/{day} 三项齐全时组合爆炸)
MAX_COMBOS_PER_FAMILY = 1500

# 子句在同一 split 内的重复上限(初始值;容量不够时会自动放宽并在报告中写明)
INIT_REPEAT_CAP = {'train': 4, 'dev': 1, 'test': 1}
MAX_REPEAT_CAP = {'train': 24, 'dev': 6, 'test': 6}

# 生成一个 split 时用的随机种子偏移。不能用 hash(name) —— Python 的字符串 hash
# 每次进程启动都不同,会让结果不可复现。
SEED_OFFSET = {'train': 101, 'dev': 202, 'test': 303}

# 采样标签组合时,不允许出现的数量分布(1/2/3 标签)
JOINERS = ('，', '。', '；')


# ---------------------------------------------------------------------------
# 同义改写:作用在「含占位符的模板字符串」上,不触碰 {goods}/{place}/{day}
# ---------------------------------------------------------------------------
PARAPHRASE_RULES = [
    ('申请退货', '办理退货'),
    ('退款', '退钱'),
    ('客服', '人工'),
    ('快递', '包裹'),
    ('发货', '寄出'),
    ('收到', '拿到'),
    ('到货', '到手'),
    ('一直', '老是'),
    ('能不能', '可不可以'),
    ('为什么', '为啥'),
    ('怎么', '如何'),
    ('还没', '到现在都还没'),
    ('没有', '没'),
    ('东西', '货'),
    ('马上', '立刻'),
    ('麻烦', '劳烦'),
    ('怎么办', '该咋处理'),
]

# 少量输入错误,模拟真实工单里的错别字。只保留「形近且不改变理解」的几种:
#   - 去掉过 '退款'→'退歀'(歀 是生僻字,看着像数据损坏而不是错别字)
#   - 去掉过 '发货'→'发火'(把"发货"变成"发火",语义变了,是坏样本而不是噪声)
TYPO_RULES = [
    ('快递', '快弟'),
    ('客服', '客福'),
    ('麻烦', '麻凡'),
]

TONE_SUFFIXES = ['啊', '呀', '吧', '呢', '哦', '了', '嘛']
NO_TONE_TAIL = '？?！!。吗呢吧啊呀了哦嘛'


# ---------------------------------------------------------------------------
# 模板 → 商品品类
# ---------------------------------------------------------------------------
# 一条模板如果写了「面料」「尺码」,它讲的必然是服饰;写「味道」「过期」,必然是食品。
# 不判断品类而随机填商品,就会生成「录音笔面料很硬穿着扎皮肤」这种句子 —— v1 语料
# 里这类错配很普遍,交付时一眼就能看出是造的数据。
#
# 匹配规则:从上到下取第一个命中的品类;都没命中就用全量商品池(通用模板)。
# 品类候选太少(heldout 段只有 2~6 个)时也回退到全量池,避免把某条模板饿死。
CATEGORY_KEYWORDS = [
    ('apparel', ['面料', '穿着', '试穿', '尺码', '码数', '版型', '上身', '掉色',
                 '起球', '线头', '缝合', '扎皮肤', '磨脚', '鞋码', '透气', '洗了',
                 '洗过', '褪色', '缩水', '鞋底', '开胶', '缝线', '走线', '拉链',
                 '领口', '袖口', '脱线', '磨破', '裤脚', '走两步']),
    ('food', ['味道', '吃着', '吃了', '喝了', '过期', '变质', '生产日期', '保质期',
              '口感', '发霉', '难吃', '怪味', '冲了', '泡了', '保质', '吃出',
              '喝了', '食用']),
    ('beauty', ['皮肤', '过敏', '涂了', '抹了', '上脸', '卸妆', '起皮', '泛红',
                '刺痛', '抹上', '涂上', '洗掉']),
    ('baby', ['宝宝', '婴儿', '小孩', '孩子', '尿', '哺', '安全座椅', '宝妈']),
    ('pet', ['猫', '狗', '宠物', '毛孩子']),
    ('digital', ['屏幕', '电池', '充电', '续航', '开机', '关机', '死机', '按键',
                 '喇叭', '摄像头', '蓝牙', '信号', '系统', '连不上', '插电', '通电',
                 '指示灯', '遥控', '连手机', '说明书', '安装', '软件', '发烫',
                 '发热', '鼓包', '插头', '接口', '短路', '漏电', '卡顿', '闪屏',
                 '待机', '充一次电', '插不进去', '充多久', '充满电', '电量',
                 '电池容量', '毫安']),
]


def detect_category(template):
    """从模板正文判断它需要哪一类商品;判断不出来返回 None(用全量池)。"""
    for cat, kws in CATEGORY_KEYWORDS:
        if any(kw in template for kw in kws):
            return cat
    return None


def _safe_replace(text, old, new):
    """普通子串替换,但跳过花括号占位符区域,避免破坏 {goods} 这类槽位。"""
    if old not in text:
        return text
    out = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            end = text.find('}', i + 1)
            if end == -1:
                out.append(text[i:])
                break
            out.append(text[i:end + 1])
            i = end + 1
            continue
        if text.startswith(old, i):
            out.append(new)
            i += len(old)
            continue
        out.append(text[i])
        i += 1
    return ''.join(out)


def _apply_rules(template, rules, rng):
    """按随机顺序尝试套用规则,每条规则最多用一次;返回 (新文本, 是否有改动)。"""
    text = template
    changed = False
    for old, new in rng.sample(rules, k=len(rules)):
        new_text = _safe_replace(text, old, new)
        if new_text != text:
            text = new_text
            changed = True
    return text, changed


def make_variants(template, rng, target):
    """把一条模板扩成至多 target 个表面变体(含模板本身),全部属于同一个 family。

    变体只做「同义换词 / 语气词 / 少量错别字」,不追加整句诉求尾巴 —— 尾巴不带标点会
    读成"客服态度特别差麻烦尽快处理"这种黏连句,反而破坏语料观感。
    """
    variants = [template]
    attempts = 0
    while len(variants) < target and attempts < target * 30:
        attempts += 1
        v = template
        roll = rng.random()
        if roll < 0.60:
            v, _ = _apply_rules(v, PARAPHRASE_RULES, rng)
        elif roll < 0.68:
            v, _ = _apply_rules(v, TYPO_RULES, rng)
        elif roll < 0.78:
            v, _ = _apply_rules(v, PARAPHRASE_RULES + TYPO_RULES, rng)
        if rng.random() < 0.55 and v and v[-1] not in NO_TONE_TAIL:
            v = v + rng.choice(TONE_SUFFIXES)
        if v != template and v not in variants:
            variants.append(v)
    return variants


def load_template_library(parts=TEMPLATE_PARTS, kind='模板'):
    """从分片文件读取模板并合并,同时做格式校验。

    parts 传 TEMPLATE_PARTS 读长句模板, 传 SHORT_TEMPLATE_PARTS 读短句模板。
    """
    library = {}
    for mod_name in parts:
        try:
            mod = __import__(mod_name)
        except ImportError as exc:
            raise SystemExit(
                f'读不到模板分片 {mod_name}.py({exc})。\n'
                f'请先确认 tools/{mod_name}.py 存在。'
            )
        for label, templates in mod.TEMPLATES.items():
            library.setdefault(label, []).extend(templates)

    missing = [lb for lb in CLASSES if not library.get(lb)]
    if missing:
        raise SystemExit(f'以下标签没有{kind}: {missing}')

    # 全局去重:模板跨标签撞车会破坏「family 互斥」的前提
    seen = {}
    for label in CLASSES:
        cleaned = []
        for t in library[label]:
            t = t.strip()
            if not t:
                continue
            if t in seen:
                print(f'[警告] {kind}重复,已剔除:「{t}」({label} 与 {seen[t]} 撞车)',
                      file=sys.stderr)
                continue
            seen[t] = label
            cleaned.append(t)
        library[label] = cleaned
    return library


def build_families(library, rng, variants_slot=6, variants_noslot=10):
    families = {}
    for label in CLASSES:
        fams = []
        for template in library[label]:
            has_slot = bool(re.search(r'\{goods\}|\{place\}|\{day\}', template))
            target = variants_slot if has_slot else variants_noslot
            fams.append(make_variants(template, rng, target))
        families[label] = fams
    return families


def assign_families(families, rng):
    """把每个 family 整体分配给一个 split。

    强制每个 split 在每个标签下至少拿到 1 个 family,否则该 split 会缺少生成该标签
    样本的素材。
    """
    pools = {s: defaultdict(list) for s in ('train', 'dev', 'test')}
    for label in CLASSES:
        fams = families[label][:]
        rng.shuffle(fams)
        n = len(fams)
        n_dev = max(1, int(round(n * FAMILY_RATIO['dev'])))
        n_test = max(1, int(round(n * FAMILY_RATIO['test'])))
        n_train = n - n_dev - n_test
        if n_train < 1:  # 模板太少时退化为均分
            n_dev = n_test = max(1, n // 3)
            n_train = max(1, n - n_dev - n_test)
        pools['train'][label] = fams[:n_train]
        pools['dev'][label] = fams[n_train:n_train + n_dev]
        pools['test'][label] = fams[n_train + n_dev:]
    return pools


def expand_family(variants, slots, rng):
    """把一个 family 展开成一批字面不同的子句。

    关键:每个变体**都能分到配额**,而不是把上限全耗在第一个变体上。
    组合空间大于配额时随机抽样,保证不同 family 抽到的槽位取值不总是同一批。
    """
    goods, places, days = slots['goods'], slots['place'], slots['day']
    per_variant = max(1, MAX_COMBOS_PER_FAMILY // max(1, len(variants)))
    clauses = []

    for v in variants:
        gs = goods if '{goods}' in v else ['']
        ps = places if '{place}' in v else ['']
        ds = days if '{day}' in v else ['']
        total = len(gs) * len(ps) * len(ds)

        if total <= per_variant:
            combos = itertools.product(gs, ps, ds)
        else:
            nd, np_ = len(ds), len(ps)
            picks = rng.sample(range(total), per_variant)
            combos = (
                (gs[k // (np_ * nd)], ps[(k // nd) % np_], ds[k % nd])
                for k in picks
            )
        clauses.extend(v.format(goods=g, place=p, day=d) for g, p, d in combos)

    rng.shuffle(clauses)
    return clauses


def build_label_tables(pools, slot_vocab, rng, stats=None):
    """为每个 (split, label) 预生成一个打乱的候选子句表。

    每个 family 先判断它需要哪一类商品,再用对应的商品池填充 —— 这是避免
    「录音笔面料很硬」这类语义错配的关键一步。
    """
    tables = {}
    for split in ('train', 'dev', 'test'):
        tables[split] = {}
        slots = slot_vocab[split]
        for label in CLASSES:
            batch = []
            for family in pools[split][label]:
                cat = detect_category(family[0])
                pool = slots.get('goods_by_cat', {}).get(cat) if cat else None
                # 品类在 heldout 段只有 2~6 个商品,太少时回退全量池,避免饿死模板
                if pool and len(pool) >= 2:
                    effective = dict(slots, goods=pool)
                else:
                    effective = slots
                if stats is not None and cat:
                    stats[cat] = stats.get(cat, 0) + 1
                batch.extend(expand_family(family, effective, rng))
            # 模板正文保证无标点,这里再兜一道:带标点的子句会让切分口径失真
            batch = [c for c in batch
                     if c and not re.search(r'[，。；！？!?;]', c)]
            rng.shuffle(batch)
            tables[split][label] = batch
    return tables


class ClauseCursor:
    """按轮转顺序发放子句,并在子句用满配额后把它移出池子。

    这样既避免了「同一个子句连着出现好几次」,又让「是否还有可用子句」变成 O(1) 的
    alive 计数 —— 10 万行规模下不能每行都全表扫描。
    """

    __slots__ = ('items', 'used', 'alive', 'i', 'cap')

    def __init__(self, clauses, cap, clause_owner, split):
        self.cap = cap
        self.used = Counter()
        self.items = [c for c in clauses if clause_owner.get(c, split) == split]
        self.alive = len(self.items)
        self.i = 0

    def _drop(self, idx):
        last = self.items.pop()
        if idx < len(self.items):
            self.items[idx] = last
        self.alive -= 1

    def take(self, clause_owner, split):
        """取一个可用子句;池子空了返回 None。"""
        if self.alive <= 0:
            return None
        for _ in range(len(self.items) + 1):
            if not self.items:
                return None
            if self.i >= len(self.items):
                self.i = 0
            cand = self.items[self.i]
            if clause_owner.get(cand, split) != split:
                self._drop(self.i)      # 被别的 split 占了(理论不该发生,兜底)
                continue
            u = self.used[cand] + 1
            if u > self.cap:
                self._drop(self.i)
                continue
            self.used[cand] = u
            if u >= self.cap:
                self._drop(self.i)      # 用满了,移出池子
            else:
                self.i += 1
            return cand
        return None


def choose_short_label(rng):
    """短句样本只带一个标签 —— 真实用户随手打的短工单不会同时吐槽好几件事。"""
    keys = list(LABEL_WEIGHT.keys())
    weights = [LABEL_WEIGHT[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def choose_labels(rng):
    """按标签数量分布和长尾权重采样一组标签;invalid 独占。"""
    n_labels = rng.choices(
        [c for c, _ in LABEL_CNT_DIST],
        weights=[w for _, w in LABEL_CNT_DIST],
        k=1,
    )[0]
    labels = []
    guard = 0
    while len(labels) < n_labels and guard < 100:
        guard += 1
        keys = list(LABEL_WEIGHT.keys())
        weights = [LABEL_WEIGHT[k] for k in keys]
        label = rng.choices(keys, weights=weights, k=1)[0]
        if label not in labels:
            labels.append(label)
    if 'invalid' in labels:
        labels = ['invalid']
    return sorted(labels, key=lambda x: CLASSES.index(x))


def generate_split(n, split, tables, short_tables, rng, clause_owner, seen_texts,
                   repeat_cap, short_ratio=SHORT_RATIO):
    """生成一个 split 的样本。返回 (rows, 因缺料放弃的次数, 短句条数)。

    short_ratio 比例的样本走「单句短文本」分支 —— 这是为了打散
    「短文本 => 无效与恶意」的长度先验(见 SHORT_TEMPLATE_PARTS 的说明)。
    """
    cursors = {lb: ClauseCursor(tables[split][lb], repeat_cap, clause_owner, split)
               for lb in CLASSES}
    short_cursors = {lb: ClauseCursor(short_tables[split][lb], repeat_cap,
                                      clause_owner, split)
                     for lb in CLASSES}
    rows = []
    starved = 0
    n_short = 0

    while len(rows) < n:
        # ---------- 分支 A: 单句短文本 ----------
        if rng.random() < short_ratio:
            label = choose_short_label(rng)
            clause = short_cursors[label].take(clause_owner, split)
            if clause is not None:
                text = clause
                if rng.random() < 0.05:
                    text += rng.choice(gen.PUNCT_NOISE)
                text = text.strip()
                if text and text not in seen_texts:
                    clause_owner[clause] = split
                    seen_texts[text] = split
                    rows.append((text, [label]))
                    n_short += 1
                    continue
            # 短句池空了(或撞了整句): 落回长句分支, 不浪费这一轮

        # ---------- 分支 B: 1~3 个子句拼接的长文本 ----------
        labels = choose_labels(rng)
        if any(cursors[lb].alive <= 0 for lb in labels):
            starved += 1
            if starved > 500 and all(c.alive <= 0 for c in cursors.values()):
                break
            if starved > 20000:
                break
            continue

        parts = [cursors[lb].take(clause_owner, split) for lb in labels]
        if any(p is None for p in parts):   # 与上面的 alive 预检重复,兜底
            continue

        text = rng.choice(JOINERS).join(parts)
        if rng.random() < 0.10:
            text += rng.choice(gen.PUNCT_NOISE)
        text = text.strip()

        if not text or text in seen_texts:
            continue

        for clause in parts:
            clause_owner[clause] = split
        seen_texts[text] = split
        rows.append((text, labels))

    return rows, starved, n_short


def split_clauses(text):
    """按标点切子句 —— 必须与泄漏检测口径完全一致。"""
    return [p.strip() for p in re.split(r'[，。；！？!?;]', text) if p.strip()]


def load_rows(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            text = parts[0].strip()
            labels = [x.strip() for x in parts[1].split(',') if x.strip()]
            if text and labels:
                rows.append((text, labels))
    return rows


def verify_from_disk(data_dir):
    """从磁盘重读三个文件做泄漏体检 —— 比在内存里统计更有说服力。"""
    splits = {name: load_rows(os.path.join(data_dir, f'{name}.txt'))
              for name in ('train', 'dev', 'test')}

    texts = {k: {t for t, _ in v} for k, v in splits.items()}
    clauses = {k: {c for t, _ in v for c in split_clauses(t)}
               for k, v in splits.items()}

    report = {
        'sizes': {k: len(v) for k, v in splits.items()},
        'exact_overlap': {
            'train_dev': len(texts['train'] & texts['dev']),
            'train_test': len(texts['train'] & texts['test']),
            'dev_test': len(texts['dev'] & texts['test']),
        },
        'clause_pool': {k: len(v) for k, v in clauses.items()},
        'clause_overlap': {
            'train_dev': len(clauses['train'] & clauses['dev']),
            'train_test': len(clauses['train'] & clauses['test']),
            'dev_test': len(clauses['dev'] & clauses['test']),
        },
        'dup_within': {k: len(v) - len(texts[k]) for k, v in splits.items()},
        'repeat_within': {},
    }

    for name, rows in splits.items():
        counter = Counter()
        for t, _ in rows:
            counter.update(set(split_clauses(t)))
        if counter:
            counts = sorted(counter.values())
            report['repeat_within'][name] = {
                'distinct_clauses': len(counter),
                'max_repeats': counts[-1],
                'median_repeats': counts[len(counts) // 2],
            }

    tc, trc = clauses['test'], clauses['train']
    covered = len(tc & trc)
    n_with = sum(1 for t, _ in splits['test'] if split_clauses(t))
    all_cov = sum(1 for t, _ in splits['test']
                  if split_clauses(t) and set(split_clauses(t)) <= trc)
    report['test_clause_covered_by_train_ratio'] = (
        round(covered / len(tc), 6) if tc else None)
    report['test_text_all_clauses_covered_by_train_ratio'] = (
        round(all_cov / n_with, 6) if n_with else None)

    return report, splits


def _overlaps_other(text, i, n, value, others, maxlen):
    """text[i:i+n] 是否与 others 里某个词的出现位置相交。

    用于排除跨词巧合:train 文本「长**沙发**货的…」里的「沙发」其实是
    「长沙」(train 城市) + 「发货」拼出来的,不是 heldout 商品「沙发」泄漏。
    """
    lo = max(0, i - maxlen)
    hi = min(len(text), i + n + maxlen)
    seg = text[lo:hi]
    for w in others:
        if w == value:          # 自己不算"别人",否则自己总和自己重叠
            continue
        s = 0
        while True:
            j = seg.find(w, s)
            if j == -1:
                break
            a, b = lo + j, lo + j + len(w)
            if a < i + n and b > i:      # 区间相交
                return True
            s = j + 1
    return False


def _hits_outside_longer(text, value, all_values, others=None, maxlen=0):
    """value 是否作为「独立的槽位取值」出现在 text 里。

    排除两类假阳性:
      1. 被更长的同类词包住 —— 「眼镜」是「墨镜」的子串、「耳机」是「蓝牙耳机」的子串;
      2. 与别的槽位取值重叠 —— 「长沙」+「发货」拼出的「沙发」。
    两类都是汉语子串匹配的固有噪声,不是真的槽位泄漏。
    """
    longer = [w for w in all_values if len(w) > len(value) and value in w]
    others = others or []
    start = 0
    while True:
        i = text.find(value, start)
        if i == -1:
            return False
        n = len(value)
        if not any(w in text[max(0, i - len(w)): i + n + len(w)] for w in longer) \
                and not _overlaps_other(text, i, n, value, others, maxlen):
            return True
        start = i + 1


def _num_hit(text, value):
    """数字槽位用边界匹配:「5」不该命中「50天」。"""
    return re.search(r'(?<!\d)' + re.escape(value) + r'(?!\d)', text) is not None


def slot_isolation_report(data_dir):
    """实测槽位隔离:train 专用槽位取值不该出现在 dev/test 文本里,反之亦然。

    这是「句式 + 槽位取值都不同」这句结论的证据。若模板正文里手写了某个 heldout
    商品名(而不是用 {goods} 占位符),这里会暴露出来。
    """
    v = vocab2.split_slot_vocab()
    raw = {}
    for name in ('train', 'dev', 'test'):
        with open(os.path.join(data_dir, f'{name}.txt'), encoding='utf-8') as f:
            raw[name] = f.read()

    stop = getattr(vocab2, 'GENERIC_NOUN_STOPLIST', set())
    # 所有槽位取值(含 train 与 heldout 两侧)用于跨词巧合掩码
    all_slot_values = []
    for s2 in ('goods', 'place', 'day'):
        all_slot_values += [x for x in v['train'][s2] if x]
        all_slot_values += [x for x in v['dev'][s2] if x]
    maxlen = max(len(x) for x in all_slot_values)

    out = {}
    for slot in ('goods', 'place', 'day'):
        train_vals = [s for s in v['train'][slot] if s]
        held_vals = [s for s in v['dev'][slot] if s]
        if slot == 'day':
            # 天数是纯数字,'5' 是 '50天' 的子串,必须用数字边界匹配
            f_test = lambda s: (_num_hit(raw['test'], s) or _num_hit(raw['dev'], s))
            f_train = lambda s: _num_hit(raw['train'], s)
        else:
            all_vals = train_vals + held_vals
            kw = dict(others=all_slot_values, maxlen=maxlen)
            f_test = lambda s: (_hits_outside_longer(raw['test'], s, all_vals, **kw)
                                or _hits_outside_longer(raw['dev'], s, all_vals, **kw))
            f_train = lambda s: _hits_outside_longer(raw['train'], s, all_vals, **kw)

        leak_in_test = sorted({s for s in train_vals if f_test(s)})
        leak_in_train = sorted({s for s in held_vals if f_train(s)})
        out[slot] = {
            'train_only_values': len(train_vals),
            'heldout_values': len(held_vals),
            'train_values_found_in_dev_test': leak_in_test[:10],
            'n_train_values_found_in_dev_test': len(leak_in_test),
            'heldout_values_found_in_train': leak_in_train[:10],
            'n_heldout_values_found_in_train': len(leak_in_train),
            # 剔除「本身就是常用名词」的商品名后的口径 —— 这些词出现在正文里指的是
            # 别的东西(如「改绑手机」),不是槽位填充泄漏
            'n_train_values_found_in_dev_test_excl_generic':
                len([s for s in leak_in_test if s not in stop]),
            'n_heldout_values_found_in_train_excl_generic':
                len([s for s in leak_in_train if s not in stop]),
            'generic_stoplist': sorted(stop),
        }
    return out


def print_report(report, slot_iso):
    print('\n' + '=' * 74)
    print('泄漏检测报告(从磁盘重读三个文件后实测)')
    print('=' * 74)
    s = report['sizes']
    print(f"数据规模: train={s['train']}  dev={s['dev']}  test={s['test']}  "
          f"合计={sum(s.values())}")
    eo = report['exact_overlap']
    print(f"整句交集: train∩dev={eo['train_dev']}  train∩test={eo['train_test']}  "
          f"dev∩test={eo['dev_test']}")
    cp = report['clause_pool']
    print(f"子句池大小: train={cp['train']}  dev={cp['dev']}  test={cp['test']}")
    co = report['clause_overlap']
    print(f"子句交集: train∩dev={co['train_dev']}  train∩test={co['train_test']}  "
          f"dev∩test={co['dev_test']}")
    print(f"test 子句在 train 中原样出现的比例: "
          f"{report['test_clause_covered_by_train_ratio'] * 100:.4f}%")
    print(f"test 中所有子句都被 train 覆盖的文本占比: "
          f"{report['test_text_all_clauses_covered_by_train_ratio'] * 100:.4f}%")

    print('\n同一 split 内的子句使用强度(衡量灌水程度,不是泄漏):')
    for name in ('train', 'dev', 'test'):
        r = report['repeat_within'][name]
        print(f"  {name:<6} 整句重复 {report['dup_within'][name]} 条 | "
              f"不同子句 {r['distinct_clauses']} 个 | "
              f"单子句最多被 {r['max_repeats']} 条文本用 | "
              f"中位数 {r['median_repeats']}")

    print('\n槽位隔离实测(train 专用取值不该出现在 dev/test,反之亦然):')
    for slot, r in slot_iso.items():
        n1 = r['n_train_values_found_in_dev_test']
        n2 = r['n_heldout_values_found_in_train']
        a1 = r['n_train_values_found_in_dev_test_excl_generic']
        a2 = r['n_heldout_values_found_in_train_excl_generic']
        print(f"  {slot:<6} train专用 {r['train_only_values']:>3} 个 / heldout "
              f"{r['heldout_values']:>3} 个")
        print(f"         train专用值出现在 dev/test: {n1} 个 "
              f"(剔除通用名词后 {a1} 个)")
        print(f"         heldout值出现在 train  : {n2} 个 "
              f"(剔除通用名词后 {a2} 个)")
        if r['train_values_found_in_dev_test']:
            print(f"         例: {r['train_values_found_in_dev_test'][:6]}")
        if r['heldout_values_found_in_train']:
            print(f"         例: {r['heldout_values_found_in_train'][:6]}")
    print(f"  通用名词豁免表: {sorted(getattr(vocab2, 'GENERIC_NOUN_STOPLIST', set()))}")
    print('  (商品名里有几个本身就是常用名词,会自然出现在模板正文里但指的不是'
          '「买的那个商品」,' )
    print('   例如「改绑手机总是收不到验证码」的「手机」指手机号。生成侧的隔离由'
          ' split_slot_vocab() 保证,此处仅做诊断。)')


def print_label_stats(name, rows):
    total = len(rows)
    label_counter = Counter()
    combo_counter = Counter()
    for _, labels in rows:
        label_counter.update(labels)
        combo_counter[','.join(labels)] += 1
    avg = sum(len(l) for _, l in rows) / max(total, 1)
    print(f'\n[{name}] {total} 条, 平均标签数 {avg:.3f}')
    for label in CLASSES:
        cnt = label_counter.get(label, 0)
        print(f'  {label:<16} {cnt:>7}  {cnt / max(total, 1) * 100:>5.1f}%')
    print('  高频标签组合 Top6:')
    for combo, cnt in combo_counter.most_common(6):
        print(f'    {combo:<44} {cnt:>7}')


def write_rows(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for text, labels in rows:
            f.write(f'{text}\t{",".join(labels)}\n')


def backup_old_data(out_dir):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join(out_dir, 'backup', f'leaky_{stamp}')
    os.makedirs(backup_dir, exist_ok=True)
    for name in ('train.txt', 'dev.txt', 'test.txt'):
        src = os.path.join(out_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, name))
    return backup_dir


def rollback_split(name, rows, clause_owner, seen_texts):
    """把某个 split 已登记的所有权和整句记录撤回(用于放宽重复上限后重来)。"""
    for text, _ in rows:
        seen_texts.pop(text, None)
        for clause in split_clauses(text):
            if clause_owner.get(clause) == name:
                del clause_owner[clause]


def main():
    parser = argparse.ArgumentParser(
        description='ShopCare 语料重建 v2:扩池 + 家族分组切分 + 槽位隔离')
    parser.add_argument('--n_train', type=int, default=100000)
    parser.add_argument('--n_dev', type=int, default=8000)
    parser.add_argument('--n_test', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default=DATA_DIR)
    parser.add_argument('--no_backup', action='store_true')
    parser.add_argument('--preview', type=int, default=0)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 74)
    print('ShopCare 语料重建 v2:扩池 + 家族分组切分 + 槽位隔离')
    print('=' * 74)

    vocab2._sanity_check()
    library = load_template_library()
    total_tpl = sum(len(v) for v in library.values())
    print(f'模板池: {total_tpl} 条')
    for label in CLASSES:
        print(f'  {label:<16} {len(library[label]):>3} 条')

    short_library = load_template_library(SHORT_TEMPLATE_PARTS, kind='短句模板')
    total_short = sum(len(v) for v in short_library.values())
    print(f'短句模板池: {total_short} 条(每个标签的短句数 '
          + ' '.join(f'{lb}={len(short_library[lb])}' for lb in CLASSES) + ')')

    rng = random.Random(args.seed)
    families = build_families(library, rng)
    pools = assign_families(families, rng)
    # 短句 family 用同一套分配规则(整体只进一个 split), 保证跨 split 子句零重合
    short_families = build_families(short_library, rng, variants_slot=6, variants_noslot=8)
    short_pools = assign_families(short_families, rng)

    fam_summary = {s: {lb: len(pools[s][lb]) for lb in CLASSES}
                   for s in ('train', 'dev', 'test')}
    print('\n每个 split 分到的 family 数量:')
    for s in ('train', 'dev', 'test'):
        print(f'  {s:<6} 合计 {sum(fam_summary[s].values()):>4}  '
              + ' '.join(f'{lb}={fam_summary[s][lb]}' for lb in CLASSES))

    slot_vocab = vocab2.split_slot_vocab()
    cat_stats = {}
    tables = build_label_tables(pools, slot_vocab, rng, stats=cat_stats)
    short_tables = build_label_tables(short_pools, slot_vocab, rng)
    print('\n模板品类匹配(避免「录音笔面料很硬」这类语义错配):')
    if cat_stats:
        for cat, n in sorted(cat_stats.items(), key=lambda x: -x[1]):
            print(f'  {cat:<10} {n:>4} 个 family 指定了该品类商品')
    total_fam = sum(len(pools[s][lb]) for s in pools for lb in CLASSES)
    print(f'  (其余 {total_fam - sum(cat_stats.values())} 个 family 为通用模板,用全量商品池)')

    print('\n候选子句容量(每个 split 每个标签可用的不同子句数):')
    for s in ('train', 'dev', 'test'):
        caps = {lb: len(tables[s][lb]) for lb in CLASSES}
        print(f'  {s:<6} 合计 {sum(caps.values()):>7}  '
              + ' '.join(f'{lb}={caps[lb]}' for lb in CLASSES))

    if args.no_backup:
        backup_dir = None
    else:
        backup_dir = backup_old_data(out_dir)
        print(f'\n旧数据已备份到: {backup_dir}')

    targets = {'train': args.n_train, 'dev': args.n_dev, 'test': args.n_test}
    limits = dict(INIT_REPEAT_CAP)
    clause_owner = {}
    seen_texts = {}
    splits = {}
    final_caps = {}
    short_stats = {}

    for name in ('train', 'dev', 'test'):
        n = targets[name]
        print(f'\n开始生成 {name}(目标 {n} 条) ...')
        while True:
            print(f'  子句重复上限 = {limits[name]}')
            rng_s = random.Random(args.seed * 7 + SEED_OFFSET[name] + limits[name])
            rows, starved, n_short = generate_split(
                n, name, tables, short_tables, rng_s, clause_owner, seen_texts,
                limits[name])
            if len(rows) >= n or limits[name] >= MAX_REPEAT_CAP[name]:
                break
            print(f'  仅生成 {len(rows)}/{n} 条,放宽重复上限到 '
                  f'{min(limits[name] * 2, MAX_REPEAT_CAP[name])} 后重试')
            rollback_split(name, rows, clause_owner, seen_texts)
            limits[name] = min(limits[name] * 2, MAX_REPEAT_CAP[name])

        if len(rows) < n:
            print(f'  [警告] {name} 只生成 {len(rows)}/{n} 条,'
                  f'已达重复上限 {limits[name]} 且容量吃满')
        final_caps[name] = limits[name]
        splits[name] = rows
        short_stats[name] = n_short
        write_rows(os.path.join(out_dir, f'{name}.txt'), rows)
        print(f'  -> 已写入 {name}.txt ({len(rows)} 条, 其中单句短文本 {n_short} 条,'
              f'占 {n_short / max(len(rows), 1) * 100:.1f}%,'
              f'缺料放弃 {starved} 次)')
        print_label_stats(name, rows)

    report, _ = verify_from_disk(out_dir)
    slot_iso = slot_isolation_report(out_dir)
    print_report(report, slot_iso)

    manifest = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'builder': 'tools/build_dataset_v2.py',
        'seed': args.seed,
        'template_pool_size': total_tpl,
        'families_per_split': fam_summary,
        'family_ratio': FAMILY_RATIO,
        'family_category_matched': cat_stats,
        'clause_repeat_cap': final_caps,
        'short_ratio_target': SHORT_RATIO,
        'short_rows': short_stats,
        'short_template_pool_size': total_short,
        'slot_vocab': {
            'train': {k: len(v) for k, v in slot_vocab['train'].items()},
            'dev_test': {k: len(v) for k, v in slot_vocab['dev'].items()},
            'disjoint': True,
            'isolation_check': slot_iso,
        },
        'leakage': report,
        'backup_dir': backup_dir,
    }
    manifest_path = os.path.join(out_dir, 'split_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'\n切分清单已保存: {manifest_path}')

    if args.preview:
        print('\n样本预览:')
        for name in ('train', 'dev', 'test'):
            print(f'  ---- [{name}] ----')
            for text, labels in splits[name][:args.preview]:
                print(f'    [{",".join(labels):<30}] {text}')

    print('\n[OK] 重建完成。建议继续执行:')
    print('      python tools/check_dataset.py')
    print('      python 02-rf/rf_train.py')
    print('      python 03-fasttext/ft_train.py')


if __name__ == '__main__':
    main()
