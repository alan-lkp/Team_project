"""
[已废弃] 请改用 tools/build_dataset_v2.py
=========================================
本脚本是第一版修复尝试,已被 `tools/build_dataset_v2.py` 取代,保留仅作历史参考。

为什么被取代: 本脚本只做了「家族分组切分」一层修复,但**模板池仍然是原来的 123 条**
(其中 61 条完全没有槽位)。模板池太小,任何切分方式都撑不起数据量 —— 生成出来的
样本量上不去,而且 test 的"新模板"本质只是 train 模板的同义改写,指标仍然偏高。

build_dataset_v2.py 在此之上补了两层:
  1. 模板池扩到 540 条(tools/_tpl_part{1,2,3}.py);
  2. 槽位词表按 split 隔离(tools/ticket_vocab_v2.py), test 用的商品名/城市名
     train 里一个都没有。

下面的原始说明保留,仅用于理解第一版思路。

--------------------------------------------------------------------------
ShopCare 语料重建工具：把“模板池随机独立生成”改成“按子句家族分组切分”

背景（为什么需要这个脚本）
============================
原 tools/generate_ticket_data.py 对 train / dev / test 分别独立随机生成。虽然整句
去重看起来没有交集，但所有 split 都从同一批子句模板池里采样，因此测试集几乎每个
子句都已经在训练集中原样出现过。模型学到的其实是“子句 -> 标签”的查表规则，
所以 RF / BERT 都拿到 Micro-F1=1.0，这个指标本身没有区分度。

本脚本的修复策略
================
1. 把每个标签的每一条模板扩成多个“表面变体”（同义换词 / 口语化 / 噪声），
   形成一个 template family；
2. 在生成任何样本之前，先把每个 family 只分配到一个 split：
       train / dev / test 各自拥有互不重叠的子句家族；
3. 生成一条多标签工单时，其每个子句都必须来自“当前 split 自己的 family 池”；
4. 生成过程中同时保证：
       - 同一个子句绝不跨 split 出现；
       - dev / test 内部不出现完全相同的子句；
       - train 内同子句重复次数受控，避免灌水刷分；
       - dev / test 的整句文本不重复，且也不与 train 整句重复；
5. 最后输出泄漏检测报告：整句交集、子句交集、test 子句被 train 覆盖比例。

运行方式
========
    python tools/rebuild_grouped_data.py
    python tools/rebuild_grouped_data.py --n_train 4000 --n_dev 600 --n_test 600
    python tools/rebuild_grouped_data.py --no_backup

注意：本脚本会直接覆盖 01-data/train.txt、dev.txt、test.txt。默认会先把旧文件
备份到 01-data/backup/ 目录，避免误操作无法恢复。
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict

# 便于从项目根目录或 tools 目录运行
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, '01-data')
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import generate_ticket_data as gen
except Exception:
    from tools import generate_ticket_data as gen


CLASSES = gen.CLASSES
LABEL_TEMPLATES = gen.LABEL_TEMPLATES
LABEL_WEIGHT = gen.LABEL_WEIGHT
LABEL_CNT_DIST = gen.LABEL_CNT_DIST
GOODS = gen.GOODS
PLACES = gen.PLACES
DAYS = gen.DAYS
PREFIXES = gen.PREFIXES
SUFFIXES = gen.SUFFIXES
TONE_WORDS = gen.TONE_WORDS
PUNCT_NOISE = gen.PUNCT_NOISE
INVALID_PREFIXES = gen.INVALID_PREFIXES


# ---------------------------------------------------------------------------
# 同义换词规则：作用在“含占位符的模板字符串”上，不触碰 {goods}/{place}/{day}
# ---------------------------------------------------------------------------
PARAPHRASE_RULES = [
    ('退款', '退钱'),
    ('申请退货', '办理退货'),
    ('客服', '人工'),
    ('一直', '老是'),
    ('还没', '到现在都还没'),
    ('能不能', '可不可以'),
    ('怎么', '如何'),
    ('没有', '没'),
    ('显示已', '页面显示已经'),
    ('快递', '包裹'),
    ('东西', '货'),
    ('收到', '拿到'),
    ('到货', '到手'),
    ('为什么', '为啥'),
    ('发货', '寄出'),
    ('不知道', '不太清楚'),
    ('应该', '应该可以'),
    ('怎么办', '该咋处理'),
]

# 常见打字/口语错误，只做少量使用，防止把语义标签本身改坏。
TYPO_RULES = [
    ('快递', '快弟'),
    ('退款', '退歀'),
    ('客服', '客福'),
    ('麻烦', '麻凡'),
    ('发货', '发火'),
]


def _safe_replace(text, old, new):
    """替换普通子串，但跳过花括号里的占位符区域，避免破坏 {goods} 等槽位。"""
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
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)


def _apply_rules(template, rules, rng):
    """随机选若干条规则应用，避免所有变体面目全非，同时保证至少一条不同。"""
    changed = False
    candidates = list(rules)
    rng.shuffle(candidates)
    for old, new in candidates:
        if old in template and (not changed or rng.random() < 0.35):
            template = _safe_replace(template, old, new)
            changed = True
        if changed and rng.random() < 0.25:
            break
    return template, changed


def _make_template_variants(template, rng):
    """把一条模板扩成若干个同族变体。

    一个 family 对应一条**原始模板**，下面的所有变体都属于同一个 family，
    因此后续 group split 时会把它们整体分到同一个 split，避免“同一条模板的
    不同变体”分别出现在 train/test 中，造成变相的模板级泄漏。

    含 goods/place/day 槽位的模板本身组合空间够大，生成 3 个表面变体即可；
    无槽位模板（发票/支付/服务/无效等）组合空间小，必须生成更多变体，
    否则 dev/test 会因为“唯一子句不够用”而生成不满目标条数。
    """
    has_slot = bool(re.search(r'\{goods\}|\{place\}|\{day\}', template))
    if has_slot:
        variants = [template]

        p, changed_p = _apply_rules(template, PARAPHRASE_RULES, rng)
        if not changed_p:
            p = template
        variants.append(p)

        t = template
        if t and t[-1] not in '？?！!。吗呢吧啊呀了':
            t = t + rng.choice(['呢', '呀', '哦'])
        t, changed_t = _apply_rules(t, TYPO_RULES + PARAPHRASE_RULES, rng)
        if not changed_t and rng.random() < 0.6:
            t = '那个' + t
        variants.append(t)

        uniq = list(dict.fromkeys(variants))
        while len(uniq) < 3:
            uniq.append(template + rng.choice(['吧', '呀', '哦']))
        return uniq[:3]

    # 无槽位模板：通过同义换词、语气词、诉求尾巴、错别字制造多个不同子句。
    # 注意后缀不要带逗号，否则“按子句切分”时会被拆成另一个可重复出现的片段。
    no_slot_tails = [
        '', '麻烦尽快处理', '请回复我', '谢谢', '急', '希望能解决',
        '一直没人管', '给个说法吧', '实在是没办法了',
    ]
    no_slot_tones = ['', '啊', '呀', '吧', '呢', '哦', '嘛', '了']
    variants = [template]
    attempts = 0
    while len(variants) < 16 and attempts < 200:
        attempts += 1
        v = template
        if rng.random() < 0.65:
            v, _ = _apply_rules(
                v,
                rng.choice([PARAPHRASE_RULES, TYPO_RULES, PARAPHRASE_RULES + TYPO_RULES]),
                rng,
            )
        if rng.random() < 0.55:
            v += rng.choice(no_slot_tails)
        elif v and v[-1] not in '？?！!。吗呢吧啊呀了':
            v += rng.choice(no_slot_tones)
        if v not in variants:
            variants.append(v)
    return variants


def _fill(template, rng):
    """填充模板槽位。"""
    return template.format(
        goods=rng.choice(GOODS),
        place=rng.choice(PLACES),
        day=rng.choice(DAYS),
    )


def _make_clause(family_template, label, rng):
    """用某个 family 模板生成一条真实子句。"""
    clause = _fill(family_template, rng)

    if label == 'invalid':
        if rng.random() < 0.75:
            prefix = rng.choice(INVALID_PREFIXES)
            clause = prefix + clause
            clause = clause.strip()
    else:
        # 轻度口语化：句尾语气词、连续标点、偶尔省略主语。
        if rng.random() < 0.18 and clause[-1] not in '？?！!。吗呢吧啊呀了':
            clause += rng.choice(TONE_WORDS)
        if rng.random() < 0.08:
            clause = clause.rstrip('，。！？!?')
            clause += rng.choice(PUNCT_NOISE)

    return clause.strip()


def _choose_labels(rng):
    """按原生成器的标签数量分布和长尾权重采样标签组合。"""
    n_labels = rng.choices(
        [c for c, _ in LABEL_CNT_DIST],
        weights=[w for _, w in LABEL_CNT_DIST],
        k=1,
    )[0]
    labels = []
    while len(labels) < n_labels:
        keys = list(LABEL_WEIGHT.keys())
        weights = [LABEL_WEIGHT[k] for k in keys]
        label = rng.choices(keys, weights=weights, k=1)[0]
        if label not in labels:
            labels.append(label)
    if 'invalid' in labels:
        labels = ['invalid']
    return sorted(labels, key=lambda x: CLASSES.index(x))


def split_clauses(text):
    """按中文/英文逗号、句号、分号把文本切成子句。"""
    parts = re.split(r'[，。；！？!?;]', text)
    return [p.strip() for p in parts if p.strip()]


def load_rows(path):
    """读取 train/dev/test 文件为 [(text, [label,...]), ...]。"""
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


def load_splits(data_dir):
    return {
        name: load_rows(os.path.join(data_dir, f'{name}.txt'))
        for name in ('train', 'dev', 'test')
    }


def exact_overlap(a_rows, b_rows):
    """整句文本交集。"""
    a_texts = {t for t, _ in a_rows}
    b_texts = {t for t, _ in b_rows}
    return a_texts & b_texts


def clause_overlap(a_rows, b_rows):
    """子句交集。"""
    a_clauses = set()
    for t, _ in a_rows:
        a_clauses.update(split_clauses(t))
    b_clauses = set()
    for t, _ in b_rows:
        b_clauses.update(split_clauses(t))
    return a_clauses & b_clauses


def leakage_report(splits):
    """输出当前 split 的泄漏体检结果。"""
    print('\n' + '=' * 72)
    print('泄漏检测报告')
    print('=' * 72)

    train_rows = splits['train']
    dev_rows = splits['dev']
    test_rows = splits['test']

    train_texts = {t for t, _ in train_rows}
    dev_texts = {t for t, _ in dev_rows}
    test_texts = {t for t, _ in test_rows}
    print(f'整句交集: train∩dev={len(train_texts & dev_texts)}, '
          f'train∩test={len(train_texts & test_texts)}, '
          f'dev∩test={len(dev_texts & test_texts)}')

    train_clauses = set()
    for t, _ in train_rows:
        train_clauses.update(split_clauses(t))
    dev_clauses = set()
    for t, _ in dev_rows:
        dev_clauses.update(split_clauses(t))
    test_clauses = set()
    for t, _ in test_rows:
        test_clauses.update(split_clauses(t))
    print(f'子句池大小: train={len(train_clauses)}, dev={len(dev_clauses)}, '
          f'test={len(test_clauses)}')
    print(f'子句交集: train∩dev={len(train_clauses & dev_clauses)}, '
          f'train∩test={len(train_clauses & test_clauses)}, '
          f'dev∩test={len(dev_clauses & test_clauses)}')

    if test_clauses:
        covered = len(test_clauses & train_clauses)
        all_covered = sum(
            1 for t, _ in test_rows
            if split_clauses(t) and set(split_clauses(t)) <= train_clauses
        )
        n_with_clause = sum(1 for t, _ in test_rows if split_clauses(t))
        print(f'test 子句在 train 中原样出现的比例: '
              f'{covered / len(test_clauses) * 100:.2f}%')
        print(f'test 中所有子句都被 train 覆盖的文本占比: '
              f'{all_covered / max(n_with_clause, 1) * 100:.2f}%')

    for name in ('train', 'dev', 'test'):
        rows = splits[name]
        dup = len(rows) - len({t for t, _ in rows})
        print(f'{name} 整句重复: {dup}')

    return {
        'exact_overlap': {
            'train_dev': len(train_texts & dev_texts),
            'train_test': len(train_texts & test_texts),
            'dev_test': len(dev_texts & test_texts),
        },
        'clause_overlap': {
            'train_dev': len(train_clauses & dev_clauses),
            'train_test': len(train_clauses & test_clauses),
            'dev_test': len(dev_clauses & test_clauses),
        },
        'test_clause_covered_by_train_ratio': (
            round(covered / len(test_clauses), 4) if test_clauses else None
        ),
        'test_text_all_clauses_covered_by_train_ratio': (
            round(all_covered / max(n_with_clause, 1), 4) if n_with_clause else None
        ),
    }


def print_label_stats(name, rows):
    total = len(rows)
    label_counter = Counter()
    combo_counter = Counter()
    for _, labels in rows:
        label_counter.update(labels)
        combo_counter[','.join(labels)] += 1
    avg_labels = sum(len(l) for _, l in rows) / max(total, 1)
    print('\n' + '-' * 72)
    print(f'[{name}] {total} 条, 平均标签数 {avg_labels:.3f}')
    for label in CLASSES:
        cnt = label_counter.get(label, 0)
        print(f'  {label:<16} {cnt:>6}  {cnt / max(total, 1) * 100:>5.1f}%')
    print('  高频标签组合 Top8:')
    for combo, cnt in combo_counter.most_common(8):
        print(f'    {combo:<44} {cnt:>6}')


def write_rows(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for text, labels in rows:
            f.write(f'{text}\t{",".join(labels)}\n')


def backup_old_data(out_dir):
    """备份当前 train/dev/test 到 backup/leaky_时间戳。"""
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join(out_dir, 'backup', f'leaky_{stamp}')
    os.makedirs(backup_dir, exist_ok=True)
    for name in ('train.txt', 'dev.txt', 'test.txt'):
        src = os.path.join(out_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, name))
    return backup_dir


def build_family_pools(seed):
    """把每个标签的模板扩成 family，并按 split 大小比例分配到 train/dev/test。

    这里的一个 family = 一条原始模板 + 它的若干表面变体。分配时以 family
    为最小单位，保证同一条原始模板不会横跨 train/dev/test。
    """
    rng = random.Random(seed)
    all_families = defaultdict(list)
    for label in CLASSES:
        for template in LABEL_TEMPLATES.get(label, []):
            variants = _make_template_variants(template, rng)
            all_families[label].append(variants)

    # 给每个 split 分配 family。每个 split 中每个标签至少 1 个 family，
    # 否则某个 split 会缺少该标签的合法生成素材。
    pools = {'train': defaultdict(list), 'dev': defaultdict(list), 'test': defaultdict(list)}
    for label in CLASSES:
        fams = all_families[label][:]
        rng.shuffle(fams)
        n = len(fams)
        # 固定比例按 split 大小：默认 4000/600/600 -> train 约 77%。
        # 对 dev/test 强制至少 1 个，剩余给 train。
        dev_n = max(1, min(n - 2, int(round(n * 0.12))))
        test_n = max(1, min(n - dev_n - 1, int(round(n * 0.12))))
        train_n = n - dev_n - test_n
        if train_n < 1:
            # 极少数模板过少时，退化为 1/1/剩余。
            dev_n = test_n = 1
            train_n = n - 2

        pools['train'][label] = fams[:train_n]
        pools['dev'][label] = fams[train_n:train_n + dev_n]
        pools['test'][label] = fams[train_n + dev_n:]

    return pools


def generate_rows(n, split_name, pools, rng, clause_owner, seen_texts,
                  max_train_clause_repeat=2, max_devtest_clause_repeat=1):
    """生成一个 split 的样本。

    参数:
        clause_owner: dict，记录每个确切子句已经属于哪个 split，用于跨 split 阻断。
        seen_texts  : dict，记录每个整句已经属于哪个 split，用于整句去重。
    """
    rows = []
    split_clause_counter = Counter()
    attempts = 0
    max_attempts = max(n * 100, 5000)
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        labels = _choose_labels(rng)
        parts = []
        ok = True

        for label in labels:
            families = pools[split_name].get(label, [])
            if not families:
                ok = False
                break
            family = rng.choice(families)
            family_template = rng.choice(family)
            clause = _make_clause(family_template, label, rng)
            if not clause:
                ok = False
                break
            # 跨 split 阻断：这个子句已经被其他 split 用过，则换一个 family 重试。
            if clause in clause_owner and clause_owner[clause] != split_name:
                ok = False
                break
            # 同 split 内部重复控制。
            repeat_limit = (max_train_clause_repeat if split_name == 'train'
                            else max_devtest_clause_repeat)
            if split_clause_counter[clause] >= repeat_limit:
                ok = False
                break
            parts.append(clause)

        if not ok:
            continue

        # 只用纯标点连接，不要用“，还有 / ，另外”这类词作连接词。
        # 否则泄漏检测会把“还有子句B”当成一个完整子句，而模型仍能看到子句B原文，
        # 导致“按子句归组”失去意义。
        joiner = rng.choice(['，', '。', '；'])
        text = joiner.join(parts)
        if rng.random() < 0.12:
            text += rng.choice(PUNCT_NOISE)
        text = text.strip()
        if not text:
            continue

        # 整句全局唯一：防止 dev/test 整句重复，也顺带保证不会跨 split 整句重复。
        if text in seen_texts:
            continue

        # 用最终文本重新切一次子句，再登记所有权。这样即使以后某个模板里带了逗号，
        # 也不会出现“登记的是 A，泄漏检测切出来的是 B”的口径偏差。
        final_clauses = split_clauses(text)
        for clause in final_clauses:
            if clause in clause_owner and clause_owner[clause] != split_name:
                ok = False
                break
        if not ok:
            continue

        for clause in final_clauses:
            clause_owner[clause] = split_name
            split_clause_counter[clause] += 1
        seen_texts[text] = split_name
        rows.append((text, labels))

    if len(rows) < n:
        print(f'[警告] {split_name} 仅生成 {len(rows)}/{n} 条。建议增加 --n_train 或扩充模板。',
              file=sys.stderr)
    return rows


def main():
    parser = argparse.ArgumentParser(description='按子句家族分组重建 ShopCare 数据集')
    parser.add_argument('--n_train', type=int, default=4000)
    parser.add_argument('--n_dev', type=int, default=600)
    parser.add_argument('--n_test', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default=DATA_DIR)
    parser.add_argument('--no_backup', action='store_true')
    parser.add_argument('--preview', type=int, default=0)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 72)
    print('ShopCare 数据集重建：按子句家族分组切分')
    print('=' * 72)
    print(f'  目标规模 : train={args.n_train}, dev={args.n_dev}, test={args.n_test}')
    print(f'  随机种子 : {args.seed}')
    print(f'  输出目录 : {out_dir}')

    backup_dir = None
    if not args.no_backup:
        backup_dir = backup_old_data(out_dir)
        print(f'  旧数据已备份到: {backup_dir}')

    rng = random.Random(args.seed)
    pools = build_family_pools(args.seed)

    family_summary = {
        split: {label: len(pools[split][label]) for label in CLASSES}
        for split in pools
    }
    print('\n每个 split 的 family 数量:')
    for split in ('train', 'dev', 'test'):
        counts = family_summary[split]
        print(f'  {split:<6} ' + ', '.join(f'{lb}={counts[lb]}' for lb in CLASSES))

    clause_owner = {}
    seen_texts = {}
    splits = {}
    for name, n in [('train', args.n_train), ('dev', args.n_dev), ('test', args.n_test)]:
        print(f'\n开始生成 {name} ...')
        rows = generate_rows(n, name, pools, rng, clause_owner, seen_texts)
        path = os.path.join(out_dir, f'{name}.txt')
        write_rows(path, rows)
        splits[name] = rows
        print_label_stats(name, rows)
        print(f'  -> 已写入 {path}')

    report = leakage_report(splits)

    manifest = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': args.seed,
        'sizes': {'train': len(splits['train']), 'dev': len(splits['dev']),
                  'test': len(splits['test'])},
        'families_per_split': family_summary,
        'leakage': report,
        'backup_dir': backup_dir,
    }
    manifest_path = os.path.join(out_dir, 'split_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'\n切分清单已保存: {manifest_path}')

    if args.preview:
        print('\n样本预览:')
        for text, labels in splits['train'][:args.preview]:
            print(f'  [{",".join(labels):<28}] {text[:64]}')

    print('\n[OK] 重建完成。建议继续执行:')
    print('      python tools/check_dataset.py')
    print('      python 01-data/data_eda.py')


if __name__ == '__main__':
    main()
