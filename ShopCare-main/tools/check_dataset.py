"""
ShopCare 数据体检 & 单标签升维多标签工具 (01-data 阶段)

功能说明:
    1. 体检模式(默认): 校验数据格式是否合法, 并输出标签分布 / 长度分布等统计;
    2. 修复模式(--fix): 清洗不规范数据(去空白、去重标签、按索引排序、统一换行符)后另存;
    3. 升维模式(--upgrade): 把**单标签**数据按关键词规则升维成**多标签**数据 ——
       公开的中文电商数据集几乎都是单标签(好/中/差、10 类商品评论等),
       本模式是"用现成公开数据快速造出多标签工单数据"的关键一步.

校验规则(与 01-data/data_format.md 一一对应):
    - 每行必须是 "文本<TAB>标签1,标签2" 两列, 用制表符分隔;
    - 文本不能为空; 标签不能为空、必须是 class.txt 中的英文标签;
    - 标签不能重复, 且应按 class.txt 索引升序排列;
    - 文件应为 UTF-8 且行尾不带 \\r(Windows 手改数据最容易踩的坑).

运行方式:
    python tools/check_dataset.py                                  # 体检 train/dev/test
    python tools/check_dataset.py --files train.txt                # 只体检 train.txt
    python tools/check_dataset.py --fix 01-data/train.txt --out 01-data/train_clean.txt
    python tools/check_dataset.py --upgrade raw_single_label.txt --out 01-data/train.txt
"""

import argparse
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, '01-data')

# ============================================================
# todo 1. 标签体系与中文别名(便于对接中文标注的真实业务数据)
# ============================================================
CLASS_FILE = os.path.join(DATA_DIR, 'class.txt')

CN2EN = {
    '物流问题': 'logistics', '物流配送': 'logistics', '物流': 'logistics',
    '商品质量': 'quality', '质量问题': 'quality', '质量': 'quality',
    '退款售后': 'after_sale', '退款退货': 'after_sale', '退款': 'after_sale', '退货': 'after_sale',
    '发票问题': 'invoice', '发票': 'invoice',
    '价格与优惠': 'price_promo', '价格保护': 'price_promo', '优惠券': 'price_promo', '价保': 'price_promo',
    '支付与账号': 'payment_account', '支付问题': 'payment_account', '账号问题': 'payment_account',
    '售前咨询': 'consult', '产品功能咨询': 'consult', '咨询': 'consult', '使用咨询': 'consult',
    '服务态度': 'service', '服务态度投诉': 'service', '投诉': 'service',
    '无效反馈': 'invalid', '无效恶意': 'invalid', '恶意反馈': 'invalid', '垃圾反馈': 'invalid',
}

# ============================================================
# todo 2. 单标签升维规则: 主标签 -> [(关键词列表, 追加标签), ...]
# ============================================================
# 设计原则: 只有当文本中出现明确的"跨诉求"线索时才追加标签, 避免把数据搞脏.
# 例如用户问"怎么申请退款"(consult) 且提到"退款", 说明这条工单确实涉及售后链路,
# 因此升维为 consult + after_sale —— 这正是真实工单"混合诉求"的来源.
UPGRADE_RULES = [
    (['退款', '退货', '退钱', '退换', '换货', '售后', '维修'], 'after_sale'),
    (['快递', '物流', '发货', '派送', '签收', '揽收', '驿站', '中转'], 'logistics'),
    (['发票', '开票', '专票', '税号', '报销'], 'invoice'),
    (['优惠券', '满减', '价保', '降价', '差价', '赠品', '券'], 'price_promo'),
    (['支付', '扣款', '付款', '账号', '登录', '验证码'], 'payment_account'),
    (['客服', '态度', '回复', '敷衍', '推诿', '不理'], 'service'),
    (['破损', '坏了', '坏了', '瑕疵', '划痕', '过期', '变质', '漏发', '少发', '质量'], 'quality'),
    (['怎么', '如何', '咨询', '有货', '教程', '保修', '参数', '支持吗'], 'consult'),
]

MAX_LABELS = 3   # 升维后最多保留几个标签(超过则按规则命中顺序截断, 保持数据干净)


def load_class_list():
    """读取 01-data/class.txt 得到标签列表(顺序即索引)"""
    if not os.path.exists(CLASS_FILE):
        raise SystemExit(f'找不到类别文件: {CLASS_FILE}')
    return [line.strip() for line in open(CLASS_FILE, encoding='utf-8') if line.strip()]


def normalize_label(raw, class_list):
    """把各种写法的标签名统一成 class.txt 中的英文标签; 无法识别返回 None"""
    raw = raw.strip()
    if raw in class_list:
        return raw
    if raw.lower() in class_list:
        return raw.lower()
    if raw in CN2EN:
        return CN2EN[raw]
    return None


def parse_line(line, class_list):
    """解析一行数据, 返回 (文本, [标签], [错误信息])"""
    errors = []
    raw_line = line.rstrip('\n')
    if raw_line.endswith('\r'):
        errors.append('行尾包含 \\r (Windows 换行符未清理), 会导致最后一个标签无法匹配')
        raw_line = raw_line.rstrip('\r')
    if not raw_line.strip():
        return None, None, ['空行']
    if '\t' not in raw_line:
        return None, None, ['缺少制表符分隔(格式应为: 文本<TAB>标签1,标签2)']
    text, label_str = raw_line.split('\t', 1)
    text = text.strip()
    if not text:
        errors.append('文本为空')
    labels = []
    for item in label_str.split(','):
        if not item.strip():
            continue
        label = normalize_label(item, class_list)
        if label is None:
            errors.append(f'未知标签: "{item.strip()}"')
        elif label not in labels:
            labels.append(label)
    if not labels:
        errors.append('没有任何有效标签')
    return text, labels, errors


def upgade_labels(text, labels, class_list):
    """按关键词规则给单标签样本追加隐含标签(升维多标签)"""
    result = list(labels)
    for keywords, extra in UPGRADE_RULES:
        if extra in result:
            continue
        if any(kw in text for kw in keywords):
            result.append(extra)
            if len(result) >= MAX_LABELS:
                break
    return sorted(result, key=lambda x: class_list.index(x))


def report(name, path, class_list, verbose_limit=10, do_fix=False, upgrade=False):
    """体检单个数据文件; do_fix / upgrade 为 True 时返回清洗后的数据行"""
    if not os.path.exists(path):
        print(f'\n[{name}] 文件不存在: {path}')
        return None

    total, bad, fixed_rows = 0, 0, []
    label_counter, error_counter, upgrade_changed = Counter(), Counter(), 0
    unsorted_rows = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            text, labels, errors = parse_line(line, class_list)
            if errors:
                bad += 1
                for e in errors:
                    error_counter[e.split(':')[0]] += 1
                if bad <= verbose_limit:
                    print(f'  [第 {total} 行] ' + '; '.join(errors))
                if text is None or not labels:
                    continue
            if upgrade:
                new_labels = upgade_labels(text, labels, class_list)
                if new_labels != labels:
                    upgrade_changed += 1
                labels = new_labels
            else:
                if labels != sorted(labels, key=lambda x: class_list.index(x)):
                    unsorted_rows += 1
                    labels = sorted(labels, key=lambda x: class_list.index(x))
            label_counter.update(labels)
            fixed_rows.append((text, labels))

    valid = len(fixed_rows)
    avg_labels = sum(len(l) for _, l in fixed_rows) / max(valid, 1)

    print('\n' + '=' * 72)
    print(f'[{name}] {path}')
    print('=' * 72)
    print(f'  总行数 {total} | 有效 {valid} | 有问题 {bad} | 平均标签数 {avg_labels:.3f}')
    if unsorted_rows:
        print(f'  [提示] {unsorted_rows} 行标签顺序不规范(已按 class.txt 索引自动排序)')
    if upgrade:
        print(f'  [升维] {upgrade_changed} 行被追加了隐含标签 (单标签 -> 多标签)')
    if error_counter:
        print('  错误类型统计:')
        for err, cnt in error_counter.most_common():
            print(f'    {err}: {cnt} 行')
    print('  标签分布:')
    for label in class_list:
        cnt = label_counter.get(label, 0)
        print(f'    {label:<16} {cnt:>6}  {cnt / max(valid, 1) * 100:>5.1f}%')

    combo = Counter(','.join(l) for _, l in fixed_rows)
    print('  高频标签组合 Top5:')
    for c, cnt in combo.most_common(5):
        print(f'    {c:<44} {cnt:>6}')

    return fixed_rows


def write_rows(path, rows):
    """把清洗/升维后的数据写出"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for text, labels in rows:
            f.write(text + '\t' + ','.join(labels) + '\n')
    print(f'  -> 已写出 {len(rows)} 条到 {path}')


def main():
    parser = argparse.ArgumentParser(description='ShopCare 数据体检 / 修复 / 单标签升维')
    parser.add_argument('--files', type=str, default='train.txt,dev.txt,test.txt',
                        help='要体检的文件名(逗号分隔, 相对 01-data 目录)')
    parser.add_argument('--path', type=str, default=None, help='直接指定要体检的文件路径(优先级高于 --files)')
    parser.add_argument('--fix', type=str, default=None, help='修复模式: 指定要清洗的文件路径')
    parser.add_argument('--upgrade', type=str, default=None, help='升维模式: 指定单标签数据文件路径')
    parser.add_argument('--out', type=str, default=None, help='输出路径(配合 --fix / --upgrade)')
    args = parser.parse_args()

    class_list = load_class_list()
    print('=' * 72)
    print(f'ShopCare 数据体检工具 | 标签体系({len(class_list)} 类): {", ".join(class_list)}')
    print('=' * 72)

    all_ok = True
    if args.fix or args.upgrade:
        src = args.fix or args.upgrade
        if not os.path.exists(src):
            raise SystemExit(f'文件不存在: {src}')
        out = args.out or (src.replace('.txt', '_clean.txt') if args.fix else os.path.join(DATA_DIR, 'train.txt'))
        name = '修复' if args.fix else '升维'
        rows = report(name, src, class_list, do_fix=bool(args.fix), upgrade=bool(args.upgrade))
        if rows:
            write_rows(out, rows)
            print(f'\n[{name}] 完成。建议接着运行: python tools/check_dataset.py --path {out}')
        return

    for name in [x.strip() for x in args.files.split(',') if x.strip()]:
        path = args.path or os.path.join(DATA_DIR, name)
        rows = report(name, path, class_list)
        if rows is None:
            all_ok = False
        elif not rows:
            all_ok = False
        if args.path:
            break

    print('\n' + '=' * 72)
    # 不用 emoji: Windows GBK 控制台(未设 PYTHONIOENCODING=utf-8)会直接 UnicodeEncodeError
    print('体检完成。' + ('数据格式无阻断性问题 [OK]' if all_ok else '[警告] 存在缺失或为空的数据集'))
    print('提示: 真实单标签数据可用 --upgrade 升维成多标签(会打印升维前后的分布与共现关系)')
    print('=' * 72)
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()