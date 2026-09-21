"""
ShopCare 数据探索分析 (EDA) —— 01-data 阶段

功能说明:
    对多标签工单数据做体检式分析, 输出四类信息:
      1. 规模与标签密度: 条数 / 平均标签数 / 单-双-三标签占比;
      2. 标签分布: 每个标签的样本数、占比, 用来观察长尾与不均衡程度;
      3. 标签共现矩阵: 哪些标签经常一起出现(多标签任务的直接证据);
      4. 文本长度分布: 用于确定 BERT 的 max_len 取值(避免过多截断).
    同时做基础异常检测: 空行、未知标签、超长文本、重复文本.
    新增: 自动剔除重复文本, 并生成 *_dedup.txt 文件.

设计说明:
    只依赖 Python 标准库 —— 这样在"还没装 pandas/torch 的环境"里也能先跑起来做数据体检,
    与 tools/check_dataset.py 的分工是: 本脚本面向"数据长什么样", check_dataset 面向"格式对不对".

运行方式:
    python 01-data/data_eda.py                    # 默认分析 train/dev/test
    python 01-data/data_eda.py --max_len 96       # 同时统计超长文本比例
"""

import argparse
import os
from collections import Counter

# 项目根目录: 本文件位于 01-data/ 下, 向上取一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, '01-data')


def load_dataset(path):
    """读取多标签数据文件

    参数: path —— 数据文件路径, 每行 "文本<TAB>标签1,标签2"
    返回: [(文本, [标签, ...]), ...] 列表
    """
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip('\n').rstrip('\r')
            if not line.strip():
                continue
            parts = line.split('\t')
            text = parts[0].strip()
            labels = [x.strip() for x in parts[1].split(',') if x.strip()] if len(parts) > 1 else []
            rows.append((text, labels))
    return rows


def deduplicate_dataset(rows):
    """基于文本内容去重"""
    seen = set()
    dedup_rows = []
    for text, labels in rows:
        if text not in seen:
            seen.add(text)
            dedup_rows.append((text, labels))
    return dedup_rows, len(rows) - len(dedup_rows)


def percentile(sorted_values, p):
    """计算分位数(不依赖 numpy 的轻量实现)"""
    if not sorted_values:
        return 0
    k = (len(sorted_values) - 1) * p
    low, high = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (k - low)


def analyze(name, path, class_list, max_len=96, top_combos=10, save_dedup=True):
    """对单个数据集做完整分析并打印报告"""
    if not os.path.exists(path):
        print(f'\n[{name}] 文件不存在: {path}')
        return None

    rows = load_dataset(path)
    raw_total = len(rows)
    if raw_total == 0:
        print(f'\n[{name}] 文件为空')
        return None

    # ================= 新增：去重逻辑 =================
    rows, dup_count = deduplicate_dataset(rows)
    total = len(rows)

    if dup_count > 0:
        print(f'\n[{name}] 发现并剔除了 {dup_count} 条重复文本。')
        if save_dedup:
            out_path = os.path.join(DATA_DIR, f'{name}_dedup.txt')
            with open(out_path, 'w', encoding='utf-8') as f:
                for text, labels in rows:
                    f.write(f"{text}\t{','.join(labels)}\n")
            print(f'[{name}] 去重后的数据已保存至: {out_path}')
    # =================================================

    label_counter = Counter()
    combo_counter = Counter()
    label_cnt_dist = Counter()
    lengths = []
    unknown = Counter()

    for text, labels in rows:
        label_counter.update(labels)
        combo_counter[','.join(labels)] += 1
        label_cnt_dist[len(labels)] += 1
        lengths.append(len(text))
        for lb in labels:
            if lb not in class_list:
                unknown[lb] += 1

    lengths.sort()
    avg_labels = sum(len(l) for _, l in rows) / total
    over_max = sum(1 for x in lengths if x > max_len)

    print('\n' + '=' * 72)
    print(f'[{name}] {path} (原始 {raw_total} 条, 去重后 {total} 条)')
    print('=' * 72)
    print(f'样本条数      : {total}')
    print(f'平均标签数    : {avg_labels:.3f}   (多标签任务参考区间 1.6 ~ 2.2)')
    print(f'标签个数分布  : ' + '  '.join(
        f'{k} 标签 {label_cnt_dist.get(k, 0)} 条({label_cnt_dist.get(k, 0) / total * 100:.1f}%)'
        for k in sorted(label_cnt_dist)))

    print('\n--- 1. 标签分布(按样本数降序) ---')
    print(f'{"标签":<18}{"样本数":>8}{"占比":>9}   {"均衡度":<20}')
    for label, cnt in label_counter.most_common():
        ratio = cnt / total * 100
        bar = '#' * int(ratio / 2)
        print(f'{label:<18}{cnt:>8}{ratio:>8.1f}%   {bar}')
    missing = [c for c in class_list if c not in label_counter]
    if missing:
        print(f'[警告] 以下标签在数据中完全没有出现: {missing}')

    print('\n--- 2. 标签共现矩阵(对角线为自身出现次数) ---')
    matrix = {a: {b: 0 for b in class_list} for a in class_list}
    for _, labels in rows:
        for a in labels:
            if a not in class_list:
                continue
            for b in labels:
                if b in class_list:
                    matrix[a][b] += 1
    header = ' ' * 18 + ''.join(f'{c[:9]:>11}' for c in class_list)
    print(header)
    for a in class_list:
        line = f'{a:<18}' + ''.join(f'{matrix[a][b]:>11}' for b in class_list)
        print(line)

    print(f'\n--- 3. 高频标签组合 Top{top_combos} ---')
    for combo, cnt in combo_counter.most_common(top_combos):
        print(f'  {combo:<44}{cnt:>8}  {cnt / total * 100:>5.1f}%')

    print(f'\n--- 4. 文本长度分布(max_len={max_len}) ---')
    print(f'  最短 {lengths[0]} / 平均 {sum(lengths) / total:.1f} / 最长 {lengths[-1]}')
    print(f'  P50 {percentile(lengths, 0.50):.0f} | P90 {percentile(lengths, 0.90):.0f} | '
          f'P99 {percentile(lengths, 0.99):.0f}')
    print(f'  超过 max_len({max_len}) 的样本: {over_max} 条 ({over_max / total * 100:.1f}%)'
          f'  —— 占比高则应调大 max_len 或做截断策略')

    print('\n--- 5. 异常检测 (基于去重后数据) ---')
    # 此时去重后，重复文本理论上为 0，保留此块用于检测无标签行和未知标签
    empty_label_rows = sum(1 for _, l in rows if not l)
    print(f'  无标签行: {empty_label_rows}')
    print(f'  未知标签: {dict(unknown) if unknown else "无"}')

    return {
        'total': total,
        'avg_labels': avg_labels,
        'label_counter': label_counter,
        'lengths': lengths,
    }


def main():
    parser = argparse.ArgumentParser(description='ShopCare 数据探索分析(EDA)')
    parser.add_argument('--max_len', type=int, default=96, help='BERT 最大长度, 用于统计截断比例')
    parser.add_argument('--files', type=str, default='train,dev,test', help='要分析的数据集名(逗号分隔)')
    parser.add_argument('--no_save', action='store_true', help='加上此参数则不保存去重后的文件，仅在内存中分析')
    args = parser.parse_args()

    class_path = os.path.join(DATA_DIR, 'class.txt')
    if not os.path.exists(class_path):
        raise SystemExit(f'找不到类别文件: {class_path}')
    class_list = [line.strip() for line in open(class_path, encoding='utf-8') if line.strip()]

    print('=' * 72)
    print('ShopCare 电商客服工单数据 EDA')
    print(f'类别体系({len(class_list)} 类): {", ".join(class_list)}')
    print('=' * 72)

    for name in [x.strip() for x in args.files.split(',') if x.strip()]:
        # 传参控制是否保存去重文件
        analyze(name, os.path.join(DATA_DIR, f'{name}.txt'), class_list,
                max_len=args.max_len, save_dedup=not args.no_save)

    print('\n' + '=' * 72)
    print('EDA 完成。若某类样本极少(长尾标签), 训练时请在损失函数中提高其 pos_weight。')
    print('=' * 72)


if __name__ == '__main__':
    main()