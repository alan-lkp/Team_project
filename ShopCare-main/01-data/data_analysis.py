# -*- coding: utf-8 -*-
"""
ShopCare 数据对比分析 —— 对 train / dev / test 三份工单数据做整体体检

与 data_eda.py 的分工:
    data_eda.py  面向「单个数据集长什么样」(逐份独立分析);
    本脚本       面向「三份数据放在一起是否一致、有没有问题」——
                 跨数据集对比 + 数据泄漏检测 + 图表输出。

分析内容:
    1. 基础校验  : 格式(制表符分隔)、未知标签、空标签行;
    2. 规模概览  : 条数、平均标签数、文本长度分位数(各 split 一列);
    3. 标签分布  : 每个标签在 train/dev/test 中的占比是否一致
                   (最大占比差 + JS 散度, 差得太多说明抽样不均匀, dev/test 指标会偏);
    4. 泄漏检测  : train↔dev、train↔test、dev↔test 之间的重复文本
                   (有重叠则测试指标虚高, 必须先处理);
    5. 标签共现  : train 上的共现矩阵(多标签关联的直接证据);
    6. 长度建议  : 按 P99 给出 BERT max_len 的参考取值;
    7. 图表输出  : 标签占比对比图 / 文本长度分布图 / 共现热力图,
                   保存到 01-data/analysis_output/, 同时控制台打印全部数值(表格视图)。

运行方式:
    python 01-data/data_analysis.py                 # 分析 train/dev/test
    python 01-data/data_analysis.py --no-charts     # 只要文字报告, 不画图
依赖:
    pandas, numpy, matplotlib (画图时才需要 matplotlib)
"""

import argparse
import os
import sys
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

# Windows 控制台/重定向默认可能是 GBK, 统一切到 UTF-8 避免乱码
_reconfigure = getattr(sys.stdout, 'reconfigure', None)
if _reconfigure is not None:
    _reconfigure(encoding='utf-8')

# 目录约定: 本文件位于 01-data/ 下
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, 'analysis_output')

# 图表配色: 三个系列固定按 train/dev/test 顺序取蓝/橙/水鸭三色
# (已通过色觉安全校验; 具体数值全部随报告打印, 不依赖颜色单独传达信息)
SERIES_COLORS = {'train': '#2a78d6', 'dev': '#eb6834', 'test': '#1baf7a'}


# ---------------------------------------------------------------- 数据读取

def load_class_list():
    """读 class.txt, 每行取第一个空白/井号前的英文标签名"""
    path = os.path.join(DATA_DIR, 'class.txt')
    classes = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            token = line.strip().split('#')[0].split()
            if token:
                classes.append(token[0])
    return classes


def load_split(name):
    """读取一份数据文件, 返回 DataFrame: [text, labels(list), n_labels]

    格式: 每行 "文本<TAB>标签1,标签2"; 同时收集格式异常行号。
    """
    path = os.path.join(DATA_DIR, f'{name}.txt')
    if not os.path.exists(path):
        return None, []
    texts, labels, bad_lines = [], [], []
    with open(path, encoding='utf-8') as f:
        for no, raw in enumerate(f, start=1):
            line = raw.rstrip('\n').rstrip('\r')
            if not line.strip():
                bad_lines.append((no, '空行'))
                continue
            parts = line.split('\t')
            if len(parts) < 2 or not parts[1].strip():
                bad_lines.append((no, '缺少标签列'))
                continue
            texts.append(parts[0].strip())
            labels.append([x.strip() for x in parts[1].split(',') if x.strip()])
    df = pd.DataFrame({'text': texts, 'labels': labels})
    df['n_labels'] = df['labels'].map(len)
    df['length'] = df['text'].map(len)
    return df, bad_lines


# ---------------------------------------------------------------- 分析

def js_divergence(p, q):
    """两个概率分布的 JS 散度(0=完全一致, 1=完全不同), 用于分布对齐程度"""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def label_distribution(dfs, classes):
    """每个标签在各 split 中的样本占比(按行数), 返回 DataFrame(index=标签)"""
    dist = {}
    for name, df in dfs.items():
        cnt = Counter(l for ls in df['labels'] for l in ls)
        dist[name] = [cnt.get(c, 0) / len(df) * 100 for c in classes]
    return pd.DataFrame(dist, index=classes)


def cooccurrence(df, classes):
    """标签共现矩阵: matrix[a][b] = 同时带标签 a 和 b 的样本数(对角线=自身总数)"""
    mat = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for labels in df['labels']:
        valid = [l for l in labels if l in classes]
        for a in valid:
            for b in valid:
                mat.loc[a, b] += 1
    return mat


def leak_report(dfs):
    """跨 split 重复文本检测: 返回 [(splitA, splitB, 重复条数), ...]"""
    sets = {name: set(df['text']) for name, df in dfs.items()}
    out = []
    for a, b in combinations(sets, 2):
        n = len(sets[a] & sets[b])
        out.append((a, b, n))
    return out


def print_summary(name, df, classes):
    """打印单个 split 的概览(规模/标签密度/长度)"""
    n = len(df)
    lengths = np.sort(df['length'].to_numpy())
    print(f'  [{name:<5}] {n:>5} 条 | 唯一文本 {df["text"].nunique():>5} | '
          f'平均标签数 {df["n_labels"].mean():.2f} | '
          f'长度 P50={np.percentile(lengths, 50):.0f} '
          f'P95={np.percentile(lengths, 95):.0f} '
          f'P99={np.percentile(lengths, 99):.0f} 最大={lengths[-1]}')
    cnt = Counter(l for ls in df['labels'] for l in ls)
    missing = [c for c in classes if cnt.get(c, 0) == 0]
    if missing:
        print(f'          [警告] 标签完全未出现: {missing}')


# ---------------------------------------------------------------- 图表

def make_charts(dfs, dist, cooc, classes):
    """输出三张 PNG: 标签占比对比 / 文本长度分布 / train 共现热力图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    os.makedirs(OUT_DIR, exist_ok=True)

    # 图 1: 标签占比分组柱状图(按标签名定位, 颜色只跟随 split)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(classes))
    width = 0.26
    for i, name in enumerate(dfs):
        ax.bar(x + (i - 1) * width, dist[name], width * 0.9,
               label=name, color=SERIES_COLORS[name])
    ax.set_xticks(x, classes, rotation=30, ha='right')
    ax.set_ylabel('样本占比 (%)')
    ax.set_title('各标签占比: train / dev / test 对比')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'label_distribution.png'), dpi=150)
    plt.close(fig)

    # 图 2: 文本长度分布, 小倍数(每 split 一格, 共用 x 轴, 避免叠色难辨)
    fig, axes = plt.subplots(1, len(dfs), figsize=(4 * len(dfs), 3.2), sharex=True)
    hi = max(int(np.percentile(np.concatenate([d['length'].to_numpy() for d in dfs.values()]), 99)), 1)
    for ax, name in zip(np.atleast_1d(axes), dfs):
        ax.hist(dfs[name]['length'].clip(upper=hi * 1.2), bins=40,
                color=SERIES_COLORS[name], edgecolor='white', linewidth=0.6)
        ax.axvline(hi, color='#52514e', linestyle='--', linewidth=1)
        ax.set_title(f'{name} (整体P99={hi})')
        ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('文本字符数')
    fig.supylabel('条数')
    fig.suptitle('文本长度分布(虚线为整体 P99)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'text_length.png'), dpi=150)
    plt.close(fig)

    # 图 3: train 标签共现热力图(单色蓝渐变, 数值直标; 对角线为自身总数, 量级大, 从配色中剔除)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    data = cooc.to_numpy().astype(float).copy()
    data[np.diag_indices_from(cooc)] = np.nan
    im = ax.imshow(np.ma.masked_invalid(data), cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    offdiag_max = np.nanmax(data)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cooc.iat[i, j]
            frac = v / offdiag_max if j != i else 0
            ax.text(j, i, str(v), ha='center', va='center', fontsize=8,
                    color='white' if frac > 0.55 else '#0b0b0b')
    ax.set_title('标签共现矩阵(train): 非对角=同现样本数, 对角=该标签总数')
    ax.set_ylabel('标签 A')
    ax.set_xlabel('标签 B')
    fig.colorbar(im, ax=ax, shrink=0.8, label='同现样本数(不含对角线)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'cooccurrence_train.png'), dpi=150)
    plt.close(fig)
    return sorted(os.listdir(OUT_DIR))


# ---------------------------------------------------------------- 主流程

def main():
    parser = argparse.ArgumentParser(description='ShopCare train/dev/test 对比分析')
    parser.add_argument('--splits', type=str, default='train,dev,test')
    parser.add_argument('--no-charts', action='store_true', help='不生成 PNG 图表')
    args = parser.parse_args()

    classes = load_class_list()
    names = [s.strip() for s in args.splits.split(',') if s.strip()]

    print('=' * 74)
    print('ShopCare 工单数据对比分析 (train / dev / test)')
    print(f'类别体系({len(classes)} 类): {", ".join(classes)}')
    print('=' * 74)

    # 1. 基础校验
    dfs, all_bad = {}, {}
    for name in names:
        df, bad = load_split(name)
        if df is None:
            print(f'[跳过] 文件不存在: 01-data/{name}.txt')
            continue
        all_bad[name] = bad
        unknown = Counter(l for ls in df['labels'] for l in ls if l not in classes)
        print(f'[{name}] 读取 {len(df)} 条 | 格式异常行 {len(bad)} | '
              f'未知标签 {dict(unknown) if unknown else "无"}')
        dfs[name] = df
    if not dfs:
        return
    names = list(dfs)

    # 2. 规模概览
    print('\n--- 1. 规模概览 ---')
    for name in names:
        print_summary(name, dfs[name], classes)

    # 3. 标签分布一致性
    print('\n--- 2. 标签占比(%) 跨数据集对比 ---')
    dist = label_distribution(dfs, classes)
    dist['最大差'] = dist.max(axis=1) - dist.min(axis=1)
    print(dist.round(1).to_string())
    if len(names) >= 2 and 'train' in dfs:
        base = dist['train'].to_numpy()
        for name in names:
            if name == 'train':
                continue
            d = js_divergence(base, dist[name].to_numpy())
            flag = '  <-- 分布差异偏大, 建议重新抽样' if d > 0.05 else ''
            print(f'  JS散度 train vs {name:<5}: {d:.4f}{flag}')
    worst = dist['最大差'].idxmax()
    print(f'  占比波动最大的标签: {worst}(最大差 {dist.loc[worst, "最大差"]:.1f} 个百分点)')

    # 4. 泄漏检测
    print('\n--- 3. 跨数据集重复文本(泄漏检测) ---')
    if len(names) >= 2:
        for a, b, n in leak_report(dfs):
            print(f'  {a} ∩ {b:<5}: {n} 条' + ('  <-- 有重叠, 需去重!' if n else '  正常'))
    else:
        print('  (只有一个数据集, 跳过)')

    # 5. 共现
    print('\n--- 4. 标签共现矩阵(train; 对角线=该标签总数) ---')
    cooc = cooccurrence(dfs.get('train', pd.DataFrame({'labels': []})), classes)
    if 'train' in dfs:
        cooc = cooccurrence(dfs['train'], classes)
        print(cooc.to_string())

    # 6. 长度与 max_len 建议
    print('\n--- 5. 文本长度与 max_len 建议 ---')
    all_len = np.concatenate([d['length'].to_numpy() for d in dfs.values()])
    p99, p95 = np.percentile(all_len, 99), np.percentile(all_len, 95)
    print(f'  整体长度: 平均 {all_len.mean():.1f} | P95 {p95:.0f} | P99 {p99:.0f} | 最大 {all_len.max()}')
    for L in (64, 96, 128, 192):
        r = (all_len > L).mean() * 100
        print(f'  max_len={L:<4} 会被截断的样本: {r:.2f}%')
    print(f'  参考: 覆盖 P99 取 max_len≈{int(np.ceil(p99 / 16) * 16)} '
          f'(按 16 对齐; 若显存紧张可退到 P95)')

    # 7. 图表
    if not args.no_charts:
        try:
            files = make_charts(dfs, dist, cooc, classes)
            print(f'\n--- 6. 图表已输出到 01-data/analysis_output/: {", ".join(files)}')
        except ImportError:
            print('\n[提示] 未安装 matplotlib, 跳过图表; 数值结论见上方表格。')

    print('\n' + '=' * 74)
    print('分析完成。重点看: ① 占比列间差异(抽样是否均匀) ② 泄漏是否为 0 ③ max_len 截断比例。')
    print('=' * 74)


if __name__ == '__main__':
    main()
