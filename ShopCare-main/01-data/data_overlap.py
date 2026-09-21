"""
ShopCare 数据集重合度检查 (01-data/data_overlap.py)

用途:
    检查 train / dev / test 三个数据集之间的文本重合度, 用来判断是否存在数据泄漏
    (测试集的句子原封不动地出现在训练集里 -> 指标虚高, 不反映真实泛化能力)。

口径:
    只比较"文本"部分(制表符前的那一段), 不看标签。
    一条文本在另一个集合里出现过, 就算一次重合。

运行方式: 在 ShopCare-main 根目录下执行
    python 01-data/data_overlap.py
"""

import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 三个数据集文件名
SPLITS = {'train': 'train.txt', 'dev': 'dev.txt', 'test': 'test.txt'}


def read_texts(filename):
    """读取一个数据集, 返回文本列表(只取制表符前的文本部分, 忽略空行和注释行)"""
    path = os.path.join(DATA_DIR, filename)
    texts = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            # 格式: 文本 \t 标签1,标签2  -> 取文本部分
            texts.append(line.split('\t', 1)[0])
    return texts


def pair_overlap(a, b):
    """计算两个集合的重合: 返回 (重合条数, a 中重合占比, b 中重合占比)"""
    sa, sb = set(a), set(b)
    inter = sa & sb
    n_inter = len(inter)
    return n_inter, n_inter / len(sa) if sa else 0.0, n_inter / len(sb) if sb else 0.0


def main():
    # 1. 读取三个数据集
    data = {name: read_texts(fname) for name, fname in SPLITS.items()}

    print('=' * 60)
    print('ShopCare 数据集重合度检查')
    print('=' * 60)
    for name, texts in data.items():
        # 顺带统计集合内重复(同一文本出现多次)
        n_dup = len(texts) - len(set(texts))
        print(f'  {name:<6}: {len(texts)} 条  (去重后 {len(set(texts))} 条, 内部重复 {n_dup} 条)')
    print('-' * 60)

    # 2. 两两算重合度
    pairs = [('train', 'dev'), ('train', 'test'), ('dev', 'test')]
    print('  两两重合(按文本):')
    print(f'  {"集合A":<8}{"集合B":<8}{"重合条数":>10}{"占A比":>10}{"占B比":>10}')
    for a, b in pairs:
        n, ra, rb = pair_overlap(data[a], data[b])
        print(f'  {a:<8}{b:<8}{n:>10}{ra:>10.2%}{rb:>10.2%}')

    # 3. 三个集合的总交集(三处都出现的文本)
    common = set(data['train']) & set(data['dev']) & set(data['test'])
    print('-' * 60)
    print(f'  三个集合都出现的文本: {len(common)} 条')

    if any(pair_overlap(data[a], data[b])[0] > 0 for a, b in pairs):
        print('\n  [提示] 存在跨集合重合, 测试指标可能虚高, 建议重新切分数据。')
    else:
        print('\n  [OK] 三个集合互无重合, 无数据泄漏风险。')


if __name__ == '__main__':
    main()
