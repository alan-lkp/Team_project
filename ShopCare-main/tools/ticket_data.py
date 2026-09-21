"""
ShopCare 工单数据读取工具 (tools)

模块说明:
    02-rf / 03-fasttext / 04-bert 三个对照模型**共用同一份数据读取逻辑**.
    这一点很重要: 如果三个模型各自实现一套读取/切分/标签映射, 很容易出现
    "读的顺序不同""标签索引错位""某个模型过滤掉空文本而另一个没过滤"这类
    隐蔽偏差, 最后横向对比出来的指标就不可信了.

数据格式(详见 01-data/data_format.md):
    每行: 文本 \t 标签1,标签2,...
    UTF-8 编码, 换行符 \n, 空行跳过, 以 # 开头的行视为注释

对外接口:
    load_class_list()   -> 9 个标签(顺序即索引)
    load_label_meta()   -> 标签中文名/归属部门/基础优先级/SLA
    read_ticket_file()  -> [(text, [label, ...]), ...]
    load_splits()       -> (train, dev, test, class_list) 一站式读取
    to_xy()             -> (texts, multi-hot 矩阵) 供 sklearn / fasttext 使用
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, '01-data')
DEFAULT_CLASS_PATH = os.path.join(DATA_DIR, 'class.txt')
DEFAULT_META_PATH = os.path.join(DATA_DIR, 'label_meta.json')

# 标签分隔符: 兼容中英文逗号, 防止人工标注时手滑用了全角逗号
LABEL_SEP = ','

# ------------------------------------------------------------
# 演示样例(02-rf / 03-fasttext / 04-bert 三个推理脚本共用同一份)
# ------------------------------------------------------------
# 前三条措辞**直接取自生成的测试语料**, 属于"同分布"样本;
# 第 4 条是语料里根本不存在的新说法(换了同义词和句式).
# 把这两类放在一起跑, 就能直观看到对照实验的真正看点:
#     n-gram 模型靠字面匹配, 语料内很高分, 一换说法就掉;
#     BERT 有预训练语义, 换说法的掉分幅度明显更小.
DEMO_SAMPLES = [
    ('麻烦问下，电饭锅收到就是坏的，根本用不了，另外客服半天不回消息，问了三遍都没人理呀',
     'quality,service'),
    ('快递放驿站也不通知我，自己去翻才找到。付款的时候一直失败，换了好几张卡都不行',
     'logistics,payment_account'),
    ('哈哈哈哈哈哈哈哈!!', 'invalid'),
    ('包裹在途中卡了半个月没有任何更新，找客服也没人搭理我', None),
]
def load_class_list(class_path=None):
    """读取标签表, 行号即索引(第 1 行 -> 0)"""
    path = class_path or DEFAULT_CLASS_PATH
    with open(path, encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError('标签文件为空: ' + path)
    return classes


def load_label_meta(meta_path=None):
    """读取标签元数据(cn / dept / base_priority / sla_hours)"""
    path = meta_path or DEFAULT_META_PATH
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_ticket_file(path, class_list=None, strict=True):
    """读取一个数据文件

    参数:
        path       : 数据文件路径
        class_list : 标签表; 传入时用于校验标签合法性
        strict     : True  -> 遇到未知标签直接报错(尽早暴露标注错误)
                     False -> 跳过未知标签并计数告警

    返回: [(text, [label, ...]), ...]
    """
    if not os.path.exists(path):
        raise FileNotFoundError('数据文件不存在: ' + path)
    known = set(class_list) if class_list else None
    pairs, skipped = [], 0
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip('\n').strip('\r')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                skipped += 1
                continue
            text = parts[0].strip()
            labels = [x.strip() for x in parts[1].replace('，', LABEL_SEP).split(LABEL_SEP) if x.strip()]
            # 去重但保持顺序, 避免"同一标签写两遍"影响 multi-hot
            labels = list(dict.fromkeys(labels))
            if not text or not labels:
                skipped += 1
                continue
            if known is not None:
                unknown = [x for x in labels if x not in known]
                if unknown:
                    if strict:
                        raise ValueError(f'第 {lineno} 行出现未知标签 {unknown}, 合法标签: {sorted(known)}')
                    labels = [x for x in labels if x in known]
                    if not labels:
                        skipped += 1
                        continue
            pairs.append((text, labels))
    if skipped:
        print(f'[警告] {os.path.basename(path)} 跳过 {skipped} 行(空行/格式错误/标签非法)')
    return pairs


def load_splits(data_dir=None, class_path=None):
    """一次性读入 train/dev/test

    返回: (train_pairs, dev_pairs, test_pairs, class_list)
    """
    data_dir = data_dir or DATA_DIR
    class_list = load_class_list(class_path)
    splits = []
    for name in ('train.txt', 'dev.txt', 'test_paraphrase.txt'):
        splits.append(read_ticket_file(os.path.join(data_dir, name), class_list))
    return splits[0], splits[1], splits[2], class_list


def to_xy(pairs, class_list):
    """转成 sklearn / numpy 友好的形式

    返回: (texts, Y) —— texts 是 str 列表, Y 是 (N, C) 的 0/1 矩阵(list of list)
    """
    idx = {c: i for i, c in enumerate(class_list)}
    texts = [t for t, _ in pairs]
    Y = []
    for _, labels in pairs:
        row = [0] * len(class_list)
        for lb in labels:
            if lb in idx:
                row[idx[lb]] = 1
        Y.append(row)
    return texts, Y


def label_statistics(pairs, class_list):
    """统计每个标签的样本数与占比, 用于发现长尾标签"""
    total = len(pairs)
    counts = {c: 0 for c in class_list}
    label_num_sum = 0
    for _, labels in pairs:
        label_num_sum += len(labels)
        for lb in labels:
            counts[lb] = counts.get(lb, 0) + 1
    stats = []
    for c in class_list:
        n = counts.get(c, 0)
        stats.append({
            'label': c,
            'count': n,
            'ratio': round(n / total, 4) if total else 0.0,
        })
    return {
        'total': total,
        'avg_labels': round(label_num_sum / total, 3) if total else 0.0,
        'per_label': stats,
    }


if __name__ == '__main__':
    tr, dv, te, classes = load_splits()
    print('=' * 72)
    print('ShopCare 数据读取自检')
    print('=' * 72)
    print(f'标签数 : {len(classes)} -> {", ".join(classes)}')
    print(f'训练集 : {len(tr)} 条    验证集 : {len(dv)} 条    测试集 : {len(te)} 条')
    st = label_statistics(tr, classes)
    print(f'平均标签数: {st["avg_labels"]}')
    print('-' * 72)
    print(f'{"标签":<16}{"训练集样本数":>12}{"占比":>10}')
    for row in st['per_label']:
        print(f'{row["label"]:<16}{row["count"]:>12}{row["ratio"]:>10.2%}')
    texts, Y = to_xy(tr, classes)
    print('-' * 72)
    print(f'to_xy 自检: texts={len(texts)}, Y={len(Y)}x{len(Y[0])}')
    print('样例:', texts[0][:40], '->', [classes[i] for i, v in enumerate(Y[0]) if v])
    print('\n[OK] ticket_data 自检通过')