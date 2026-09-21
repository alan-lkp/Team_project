import os
import re
import random
from collections import defaultdict

# ==================== 配置区 ====================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(DATA_DIR, 'train.txt')
DEV_FILE = os.path.join(DATA_DIR, 'dev.txt')
TEST_FILE = os.path.join(DATA_DIR, 'test.txt')

# 目标划分比例 (可调整，目前是 8:1:1)
SPLIT_RATIO = {'train': 0.8, 'dev': 0.1, 'test': 0.1}
RANDOM_SEED = 42


# ===============================================

def parse_line(line):
    """解析数据行：文本\t标签1,标签2"""
    line = line.strip()
    if not line:
        return None, None
    parts = line.split('\t')
    text = parts[0].strip()
    labels = parts[1].strip() if len(parts) > 1 else ""
    return text, labels


def extract_clauses(text):
    """按中英文标点切分提取子句，并去重排序"""
    # 按逗号、句号、问号、叹号、分号、换行等切分
    parts = re.split(r'[,，.。?？!！;；\n\r]+', text)
    clauses = [p.strip() for p in parts if p.strip()]
    if not clauses:
        return text.strip()  # 兜底：如果没有标点，整句作为唯一子句
    # 去重并排序，保证不同顺序但相同子句组成的句子归入同一组
    return '|'.join(sorted(list(set(clauses))))


def load_all_data():
    """读取所有数据，并按照子句模板进行分组"""
    all_data = []
    for file_path in [TRAIN_FILE, DEV_FILE, TEST_FILE]:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    text, labels = parse_line(line)
                    if text:
                        all_data.append({'text': text, 'labels': labels})

    # 1. 全局整句去重 (防止完全重复的句子)
    unique_data = []
    seen_texts = set()
    for item in all_data:
        if item['text'] not in seen_texts:
            seen_texts.add(item['text'])
            unique_data.append(item)

    print(f"原始数据总条数: {len(all_data)}，全局整句去重后: {len(unique_data)}")

    # 2. 按子句模板分组 (Grouped Split 的核心)
    groups = defaultdict(list)
    for item in unique_data:
        template = extract_clauses(item['text'])
        groups[template].append(item)

    print(f"去重后提取出 {len(groups)} 个唯一子句模板组合。")

    # 将字典转为列表并打乱
    group_list = list(groups.values())
    random.seed(RANDOM_SEED)
    random.shuffle(group_list)

    return group_list


def split_groups(group_list):
    """按组分配数据到 train/dev/test，保证组不被切碎"""
    total = sum(len(g) for g in group_list)
    targets = {k: int(v * total) for k, v in SPLIT_RATIO.items()}

    split_data = {'train': [], 'dev': [], 'test': []}
    current_counts = {'train': 0, 'dev': 0, 'test': 0}

    # 贪心分配：保证每个组完整分给某一个集合
    for group in group_list:
        group_size = len(group)
        # 找到当前最缺数据的集合 (按目标比例和当前数量的差距)
        best_split = None
        best_diff = float('inf')
        for split_name in ['train', 'dev', 'test']:
            diff = targets[split_name] - current_counts[split_name]
            if diff > 0 and diff < best_diff:
                best_diff = diff
                best_split = split_name

        if best_split is None:
            best_split = 'train'  # 默认兜底

        split_data[best_split].extend(group)
        current_counts[best_split] += group_size

    return split_data


def save_data(split_data):
    """保存数据到原文件"""
    for split_name, items in split_data.items():
        file_path = os.path.join(DATA_DIR, f'{split_name}.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(f"{item['text']}\t{item['labels']}\n")
        print(f"已保存 {split_name}.txt: {len(items)} 条")


if __name__ == '__main__':
    print("开始修复数据泄漏...")
    groups = load_all_data()
    split_data = split_groups(groups)
    save_data(split_data)
    print("修复完成！请重新检查 02-rf 和 04-bert 的评估结果。")