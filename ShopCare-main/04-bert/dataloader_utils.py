"""
数据集与 DataLoader 工具 (04-bert)

模块说明:
    1. TicketDataset      —— 多标签工单数据集, 每行 "文本<TAB>标签1,标签2" 转成
                             (input_ids, attention_mask, multi-hot labels);
    2. compute_pos_weight —— 统计每个标签的"负样本/正样本"比值, 供 BCEWithLogitsLoss 使用,
                             缓解 invoice / payment_account 等长尾标签的欠拟合;
    3. build_dataloader / build_all_dataloader —— 构建训练/验证/测试 DataLoader,
       训练集支持传入 WeightedRandomSampler(难例动态加权采样, 创新点 1).

多标签的关键点(新手易错):
    标签必须转成 **multi-hot 向量**(长度 = 类别数, 命中位置为 1),
    而不是单标签互斥的类别索引 —— 后者会把多标签问题强行退化成单标签问题.
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# todo 1. 多标签工单数据集
# ============================================================
class TicketDataset(Dataset):
    """电商客服工单多标签数据集

    参数:
        path      : 数据文件路径(每行 "文本<TAB>标签1,标签2")
        tokenizer : HuggingFace 分词器
        class2id  : 标签名 -> 索引 的映射
        max_len   : 文本最大长度(短补长截)
    """

    def __init__(self, path, tokenizer, class2id, max_len=96):
        assert os.path.exists(path), f'数据文件不存在: {path}'
        self.path = path
        self.tokenizer = tokenizer
        self.class2id = class2id
        self.max_len = max_len
        self.num_labels = len(class2id)

        self.texts = []
        self.label_lists = []      # 原始标签名列表(评估与难例采样时需要)
        self.label_vectors = []    # multi-hot 向量
        skipped = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n').rstrip('\r')
                if not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    skipped += 1
                    continue
                text = parts[0].strip()
                labels = [x.strip() for x in parts[1].split(',') if x.strip() in class2id]
                if not text or not labels:
                    skipped += 1
                    continue
                vector = [0.0] * self.num_labels
                for lb in labels:
                    vector[self.class2id[lb]] = 1.0
                self.texts.append(text)
                self.label_lists.append(labels)
                self.label_vectors.append(vector)

        if skipped:
            print(f'  [提示] {os.path.basename(path)} 跳过 {skipped} 行无效数据(空文本/无有效标签)')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        """返回一条样本: 分词结果 + multi-hot 标签 + 原始索引(难例采样与错误分析需要)"""
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item['labels'] = torch.tensor(self.label_vectors[index], dtype=torch.float)
        item['index'] = torch.tensor(index, dtype=torch.long)
        return item

    def label_matrix(self):
        """返回全部样本的 multi-hot 标签矩阵 (N, C), 供 pos_weight 与难例权重计算"""
        return torch.tensor(self.label_vectors, dtype=torch.float)


# ============================================================
# todo 2. 类别权重(pos_weight)统计
# ============================================================
def compute_pos_weight(dataset, eps=1e-6):
    """计算 BCEWithLogitsLoss 的 pos_weight = 负样本数 / 正样本数

    参数: dataset —— TicketDataset
    返回: (num_labels,) 的张量, 值越大表示该标签越稀少、损失权重越高
    """
    labels = dataset.label_matrix()                       # (N, C)
    pos = labels.sum(dim=0)                               # 每个标签的正样本数
    neg = labels.shape[0] - pos
    weight = neg / (pos + eps)
    # 上限截断: 避免极端长尾标签把损失放大到不可控
    return torch.clamp(weight, min=1.0, max=50.0)


# ============================================================
# todo 3. DataLoader 构建
# ============================================================
def build_dataloader(dataset, batch_size=32, shuffle=False, sampler=None, num_workers=0):
    """构建 DataLoader(默认 collate 即可处理 dict 形式的样本)"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),   # 使用 sampler 时不能再开 shuffle
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
    )


def build_all_dataloader(config, tokenizer, train_sampler=None):
    """一次性构建训练/验证/测试 DataLoader

    参数:
        config       : 04-bert/config.py 的 Config 实例
        tokenizer    : 分词器
        train_sampler: 训练集采样器(难例动态加权采样), 为 None 时按顺序随机打乱
    返回:
        (train_dataset, train_loader, dev_loader, test_loader)
    """
    train_dataset = TicketDataset(config.train_path, tokenizer, config.class2id, config.max_len)
    dev_dataset = TicketDataset(config.dev_path, tokenizer, config.class2id, config.max_len)
    test_dataset = TicketDataset(config.test_path, tokenizer, config.class2id, config.max_len) \
        if os.path.exists(config.test_path) else None

    train_loader = build_dataloader(train_dataset, config.batch_size,
                                    shuffle=True, sampler=train_sampler,
                                    num_workers=config.num_workers)
    dev_loader = build_dataloader(dev_dataset, config.batch_size, shuffle=False,
                                  num_workers=config.num_workers)
    test_loader = build_dataloader(test_dataset, config.batch_size, shuffle=False,
                                   num_workers=config.num_workers) if test_dataset else None

    print(f'  数据加载完成: train={len(train_dataset)}  dev={len(dev_dataset)}  '
          f'test={len(test_dataset) if test_dataset else 0}')
    return train_dataset, train_loader, dev_loader, test_loader


def load_tokenizer(config):
    """加载分词器(强制本地目录, 缺失时给出明确提示, 绝不联网下载)"""
    from transformers import BertTokenizer
    if not os.path.exists(os.path.join(config.bert_dir, 'vocab.txt')):
        raise SystemExit(
            f'[错误] 分词器文件缺失: {os.path.join(config.bert_dir, "vocab.txt")}\n'
            f'       本项目不自动联网下载模型, 请手动放置 bert-base-chinese 到:\n'
            f'       {config.bert_dir}\n'
            f'       或设置环境变量 BERT_MODEL_DIR 指向已有的本地模型目录.'
        )
    return BertTokenizer.from_pretrained(config.bert_dir)


if __name__ == '__main__':
    # 本模块是库而不是可执行脚本: 数据集与分词器都依赖 config 与真实模型目录,
    # 端到端自测请运行: python 04-bert/train_bert.py --dry_run true (只跑一个 batch 验证链路)
    print('dataloader_utils.py 是库模块, 请通过 04-bert/train_bert.py 使用.')
    print(f'当前可用的 DataLoader 构建函数: build_dataloader / build_all_dataloader')
