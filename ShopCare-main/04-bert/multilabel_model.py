"""
多标签分类模型 (04-bert) —— BERT 主干 + LoRA + 多标签输出头

模块说明:
    1. load_backbone_local  —— 从本地目录加载预训练主干(绝不联网);
    2. BertMultiLabelClassifier —— 主干(pooler_output) -> Dropout -> Linear(768, 9);
    3. build_loss           —— BCEWithLogitsLoss + 每类 pos_weight(治长尾不均衡);
    4. build_model          —— 一步构建"注入 LoRA + 冻结主干 + 上设备 + 打印参数量";
    5. load_trained_model   —— 推理时加载 LoRA 权重(只读适配器, 不重复加载主干文件).

为什么用 BCEWithLogitsLoss 而不是 CrossEntropyLoss:
    CrossEntropyLoss 隐含 "9 个类别互斥、概率和为 1" 的假设, 与多标签语义冲突 ——
    一条工单可以同时是"物流问题"和"服务态度问题". 因此对 9 个标签各自做独立二分类,
    推理时逐标签 sigmoid 得到独立概率, 也**不做 softmax**.

自测:
    python 04-bert/multilabel_model.py    # 需要本地已放置 bert-base-chinese
"""

import os

import torch
import torch.nn as nn

from lora_utils import (inject_lora, mark_only_lora_trainable, print_trainable_summary,
                        get_lora_state_dict, save_lora, load_lora)


# ============================================================
# todo 1. 本地加载预训练主干
# ============================================================
def load_backbone_local(bert_dir):
    """从本地目录加载 BERT 主干; 目录不存在时给出可执行的修复提示(不联网下载)"""
    from transformers import BertModel
    config_file = os.path.join(bert_dir, 'config.json')
    if not os.path.exists(config_file):
        raise SystemExit(
            '[错误] 找不到本地预训练模型:\n'
            f'    {bert_dir}\n'
            f'    缺少 config.json\n'
            '  修复方式(二选一):\n'
            '    1) 把 bert-base-chinese 整个目录放到 04-bert/bert-base-chinese/ 下;\n'
            '    2) 设置环境变量 BERT_MODEL_DIR 指向已有的本地模型目录.'
        )
    # 传入本地目录时 transformers 不会联网下载
    return BertModel.from_pretrained(bert_dir)


# ============================================================
# todo 2. 多标签分类模型
# ============================================================
class BertMultiLabelClassifier(nn.Module):
    """BERT + LoRA + 多标签分类头

    结构:
        input_ids/attention_mask
            -> BERT 主干(冻结; query/value 上挂 LoRA 低秩旁路)
            -> pooler_output (hidden_size, 中文 BERT 为 768)
            -> Dropout(cfg.dropout)
            -> Linear(hidden_size, 9) -> logits

    说明: forward 只返回 logits(未过 sigmoid), 因为 BCEWithLogitsLoss 内部自带
    sigmoid 且数值更稳定; 推理时用 predict_proba() 取概率.
    """

    def __init__(self, cfg, backbone=None):
        super().__init__()
        self.cfg = cfg
        self.num_labels = cfg.num_labels

        # 主干: 允许外部传入(便于 06-蒸馏 阶段复用同一份主干)
        self.bert = backbone if backbone is not None else load_backbone_local(cfg.bert_dir)
        hidden_size = self.bert.config.hidden_size

        # 注入 LoRA(可选): 只改 query/value 两个投影层
        self.lora_modules = []
        if cfg.use_lora:
            self.lora_modules = inject_lora(
                self.bert,
                target_modules=cfg.lora_target_modules,
                r=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )
            print(f'  [LoRA] 已注入 {len(self.lora_modules)} 个低秩旁路 '
                  f'(r={cfg.lora_r}, alpha={cfg.lora_alpha})')

        # 多标签输出头
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # 冻结主干, 只训练 LoRA + 分类头
        if cfg.use_lora:
            mark_only_lora_trainable(self, extra_trainable=('classifier',))
            print_trainable_summary(self)
        else:
            # 对照组: 不注入 LoRA, 只冻结主干, 训练分类头
            for param in self.bert.parameters():
                param.requires_grad = False
            print('  [LoRA] 已关闭, 仅训练分类头(对照组设置)')
            print_trainable_summary(self)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """前向传播, 返回 (batch, num_labels) 的 logits"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # pooler_output = [CLS] 位置经一层 tanh 变换后的句向量
        pooled = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled))
        return logits

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask=None, token_type_ids=None):
        """推理: 逐标签 sigmoid 得到独立概率(不做 softmax, 见模块 docstring 说明)"""
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.sigmoid(logits)

    def save(self, path, extra=None):
        """保存 LoRA 增量 + 分类头(主干权重复用本地预训练模型, 不重复保存)"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        meta = {
            'num_labels': self.num_labels,
            'class_list': self.cfg.class_list,
            'bert_dir': self.cfg.bert_dir,
            'use_lora': self.cfg.use_lora,
            'lora_r': self.cfg.lora_r,
            'lora_alpha': self.cfg.lora_alpha,
            'lora_target_modules': list(self.cfg.lora_target_modules),
            'max_len': self.cfg.max_len,
            'label_threshold': self.cfg.label_threshold,
            'global_threshold': self.cfg.global_threshold,
        }
        meta.update(extra or {})
        save_lora(self, path, extra=meta)
        return path


# ============================================================
# todo 3. 损失函数
# ============================================================
def build_loss(cfg, pos_weight=None):
    """构建多标签损失: BCEWithLogitsLoss(+ 每类 pos_weight)

    参数:
        cfg       : Config 实例
        pos_weight: (num_labels,) 张量; 传 None 表示各类等权
    """
    if pos_weight is not None:
        pos_weight = pos_weight.to(cfg.device) if hasattr(pos_weight, 'to') else pos_weight
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def load_pos_weight(cfg, path=None):
    """从磁盘读取训练时保存的 pos_weight(推理阶段一般不需要, 仅评估复现时用)"""
    path = path or cfg.pos_weight_path
    if os.path.exists(path):
        return torch.load(path, map_location='cpu')
    return None


# ============================================================
# todo 4. 模型构建入口
# ============================================================
def build_model(cfg, pos_weight=None):
    """构建模型 + 损失, 并移动到指定设备

    返回: (model, criterion)
    """
    model = BertMultiLabelClassifier(cfg)
    model.to(cfg.device)
    criterion = build_loss(cfg, pos_weight)
    return model, criterion


def load_trained_model(cfg, path=None, device=None):
    """加载训练好的模型用于推理/评估

    流程: 构建同结构模型 -> 加载 LoRA + 分类头 -> 切到 eval 模式 -> 上设备
    """
    path = path or cfg.model_save_path
    if not os.path.exists(path):
        raise SystemExit(
            f'[错误] 未找到训练好的模型: {path}\n'
            f'       请先运行: python 04-bert/train_bert.py'
        )
    model = BertMultiLabelClassifier(cfg)
    meta = load_lora(model, path, strict=False)
    model.to(device or cfg.device)
    model.eval()
    return model, meta


# ============================================================
# todo 5. 自测
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config

    cfg = Config()
    problems = cfg.check_files(need_model=True)
    if problems:
        print('=' * 72)
        print('自测无法进行, 请先解决以下问题:')
        for p in problems:
            print('  - ' + p)
        print('=' * 72)
        sys.exit(0)

    print('=' * 72)
    print('多标签模型自测')
    print('=' * 72)
    model, criterion = build_model(cfg)
    print(f'\n  模型结构: {type(model).__name__}, 类别数={model.num_labels}')
    print(f'  损失函数: {criterion}')

    # 造两条假输入, 验证前向形状与概率范围
    batch_size, seq_len = 2, cfg.max_len
    fake_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(cfg.device),
        'attention_mask': torch.ones((batch_size, seq_len), dtype=torch.long).to(cfg.device),
        'token_type_ids': torch.zeros((batch_size, seq_len), dtype=torch.long).to(cfg.device),
    }
    model.eval()
    with torch.no_grad():
        logits = model(**fake_input)
        probs = model.predict_proba(**fake_input)
    print(f'\n  logits 形状: {tuple(logits.shape)}  (期望 ({batch_size}, {cfg.num_labels}))')
    print(f'  概率范围   : [{probs.min().item():.4f}, {probs.max().item():.4f}] (应落在 0~1)')
    print('\n[OK] 模型自测通过(以上为随机权重输出, 不代表真实效果)')
