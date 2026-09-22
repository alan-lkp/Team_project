# ShopCare 04-bert 代码汇总

> 本文档汇总 `04-bert/` 目录下的全部 10 个 Python 源文件，便于查阅与答辩。
>
> 模块依赖顺序：
> `config.py` → `lora_utils.py` → `multilabel_model.py` → `dataloader_utils.py` → `hard_example_sampler.py` / `reject_utils.py` / `llm_fallback.py` → `model2dev_utils.py` → `train_bert.py` → `bert_predict_fun.py`

## 目录

| # | 文件 | 作用 |
|---|---|---|
| 1 | [config.py](#1-configpy) | 全局配置：路径 / 类别 / LoRA / 训练 / 拒识阈值 |
| 2 | [lora_utils.py](#2-lora_utilspy) | LoRA 低秩适配手写实现（不依赖 peft） |
| 3 | [multilabel_model.py](#3-multilabel_modelpy) | BERT + LoRA + 多标签输出头 |
| 4 | [dataloader_utils.py](#4-dataloader_utilspy) | TicketDataset / pos_weight / DataLoader |
| 5 | [hard_example_sampler.py](#5-hard_example_samplerpy) | 多标签难例动态加权采样（创新点 1） |
| 6 | [reject_utils.py](#6-reject_utilspy) | 双阈值置信度拒识（创新点 2） |
| 7 | [llm_fallback.py](#7-llm_fallbackpy) | 拒识工单的 LLM 语义兜底 |
| 8 | [model2dev_utils.py](#8-model2dev_utilspy) | 统一评估工具：学术指标 + 业务指标 |
| 9 | [train_bert.py](#9-train_bertpy) | BERT + LoRA 训练主脚本 |
| 10 | [bert_predict_fun.py](#10-bert_predict_funpy) | 对外唯一推理入口 |

---

## 1. config.py

```python
"""
ShopCare 核心模型配置 (04-bert)

模块说明:
    集中管理 04-bert 阶段的所有路径、模型超参数、LoRA 参数与拒识阈值,
    供训练脚本(train_bert.py)、推理入口(bert_predict_fun.py)、
    以及复用它的 05/06/07 优化阶段统一调用.

设计说明(与参考项目 MindCare 的一个关键差异):
    本 Config **不在 __init__ 里加载 BERT 权重**, 只保存路径与超参数.
    原因: 读一个路径不该触发几百 MB 的模型加载; 而且模型目录还没放好时,
    import 本文件不应该直接崩. 真正加载模型统一走 load_pretrained() ——
    该函数**强制本地目录加载, 绝不联网下载**.

运行方式:
    python 04-bert/config.py     # 打印当前配置摘要
"""

import os

# 项目根目录: 本文件位于 04-bert/ 下, 向上取一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """04-bert 全局配置: 路径 / 数据 / 模型 / LoRA / 训练 / 拒识阈值"""

    def __init__(self):
        # ==================== todo 1. 基础路径 ====================
        self.root_path = PROJECT_ROOT.replace('\\', '/') + '/'
        self.model_name = 'bert-lora'

        # 数据路径(与 01-data/data_format.md 一致)
        self.train_path = os.path.join(PROJECT_ROOT, '01-data', 'train.txt')
        self.dev_path = os.path.join(PROJECT_ROOT, '01-data', 'dev.txt')
        self.test_path = os.path.join(PROJECT_ROOT, '01-data', 'test.txt')
        self.class_path = os.path.join(PROJECT_ROOT, '01-data', 'class.txt')

        # 模型与产物路径
        self.save_dir = os.path.join(PROJECT_ROOT, '04-bert', 'save_models')
        self.result_dir = os.path.join(PROJECT_ROOT, '04-bert', 'result')
        self.model_save_path = os.path.join(self.save_dir, 'bert_lora_multilabel.pt')
        self.pos_weight_path = os.path.join(self.save_dir, 'pos_weight.pt')
        self.train_log_path = os.path.join(self.result_dir, 'train_log.json')
        self.hard_example_log_path = os.path.join(self.result_dir, 'hard_example_log.json')

        # ==================== todo 2. 类别体系 ====================
        # 每行一个标签, 行号即索引(第 1 行索引 0)
        self.class_list = [line.strip() for line in open(self.class_path, encoding='utf-8') if line.strip()]
        self.num_labels = len(self.class_list)
        self.id2class = {i: c for i, c in enumerate(self.class_list)}
        self.class2id = {c: i for i, c in enumerate(self.class_list)}

        # ==================== todo 3. 预训练模型(本地, 不联网) ====================
        # 可用环境变量 BERT_MODEL_DIR 指向任意本地模型目录(例如换成 RoBERTa)
        self.bert_dir = os.environ.get(
            'BERT_MODEL_DIR',
            os.path.join(PROJECT_ROOT, '04-bert', 'bert-base-chinese')
        )
        self.max_len = 96          # 工单多为短文本, 96 足够; 由 data_eda 的截断比例决定是否需要调大

        # ==================== todo 4. LoRA 参数 ====================
        self.use_lora = True                 # 关掉即退化为"冻结主干 + 只训分类头"的对照组
        self.lora_r = 8                      # 低秩矩阵的秩 r
        self.lora_alpha = 16                 # 缩放系数 alpha, 实际缩放 = alpha / r
        self.lora_dropout = 0.1
        self.lora_target_modules = ['query', 'value']   # 注入注意力的 Q/V 投影

        # ==================== todo 5. 训练超参数 ====================
        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 2e-4            # LoRA 专用学习率(比全量微调高一个量级)
        self.weight_decay = 0.01
        self.dropout = 0.3                   # 分类头前的 dropout
        self.patience = 2                    # 早停: dev Micro-F1 连续 N 轮不升则停
        self.seed = 42
        self.num_workers = 0                 # Windows 下建议 0, 避免多进程 DataLoader 问题

        # ==================== todo 6. 多标签难例动态采样(创新点 1) ====================
        self.use_hard_sampling = True
        self.hard_w_label_cnt = 0.5          # 标签数信号权重
        self.hard_w_miss = 1.0               # 漏判率信号权重
        self.hard_w_conf = 0.8               # 置信度缺口信号权重
        self.hard_w_max = 5.0                # 单样本权重上限(防止个别难例主导训练)

        # ==================== todo 7. 双阈值拒识(创新点 2) ====================
        self.label_threshold = float(os.environ.get('LABEL_THRESHOLD', 0.5))    # 单标签激活阈值
        self.global_threshold = float(os.environ.get('GLOBAL_THRESHOLD', 0.8))  # 全局平均置信度阈值

        # ==================== todo 8. 设备 ====================
        # 延迟导入 torch: 让"只想看配置"的场景不必装 torch
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            self.device = 'cpu'

    # ------------------------------------------------------------------
    def check_files(self, need_model=True):
        """启动前的自检: 数据与模型目录是否存在, 缺失时给出明确的修复提示"""
        problems = []
        for name, path in [('训练集', self.train_path), ('验证集', self.dev_path),
                           ('类别文件', self.class_path)]:
            if not os.path.exists(path):
                problems.append(f'{name}缺失: {path}')
        if need_model and not os.path.isdir(self.bert_dir):
            problems.append(
                f'预训练模型目录缺失: {self.bert_dir}\n'
                f'    本项目不自动联网下载模型, 请手动放置 bert-base-chinese 目录, 或设置环境变量 BERT_MODEL_DIR'
            )
        elif need_model and not os.path.exists(os.path.join(self.bert_dir, 'config.json')):
            problems.append(f'模型目录里缺少 config.json: {self.bert_dir}')
        return problems

    def summary(self):
        """返回配置摘要文本(训练开始时打印, 便于复现实验)"""
        return '\n'.join([
            f'  项目根目录   : {self.root_path}',
            f'  标签体系     : {self.num_labels} 类 -> {", ".join(self.class_list)}',
            f'  预训练模型   : {self.bert_dir}',
            f'  最大长度     : {self.max_len}',
            f'  LoRA         : r={self.lora_r}, alpha={self.lora_alpha}, '
            f'dropout={self.lora_dropout}, target={self.lora_target_modules}',
            f'  训练         : epochs={self.epochs}, batch={self.batch_size}, '
            f'lr={self.learning_rate}, patience={self.patience}',
            f'  难例动态采样 : {self.use_hard_sampling} '
            f'(α={self.hard_w_label_cnt}, β={self.hard_w_miss}, γ={self.hard_w_conf})',
            f'  拒识阈值     : 单标签 {self.label_threshold} / 全局 {self.global_threshold}',
            f'  设备         : {self.device}',
        ])


if __name__ == '__main__':
    c = Config()
    print('=' * 72)
    print('ShopCare 04-bert 配置摘要')
    print('=' * 72)
    print(c.summary())
    problems = c.check_files(need_model=False)
    if problems:
        print('\n[警告] 文件自检发现问题:')
        for p in problems:
            print('  - ' + p)
    else:
        print('\n[OK] 文件自检通过')
```

---

## 2. lora_utils.py

```python
"""
LoRA 低秩适配工具 (04-bert) —— 手写实现, 不依赖 peft

模块说明:
    实现 LoRA (Low-Rank Adaptation) 的完整生命周期:
      1. LoRALinear   —— 把 nn.Linear 包成 "原权重(冻结) + 低秩旁路(可训练)";
      2. inject_lora  —— 按目标模块名把主干里的 Linear 批量替换为 LoRALinear;
      3. mark_only_lora_trainable —— 冻结主干, 只放开 LoRA 参数与分类头;
      4. count_trainable_params   —— 打印可训练参数占比(验证冻结是否生效);
      5. get_lora_state_dict / save_lora / load_lora —— 只存 LoRA 增量(几十 KB);
      6. merge_lora_weights       —— 把 delta 合并回原权重, 供 05-量化/07-剪枝阶段使用.

原理(一句话):
    冻结原始权重 W, 旁路学习一个低秩增量 ΔW = B·A, 前向变为
        h = W·x + (alpha / r) · B · A · x
    其中 A: (r, d_in) 用随机初始化, B: (d_out, r) **初始化为全零** ——
    这样训练开始时旁路输出恒为 0, 模型行为与预训练权重完全一致, 不会一开始就把模型带偏.

为什么不用 peft:
    1. 本项目要能把"低秩分解"讲清楚, 手写实现更透明;
    2. 避免 peft / transformers / torch 三者版本互相锁死(教学与部署环境往往版本不齐);
    3. 只有 100 多行, 便于在 05/06/07 阶段继续基于它做量化、蒸馏、剪枝.

自测:
    python 04-bert/lora_utils.py     # 用一个小 MLP 演示注入/冻结/合并全流程
"""

import math
import os

import torch
import torch.nn as nn


# ============================================================
# todo 1. 定义 LoRA 线性层
# ============================================================
class LoRALinear(nn.Module):
    """带 LoRA 旁路的线性层

    结构:
        输入 x -> ┌─ base_layer (原 nn.Linear, 权重冻结) ─────────────┐
                  └─ lora_dropout -> lora_a -> lora_b -> ×scaling ───┴─> 相加输出
    """

    def __init__(self, base_layer, r=8, alpha=16, dropout=0.1):
        """
        参数:
            base_layer: 被包装的原始 nn.Linear(复用其权重, 不复制)
            r         : 低秩矩阵的秩
            alpha     : 缩放系数, 实际缩放倍数为 alpha / r
            dropout   : 旁路 dropout 概率
        """
        super().__init__()
        assert isinstance(base_layer, nn.Linear), 'base_layer 必须是 nn.Linear'

        self.base_layer = base_layer
        # 冻结原始权重: 只作为"不可变的先验知识"
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r          # LoRA 论文中的缩放, 保证 r 变化时学习率尺度稳定
        self.lora_dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        # 低秩矩阵: A (r, in_features) 随机初始化, B (out_features, r) 全零初始化
        self.lora_a = nn.Parameter(torch.empty(r, base_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base_layer.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        """前向传播: 原权重输出 + 低秩增量输出"""
        base_out = self.base_layer(x)
        delta = self.lora_dropout(x) @ self.lora_a.t() @ self.lora_b.t()
        return base_out + delta * self.scaling

    @torch.no_grad()
    def merge(self):
        """把低秩增量合并进原权重(推理部署用), 合并后本层退化为普通 nn.Linear 语义"""
        delta = (self.lora_b @ self.lora_a) * self.scaling
        self.base_layer.weight.data += delta
        # 合并后旁路归零, 保持输出不变(避免重复叠加)
        self.lora_b.data.zero_()

    def extra_repr(self):
        return f'r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}'


# ============================================================
# todo 2. 注入 LoRA 到主干模型
# ============================================================
def inject_lora(model, target_modules=('query', 'value'), r=8, alpha=16, dropout=0.1):
    """把主干中名字匹配 target_modules 的 nn.Linear 替换为 LoRALinear

    参数:
        model          : HuggingFace 的 BERT/RoBERTa 模型
        target_modules : 目标子模块名(默认注意力的 query / value 投影)
        r / alpha / dropout: LoRA 超参数
    返回:
        被替换的模块名列表(用于日志打印, 便于确认注入位置是否符合预期)
    """
    replaced = []
    # 先取快照, 避免在遍历过程中修改模块树
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name in target_modules:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(f'{module_name}.{child_name}' if module_name else child_name)
    return replaced


# ============================================================
# todo 3. 冻结主干, 只训练 LoRA 参数与分类头
# ============================================================
def mark_only_lora_trainable(model, extra_trainable=('classifier',)):
    """冻结除 LoRA 参数与指定前缀(分类头)之外的所有参数

    参数:
        model            : 已注入 LoRA 的模型
        extra_trainable  : 额外需要训练的参数名前缀(默认分类头 classifier)
    返回:
        (可训练参数量, 总参数量, 可训练占比)
    """
    for name, param in model.named_parameters():
        is_lora = ('lora_a' in name) or ('lora_b' in name)
        is_extra = any(name.startswith(prefix) for prefix in extra_trainable)
        param.requires_grad = bool(is_lora or is_extra)
    return count_trainable_params(model)


def count_trainable_params(model):
    """统计可训练参数量, 返回 (可训练, 总量, 占比)"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, (trainable / total if total else 0.0)


def print_trainable_summary(model):
    """打印可训练参数占比 —— 正常情况下应低于 1%, 否则说明冻结没生效"""
    trainable, total, ratio = count_trainable_params(model)
    print(f'  可训练参数: {trainable:,} / {total:,}  ({ratio * 100:.4f}%)')
    if ratio > 0.05:
        print('  [警告] 可训练参数占比过高, 请检查 mark_only_lora_trainable 是否生效')
    return ratio


# ============================================================
# todo 4. 只保存 / 加载 LoRA 增量(不重复保存主干权重)
# ============================================================
def get_lora_state_dict(model):
    """提取 LoRA 与分类头的权重(主干权重不入库, 复用本地预训练模型)"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if ('lora_a' in k) or ('lora_b' in k) or k.startswith('classifier')}


def save_lora(model, path, extra=None):
    """保存 LoRA 适配器 + 分类头(默认几十 KB ~ 几 MB)"""
    payload = {'lora_state_dict': get_lora_state_dict(model), 'extra': extra or {}}
    torch.save(payload, path)
    return path


def load_lora(model, path, strict=True):
    """把 LoRA 增量加载回模型(要求模型已经注入过 LoRA, 且结构一致)"""
    payload = torch.load(path, map_location='cpu')
    state = payload.get('lora_state_dict', payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        print(f'  [提示] 加载 LoRA 时存在未匹配项: missing={len(missing)}, unexpected={len(unexpected)}')
    return payload.get('extra', {})


# ============================================================
# todo 5. 合并 LoRA(部署时可用, 05-量化/07-剪枝会调用)
# ============================================================
def merge_lora_weights(model):
    """把所有 LoRALinear 的增量合并进原权重, 返回合并的层数

    合并后模型结构与原预训练模型完全一致(不再有旁路), 便于:
      - 导出通用权重(不依赖本项目代码即可推理);
      - 做动态量化(量化对象变回标准 nn.Linear, 兼容性最好).
    """
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            merged += 1
    return merged


# ============================================================
# todo 6. 自测: 用一个小 MLP 走通全流程(不需要 transformers)
# ============================================================
if __name__ == '__main__':
    torch.manual_seed(0)
    print('=' * 72)
    print('LoRA 工具自测')
    print('=' * 72)

    # 1. 造一个带 query/value 子模块的小模型, 模拟注意力结构
    class FakeAttention(nn.Module):
        def __init__(self, dim=16):
            super().__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)

        def forward(self, x):
            return self.query(x) + self.key(x) + self.value(x)

    class FakeBackbone(nn.Module):
        def __init__(self, dim=16):
            super().__init__()
            self.attention = FakeAttention(dim)
            self.classifier = nn.Linear(dim, 9)

        def forward(self, x):
            return self.classifier(self.attention(x))

    model = FakeBackbone()
    x = torch.randn(4, 16)

    # 2. 注入前: 记录输出, 注入后应保持一致(B 全零初始化 → 旁路输出为 0)
    before = model(x).detach().clone()
    replaced = inject_lora(model, target_modules=('query', 'value'), r=8, alpha=16, dropout=0.1)
    print(f'\n  [1] 注入 LoRA 的模块({len(replaced)} 个):')
    for name in replaced:
        print(f'      {name}')

    after = model(x).detach()
    diff = (before - after).abs().max().item()
    print(f'\n  [2] 注入前后输出最大差异: {diff:.2e} (应为 0, 证明 B 零初始化生效)')
    assert diff < 1e-6, '注入 LoRA 后输出发生变化, 请检查 B 的初始化'

    # 3. 冻结与参数量统计
    trainable, total, ratio = mark_only_lora_trainable(model)
    print(f'\n  [3] 可训练参数: {trainable} / {total} ({ratio * 100:.2f}%)')

    # 4. 只训练 LoRA 也能改变输出
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    after_step = model(x).detach()
    print(f'  [4] 单步训练后输出变化量: {(before - after_step).abs().max().item():.4f} (应 > 0)')

    # 5. 保存 / 加载 / 合并
    save_lora(model, 'lora_selftest.pt')
    size_kb = os.path.getsize('lora_selftest.pt') / 1024
    print(f'  [5] LoRA 权重文件大小: {size_kb:.1f} KB (主干权重未重复保存)')
    merged = merge_lora_weights(model)
    print(f'  [6] 合并 LoRA 层数: {merged}')
    os.remove('lora_selftest.pt')
    print('\n[OK] LoRA 工具自测通过')
```

---

## 3. multilabel_model.py

```python
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
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long).to(cfg.device),
        'token_type_ids': torch.zeros(batch_size, seq_len, dtype=torch.long).to(cfg.device),
    }
    model.eval()
    with torch.no_grad():
        logits = model(**fake_input)
        probs = model.predict_proba(**fake_input)
    print(f'\n  logits 形状: {tuple(logits.shape)}  (期望 ({batch_size}, {cfg.num_labels}))')
    print(f'  概率范围   : [{probs.min().item():.4f}, {probs.max().item():.4f}] (应落在 0~1)')
    print('\n[OK] 模型自测通过(以上为随机权重输出, 不代表真实效果)')
```

---

## 4. dataloader_utils.py

```python
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
```

---

## 5. hard_example_sampler.py

```python
"""
多标签难例动态加权采样 (04-bert) —— 创新点 1

要解决的问题:
    均匀采样下, 模型把大量算力花在"一句话只讲物流"的简单样本上;
    而真正影响业务指标的是**多诉求混杂、标签漏判**的难例工单.
    本模块在训练过程中动态识别难例并提高其采样权重, 让模型重点学习这些硬骨头.

难例的三个信号(全部来自"上一轮模型自己的预测"):
    1. 标签数信号 : 标签越多越可能是混合诉求难例          -> 权重 alpha(默认 0.5)
    2. 漏判率信号 : 真实标签中概率低于激活阈值的比例      -> 权重 beta (默认 1.0)
    3. 置信度缺口 : 1 - 真实标签上的平均概率              -> 权重 gamma(默认 0.8)

    样本权重 w = 1 + alpha*(标签数-1)/2 + beta*漏判率 + gamma*置信度缺口
    再做 [1, w_max] 截断(默认上限 5.0), 防止个别样本被反复采样导致过拟合.

迭代流程(每个 epoch 结束后执行一次):
    用当前模型对**训练集**重新打分 -> 更新权重 -> 重建 WeightedRandomSampler -> 下一轮训练
    注意: 只在训练集上打分, dev/test 绝不参与权重计算(避免信息泄露).

日志:
    每轮的权重分布与 Top 难例索引写入 04-bert/result/hard_example_log.json,
    这是消融实验"难例采样是否有效"的直接证据.
"""

import json
import os

import torch
from torch.utils.data import WeightedRandomSampler


# ============================================================
# todo 1. 难例采样器
# ============================================================
class HardExampleSampler:
    """多标签难例动态加权采样器

    用法(训练脚本中):
        sampler = HardExampleSampler(cfg)
        for epoch in range(epochs):
            if cfg.use_hard_sampling and epoch > 0:
                train_loader.sampler = sampler.refresh(model, train_dataset, cfg)
            ... 训练 ...
            sampler.log_epoch(epoch, weights)
    """

    def __init__(self, cfg, log_path=None):
        self.cfg = cfg
        self.log_path = log_path or cfg.hard_example_log_path
        self.history = []          # 每轮的统计信息
        self.weights = None        # 当前样本权重 (N,)

    # ---------------- 核心: 计算难例权重 ----------------
    def compute_weights(self, probs, y_true):
        """根据模型预测概率计算每个训练样本的采样权重

        参数:
            probs : (N, C) 模型输出的标签概率
            y_true: (N, C) multi-hot 真值
        返回:
            (N,) 的权重张量
        """
        cfg = self.cfg
        probs = probs.detach().float()
        y_true = y_true.detach().float()

        # 1. 标签数信号: 单标签 0 分, 双标签 0.5 分, 三标签 1 分(线性映射)
        num_labels = y_true.sum(dim=1)
        label_signal = torch.clamp(num_labels - 1, min=0) / 2.0

        # 2. 漏判率: 真实标签里 "模型概率低于激活阈值" 的比例 = 模型漏掉的比例
        below_threshold = ((probs < cfg.label_threshold).float() * y_true).sum(dim=1)
        miss_rate = below_threshold / torch.clamp(num_labels, min=1)

        # 3. 置信度缺口: 真实标签上的平均概率越低, 说明模型越犹豫
        mean_true_prob = (probs * y_true).sum(dim=1) / torch.clamp(num_labels, min=1)
        conf_gap = 1.0 - mean_true_prob

        weights = (1.0
                   + cfg.hard_w_label_cnt * label_signal
                   + cfg.hard_w_miss * miss_rate
                   + cfg.hard_w_conf * conf_gap)
        # 截断: 下限 1.0 保证简单样本也不会被完全忽略, 上限防止个别难例主导训练
        weights = torch.clamp(weights, min=1.0, max=cfg.hard_w_max)
        return weights

    # ---------------- 构建采样器 ----------------
    def build_sampler(self, weights):
        """把权重包装成 WeightedRandomSampler(有放回采样)"""
        weights = weights.detach().cpu().double().clamp(min=1e-6)
        self.weights = weights
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    @torch.no_grad()
    def refresh(self, model, train_dataset, cfg, batch_size=None):
        """用当前模型对训练集重新打分, 返回新的采样器

        参数:
            model         : 当前轮次的模型
            train_dataset : TicketDataset(需要能取到全部 multi-hot 标签)
            cfg           : Config
        返回: sampler, weights
        """
        from dataloader_utils import build_dataloader

        batch_size = batch_size or cfg.batch_size
        loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=cfg.num_workers)
        model.eval()
        probs_list, idx_list = [], []
        for batch in loader:
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(cfg.device)
            probs_list.append(model.predict_proba(input_ids, attention_mask, token_type_ids).cpu())
            idx_list.append(batch['index'])

        probs = torch.cat(probs_list)
        indices = torch.cat(idx_list)
        y_true = train_dataset.label_matrix()[indices]          # 按采样顺序对齐真值
        weights = self.compute_weights(probs, y_true)
        return self.build_sampler(weights), weights

    # ---------------- 统计与日志 ----------------
    def stats(self, weights, top_ratio=0.1):
        """统计权重分布, 便于观察"难例是否真的被加权" """
        weights = weights.detach().cpu()
        k = max(int(len(weights) * top_ratio), 1)
        top_values, top_indices = torch.topk(weights, k)
        return {
            'mean': round(float(weights.mean()), 4),
            'max': round(float(weights.max()), 4),
            'min': round(float(weights.min()), 4),
            'hard_sample_ratio': round(float((weights > 1.5).float().mean()), 4),  # 被明显加权的样本占比
            'top_hard_indices': top_indices.tolist()[:50],                        # 保存前 50 个难例下标
            'top_hard_mean_weight': round(float(top_values.mean()), 4),
        }

    def log_epoch(self, epoch, weights):
        """把本轮权重统计写入日志文件(同时保留在内存 history 中)"""
        info = {'epoch': epoch}
        info.update(self.stats(weights))
        self.history.append(info)

        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'hard_w_label_cnt': self.cfg.hard_w_label_cnt,
                    'hard_w_miss': self.cfg.hard_w_miss,
                    'hard_w_conf': self.cfg.hard_w_conf,
                    'hard_w_max': self.cfg.hard_w_max,
                    'label_threshold': self.cfg.label_threshold,
                },
                'history': self.history,
            }, f, ensure_ascii=False, indent=2)
        return info


# ============================================================
# todo 2. 自测(不依赖真实模型: 用构造的概率与真值验证权重公式)
# ============================================================
if __name__ == '__main__':
    import os as _os
    import sys
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from config import Config

    cfg = Config()
    cfg.use_hard_sampling = True
    sampler = HardExampleSampler(cfg, log_path='_selftest_hard_log.json')

    # 构造 4 条样本: 简单单标签 / 难例多标签 / 全漏判 / 一般
    y_true = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # 简单: 单标签且预测正确
        [1, 0, 1, 0, 0, 0, 0, 0, 0],   # 难例: 双标签
        [0, 1, 0, 1, 0, 0, 0, 0, 1],   # 难例: 三标签且全漏判
        [0, 0, 0, 0, 0, 0, 1, 0, 0],   # 一般: 单标签, 概率偏低
    ], dtype=torch.float)
    probs = torch.tensor([
        [0.95, 0.01, 0.02, 0.01, 0.01, 0.01, 0.05, 0.02, 0.01],
        [0.88, 0.02, 0.45, 0.01, 0.01, 0.01, 0.10, 0.03, 0.01],
        [0.05, 0.20, 0.10, 0.30, 0.05, 0.02, 0.15, 0.08, 0.25],
        [0.02, 0.01, 0.03, 0.02, 0.02, 0.01, 0.55, 0.03, 0.02],
    ])

    weights = sampler.compute_weights(probs, y_true)
    print('=' * 72)
    print('难例动态加权采样自测')
    print('=' * 72)
    names = ['简单单标签(预测正确)', '双标签(一个漏判)', '三标签(全部漏判)', '单标签(概率偏低)']
    for i, name in enumerate(names):
        print(f'  {name:<24} 权重 = {weights[i]:.3f}')
    print('\n  预期: 三标签全漏判 > 双标签 > 单标签偏低 > 简单样本(=1.0)')
    assert weights[2] > weights[1] > weights[0]
    assert weights[3] > weights[0]
    assert abs(weights[0].item() - 1.0) < 1e-6

    info = sampler.log_epoch(0, weights)
    print(f'\n  权重统计: {info}')
    _os.remove('_selftest_hard_log.json')
    print('\n[OK] 难例采样模块自测通过')
```

---

## 6. reject_utils.py

```python
"""
双阈值置信度拒识 (04-bert) —— 创新点 2 的核心判定逻辑

模块说明:
    纯函数实现(不依赖 torch / numpy), 因此:
      * 训练评估(model2dev_utils)、在线服务(backend)、LLM 兜底(llm_fallback)可以
        复用**同一份判定逻辑** —— 线上线下的拒识行为不会出现"两套标准";
      * 方便单独单测。

判定规则(docs 中的双阈值机制):
    1. 单标签激活阈值 label_threshold(默认 0.5):
       概率 ≥ 0.5 的标签视为"激活标签", 保留多标签组合输出;
    2. 全局置信度阈值 global_threshold(默认 0.8):
       激活标签的**平均置信度** ≥ 0.8 → 自动输出并分流;
       平均置信度 < 0.8 → 判定为"复杂模糊工单", 拒识;
    3. 一个标签都没激活 → 直接拒识(工单无有效诉求)。

拒识后怎么办(由调用方决定):
    打开 LLM 兜底 → 交给 llm_fallback 解析; 关闭或解析失败 → 进入人工复核队列.

为什么需要全局阈值:
    单阈值只能回答"这个标签算不算命中", 无法表达"整条工单的判断是否可靠".
    真实场景常见 "三个标签概率都在 0.5~0.6" 的糊状输出 —— 标签看着都有点像,
    但整体不可信, 此时必须拒识, 否则就是线上误判的来源.

自测:
    python 04-bert/reject_utils.py
"""

# ============================================================
# todo 1. 拒识原因常量(前后端共用, 避免各处硬编码字符串)
# ============================================================
REJECT_NONE = None
REJECT_NO_LABEL = 'no_label_activated'   # 没有任何标签达到激活阈值
REJECT_LOW_CONF = 'low_confidence'       # 有激活标签, 但平均置信度不足

REJECT_REASON_CN = {
    REJECT_NO_LABEL: '未识别到有效诉求(所有标签概率均低于激活阈值)',
    REJECT_LOW_CONF: '诉求模糊(激活标签的平均置信度未达到全局阈值)',
}


def _to_list(row):
    """把 torch.Tensor / numpy.ndarray / list 统一转成 Python list"""
    if hasattr(row, 'tolist'):
        return row.tolist()
    return list(row)


# ============================================================
# todo 2. 单条工单的拒识判定
# ============================================================
def decide(probs_row, class_list, label_threshold=0.5, global_threshold=0.8, max_labels=None):
    """对一条工单的概率向量做双阈值判定

    参数:
        probs_row        : 长度 = 类别数 的概率序列(torch.Tensor / numpy / list 均可)
        class_list       : 标签名列表(顺序与概率一致)
        label_threshold  : 单标签激活阈值(默认 0.5)
        global_threshold : 全局平均置信度阈值(默认 0.8)
        max_labels       : 最多保留几个激活标签(默认 None 表示不限制)
    返回:
        dict(labels, confidences, avg_confidence, rejected, reject_reason,
             reject_reason_cn, need_human_review, all_scores)
    """
    scores = [float(x) for x in _to_list(probs_row)]
    pairs = sorted(zip(class_list, scores), key=lambda t: t[1], reverse=True)

    # 1. 按单标签阈值筛出激活标签
    activated = [(name, score) for name, score in pairs if score >= label_threshold]
    if max_labels:
        activated = activated[:max_labels]

    # 2. 一个都没激活 -> 直接拒识
    if not activated:
        return {
            'labels': [],
            'confidences': {},
            'avg_confidence': 0.0,
            'rejected': True,
            'reject_reason': REJECT_NO_LABEL,
            'reject_reason_cn': REJECT_REASON_CN[REJECT_NO_LABEL],
            'need_human_review': True,
            'all_scores': {name: round(score, 4) for name, score in pairs},
        }

    # 3. 计算激活标签的平均置信度, 与全局阈值比较
    avg_confidence = sum(score for _, score in activated) / len(activated)
    rejected = avg_confidence < global_threshold

    return {
        'labels': [name for name, _ in activated],
        'confidences': {name: round(score, 4) for name, score in activated},
        'avg_confidence': round(avg_confidence, 4),
        'rejected': bool(rejected),
        'reject_reason': REJECT_LOW_CONF if rejected else REJECT_NONE,
        'reject_reason_cn': REJECT_REASON_CN[REJECT_LOW_CONF] if rejected else None,
        'need_human_review': bool(rejected),
        'all_scores': {name: round(score, 4) for name, score in pairs},
    }


def batch_decide(probs, cfg, class_list=None, max_labels=None):
    """批量判定(评估时用): probs 为 (N, C) 的概率矩阵"""
    class_list = class_list or cfg.class_list
    probs = probs.tolist() if hasattr(probs, 'tolist') else probs
    return [decide(row, class_list,
                   label_threshold=cfg.label_threshold,
                   global_threshold=cfg.global_threshold,
                   max_labels=max_labels) for row in probs]


# ============================================================
# todo 3. 是否需要走 LLM 兜底
# ============================================================
def should_use_llm(decision, llm_enabled=True):
    """只有"拒识"且"兜底开关打开"时才升级到 LLM —— 这是控制成本的关键"""
    return bool(llm_enabled and decision.get('rejected'))


# ============================================================
# todo 4. 阈值标定辅助(在 dev 集上网格搜索最优阈值)
# ============================================================
def grid_search_thresholds(probs, y_true, class_list,
                           label_candidates=(0.3, 0.4, 0.5, 0.6, 0.7),
                           global_candidates=(0.6, 0.7, 0.75, 0.8, 0.85, 0.9),
                           target_reject_rate=0.15):
    """在验证集上网格搜索双阈值

    目标: 在"拒识率不超过 target_reject_rate"的前提下, 让自动分流的 Micro-F1 尽量高.
    返回: 候选结果列表(按 micro_f1 降序), 每项含阈值组合与对应指标.
    """
    from sklearn.metrics import f1_score
    import numpy as _np

    yt = _np.asarray(y_true.tolist() if hasattr(y_true, 'tolist') else y_true).astype(int)
    results = []
    for label_thr in label_candidates:
        for global_thr in global_candidates:
            decisions = [decide(row, class_list, label_thr, global_thr)
                         for row in (probs.tolist() if hasattr(probs, 'tolist') else probs)]
            keep = [i for i, d in enumerate(decisions) if not d['rejected']]
            reject_rate = 1 - len(keep) / max(len(decisions), 1)
            if not keep:
                continue
            yp = _np.zeros_like(yt)
            for i in keep:
                for name in decisions[i]['labels']:
                    yp[i, class_list.index(name)] = 1
            micro = f1_score(yt[keep], yp[keep], average='micro', zero_division=0)
            results.append({
                'label_threshold': label_thr,
                'global_threshold': global_thr,
                'reject_rate': round(reject_rate, 4),
                'auto_micro_f1': round(float(micro), 4),
                'auto_samples': len(keep),
                'feasible': reject_rate <= target_reject_rate,
            })
    results.sort(key=lambda r: (r['feasible'], r['auto_micro_f1']), reverse=True)
    return results


# ============================================================
# todo 5. 自测
# ============================================================
if __name__ == '__main__':
    class_list = ['logistics', 'quality', 'after_sale', 'invoice', 'price_promo',
                  'payment_account', 'consult', 'service', 'invalid']

    cases = [
        ('清晰工单(物流+服务)', [0.96, 0.10, 0.05, 0.02, 0.03, 0.01, 0.20, 0.88, 0.01]),
        ('模糊工单(全都五六成)', [0.55, 0.58, 0.52, 0.51, 0.54, 0.50, 0.56, 0.53, 0.49]),
        ('无有效诉求',          [0.10, 0.12, 0.08, 0.05, 0.06, 0.04, 0.11, 0.09, 0.07]),
        ('单标签高置信',        [0.05, 0.03, 0.04, 0.91, 0.02, 0.01, 0.06, 0.05, 0.01]),
    ]

    print('=' * 72)
    print('双阈值拒识自测 (激活阈值 0.5 / 全局阈值 0.8)')
    print('=' * 72)
    for name, probs in cases:
        d = decide(probs, class_list)
        tag = '拒识' if d['rejected'] else '自动分流'
        print(f'\n  [{name}] -> {tag}')
        print(f'    激活标签   : {d["labels"] or "无"}')
        print(f'    平均置信度 : {d["avg_confidence"]}')
        if d['rejected']:
            print(f'    拒识原因   : {d["reject_reason_cn"]}')
            print(f'    是否走 LLM : {should_use_llm(d, llm_enabled=True)}')
            print(f'    是否转人工 : {d["need_human_review"]}')

    print('\n[OK] 拒识模块自测通过')
```

---

## 7. llm_fallback.py

```python
"""
LLM 兜底解析 (04-bert) —— 拒识工单的语义兜底

定位:
    双阈值拒识把"本地模型没把握"的工单挑出来后, 交给大模型做语义兜底解析.
    这是三级兜底机制(must: 本地模型 -> LLM -> 人工复核)的第二级.

成本控制设计(为什么不能全量走 LLM):
    1. 只有 rejected=True 的工单才会触发(见 reject_utils.should_use_llm);
    2. 未配置 DEEPSEEK_API_KEY 时**自动禁用**, 服务照样能跑(走人工复核);
    3. 提示词严格限定 9 类 + 强制 JSON 输出, 减少无效 token 与解析失败重试;
    4. 相同文本在服务层会被 Redis 缓存(见 backend/middleware), 避免重复计费.

可靠性设计:
    - 返回结果必须通过"标签合法性校验"(只能是我们体系内的 9 个标签);
    - 解析失败/超时/标签非法 -> 返回 None, 由上层转入人工复核, 绝不返回脏数据;
    - 目前不引入 json_repair 等额外依赖, 用手写提取(去掉 ```json 包裹、取首尾大括号).

自测(不需要 API Key):
    python 04-bert/llm_fallback.py
"""

import json
import os
import re

# ============================================================
# todo 0. 配置(全部可用环境变量覆盖)
# ============================================================
DEFAULT_MODEL = os.environ.get('LLM_MODEL', 'deepseek-chat')
DEFAULT_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.deepseek.com')

# 标签中文名(用于提示词里做类别说明; 与 backend/common/shop_utils.py 保持一致)
LABEL_CN = {
    'logistics': '物流配送(未发货、运输停滞、派送延迟、丢件、地址修改)',
    'quality': '商品质量(破损、瑕疵、功能故障、材质不符、过期变质、少件漏发)',
    'after_sale': '退款退货(退款、退货、换货、退款未到账、维修换新、售后维权)',
    'invoice': '发票问题(开票申请、发票信息错误、重开、发票未收到)',
    'price_promo': '价格与优惠(价保补差、优惠券失效、活动规则、多扣款、赠品未发)',
    'payment_account': '支付与账号(支付失败、重复扣款、订单异常、账号封禁或登录异常)',
    'consult': '售前与使用咨询(使用方法、参数规格、适配性、库存时效、安装教程)',
    'service': '服务态度(回复慢、态度差、敷衍、承诺未兑现、推诿)',
    'invalid': '无效与恶意(灌水、广告、辱骂、与商品无关、恶意差评威胁)',
}


def build_system_prompt(class_list):
    """构造系统提示词: 角色 + 类别定义 + 判定规则 + 少样本示例 + 强约束输出格式"""
    options = '\n'.join(f'- {name}: {LABEL_CN.get(name, "")}' for name in class_list)
    return f"""你是电商客服工单的多标签分类助手。请阅读用户反馈, 从下列固定类别中选出所有命中的标签。

# 可选类别(严格限定, 不得新增)
{options}

# 判定规则
1. 一条工单可以同时命中多个类别(例如"货坏了要退款, 客服还不理人" = quality + after_sale + service), 请全部选出;
2. 只有文本里**确实提到**某个诉求时才给对应标签, 不要脑补;
3. 纯灌水、广告、辱骂且无任何具体诉求的工单, 只输出 invalid;
4. 若同时命中 invalid 与其它诉求, 以具体诉求为准, 不要输出 invalid;
5. 发票、价保、优惠券、支付失败、账号问题都是独立类别, 不要都归到 after_sale。

# 少样本示例
输入: 快递到广州十天了还没动静, 客服也不回复
输出: {{"labels": ["logistics", "service"], "confidence": 0.9, "reason": "物流停滞叠加客服未响应"}}

输入: 收到的电饭锅是坏的, 我要退款, 发票也一直没开
输出: {{"labels": ["quality", "after_sale", "invoice"], "confidence": 0.92, "reason": "质量问题同时要求退款并催开发票"}}

输入: 哈哈哈哈哈哈
输出: {{"labels": ["invalid"], "confidence": 0.95, "reason": "无意义内容"}}

# 输出格式(必须是合法 JSON, 不要输出任何解释文字或代码块标记)
{{"labels": ["标签1", "标签2"], "confidence": 0.0~1.0 之间的数字, "reason": "一句话理由"}}"""


# ============================================================
# todo 1. LLM 兜底客户端
# ============================================================
class LLMFallback:
    """拒识工单的 LLM 兜底解析器

    参数:
        class_list : 合法标签列表(用于提示词与结果校验)
        api_key    : 不传则读环境变量 DEEPSEEK_API_KEY; 为空时自动禁用
        base_url   : OpenAI 兼容接口地址(DeepSeek 官方为 https://api.deepseek.com)
        model      : 模型名
        timeout    : 单次请求超时(秒)
    说明:
        未安装 openai SDK 或未配置 Key 时, available 属性为 False, parse() 直接返回 None,
        调用方据此走人工复核 —— 保证"没有 LLM 也能完整演示".
    """

    def __init__(self, class_list, api_key=None, base_url=None, model=None, timeout=20):
        self.class_list = list(class_list)
        # 注意: api_key=None 表示"未指定, 去读环境变量"; api_key='' 表示"显式禁用"(仅测试用)
        if api_key is None:
            self.api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
        else:
            self.api_key = api_key.strip()
        self.base_url = base_url or DEFAULT_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.system_prompt = build_system_prompt(self.class_list)

        switch = os.environ.get('LLM_FALLBACK_ENABLED', 'true').lower()
        self.enabled_by_env = switch not in ('0', 'false', 'no', 'off')
        self._client = None
        self.disable_reason = None

        if not self.api_key:
            self.disable_reason = '未配置 DEEPSEEK_API_KEY(环境变量), LLM 兜底已自动禁用'
        elif not self.enabled_by_env:
            self.disable_reason = '环境变量 LLM_FALLBACK_ENABLED=false, LLM 兜底已禁用'

    # ---------------- 可用性 ----------------
    @property
    def available(self):
        return bool(self.api_key) and self.enabled_by_env

    def _get_client(self):
        """延迟创建客户端(避免没装 openai 时 import 就报错)"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    '未安装 openai SDK, 无法使用 LLM 兜底; 请执行 pip install openai, '
                    '或关闭兜底开关直接走人工复核'
                ) from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    # ---------------- 结果解析 ----------------
    @staticmethod
    def extract_json(text):
        """从模型回复里提取 JSON 对象(兼容 ```json 包裹与前后多余文字)"""
        if not text:
            return None
        cleaned = text.strip()
        cleaned = re.sub(r'^```(?:json)?', '', cleaned).strip()
        cleaned = re.sub(r'```$', '', cleaned).strip()
        start, end = cleaned.find('{'), cleaned.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None

    def validate(self, payload):
        """校验并规范化 LLM 返回: 标签必须在合法集合内, 去重, 按 class_list 顺序排序"""
        if not isinstance(payload, dict):
            return None
        raw_labels = payload.get('labels')
        if isinstance(raw_labels, str):
            raw_labels = re.split(r'[,，、\s]+', raw_labels)
        if not isinstance(raw_labels, list):
            return None
        labels = []
        for item in raw_labels:
            if not isinstance(item, str):
                continue
            name = item.strip().lower()
            if name in self.class_list and name not in labels:
                labels.append(name)
        if not labels:
            return None
        labels = sorted(labels, key=self.class_list.index)
        try:
            confidence = float(payload.get('confidence', 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            'labels': labels,
            'confidence': max(0.0, min(1.0, confidence)),
            'reason': str(payload.get('reason', ''))[:200],
        }

    # ---------------- 主入口 ----------------
    def parse(self, text, temperature=0.0, max_tokens=200):
        """解析一条工单; 任何异常都返回 None(由上层转人工复核), 不抛给业务层

        返回: {'labels': [...], 'confidence': float, 'reason': str, 'source': 'llm'}
              或 None(未启用 / 请求失败 / 结果非法)
        """
        if not self.available or not text or not text.strip():
            return None
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': text.strip()[:1000]},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            content = response.choices[0].message.content
        except Exception as exc:                     # 网络/超时/鉴权失败统一兜住
            print(f'  [LLM 兜底] 调用失败, 转人工复核: {type(exc).__name__}: {exc}')
            return None

        payload = self.extract_json(content)
        result = self.validate(payload)
        if result is None:
            print(f'  [LLM 兜底] 返回内容无法解析为合法标签, 转人工复核: {str(content)[:120]}')
            return None
        result['source'] = 'llm'
        result['model'] = self.model
        return result


# ============================================================
# todo 2. 模块级便捷入口(供 backend 直接调用, 单例复用)
# ============================================================
_default_parser = None


def get_default_parser(class_list):
    """获取默认解析器(进程内单例, 避免每次请求都重建提示词与客户端)"""
    global _default_parser
    if _default_parser is None or _default_parser.class_list != list(class_list):
        _default_parser = LLMFallback(class_list)
    return _default_parser


def llm_parse(text, class_list):
    """便捷函数: 用默认解析器解析一条文本"""
    return get_default_parser(class_list).parse(text)


# ============================================================
# todo 3. 自测(离线可跑: 验证提示词与 JSON 解析; 有 Key 时会真的调用一次)
# ============================================================
if __name__ == '__main__':
    class_list = ['logistics', 'quality', 'after_sale', 'invoice', 'price_promo',
                  'payment_account', 'consult', 'service', 'invalid']

    print('=' * 72)
    print('LLM 兜底模块自测')
    print('=' * 72)

    prompt = build_system_prompt(class_list)
    print(f'\n[1] 系统提示词长度: {len(prompt)} 字符, 覆盖类别 {len(class_list)} 个')
    assert 'invalid' in prompt and 'JSON' in prompt

    print('\n[2] JSON 提取与校验测试:')
    cases = [
        ('```json\n{"labels": ["logistics", "service"], "confidence": 0.9, "reason": "物流+客服"}\n```',
         ['logistics', 'service']),
        ('模型啰嗦的解释... {"labels": ["INVOICE"], "confidence": 1.5, "reason": "发票"} 结束',
         ['invoice']),
        ('{"labels": "物流, 客服", "confidence": 0.8}', None),      # 中文标签非法 -> 校验应拦截
        ('{"labels": ["不存在的标签"], "confidence": 0.8}', None),   # 非法标签 -> 拦截
        ('完全没有 JSON 的回复', None),
    ]
    parser = LLMFallback(class_list, api_key='')     # 显式禁用, 只测离线逻辑
    for raw, expect in cases:
        result = parser.validate(parser.extract_json(raw))
        labels = result['labels'] if result else None
        status = 'OK' if (labels == expect if expect is not None else result is None) else 'FAIL'
        print(f'    [{status}] 提取结果={labels}  (期望={expect})')

    print(f'\n[3] 兜底可用性: available={parser.available}')
    print(f'    禁用原因: {parser.disable_reason}')

    real = LLMFallback(class_list)
    if real.available:
        print('\n[4] 检测到 DEEPSEEK_API_KEY, 尝试真实调用一次...')
        out = real.parse('快递到广州十天了还没动静，客服也不回复')
        print(f'    真实返回: {out}')
    else:
        print('\n[4] 未配置 DEEPSEEK_API_KEY, 跳过真实调用(服务会自动走人工复核路径)')

    print('\n[OK] LLM 兜底模块自测通过')
```

---

## 8. model2dev_utils.py

```python
"""
统一评估工具 (04-bert) —— 学术指标 + 业务指标

模块说明:
    把 "跑一遍数据集 -> 算指标 -> 打印/存盘" 收敛成一个函数, 保证:
      * 训练时的早停、测试集的最终汇报、页面上展示的效果, 用的是**同一套口径**;
      * 02-rf / 03-fasttext / 04-bert 三个对照模型共用本模块(它们的 predict 函数
        统一返回 (N, C) 概率矩阵, 直接调用这里的 compute_metrics 即可横向对比).

学术指标:
    Micro-F1(多标签主指标) / Macro-F1(考察长尾标签) / Subset Accuracy(整单全对率) /
    Hamming Loss(错判标签位占比) / 逐标签 P-R-F1 / 标签匹配率(Jaccard 均值)
业务指标:
    自动分流率 / 拒识率 / 平均置信度 / 平均激活标签数 / P95 耗时(在服务层统计)
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, hamming_loss


def _to_np(x):
    """兼容 torch.Tensor / numpy.ndarray / list"""
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


# ============================================================
# todo 1. 跑一遍数据集, 拿到概率矩阵
# ============================================================
@torch.no_grad()
def run_inference(model, dataloader, cfg):
    """对数据集做一次完整推理

    参数: model / dataloader / cfg(取 device)
    返回: (probs, y_true, indices)
          probs   : (N, C) 每个标签的独立概率
          y_true  : (N, C) multi-hot 真值
          indices : (N,)   样本在原数据集中的下标(用于反查文本, 供难例采样与 LLM 兜底使用)
    """
    model.eval()
    probs_list, true_list, idx_list = [], [], []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(cfg.device)
        attention_mask = batch['attention_mask'].to(cfg.device)
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(cfg.device)

        probs = model.predict_proba(input_ids, attention_mask, token_type_ids)
        probs_list.append(probs.detach().cpu())
        true_list.append(batch['labels'].cpu())
        idx_list.append(batch['index'].cpu())

    if not probs_list:
        return torch.zeros((0, cfg.num_labels)), torch.zeros((0, cfg.num_labels)), torch.zeros((0,), dtype=torch.long)
    return torch.cat(probs_list), torch.cat(true_list), torch.cat(idx_list)


# ============================================================
# todo 2. 学术指标
# ============================================================
def compute_metrics(y_true, y_pred, class_list, threshold=0.5):
    """根据概率矩阵/0-1 矩阵计算多标签学术指标

    参数:
        y_true / y_pred : (N, C) 的 0-1 矩阵(或概率, 概率会按 threshold 二值化)
        class_list      : 标签名列表(顺序与列一致)
        threshold       : 二值化阈值
    返回: 指标字典
    """
    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)
    # 统一成 0-1 整数矩阵: 真值按 0.5 二值化, 预测若为概率则按阈值二值化
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred >= threshold).astype(int) if y_pred.max() <= 1.0 else y_pred.astype(int)
    n, c = y_true.shape

    # 1. 整体指标: Micro 看全局(受高频标签主导), Macro 看每类平均(考察长尾)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    subset_acc = float((y_true == y_pred).all(axis=1).mean())     # 9 个标签全对才算对
    hamming = hamming_loss(y_true, y_pred)

    # 2. 逐标签 P/R/F1: 定位到底是哪个标签拖后腿
    per_label = {}
    for i, name in enumerate(class_list[:c]):
        tp = int(((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum())
        fp = int(((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum())
        fn = int(((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[name] = {
            'support': int(y_true[:, i].sum()),   # 该标签真实出现次数
            'pred': int(y_pred[:, i].sum()),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }

    # 3. 标签匹配率: 预测集合与真实集合的 Jaccard 相似度均值(业务视角的"匹配得有多准")
    jaccard_sum = 0.0
    for i in range(n):
        inter = int((y_true[i] & y_pred[i]).sum())
        union = int((y_true[i] | y_pred[i]).sum())
        jaccard_sum += (inter / union) if union else 1.0

    return {
        'micro_f1': round(float(micro_f1), 4),
        'macro_f1': round(float(macro_f1), 4),
        'subset_accuracy': round(float(subset_acc), 4),
        'hamming_loss': round(float(hamming), 4),
        'label_match_accuracy': round(jaccard_sum / max(n, 1), 4),
        'avg_true_labels': round(float(y_true.sum(axis=1).mean()), 3),
        'avg_pred_labels': round(float(y_pred.sum(axis=1).mean()), 3),
        'per_label': per_label,
        'samples': n,
    }


# ============================================================
# todo 3. 业务指标(拒识机制的表现)
# ============================================================
def compute_business_metrics(probs, y_true, cfg, class_list):
    """统计拒识相关业务指标(直接复用 reject_utils 的判定逻辑, 保证线上线下一致)"""
    from reject_utils import batch_decide

    decisions = batch_decide(probs, cfg, class_list)
    n = len(decisions)
    rejected = sum(1 for d in decisions if d['rejected'])
    auto_dispatch = n - rejected
    confidences = [d['avg_confidence'] for d in decisions if d['labels']]

    # 拒识子集的"本该判对率": 用来判断拒识是否把可分样本也一起拒掉了
    rejected_idx = [i for i, d in enumerate(decisions) if d['rejected']]
    if rejected_idx and y_true is not None:
        yt = _to_np(y_true)[rejected_idx].astype(int)
        yp = np.array([[1 if class_list[j] in decisions[i]['labels'] else 0
                       for j in range(len(class_list))] for i in rejected_idx])
        micro_on_rejected = float(f1_score(yt, yp, average='micro', zero_division=0))
    else:
        micro_on_rejected = None

    return {
        'auto_dispatch_rate': round(auto_dispatch / max(n, 1), 4),   # 自动分流率
        'reject_rate': round(rejected / max(n, 1), 4),               # 拒识率
        'avg_confidence': round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        'avg_labels_per_ticket': round(sum(len(d['labels']) for d in decisions) / max(n, 1), 3),
        'rejected_micro_f1': round(micro_on_rejected, 4) if micro_on_rejected is not None else None,
        'samples': n,
    }


# ============================================================
# todo 4. 打印与存盘
# ============================================================
def print_metrics(metrics, title='评估结果'):
    """把指标打印成可读的报表"""
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)
    print(f'  Micro-F1          : {metrics["micro_f1"]:.4f}   <- 多标签主指标')
    print(f'  Macro-F1          : {metrics["macro_f1"]:.4f}   <- 长尾标签是否被牺牲')
    print(f'  Subset Accuracy   : {metrics["subset_accuracy"]:.4f}   <- 9 个标签全对的比例')
    print(f'  Hamming Loss      : {metrics["hamming_loss"]:.4f}   <- 标签位错判比例, 越低越好')
    print(f'  标签匹配率(Jaccard): {metrics["label_match_accuracy"]:.4f}')
    print(f'  平均真实/预测标签数 : {metrics["avg_true_labels"]} / {metrics["avg_pred_labels"]}')

    if 'reject_rate' in metrics:
        print('  ---- 业务指标 ----')
        print(f'  自动分流率        : {metrics["auto_dispatch_rate"]:.4f}')
        print(f'  拒识率            : {metrics["reject_rate"]:.4f}')
        print(f'  平均置信度        : {metrics["avg_confidence"]:.4f}')
        if metrics.get('rejected_micro_f1') is not None:
            print(f'  被拒识样本的 Micro-F1: {metrics["rejected_micro_f1"]:.4f}'
                  f'   <- 越接近整体指标, 说明拒识越"只拒模糊样本"')

    print('  ---- 逐标签表现 ----')
    print(f'  {"标签":<16}{"support":>8}{"precision":>11}{"recall":>9}{"f1":>8}')
    for name, m in metrics['per_label'].items():
        print(f'  {name:<16}{m["support"]:>8}{m["precision"]:>11.4f}{m["recall"]:>9.4f}{m["f1"]:>8.4f}')


def save_metrics(metrics, path):
    """把指标存成 JSON(用于消融实验汇总与前端看板展示)"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


# ============================================================
# todo 5. 一站式评估入口(训练/测试都在用这个)
# ============================================================
def evaluate_model(model, dataloader, cfg, title='评估结果', use_reject=True, save_path=None):
    """跑推理 + 算学术指标 + (可选)算业务指标 + 打印 + 存盘

    返回: 合并后的指标字典
    """
    probs, y_true, _ = run_inference(model, dataloader, cfg)
    metrics = compute_metrics(y_true, probs, cfg.class_list, threshold=cfg.label_threshold)
    metrics['threshold'] = cfg.label_threshold
    if use_reject:
        metrics.update(compute_business_metrics(probs, y_true, cfg, cfg.class_list))
    print_metrics(metrics, title=title)
    if save_path:
        save_metrics(metrics, save_path)
    return metrics


if __name__ == '__main__':
    # 用随机概率自测指标计算逻辑(不需要模型与数据)
    torch.manual_seed(0)
    class_list = ['a', 'b', 'c']
    y_true = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float)
    y_prob = torch.tensor([[0.9, 0.7, 0.2], [0.1, 0.3, 0.85], [0.6, 0.4, 0.1], [0.2, 0.9, 0.3]])
    m = compute_metrics(y_true, y_prob, class_list, threshold=0.5)
    print_metrics(m, title='指标模块自测(构造数据)')
    print('\n[OK] model2dev_utils 自测通过')
```

---

## 9. train_bert.py

```python
"""
BERT + LoRA 多标签训练主脚本 (04-bert)

功能说明:
    一条命令跑完 "读数据 -> 注入LoRA -> 多标签训练(含难例动态加权采样) -> 早停 -> 保存最优 -> 测试集评估"

核心流程:
    1. 读 01-data/train|dev|test.txt, 转 multi-hot 标签;
    2. 统计 pos_weight(负/正样本比), 交给 BCEWithLogitsLoss 治长尾不均衡;
    3. 注入 LoRA(query/value) 并冻结主干, 只训练 LoRA + 分类头(可训练参数 < 1%);
    4. 训练: 每轮结束在 dev 上算 Micro-F1, 连续 patience 轮不升则早停, 只保存最优权重;
    5. 创新点 1: 从第 2 轮开始, 用上一轮模型对训练集重新打分 -> 更新难例权重 -> 重建采样器;
    6. 训练结束用最优权重在 test 集上汇报最终指标(学术 + 业务), 并写入 result/ 目录.

运行方式:
    python 04-bert/train_bert.py                                  # 默认配置
    python 04-bert/train_bert.py --epochs 3 --batch_size 16       # 小显存
    python 04-bert/train_bert.py --use_hard_sampling false        # 消融: 关闭难例采样
    python 04-bert/train_bert.py --dry_run true                   # 只跑 1 个 batch 验证链路

产物:
    04-bert/save_models/bert_lora_multilabel.pt   最优模型(LoRA + 分类头, 含元信息)
    04-bert/save_models/pos_weight.pt             类别权重(复现实验用)
    04-bert/result/train_log.json                 每轮指标
    04-bert/result/hard_example_log.json          难例权重演化轨迹(创新点 1 的证据)
    04-bert/result/test_metrics.json              测试集最终指标
"""

import argparse
import json
import os
import random
import sys
import time

import torch

# 让脚本无论从哪个目录启动, 都能 import 同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataloader_utils import build_all_dataloader, compute_pos_weight, load_tokenizer
from hard_example_sampler import HardExampleSampler
from model2dev_utils import compute_metrics, run_inference, save_metrics, print_metrics
from multilabel_model import BertMultiLabelClassifier, build_loss, build_model


def str2bool(value):
    """命令行布尔参数解析: --use_hard_sampling true/false/1/0"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def set_seed(seed):
    """固定随机种子, 保证实验可复现"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, criterion, optimizer, scheduler, dataloader, cfg, epoch, total_epochs):
    """训练一个 epoch, 返回平均损失与耗时"""
    model.train()
    total_loss, steps = 0.0, 0
    start = time.time()
    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch['input_ids'].to(cfg.device)
        attention_mask = batch['attention_mask'].to(cfg.device)
        token_type_ids = batch.get('token_type_ids')
        token_type_ids = token_type_ids.to(cfg.device) if token_type_ids is not None else None
        labels = batch['labels'].to(cfg.device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps += 1
        if step % 50 == 0:
            print(f'    epoch {epoch}/{total_epochs} | step {step}/{len(dataloader)} | '
                  f'loss {total_loss / steps:.4f}')

    return total_loss / max(steps, 1), time.time() - start


def main():
    parser = argparse.ArgumentParser(description='ShopCare BERT+LoRA 多标签训练')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数(默认取 config)')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小(默认取 config)')
    parser.add_argument('--lr', type=float, default=None, help='学习率(默认取 config)')
    parser.add_argument('--use_lora', type=str, default='true', help='是否使用 LoRA(对照组可设 false)')
    parser.add_argument('--use_hard_sampling', type=str, default='true', help='难例动态加权采样开关')
    parser.add_argument('--dry_run', type=str, default='false', help='只跑 1 个 batch 验证链路')
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    cfg.use_lora = str2bool(args.use_lora)
    cfg.use_hard_sampling = str2bool(args.use_hard_sampling)
    dry_run = str2bool(args.dry_run)

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    print('=' * 72)
    print('ShopCare 04-bert 训练  (BERT + LoRA 多标签 + 难例动态加权采样)')
    print('=' * 72)
    print(cfg.summary())

    # todo 1. 自检: 数据与本地模型目录
    problems = cfg.check_files(need_model=True)
    if problems:
        print('\n[错误] 启动自检未通过:')
        for p in problems:
            print('  - ' + p)
        sys.exit(1)

    # todo 2. 数据与分词器
    tokenizer = load_tokenizer(cfg)
    print('\n[1/6] 加载数据...')
    train_dataset, train_loader, dev_loader, test_loader = build_all_dataloader(cfg, tokenizer)

    # todo 3. 类别权重(治长尾不均衡)
    print('\n[2/6] 统计类别权重 pos_weight...')
    pos_weight = compute_pos_weight(train_dataset)
    torch.save(pos_weight, cfg.pos_weight_path)
    print('  按标签顺序: ' + ', '.join(f'{name}={w:.2f}'
                                     for name, w in zip(cfg.class_list, pos_weight.tolist())))

    # todo 4. 模型 / 损失 / 优化器
    print('\n[3/6] 构建模型...')
    model, criterion = build_model(cfg, pos_weight)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = None
    try:
        from transformers import get_linear_schedule_with_warmup
        total_steps = max(len(train_loader) * cfg.epochs, 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
        print(f'  学习率调度: 线性 warmup(10%) + 线性衰减, 总步数 {total_steps}')
    except ImportError:
        print('  [提示] 未找到 get_linear_schedule_with_warmup, 使用固定学习率')

    # todo 5. dry run: 只跑一个 batch, 快速验证链路是否通
    if dry_run:
        print('\n[dry_run] 只跑 1 个 batch, 验证前向/反向是否正常...')
        model.train()
        batch = next(iter(train_loader))
        inputs = {k: v.to(cfg.device) for k, v in batch.items()
                  if k in ('input_ids', 'attention_mask', 'token_type_ids')}
        logits = model(**inputs)
        loss = criterion(logits, batch['labels'].to(cfg.device))
        loss.backward()
        print(f'  logits 形状 {tuple(logits.shape)} | loss {loss.item():.4f} | 反向传播正常')
        print('\n[OK] dry_run 通过: 链路可用, 去掉 --dry_run 即可正式训练')
        return

    # todo 6. 训练主循环(含早停 + 难例动态采样)
    print('\n[4/6] 开始训练...')
    hard_sampler = HardExampleSampler(cfg) if cfg.use_hard_sampling else None
    best_micro_f1, best_epoch, no_improve = -1.0, 0, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        # 创新点 1: 从第 2 轮起, 用上一轮模型重新计算难例权重并重建采样器
        if cfg.use_hard_sampling and epoch > 1:
            print(f'\n  [难例采样] 第 {epoch} 轮: 用上一轮模型对训练集重新打分并更新采样权重...')
            sampler, weights = hard_sampler.refresh(model, train_dataset, cfg)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.batch_size, sampler=sampler,
                num_workers=cfg.num_workers)
            info = hard_sampler.log_epoch(epoch - 1, weights)
            print(f'    权重均值 {info["mean"]} / 最大 {info["max"]} / '
                  f'被明显加权的难例占比 {info["hard_sample_ratio"] * 100:.1f}%')

        loss, cost = train_one_epoch(model, criterion, optimizer, scheduler,
                                     train_loader, cfg, epoch, cfg.epochs)

        # 每轮在 dev 上评估(Micro-F1 作为早停依据)
        probs, y_true, _ = run_inference(model, dev_loader, cfg)
        metrics = compute_metrics(y_true, probs, cfg.class_list, threshold=cfg.label_threshold)
        print(f'  -> epoch {epoch}: train_loss={loss:.4f} | dev Micro-F1={metrics["micro_f1"]:.4f} '
              f'| dev Macro-F1={metrics["macro_f1"]:.4f} | 耗时 {cost:.1f}s')

        history.append({
            'epoch': epoch, 'train_loss': round(loss, 4),
            'dev_micro_f1': metrics['micro_f1'], 'dev_macro_f1': metrics['macro_f1'],
            'dev_subset_accuracy': metrics['subset_accuracy'], 'seconds': round(cost, 1),
        })

        # 保存最优 + 早停
        if metrics['micro_f1'] > best_micro_f1:
            best_micro_f1, best_epoch, no_improve = metrics['micro_f1'], epoch, 0
            model.save(cfg.model_save_path, extra={
                'best_epoch': epoch,
                'dev_micro_f1': best_micro_f1,
                'pos_weight': pos_weight.tolist(),
            })
            print(f'     [保存] dev Micro-F1 提升至 {best_micro_f1:.4f}, 已保存最优模型')
        else:
            no_improve += 1
            print(f'     [早停计数] 第 {no_improve}/{cfg.patience} 轮未提升')
            if no_improve >= cfg.patience:
                print(f'  [早停] 连续 {cfg.patience} 轮未提升, 提前结束训练')
                break

        with open(cfg.train_log_path, 'w', encoding='utf-8') as f:
            json.dump({'config': cfg.summary().split('\n'), 'history': history},
                      f, ensure_ascii=False, indent=2)

    print(f'\n[5/6] 训练结束: 最优 epoch={best_epoch}, dev Micro-F1={best_micro_f1:.4f}')

    # todo 7. 用最优权重在测试集上做最终评估
    print('\n[6/6] 加载最优模型, 在测试集上评估...')
    from multilabel_model import load_trained_model
    best_model, meta = load_trained_model(cfg, cfg.model_save_path)
    if test_loader is not None:
        probs, y_true, _ = run_inference(best_model, test_loader, cfg)
        test_metrics = compute_metrics(y_true, probs, cfg.class_list, threshold=cfg.label_threshold)
        test_metrics['threshold'] = cfg.label_threshold
        from model2dev_utils import compute_business_metrics
        test_metrics.update(compute_business_metrics(probs, y_true, cfg, cfg.class_list))
        print_metrics(test_metrics, title='测试集最终指标')
        save_metrics(test_metrics, os.path.join(cfg.result_dir, 'test_metrics.json'))
    else:
        test_metrics = None
        print('  [提示] 未找到 test.txt, 跳过测试集评估')

    print('\n' + '=' * 72)
    print('训练完成 [OK]')
    print(f'  最优模型 : {cfg.model_save_path}')
    print(f'  训练日志 : {cfg.train_log_path}')
    if test_metrics:
        print(f'  测试指标 : Micro-F1={test_metrics["micro_f1"]:.4f}  '
              f'Macro-F1={test_metrics["macro_f1"]:.4f}  '
              f'拒识率={test_metrics.get("reject_rate", 0):.4f}')
    print('  下一步   : python 04-bert/bert_predict_fun.py   # 单条推理自测')
    print('=' * 72)


if __name__ == '__main__':
    main()
```

---

## 10. bert_predict_fun.py

```python
"""
对外唯一推理入口 (04-bert)

定位:
    backend(FastAPI)只通过本模块的 predict_fun() 调用 BERT 模型, 不直接碰模型权重.
    好处: 模型层可以被单独训练/评估/替换, 不会和 Web 框架耦合
    (这也是参考项目 MindCare 里 predict_fun 被 backend 复用的做法).

完整推理链路(一条工单进, 一个业务决策出):
    文本 -> tokenizer -> BERT+LoRA 前向 -> 9 个标签概率(独立 sigmoid)
         -> 双阈值拒识判定(reject_utils)
             ├─ 通过     -> 直接输出多标签结果(标 resolved_by='model')
             └─ 拒识     -> 若开启 LLM 兜底且可用 -> llm_fallback 解析
                              ├─ 成功 -> 输出结果(标 resolved_by='llm')
                              └─ 失败 -> 转人工复核(标 resolved_by='human')

返回结构里 labels 的每一项都带 (label, cn, score, dept), 前端可以直接按部门分组展示.

运行方式:
    python 04-bert/bert_predict_fun.py        # 交互式输入工单文本, 查看判定结果
"""

import json
import os
import sys
import time

# 让脚本无论从哪个目录启动都能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from reject_utils import decide, should_use_llm
from llm_fallback import LLMFallback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(PROJECT_ROOT, '01-data', 'label_meta.json')

# the three inference scripts share one demo set (see tools/ticket_data.py)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from tools.ticket_data import DEMO_SAMPLES        # noqa: E402


def load_label_meta(path=None):
    """读取标签元数据(中文名 / 责任部门 / 基础优先级), 缺失时降级为空表"""
    path = path or META_PATH
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f).get('labels', {})


# ============================================================
# todo 1. 预测器(模型懒加载 + 单例复用)
# ============================================================
class TicketPredictor:
    """工单分类预测器: 封装 tokenizer / 模型 / 拒识 / LLM 兜底

    参数:
        cfg         : Config 实例(不传则新建)
        enable_llm  : 是否允许 LLM 兜底(逐次调用还可用 use_llm_fallback 覆盖)
    说明:
        模型采用**懒加载**: 只有第一次调用 predict() 时才真正加载权重,
        这样 backend 启动就很快, 且模型文件缺失也不会导致服务起不来(只影响该模型可用性).
    """

    def __init__(self, cfg=None, enable_llm=True):
        self.cfg = cfg or Config()
        self.enable_llm = enable_llm
        self.label_meta = load_label_meta()

        self.tokenizer = None
        self.model = None
        self.llm = None
        self._loaded = False
        self.load_error = None
        self.load_seconds = None

    # ---------------- 模型加载 ----------------
    def load(self, force=False):
        """加载 tokenizer + 模型 + LLM 兜底客户端(重复调用只加载一次)"""
        if self._loaded and not force:
            return self
        start = time.time()
        try:
            import torch                                   # noqa: F401  (确认 torch 可用)
            from dataloader_utils import load_tokenizer
            from multilabel_model import load_trained_model

            self.tokenizer = load_tokenizer(self.cfg)
            self.model, self.meta = load_trained_model(self.cfg)
            self._loaded = True
            self.load_error = None
        except SystemExit as exc:                          # 模型缺失时给出明确原因
            self.load_error = str(exc)
        except Exception as exc:                           # noqa: BLE001
            self.load_error = f'{type(exc).__name__}: {exc}'

        if self.enable_llm and self.llm is None:
            self.llm = LLMFallback(self.cfg.class_list)
        self.load_seconds = round(time.time() - start, 2)
        return self

    @property
    def status(self):
        """返回模型/兜底可用状态(供 /health 与 /models 接口使用)"""
        return {
            'model': 'bert',
            'loaded': self._loaded,
            'load_error': self.load_error,
            'load_seconds': self.load_seconds,
            'model_path': self.cfg.model_save_path,
            'model_exists': os.path.exists(self.cfg.model_save_path),
            'llm_fallback_available': bool(self.llm and self.llm.available),
            'llm_disabled_reason': (self.llm.disable_reason if self.llm else 'LLM 兜底未初始化'),
            'label_threshold': self.cfg.label_threshold,
            'global_threshold': self.cfg.global_threshold,
        }

    # ---------------- 核心预测 ----------------
    def predict(self, text, use_llm_fallback=True, top_k=3):
        """对一条工单做完整判定

        参数:
            text             : 工单文本
            use_llm_fallback : 本次调用是否允许 LLM 兜底(前端有开关)
            top_k            : 返回概率最高的前 k 个标签(便于前端展示"模型在犹豫什么")
        返回: 业务决策字典(见模块 docstring)
        """
        text = (text or '').strip()
        start = time.time()
        if not text:
            return {'text': text, 'model_used': 'bert', 'labels': [], 'rejected': True,
                    'reject_reason': 'empty_text', 'reject_reason_cn': '文本为空',
                    'llm_fallback': False, 'need_human_review': False,
                    'resolved_by': 'none', 'avg_confidence': 0.0,
                    'confidences': {}, 'all_scores': {}, 'latency_ms': 0.0}

        if not self._loaded:
            self.load()
        if not self._loaded:
            # 模型不可用: 明确报错, 不假装成功(由 backend 决定是否降级到其它模型)
            raise RuntimeError(f'BERT 模型不可用: {self.load_error}')

        import torch
        # 1. 分词 -> 张量 -> 设备
        encoded = self.tokenizer(text, max_length=self.cfg.max_len, padding='max_length',
                                 truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.cfg.device)
        attention_mask = encoded['attention_mask'].to(self.cfg.device)
        token_type_ids = encoded.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.cfg.device)

        # 2. 逐标签独立 sigmoid 概率
        with torch.no_grad():
            probs = self.model.predict_proba(input_ids, attention_mask, token_type_ids)[0].cpu()

        # 3. 双阈值拒识判定
        decision = decide(probs, self.cfg.class_list,
                          label_threshold=self.cfg.label_threshold,
                          global_threshold=self.cfg.global_threshold)

        result = {
            'text': text,
            'model_used': 'bert',
            'labels': self._enrich(decision['labels'], decision['confidences']),
            'confidences': decision['confidences'],
            'avg_confidence': decision['avg_confidence'],
            'rejected': decision['rejected'],
            'reject_reason': decision['reject_reason'],
            'reject_reason_cn': decision['reject_reason_cn'],
            'llm_fallback': False,
            'llm_reason': None,
            'need_human_review': decision['rejected'],
            'resolved_by': 'model' if not decision['rejected'] else 'human',
            'top_k_scores': self._top_k(decision['all_scores'], top_k),
            'latency_ms': 0.0,
        }

        # 4. 拒识 + 兜底开关 -> LLM 兜底(失败则保持人工复核)
        llm_enabled = bool(use_llm_fallback and self.enable_llm and self.llm and self.llm.available)
        if should_use_llm(decision, llm_enabled):
            llm_result = self.llm.parse(text)
            if llm_result:
                result['labels'] = self._enrich(llm_result['labels'],
                                                {lb: llm_result['confidence'] for lb in llm_result['labels']})
                result['confidences'] = {lb: llm_result['confidence'] for lb in llm_result['labels']}
                result['avg_confidence'] = llm_result['confidence']
                result['llm_fallback'] = True
                result['llm_reason'] = llm_result['reason']
                result['need_human_review'] = False
                result['resolved_by'] = 'llm'
            else:
                result['llm_fallback'] = True          # 尝试过兜底但失败
                result['llm_reason'] = 'LLM 兜底解析失败, 已转人工复核'
                result['need_human_review'] = True
                result['resolved_by'] = 'human'

        result['latency_ms'] = round((time.time() - start) * 1000, 2)
        return result

    def predict_batch(self, texts, use_llm_fallback=True, top_k=3):
        """批量预测(逐条循环; 批量推理可后续按需优化)"""
        return [self.predict(t, use_llm_fallback=use_llm_fallback, top_k=top_k) for t in texts]

    # ---------------- 内部工具 ----------------
    def _enrich(self, labels, confidences):
        """给标签补上中文名与责任部门(前端展示与业务分流都要用)"""
        enriched = []
        for name in labels:
            meta = self.label_meta.get(name, {})
            enriched.append({
                'label': name,
                'cn': meta.get('cn', name),
                'dept': meta.get('dept', '未分配'),
                'base_priority': meta.get('base_priority', 'P2'),
                'score': round(float(confidences.get(name, 0.0)), 4),
            })
        return sorted(enriched, key=lambda x: x['score'], reverse=True)

    @staticmethod
    def _top_k(all_scores, k):
        """取概率最高的 k 个标签(含未激活的), 让前端能展示"模型在犹豫什么" """
        items = sorted(all_scores.items(), key=lambda t: t[1], reverse=True)
        return [{'label': name, 'score': score} for name, score in items[:k]]


# ============================================================
# todo 2. 模块级单例入口(backend 与脚本都用这两个)
# ============================================================
_predictor = None


def get_predictor(cfg=None, enable_llm=True):
    """获取全局预测器单例(避免每次请求都重新加载模型)"""
    global _predictor
    if _predictor is None:
        _predictor = TicketPredictor(cfg, enable_llm=enable_llm)
    return _predictor


def predict_fun(text, cfg=None, use_llm_fallback=True, top_k=3):
    """便捷函数: 单条工单预测(backend 调用这一行即可)"""
    return get_predictor(cfg).predict(text, use_llm_fallback=use_llm_fallback, top_k=top_k)


# ============================================================
# todo 3. 交互式自测
# ============================================================
if __name__ == '__main__':
    cfg = Config()
    problems = cfg.check_files(need_model=True)
    if not os.path.exists(cfg.model_save_path):
        problems.append(f'未找到训练好的模型: {cfg.model_save_path} (请先运行 train_bert.py)')

    print('=' * 72)
    print('ShopCare BERT+LoRA 工单分类 —— 交互式推理')
    print('=' * 72)
    print(cfg.summary())

    if problems:
        print('\n[警告] 以下问题会导致无法推理:')
        for p in problems:
            print('  - ' + p)
        print('\n可用性检查(仅查看状态):')
        print(get_predictor(cfg, enable_llm=False).status)
        sys.exit(0)

    print('\n正在加载模型...')
    predictor = get_predictor(cfg, enable_llm=True)
    print(f'加载完成, 耗时 {predictor.load_seconds}s')

    print('\n---- 内置样例(前三条同分布, 第 4 条为语料外新说法) ----')
    for text, gold in DEMO_SAMPLES:
        r = predictor.predict(text)
        labels = ', '.join(f'{x["cn"]}({x["score"]:.2f})' for x in r['labels']) or '无'
        print(f'\n  文本: {text}')
        if gold:
            print(f'  真实: {gold}')
        print(f'  预测: {labels}')
        print(f'  平均置信度: {r["avg_confidence"]} | 判定: {"拒识" if r["rejected"] else "自动分流"}'
              f' | 处理方: {r["resolved_by"]} | 耗时: {r["latency_ms"]}ms')

    print('\n---- 交互输入(直接回车退出) ----')
    while True:
        try:
            text = input('\n请输入工单文本: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        r = predictor.predict(text)
        print(json.dumps(r, ensure_ascii=False, indent=2))
```

---

> 文档生成时间：2026-09-21
