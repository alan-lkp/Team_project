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
