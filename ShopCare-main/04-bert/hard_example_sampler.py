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