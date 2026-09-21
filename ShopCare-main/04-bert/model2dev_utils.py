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