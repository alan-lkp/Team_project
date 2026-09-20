"""
ShopCare 通用指标与拒识工具 (tools, 纯 numpy 实现)

模块说明:
    04-bert/model2dev_utils.py 里的同口径指标依赖 torch, 而 02-rf / 03-fasttext
    以及 backend 服务层**不应该为了算几个指标就装上 torch**(几百 MB).
    所以这里用 numpy 重新实现一份**口径完全一致**的版本:

        02-rf  ─┐
        03-fasttext ─┼─> tools/ml_metrics.py  <─ 同一套阈值/同一套公式
        04-bert ─┘(torch 版, 见 model2dev_utils)

    关键口径(三个对照模型必须一致, 否则对比无意义):
        * 激活规则   : 独立概率 >= label_threshold
        * 拒识规则   : 激活标签的平均置信度 < global_threshold  -> 拒识
        * Micro-F1   : 所有 (样本,标签) 位置拉平后算全局 F1(多标签主指标)
        * Macro-F1   : 逐标签 F1 的算术平均(考察长尾标签)
        * 子集准确率 : 一条工单的全部标签都对才算对(最严格)
        * Hamming    : 错判标签位 / 总标签位(越低越好)

对外接口:
    compute_metrics()           -> 学术指标字典
    decide()                    -> 单条工单的激活 + 拒识判定
    batch_decide()              -> 批量判定
    compute_business_metrics()  -> 业务指标(自动分流率/拒识率/平均置信度)
    grid_search_thresholds()    -> 在 dev 上标定双阈值
"""

import os
import sys

import numpy as np

__all__ = [
    'REJECT_NONE', 'REJECT_NO_LABEL', 'REJECT_LOW_CONF', 'REJECT_REASON_CN',
    'compute_metrics', 'decide', 'batch_decide', 'should_use_llm',
    'compute_business_metrics', 'grid_search_thresholds',
]

# 拒识原因码(与 04-bert/reject_utils.py 保持一致)
REJECT_NONE = 'none'
REJECT_NO_LABEL = 'no_label_activated'
REJECT_LOW_CONF = 'low_confidence'

REJECT_REASON_CN = {
    REJECT_NONE: '正常激活',
    REJECT_NO_LABEL: '无标签激活(9 个标签都没超过单标签阈值, 疑似新问题或灌水)',
    REJECT_LOW_CONF: '诉求模糊(激活标签的平均置信度未达到全局阈值)',
}


# ============================================================
# todo 1. 输入规整
# ============================================================
def _as_2d(x):
    """把 torch.Tensor / numpy / list 统一成 float 的 (N, C) numpy 矩阵"""
    if hasattr(x, 'detach'):          # torch.Tensor
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    return a


# ============================================================
# todo 2. 学术指标
# ============================================================
def _f1_from_counts(tp, fp, fn, eps=1e-12):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def compute_metrics(y_true, y_prob, class_list, threshold=0.5):
    """计算多标签分类的学术指标

    参数:
        y_true    : (N, C) multi-hot 真值
        y_prob    : (N, C) 每标签独立概率
        class_list: 标签名(长度必须等于 C)
        threshold : 单标签激活阈值
    返回: dict —— micro/macro/subset/hamming + 逐标签明细
    """
    yt = (_as_2d(y_true) >= 0.5).astype(int)
    yp = (_as_2d(y_prob) >= threshold).astype(int)
    if yt.shape != yp.shape:
        raise ValueError(f'真值 {yt.shape} 与预测 {yp.shape} 形状不一致')
    if yt.shape[1] != len(class_list):
        raise ValueError(f'标签数 {yt.shape[1]} 与 class_list 长度 {len(class_list)} 不一致')

    tp_all = int(((yt == 1) & (yp == 1)).sum())
    fp_all = int(((yt == 0) & (yp == 1)).sum())
    fn_all = int(((yt == 1) & (yp == 0)).sum())
    micro_p, micro_r, micro_f1 = _f1_from_counts(tp_all, fp_all, fn_all)

    per_label, macro_f1_sum = [], 0.0
    for i, name in enumerate(class_list):
        tp = int(((yt[:, i] == 1) & (yp[:, i] == 1)).sum())
        fp = int(((yt[:, i] == 0) & (yp[:, i] == 1)).sum())
        fn = int(((yt[:, i] == 1) & (yp[:, i] == 0)).sum())
        support = int(yt[:, i].sum())
        p, r, f1 = _f1_from_counts(tp, fp, fn)
        # 支持数>0 才计入 macro, 否则"训练集里根本没这个标签"会把 macro 拉低
        if support > 0:
            macro_f1_sum += f1
        per_label.append({
            'label': name, 'support': support,
            'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4),
        })
    n_valid_labels = sum(1 for x in per_label if x['support'] > 0) or 1

    subset_acc = float((yt == yp).all(axis=1).mean())
    hamming = float((yt != yp).mean())
    # 标签匹配率(Jaccard): 预测集合与真实集合的交并比, 对"部分正确"更宽容
    inter = ((yt == 1) & (yp == 1)).sum(axis=1)
    union = ((yt == 1) | (yp == 1)).sum(axis=1)
    jaccard = float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0)))

    return {
        'micro_f1': round(float(micro_f1), 4),
        'micro_precision': round(float(micro_p), 4),
        'micro_recall': round(float(micro_r), 4),
        'macro_f1': round(float(macro_f1_sum / n_valid_labels), 4),
        'subset_accuracy': round(subset_acc, 4),
        'hamming_loss': round(hamming, 4),
        'jaccard_mean': round(jaccard, 4),
        'avg_pred_labels': round(float(yp.sum(axis=1).mean()), 3),
        'avg_true_labels': round(float(yt.sum(axis=1).mean()), 3),
        'threshold': threshold,
        'num_samples': int(yt.shape[0]),
        'per_label': per_label,
    }


# ============================================================
# todo 3. 双阈值拒识
# ============================================================
def decide(probs_row, class_list, label_threshold=0.5, global_threshold=0.8, max_labels=None):
    """单条工单的激活 + 拒识判定(语义与 04-bert/reject_utils.decide 完全一致)

    返回: dict(labels, confidences, avg_confidence, rejected, reject_reason,
               reject_reason_cn, need_human_review, all_scores)
    """
    scores = [float(x) for x in np.asarray(probs_row, dtype=float).ravel()]
    pairs = sorted(zip(class_list, scores), key=lambda t: t[1], reverse=True)

    activated = [(n, s) for n, s in pairs if s >= label_threshold]
    if max_labels:
        activated = activated[:max_labels]

    if not activated:
        return {
            'labels': [], 'confidences': {}, 'avg_confidence': 0.0,
            'rejected': True,
            'reject_reason': REJECT_NO_LABEL,
            'reject_reason_cn': REJECT_REASON_CN[REJECT_NO_LABEL],
            'need_human_review': True,
            'all_scores': {n: round(s, 4) for n, s in pairs},
        }

    avg_conf = sum(s for _, s in activated) / len(activated)
    rejected = avg_conf < global_threshold
    return {
        'labels': [n for n, _ in activated],
        'confidences': {n: round(s, 4) for n, s in activated},
        'avg_confidence': round(avg_conf, 4),
        'rejected': bool(rejected),
        'reject_reason': REJECT_LOW_CONF if rejected else REJECT_NONE,
        'reject_reason_cn': REJECT_REASON_CN[REJECT_LOW_CONF] if rejected else None,
        'need_human_review': bool(rejected),
        'all_scores': {n: round(s, 4) for n, s in pairs},
    }


def batch_decide(probs, class_list, label_threshold=0.5, global_threshold=0.8, max_labels=None):
    """批量判定: probs 为 (N, C)"""
    class_list = list(class_list)
    return [decide(row, class_list, label_threshold, global_threshold, max_labels)
            for row in _as_2d(probs)]


def should_use_llm(decision, llm_enabled=True):
    """只有"拒识"且"兜底开关打开"时才升级到 LLM —— 控制成本的关键开关"""
    return bool(llm_enabled and decision.get('rejected'))


# ============================================================
# todo 4. 业务指标
# ============================================================
def compute_business_metrics(y_prob, y_true=None, class_list=None,
                             label_threshold=0.5, global_threshold=0.8):
    """服务层关心的指标: 自动分流率 / 拒识率 / 平均置信度 / 平均激活标签数

    自动化率是这套系统最核心的业务数字 —— 它直接决定能省下多少人工.
    """
    probs = _as_2d(y_prob)
    class_list = list(class_list) if class_list is not None else [str(i) for i in range(probs.shape[1])]
    decisions = batch_decide(probs, class_list, label_threshold, global_threshold)
    n = len(decisions)
    rejected = sum(1 for d in decisions if d['rejected'])
    out = {
        'num_samples': n,
        'auto_rate': round((n - rejected) / n, 4) if n else 0.0,
        'reject_rate': round(rejected / n, 4) if n else 0.0,
        'avg_confidence': round(float(np.mean([d['avg_confidence'] for d in decisions])), 4) if n else 0.0,
        'avg_activated_labels': round(float(np.mean([len(d['labels']) for d in decisions])), 3) if n else 0.0,
        'reject_reason_dist': {},
    }
    for d in decisions:
        if d['rejected']:
            out['reject_reason_dist'][d['reject_reason']] = out['reject_reason_dist'].get(d['reject_reason'], 0) + 1

    if y_true is not None:
        yt = (_as_2d(y_true) >= 0.5).astype(int)
        keep = np.array([not d['rejected'] for d in decisions])
        if keep.sum() > 0:
            auto_metrics = compute_metrics(yt[keep], probs[keep], class_list, label_threshold)
            out['auto_micro_f1'] = auto_metrics['micro_f1']
            out['auto_macro_f1'] = auto_metrics['macro_f1']
        else:
            out['auto_micro_f1'] = 0.0
            out['auto_macro_f1'] = 0.0
    return out


# ============================================================
# todo 5. 阈值标定(在 dev 集上网格搜索)
# ============================================================
def grid_search_thresholds(y_prob, y_true, class_list,
                           label_candidates=(0.3, 0.4, 0.5, 0.6, 0.7),
                           global_candidates=(0.6, 0.7, 0.75, 0.8, 0.85, 0.9),
                           target_reject_rate=0.15):
    """在验证集上标定双阈值

    目标: 在"拒识率 <= target_reject_rate"的前提下, 让自动分流的 Micro-F1 尽量高.
    返回: 按 micro_f1 降序排列的候选列表
    """
    yt = (_as_2d(y_true) >= 0.5).astype(int)
    probs = _as_2d(y_prob)
    results = []
    for lt in label_candidates:
        for gt in global_candidates:
            biz = compute_business_metrics(probs, yt, class_list, lt, gt)
            if biz['reject_rate'] > target_reject_rate:
                continue
            results.append({
                'label_threshold': lt,
                'global_threshold': gt,
                'auto_rate': biz['auto_rate'],
                'reject_rate': biz['reject_rate'],
                'auto_micro_f1': biz.get('auto_micro_f1', 0.0),
                'auto_macro_f1': biz.get('auto_macro_f1', 0.0),
                'avg_confidence': biz['avg_confidence'],
            })
    results.sort(key=lambda r: (-r['auto_micro_f1'], r['reject_rate']))
    return results


# ============================================================
# todo 6. 打印
# ============================================================
def print_metrics(metrics, title='评估结果'):
    """把指标打印成一张对齐的表(中文标签用 ljust 会有宽度问题, 这里用固定列宽)"""
    print('=' * 72)
    print(title)
    print('=' * 72)
    print(f'  样本数        : {metrics.get("num_samples")}')
    print(f'  Micro-F1      : {metrics.get("micro_f1")}   (主指标, 全局拉平)')
    print(f'  Micro-P / R   : {metrics.get("micro_precision")} / {metrics.get("micro_recall")}')
    print(f'  Macro-F1      : {metrics.get("macro_f1")}   (长尾标签体检)')
    print(f'  子集准确率    : {metrics.get("subset_accuracy")}   (整单全对率)')
    print(f'  Hamming Loss  : {metrics.get("hamming_loss")}')
    print(f'  标签匹配率    : {metrics.get("jaccard_mean")}')
    print(f'  平均预测标签数: {metrics.get("avg_pred_labels")}   平均真实标签数: {metrics.get("avg_true_labels")}')
    if 'auto_rate' in metrics:
        print('-' * 72)
        print('  业务指标')
        print(f'    自动分流率  : {metrics.get("auto_rate"):.2%}   (核心业务数字)')
        print(f'    拒识率      : {metrics.get("reject_rate"):.2%}')
        print(f'    自动分流F1  : {metrics.get("auto_micro_f1")} / Macro {metrics.get("auto_macro_f1")}')
    if 'per_label' in metrics:
        print('-' * 72)
        print(f'  {"标签":<18}{"support":>9}{"P":>9}{"R":>9}{"F1":>9}')
        for row in metrics['per_label']:
            print(f'  {row["label"]:<18}{row["support"]:>9}{row["precision"]:>9}{row["recall"]:>9}{row["f1"]:>9}')


if __name__ == '__main__':
    # 自测: 构造 4 条样本 3 个标签, 概率是"故意半对半错"的确定值, 便于人肉核对
    # 第 2 行故意多判一个 quality(FP), 第 3 行故意漏判 quality(FN) -> 手算 P=R=4/5, F1=0.8
    classes = ['logistics', 'quality', 'after_sale']
    y_true = [[1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    y_prob = [[0.9, 0.7, 0.2], [0.1, 0.3, 0.85], [0.6, 0.55, 0.1], [0.2, 0.4, 0.3]]
    m = compute_metrics(y_true, y_prob, classes, threshold=0.5)
    print_metrics(m, title='ml_metrics 自测(构造数据: 1 个 FP + 1 个 FN -> Micro-F1 应为 0.8)')
    assert abs(m['micro_f1'] - 0.8) < 1e-6, m['micro_f1']
    assert abs(m['subset_accuracy'] - 0.5) < 1e-6, m['subset_accuracy']
    assert abs(m['hamming_loss'] - 2 / 12) < 1e-4, m['hamming_loss']
    assert abs(m['jaccard_mean'] - 0.625) < 1e-6, m['jaccard_mean']

    d = decide([0.9, 0.51, 0.1], classes, 0.5, 0.8)
    assert d['labels'] == ['logistics', 'quality'], d['labels']
    assert d['rejected'] is True, '平均置信度 0.705 < 0.8 应该拒识'
    print('\n拒识自测:', {k: d[k] for k in ('labels', 'avg_confidence', 'rejected', 'reject_reason')})

    d2 = decide([0.91, 0.88, 0.02], classes, 0.5, 0.8)
    assert d2['rejected'] is False and d2['labels'] == ['logistics', 'quality']
    print('正常自测:', {k: d2[k] for k in ('labels', 'avg_confidence', 'rejected')})

    d3 = decide([0.4, 0.3, 0.2], classes, 0.5, 0.8)
    assert d3['reject_reason'] == REJECT_NO_LABEL
    print('无标签自测:', d3['reject_reason'], '->', d3['reject_reason_cn'])

    grid = grid_search_thresholds(y_prob, y_true, classes)
    print('阈值搜索候选数:', len(grid), '| 最优:', grid[0] if grid else None)
    print('\n[OK] ml_metrics 自测通过')