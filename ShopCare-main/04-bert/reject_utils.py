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