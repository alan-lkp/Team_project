# -*- coding: utf-8 -*-
"""
概率校准 —— 让「模型说 0.8」真的意味着「大约 80% 是对的」

为什么需要这个
==============
02-rf 和 03-fasttext 输出的原始概率**尺度完全不同**,都不能当"可信度"直接读:

    同一条工单「发票开错了, 抬头写成公司旧名字, 能重开吗」
        02-rf     : invoice = 0.419   (其余标签最高 0.33)
        03-fasttext: invoice = 1.000   (其余标签 ~0.000)

两个模型**标签都判对了**,但一个显示 42%、一个显示 100%。原因是:

  * 随机森林的概率是 300 棵树在叶子上的投票比例,天然被压向中间,
    实测**偏保守**(它说 0.6~0.8 时,实际命中率是 100%);
  * FastText 用 ova(每标签一个独立 sigmoid),在 10 万条语料上训练 25 轮后
    会**饱和**,5.9% 的标签概率正好是 1.0(它说 0.9+ 时实际命中率 88%)。

后果有两个,都是实打实的:

  1. 前端把原始概率当"置信度"展示 —— FastText 的 100.0% 会让评审直接怀疑数据泄漏,
     RF 的 42% 又像是模型坏了,两个数字**不可比**;
  2. 双阈值拒识链路拿这些数字做判断 —— 一个说 1.0、一个说 0.4,
     全局阈值 0.5 对两者根本不代表同一件事。

校准做什么
==========
在**验证集**上为每个标签单独拟合一个单调映射,把原始分数映射成经验正确率。
单调保证**不改变排序**,所以 Micro-F1 不会因为校准而变(阈值会被重新标定),
变的是"概率"这个数字本身的含义。

支持三种方法:

  * `platt`     : 对 logit(p) 做一维逻辑回归, 2 个参数/标签。平滑、不会输出正好 0/1。
  * `isotonic`  : 保序回归, 直接拟合经验正确率。非参数, 但会输出阶梯函数
                  (同一个台阶内的样本概率完全相同)。
  * `equal_freq`: **等频分箱** + 平滑。按分位数切箱, 每箱样本数相同。见下面的说明。

**默认是 `platt`**, 但 02-rf 显式改用 `equal_freq` —— 它治的是 platt 治不了的病。

为什么 RF 需要 equal_freq
-------------------------
随机森林的概率被压缩在一个很窄的区间(实测 02-rf 只有 [0.055, 0.78])。
platt 是一条全局 sigmoid, 它要在**整个区间**上拟合, 而高分段几乎没有样本:

    dev 上 quality 标签: 原始概率 >0.45 的 148 条, >0.5 的 35 条, >0.55 的 3 条

platt 于是拟合出一条极陡的曲线来迁就这条尾部(实测 a=7.19), 把原始 0.482
**外推**成 0.9649; 而 dev 的尾部本身并不单调(logistics 在 [0.55,0.70) 的命中率
0.50 反而低于 [0.50,0.55) 的 0.79), 说明那段基本是噪声 —— 在噪声上拟合出的
陡峭斜率, 就是"前端显示 99%"的来源。

等频分箱从两个方向解决它:

  1. **每箱样本数相同**。定宽分箱会让高分段空掉, 等频分箱保证尾部拿到的样本量
     和低分段一样多, 从根上消除"从个位数样本外推"。
  2. **落箱时夹到端点**。观测范围之外的输入夹到首/尾箱, 绝不外推。

代价是输出变成阶梯(同一箱内概率相同), 但因此**永远到不了 1.0** —— 实测 02-rf
最大输出从 platt 的 1.000 降到 0.963, 而 test Micro-F1 反而从 0.6154 升到 0.6330。

实测对比(02-rf, 校准器拟合于 dev, 指标算在 test):

    方法                     全局ECE   最大输出   ≥0.99 那一档的实际命中率
    platt(不校准前 0.2026)    0.0218    1.000     0.8401   ← 声称 99.6%, 实际 84%
    isotonic                 0.0239    1.000     0.9610
    equal_freq(20箱 prior=20) 0.0251    0.963       (无此档)

方法必须**显式指定**, 不提供"自动挑一个"的选项。原因: 唯一客观的挑选标准是 dev 上的
对数损失, 而实测这个标准会选中 isotonic —— 它在 dev 上确实最低, 但代价是把 dev 里恰好
100% 命中的分箱直接映射成 1.000, 换到 test 上就成了新的过度自信(2.3% 的标签输出正好
1.0, 而 platt 只有 0.3%)。用一个会在评测集上变差的指标去自动选校准器, 不如按每个模型
自己的输出分布手工定一个, 而且在报告里说得清"哪套模型用了哪种方法、为什么"。

用法
====
    from tools.prob_calibration import (fit_calibrators, apply_calibrators,
                                        expected_calibration_error, summary)

    calib = fit_calibrators(probs_dev, Y_dev, class_list)            # 默认 platt
    calib = fit_calibrators(probs_dev, Y_dev, class_list,            # 02-rf 用这个
                            method='equal_freq', n_bins=20, prior=20.0)
    probs_test_c = apply_calibrators(probs_test, calib)
    print(summary(probs_test, probs_test_c, Y_test, calib))

推理端只读产物里的 `calibrators` 列表(逐标签带 `kind`), 不解析 `method` 字符串,
所以新旧方法的产物可以共存, 换方法不需要动任何推理代码。

校准器只存纯数字(JSON 可序列化), 能直接塞进 02-rf 的 joblib 包和
03-fasttext 的 model_meta.json, 推理端 apply 一下即可。
"""

import numpy as np

# 校准器内部对概率做 logit 变换前先截断, 避免 log(0) = -inf
_EPS = 1e-6

METHODS = ('platt', 'isotonic', 'equal_freq')


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _fit_platt(p, y):
    """一维逻辑回归: logit(p) -> y。返回 (a, b), 校准后 = sigmoid(a * logit(p) + b)。"""
    from sklearn.linear_model import LogisticRegression

    z = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    lr.fit(z, y)
    return {'a': float(lr.coef_[0][0]), 'b': float(lr.intercept_[0])}


def _apply_platt(p, params):
    return _sigmoid(params['a'] * _logit(p) + params['b'])


def _fit_isotonic(p, y):
    """保序回归: 直接拟合「分数 -> 经验正确率」的单调阶梯。"""
    from sklearn.isotonic import IsotonicRegression

    if len(np.unique(y)) < 2:
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(p, y)
    # sklearn 版本差异: 新版本把断点挂在 IsotonicRegression 自身(X_thresholds_),
    # 老版本在 iso.f_ 上。两边都试一下, 避免被环境卡住。
    x = getattr(iso, 'X_thresholds_', None)
    yy = getattr(iso, 'y_thresholds_', None)
    if x is None or yy is None:
        x, yy = iso.f_.x_thresholds_, iso.f_.y_thresholds_
    return {'x': [float(v) for v in x], 'y': [float(v) for v in yy]}


def _apply_isotonic(p, params):
    return np.interp(np.asarray(p, dtype=float), params['x'], params['y'])


def _fit_binned(p, y, n_bins=20, prior=20.0):
    """等频分箱校准: 按分位数切箱, 每箱样本数相同。

    为什么不能用定宽分箱
    --------------------
    随机森林的概率被压在一个很窄的区间(实测 02-rf 只有 [0.055, 0.78]),
    定宽分箱会让高分段基本空掉 —— dev 上 quality 标签原始概率 >0.5 的只有
    35 条、>0.55 的只有 3 条。在这种区间上做任何拟合都是从个位数样本外推。
    等频分箱保证每箱样本数都是 n/箱数, 尾部拿到的样本量和低分段一样多。

    为什么要平滑
    ------------
    一箱恰好 100% 命中时, 直接取经验正确率会输出 1.000 —— 那正是这套系统想
    避免的"声称一个测不出来的精度"。所以向全局正例率 base 收缩:

        rate = (hits + prior * base) / (n + prior)

    prior 是"先验的等效样本数": 一箱至少要有 prior 条样本, 才能把估计从全局
    先验明显拉开。prior 越大越保守。

    返回的 dict 直接 JSON 可序列化, 会被写进模型产物。
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    qs = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(qs) < 3:
        # 概率取值太少(RF 可能只输出少数几个离散值), 连一箱都切不出来
        return None
    edges = qs[1:-1]                    # 内部切点; 两端不设界, 范围外自然夹到端点箱
    idx = np.clip(np.searchsorted(edges, p, side='right'), 0, len(edges))
    base = float(y.mean())
    rates = []
    for k in range(len(edges) + 1):
        m = idx == k
        n, hits = int(m.sum()), float(y[m].sum())
        rates.append((hits + prior * base) / (n + prior) if n + prior > 0 else base)
    # 保序: 分箱是逐箱独立估的, 噪声可能让相邻箱非单调, 累积取最大即恢复单调。
    # 单调才不会打乱排序, 阈值语义(分数越高越可能为正)才成立。
    rates = np.maximum.accumulate(np.asarray(rates, dtype=float))
    return {'kind': 'equal_freq',
            'edges': [float(v) for v in edges],
            'rates': [float(v) for v in rates],
            'n_bins': int(n_bins), 'prior': float(prior)}


def _apply_binned(p, params):
    """落箱: searchsorted 找箱号, clip 保证落在观测范围外的输入夹到端点箱, 不外推。"""
    edges = np.asarray(params['edges'], dtype=float)
    rates = np.asarray(params['rates'], dtype=float)
    idx = np.clip(np.searchsorted(edges, np.asarray(p, dtype=float), side='right'),
                  0, len(rates) - 1)
    return rates[idx]


def _apply_one(p, params):
    if not params:
        return np.asarray(p, dtype=float)      # 退化标签: 原样返回
    kind = params['kind']
    if kind == 'platt':
        return _apply_platt(p, params)
    if kind == 'isotonic':
        return _apply_isotonic(p, params)
    if kind == 'equal_freq':
        return _apply_binned(p, params)
    # 白名单之外必须报错, 不能兜底。老版本这里是 `else: return _apply_isotonic(...)`,
    # 于是任何新的 kind 都会掉进 isotonic 分支 —— 运气好读 params['x'] 时 KeyError,
    # 运气不好(产物里恰好有同名 key)则静默给出一个看起来正常、实际完全错误的概率。
    raise ValueError(f'未知的校准器类型: {kind!r}')


def apply_calibrators(probs, calib):
    """按校准器把 (N, C) 的原始概率矩阵映射成校准后的概率矩阵。"""
    probs = np.asarray(probs, dtype=float)
    out = np.zeros_like(probs)
    for i, params in enumerate(calib['calibrators']):
        out[:, i] = _apply_one(probs[:, i], params)
    return out


def _log_loss(probs, y_true):
    # y_true 从 to_xy 出来是 list of list, 这里统一转成 ndarray 再算
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(probs, y_true, bins=15):
    """ECE: 把预测概率分箱, 算 |平均置信度 - 实际正确率| 的加权平均。

    0 表示完美校准; 0.1 以上说明"模型说的概率"和"实际正确率"差得比较远。
    """
    probs = np.asarray(probs, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(probs)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(probs[mask].mean() - y_true[mask].mean())
    return round(float(ece), 4)


def fit_calibrators(probs, y_true, class_list=None, method='platt',
                    n_bins=20, prior=20.0):
    """在验证集上为每个标签拟合校准器。

    method: 'platt' / 'isotonic' / 'equal_freq', **必须显式指定**(没有自动挑选, 理由见模块
            docstring)。当前三个调用点: 02-rf 显式用 'equal_freq', 03-fasttext 与 04-bert 用
            默认的 'platt'。
    n_bins, prior: 仅 'equal_freq' 使用 —— 分箱数与平滑先验的等效样本数。
   返回一个 JSON 可序列化的 dict。
    """
    if method not in METHODS:
        raise ValueError(f'method 必须是 {METHODS} 之一, 收到 {method!r}')

    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if probs.ndim != 2 or probs.shape != y_true.shape:
        raise ValueError(f'probs 与 y_true 形状必须一致且为二维, '
                         f'收到 {probs.shape} / {y_true.shape}')

    n_labels = probs.shape[1]
    names = list(class_list) if class_list is not None else [str(i) for i in range(n_labels)]
    calibrators = []
    per_label = []

    for i in range(n_labels):
        if method == 'platt':
            params = _fit_platt(probs[:, i], y_true[:, i])
        elif method == 'isotonic':
            params = _fit_isotonic(probs[:, i], y_true[:, i])
        else:
            params = _fit_binned(probs[:, i], y_true[:, i], n_bins, prior)

        if params:
            params = dict(params, kind=method)
            calibrators.append(params)
            lo, hi = float(probs[:, i].min()), float(probs[:, i].max())
            # equal_freq 额外记下分箱数与平滑先验, 便于复现和排查
            extra = ({'n_bins': params['n_bins'], 'prior': params['prior']}
                     if method == 'equal_freq' else {})
            per_label.append({'label': names[i], 'kind': method,
                              'raw_range': [round(lo, 4), round(hi, 4)], **extra})
        else:
            calibrators.append(None)
            per_label.append({'label': names[i], 'kind': None,
                              'note': '训练集里该标签只有单一类别, 不校准'})

    return {
        'method': method,
        'method_reason': '显式指定',
        'calibrators': calibrators,
        'per_label': per_label,
    }


def summary(probs_raw, probs_cal, y_true, calib, title='概率校准结果'):
    """打印校准前后的 ECE / 对数损失, 以及几个典型分档的变化。"""
    lines = ['']
    lines.append('=' * 72)
    lines.append(title)
    lines.append('=' * 72)
    lines.append(f"  方法: {calib['method']}  ({calib['method_reason']})")
    lines.append('')
    lines.append(f"  {'':<10}{'ECE(越低越好)':>18}{'对数损失':>14}")
    lines.append(f"  {'校准前':<10}{expected_calibration_error(probs_raw, y_true):>18}"
                 f"{_log_loss(probs_raw, y_true):>14.4f}")
    lines.append(f"  {'校准后':<10}{expected_calibration_error(probs_cal, y_true):>18}"
                 f"{_log_loss(probs_cal, y_true):>14.4f}")

    # 展示"模型说什么 -> 校准后说什么", 直观看饱和有没有被修掉
    lines.append('')
    lines.append('  原始分数 -> 校准后概率(举例):')
    flat_raw = np.asarray(probs_raw, dtype=float).ravel()
    flat_cal = np.asarray(probs_cal, dtype=float).ravel()
    for probe in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        mask = np.abs(flat_raw - probe) < 0.005
        if mask.sum() == 0:
            continue
        lines.append(f'    {probe:.2f}  ({int(mask.sum()):>6} 个标签)  ->  '
                     f'{flat_cal[mask].mean():.3f}')

    n_sat_before = int((np.asarray(probs_raw) >= 0.999).sum())
    n_sat_after = int((np.asarray(probs_cal) >= 0.999).sum())
    lines.append('')
    lines.append(f'  概率正好 1.000 的标签数: 校准前 {n_sat_before} -> 校准后 {n_sat_after}')
    return '\n'.join(lines)
