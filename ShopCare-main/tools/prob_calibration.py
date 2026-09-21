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

支持两种方法:

  * `platt`   : 对 logit(p) 做一维逻辑回归, 2 个参数/标签。平滑、不会输出正好 0/1。
  * `isotonic`: 保序回归, 直接拟合经验正确率。非参数、更贴合数据, 但会输出阶梯
                函数(同一个台阶内的样本概率完全相同)。

默认用 **platt**, 这是实测选出来的, 不是拍脑袋:

    02-rf 上两种方法的对比(校准器拟合于 dev, 指标在 test 上算)
        platt     test ECE 0.0806 -> 0.0270    概率正好 1.0 的标签占 0.3%
        isotonic  test ECE 0.0806 -> 0.0352    概率正好 1.0 的标签占 2.3%

isotonic 在 dev 上的对数损失更低, 但那是**拟合过度**: dev 里恰好 100% 命中的分箱
会被直接映射成 1.000, 换到 test 上就是新的过度自信 —— 而"显示 100%"正是这套系统
想避免的东西。platt 平滑、不外推饱和, test 上反而更准。

用法
====
    from tools.prob_calibration import fit_calibrators, apply_calibrators, summary

    calib = fit_calibrators(probs_dev, Y_dev, class_list, method='auto')
    probs_dev_c  = apply_calibrators(probs_dev, calib)
    probs_test_c = apply_calibrators(probs_test, calib)
    print(summary(probs_dev, probs_test, Y_dev, Y_test, calib))

校准器只存纯数字(JSON 可序列化), 能直接塞进 02-rf 的 joblib 包和
03-fasttext 的 model_meta.json, 推理端 apply 一下即可。
"""

import numpy as np

# 校准器内部对概率做 logit 变换前先截断, 避免 log(0) = -inf
_EPS = 1e-6

METHODS = ('auto', 'platt', 'isotonic')


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


def _apply_one(p, params):
    if not params:
        return np.asarray(p, dtype=float)      # 退化标签: 原样返回
    if params['kind'] == 'platt':
        return _apply_platt(p, params)
    return _apply_isotonic(p, params)


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


def fit_calibrators(probs, y_true, class_list=None, method='platt'):
    """在验证集上为每个标签拟合校准器。

    method: 'platt' / 'isotonic' / 'auto'(两种都试, 按对数损失选更好的)
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
    chosen = method
    calibrators = []
    per_label = []

    # 'auto' 需要先把两种方法都拟合出来, 再统一比较, 所以这里分两轮
    candidates = {}
    for kind in ('platt', 'isotonic'):
        if method not in ('auto', kind):
            continue
        params_list = []
        for i in range(n_labels):
            fitter = _fit_platt if kind == 'platt' else _fit_isotonic
            params = fitter(probs[:, i], y_true[:, i])
            if params:
                params = dict(params, kind=kind)
            params_list.append(params)
        candidates[kind] = params_list

    if method == 'auto':
        scores = {}
        for kind, params_list in candidates.items():
            calib = {'method': kind, 'calibrators': params_list}
            scores[kind] = _log_loss(apply_calibrators(probs, calib), y_true)
        chosen = min(scores, key=scores.get)
        reason = ', '.join(f'{k}={v:.4f}' for k, v in scores.items())
    else:
        reason = f'指定 {method}'

    for i, params in enumerate(candidates[chosen]):
        if params:
            calibrators.append(params)
            lo, hi = float(probs[:, i].min()), float(probs[:, i].max())
            per_label.append({'label': names[i], 'kind': params['kind'],
                              'raw_range': [round(lo, 4), round(hi, 4)]})
        else:
            calibrators.append(None)
            per_label.append({'label': names[i], 'kind': None,
                              'note': '训练集里该标签只有单一类别, 不校准'})

    return {
        'method': chosen,
        'method_reason': reason,
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
