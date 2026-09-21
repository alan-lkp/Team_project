"""
ShopCare 02-rf 测试集评估脚本

加载已训练好的随机森林模型, 在 01-data/test.txt 上评估并输出:
  准确率 / 精准率 / 召回率 / F1 (micro + macro + 逐标签明细)

说明:
    * 准确率这里用"子集准确率"——多标签场景下最严格的口径, 一条工单的全部标签都判对才算对。
    * micro 是把所有 (样本,标签) 位置拉平后算的全局 P/R/F1, 多标签主指标。
    * macro 是逐标签算 F1 再取平均, 更反映长尾标签的表现。

运行方式: 在 ShopCare-main 根目录下执行
    python 02-rf/rf_test.py
"""

import os
import sys

# 把 02-rf 目录和项目根目录加进 sys.path, 让 from config / from tools / from rf_train 都能找到
_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_STAGE_DIR)
for _p in (_STAGE_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib                                              # noqa: E402
from tools.ticket_data import load_splits, to_xy           # noqa: E402
from tools.ml_metrics import compute_metrics, print_metrics  # noqa: E402
from rf_train import predict_proba_matrix                   # noqa: E402

# 已训练模型的路径(由 rf_train.py 训练后生成)
MODEL_PATH = os.path.join(_STAGE_DIR, 'save_models', 'rf_tfidf.joblib')


def main():
    # 1. 加载模型包(内含 vectorizer / 9 个二分类器 / 阈值 / 标签表)
    bundle = joblib.load(MODEL_PATH)
    vectorizer = bundle['vectorizer']
    estimators = bundle['estimators']
    constants = bundle['constants']
    class_list = bundle['class_list']
    threshold = bundle.get('label_threshold', 0.5)   # 单标签激活阈值

    print('=' * 72)
    print('ShopCare 02-rf 测试集评估')
    print('=' * 72)
    print(f'  模型文件     : {MODEL_PATH}')
    print(f'  特征模式     : {bundle.get("feature_mode")}')
    print(f'  激活阈值     : {threshold}')
    print(f'  训练时间     : {bundle.get("trained_at")}')

    # 2. 读取测试集, 转成 (文本列表, multi-hot 真值矩阵)
    _, _, test_pairs, _ = load_splits()
    X_test_text, Y_test = to_xy(test_pairs, class_list)
    print(f'  测试集样本数 : {len(X_test_text)}')
    print('-' * 72)

    # 3. 用模型算出每条工单在 9 个标签上的独立概率 -> (N, 9) 矩阵
    probs = predict_proba_matrix(vectorizer, estimators, constants, X_test_text)

    # 4. 算指标(micro/macro/subset/hamming + 逐标签 P/R/F1)
    metrics = compute_metrics(Y_test, probs, class_list, threshold=threshold)
    print_metrics(metrics, title='02-rf 测试集结果')

    # 5. 单独把用户关心的四项抽出来再打一遍, 方便直接引用
    print('\n  核心指标汇总:')
    print(f'  准确率 (subset_accuracy) : {metrics["subset_accuracy"]}')
    print(f'  精准率 (micro_precision)  : {metrics["micro_precision"]}')
    print(f'  召回率 (micro_recall)     : {metrics["micro_recall"]}')
    print(f'  F1     (micro_f1)         : {metrics["micro_f1"]}')
    print(f'  F1     (macro_f1)         : {metrics["macro_f1"]}')


if __name__ == '__main__':
    main()
