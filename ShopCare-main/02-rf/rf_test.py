"""
ShopCare 02-rf 测试集评估脚本(简洁版, 使用 tools 读取数据)

运行方式:
    python 02-rf/rf_test.py
"""

import os
import sys
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 把项目根目录加进 sys.path, 让 pickle 加载模型时能找到 tools 模块
_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_STAGE_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# 使用 tools 读取数据，避开手动解析 txt 文件的列名问题
from tools.ticket_data import load_splits, to_xy

# TODO 1. 加载模型包(tfidf / 9 个二分类器 / 常量 / 标签表 / 阈值)
bundle = joblib.load(os.path.join(_STAGE_DIR, 'save_models', 'rf_tfidf.joblib'))
vectorizer = bundle['vectorizer']
estimators = bundle['estimators']
constants = bundle['constants']
class_list = bundle['class_list']
threshold = bundle.get('label_threshold', 0.5)

# TODO 2. 使用 tools 加载测试集，拿到文本列表和 multi-hot 真值矩阵
_, _, test_pairs, _ = load_splits()
x_text, Y_true = to_xy(test_pairs, class_list)

# TODO 3. TF-IDF 向量化文本
X_test = vectorizer.transform(x_text)

# TODO 4. 9 个二分类器各出概率，再按阈值二值化
probs = np.column_stack([
    clf.predict_proba(X_test)[:, 1] if clf is not None else np.full(X_test.shape[0], c)
    for clf, c in zip(estimators, constants)
])
Y_pred = (probs >= threshold).astype(int)

# TODO 5. 直接使用 sklearn 评估，输出四项核心指标
print(f"准确率:{accuracy_score(Y_true, Y_pred)}")
print(f"精确率:{precision_score(Y_true, Y_pred, average='micro', zero_division=0)}")
print(f"召回率:{recall_score(Y_true, Y_pred, average='micro', zero_division=0)}")
print(f"f1分数:{f1_score(Y_true, Y_pred, average='micro', zero_division=0)}")