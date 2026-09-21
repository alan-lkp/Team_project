"""
ShopCare 03-fasttext 测试集评估脚本(简洁版, 使用 tools 读取数据)

运行方式:
    python 03-fasttext/ft_test.py
"""

import os
import sys
import numpy as np
import fasttext
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 把项目根目录加进 sys.path
_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_STAGE_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# 使用 tools 读取数据，避开手动解析 txt 文件的列名问题
from tools.ticket_data import load_splits, to_xy
from config import FTConfig  # 导入 FastText 的配置类

# TODO 1. 提前创建配置对象，加载训练好的 FastText 模型
config = FTConfig()
model = fasttext.load_model(config.model_save_path)

# 获取配置里的标签列表和阈值
class_list = config.class_list
threshold = config.label_threshold
prefix_len = len(config.label_prefix)

# TODO 2. 使用 tools 加载测试集，拿到文本列表和 multi-hot 真值矩阵
_, _, test_pairs, _ = load_splits()
x_text, Y_true = to_xy(test_pairs, class_list)

# TODO 3. 获取类别到索引的映射，方便后续把预测结果转成矩阵
label2id = {c: i for i, c in enumerate(class_list)}

# TODO 4. 遍历文本，进行预测并按阈值二值化
Y_pred = np.zeros_like(Y_true)
for i, text in enumerate(x_text):
    # k=-1 表示输出所有标签及其概率
    labels, probs = model.predict(text, k=-1)
    for lb, prob in zip(labels, probs):
        lb = lb[prefix_len:]  # 去掉 '__label__' 前缀
        if lb in label2id and prob >= threshold:
            Y_pred[i][label2id[lb]] = 1

# TODO 5. 直接使用 sklearn 评估，输出四项核心指标
print(f"准确率:{accuracy_score(Y_true, Y_pred)}")
print(f"精确率:{precision_score(Y_true, Y_pred, average='micro', zero_division=0)}")
print(f"召回率:{recall_score(Y_true, Y_pred, average='micro', zero_division=0)}")
print(f"f1分数:{f1_score(Y_true, Y_pred, average='micro', zero_division=0)}")