# ShopCare 多标签 OvR 随机森林 主流程 + 对应代码 + 通俗例子

> 任务：工单多标签分类，9 个标签，一条工单可同时打多个标签（退款、物流可同时存在） 核心方案：**9 个独立二分类随机森林（One-vs-Rest）**

## 流程 1：读取数据集，构造二元标签矩阵 Y

**作用**：读取工单文本 + 原始标签列表，转换成模型能识别的文本列表和 0/1 标签矩阵

```
# 代码片段
train_pairs, dev_pairs, test_pairs, class_list = load_splits()
X_train_text, Y_train = to_xy(train_pairs, class_list)
```

✅例子： 工单`{"text":"不发货，要求退款","labels":["退款","物流"]}` `class_list = ["退款","物流","发票","质量","售后","活动","态度","配件","其他"]` `to_xy`生成一行：`[1,1,0,0,0,0,0,0,0]` 多条样本堆叠得到 Y 矩阵（N 行 9 列，行 = 样本，列 = 标签）

## 流程 2：文本清洗 + TF-IDF 向量化

**作用**：原始文本清洗，转为 TF-IDF 稀疏特征矩阵；**仅训练集 fit，验证 / 测试集只 transform，防止数据泄露**

```
# 代码片段
vectorizer = build_vectorizer(cfg, mode, _tok)
X_train = vectorizer.fit_transform(clean_texts(X_train_text))
```

✅例子：文本`"商家一直不发货，我要求退款"` →清洗→TF-IDF 稀疏向量，作为 RF 输入特征 X

## 流程 3：逐标签训练 OvR 随机森林（train_ovr）

**作用**：循环标签矩阵**每一列**，每一列单独训练 1 个二分类随机森林；处理退化边界

```
# 代码片段
for i, name in enumerate(cfg.class_list):
    y = Y[:, i]  #取出第i列，该标签全部样本0/1标签
    # 退化判断：该列全0或全1，不训练模型，使用拉普拉斯常数概率
    if n_pos == 0 or n_pos == n_all:
        prior = (n_pos + 1) / (n_all + 2)
        estimators.append(None)
    else:
        clf = RandomForestClassifier(...)
        clf.fit(X, y)
        estimators.append(clf)
```

✅例子：

- 第 0 列（退款标签）：训练 RF，判断工单是不是退款问题
- 第 1 列（物流标签）：训练另一个独立 RF，判断工单是不是物流问题 一共 9 个独立 RF。**每个模型互不干扰**

## 流程 4：预测，输出 N×9 概率矩阵（predict_proba_matrix）

**作用**：输入文本，9 个模型各自输出`P(标签=1)`，拼接成概率矩阵；**无 softmax**

```
# 代码片段
for i, (clf, const) in enumerate(zip(estimators, constants)):
    if clf is None:
        probs[:, i] = const['prob']
    else:
        probs[:, i] = clf.predict_proba(X)[:, classes.index(1)]
```

✅例子： 输入工单`"不发货，申请退款"`，得到概率向量： `[0.92,0.87,0.04,0.07,0.25,0.25,0.25,0.25,0.25]`

> 9 个概率相互独立，总和不需要等于 1

## 流程 5：验证集网格搜索双阈值

**作用**：在 dev 集搜索最优单标签阈值、全局拒识阈值，平衡自动识别率和拒识率

```
# 代码片段
grid = grid_search_thresholds(probs_dev, Y_dev, class_list, target_reject_rate=cfg.target_reject_rate)
```

✅例子：找到`label_threshold=0.5`，概率≥0.5 判定标签为 1；低于阈值不打标签，部分样本直接拒识人工处理

## 流程 6：测试集评估

**作用**：在未见过的测试集计算 Micro-F1、业务指标（自动分流率、拒识率）

```
# 代码片段
probs_test = predict_proba_matrix(vectorizer, estimators, constants, X_test_text)
metrics = compute_metrics(Y_test, probs_test, class_list, threshold=cfg.label_threshold)
biz = compute_business_metrics(...)
```

## 流程 7：保存模型与指标文件

**作用**：保存 TF-IDF 向量化器、9 个 RF 模型、阈值、特征重要性，用于后续推理

```
# 代码片段
bundle = {
    'vectorizer': vectorizer,
    'estimators': estimators,
    'constants': constants,
    'label_threshold': cfg.label_threshold
}
joblib.dump(bundle, cfg.model_save_path, compress=3)
```

# 一句话总链路

读取工单→构建 0/1 二元标签矩阵 Y →文本清洗 + TF-IDF 生成特征 X →**9 个独立二分类 RF 分别训练**→预测输出 9 个独立概率→dev 搜索双阈值→测试集评估→保存整套模型。