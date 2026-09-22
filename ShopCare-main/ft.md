# ShopCare FastText 多标签 主流程 + 代码片段 + 通俗例子

> 任务：工单多标签分类，9 个标签，一条工单可同时打多个标签；核心方案：**FastText OVA (One-Vs-All)，loss=ova，每个标签独立 sigmoid，无 softmax** 和前面 02-RF 输出概率矩阵口径完全对齐，方便横向对比

## 流程 1：读取数据集，构造 0/1 二元标签矩阵 Y

**作用**：读取工单文本和标签列表，得到文本列表与多热 0/1 标签矩阵

```
#代码片段
train_pairs, dev_pairs, test_pairs, class_list = load_splits()
X_train_text, Y_train = to_xy(train_pairs, class_list)
```

✅例子： 工单`{"text":"不发货，要求退款","labels":["退款","物流"]}` `class_list = ["退款","物流","发票","质量","售后","活动","态度","配件","其他"]` `to_xy`生成一行`[1,1,0,0,0,0,0,0,0]`，多条样本堆叠得到 N×9 的 Y 矩阵

## 流程 2：清洗分词，转换成 FastText 专用语料文件

**作用**：把文本 + 标签矩阵，转为 FastText 要求的`__label__xxx`格式文本文件；一行可以带多个标签

```
#代码片段
def to_fasttext_line(text, labels, cfg, tokenizer):
    tags = ' '.join(cfg.label_prefix + lb for lb in labels)
    return f'{tags} {" ".join(tokenize_for_fasttext(text, tokenizer))}'
write_corpus(train_corpus, xy_to_pairs(X_train_text, Y_train, class_list), cfg, tokenizer)
```

✅例子： 原工单：`"商家一直不发货，我要求退款"`，标签`["退款","物流"]` 处理后语料行： `__label__退款 __label__物流 商家 一直 不 发货 我 要求 退款`

## 流程 3：FastText 训练（loss="ova"）

**作用**：读取语料文件训练，OVA 模式下**每个标签单独 sigmoid 二分类**，一个模型内部包含 9 套独立分类器，不用像 RF 那样手动循环训练 9 个模型

```
#代码片段
model = fasttext.train_supervised(
    input=corpus_path,
    lr=cfg.lr,
    dim=cfg.dim,
    loss=cfg.loss, # ova：每个标签独立sigmoid，不是softmax
)
```

✅例子：训练完成后，模型内部同时学习：是不是退款、是不是物流、是不是发票等 9 个独立判断。

## 流程 4：批量推理，输出 N×9 概率矩阵（和 RF 同口径）

**作用**：输入文本，模型一次性输出所有标签的 sigmoid 概率，拼成 N 行 9 列概率矩阵，**概率互相独立，总和不需要等于 1**

```
#代码片段
def predict_proba_matrix(model, texts, class_list, tokenizer, cfg):
    for row, text in enumerate(texts):
        tokens = tokenize_for_fasttext(text, tokenizer)
        labels, scores = model.predict(' '.join(tokens), k=k)
```

✅例子： 输入工单`"不发货，申请退款"`，得到概率向量： `[0.91,0.86,0.05,0.06,0.24,0.24,0.24,0.24,0.24]`

## 流程 5：验证集网格搜索双阈值

**作用**：在 dev 集搜索最优单标签阈值、全局拒识阈值，和 02-RF 共用同一套阈值逻辑

```
#代码片段
grid = grid_search_thresholds(probs_dev, Y_dev, class_list, target_reject_rate=cfg.target_reject_rate)
```

✅例子：找到`label_threshold=0.5`，概率≥0.5 判定标签为 1；低置信样本直接拒识，交给人工。

## 流程 6：测试集评估

**作用**：在测试集计算 Micro-F1、业务指标（自动分流率、拒识率），和 RF 指标对齐，方便对比

```
#代码片段
probs_test = predict_proba_matrix(model, X_test_text, class_list, tokenizer, cfg)
metrics = compute_metrics(Y_test, probs_test, class_list, threshold=cfg.label_threshold)
biz = compute_business_metrics(...)
```

## 流程 7：保存模型 + 元数据 json（重点！和 RF 不一样）

**作用**：保存`.bin`模型文件；额外保存`model_meta.json`，记录标签列表、分词方式、阈值等配置。

> 原因：FastText 的 bin 文件只存词向量和权重，**不带分词函数**，推理必须使用完全一致的分词规则，否则效果变差。

```
#代码片段
model.save_model(cfg.model_save_path)
meta = {'class_list': class_list, 'label_threshold': cfg.label_threshold, ...}
json.dump(meta, f, ensure_ascii=False, indent=2)
```

# 一句话总链路

读取工单→构建 0/1 二元标签矩阵 Y →文本清洗分词→生成`__label__`格式语料文件 →FastText OVA 训练（单模型内置 9 个独立 sigmoid 分类器）→推理输出 9 个独立概率→dev 搜索双阈值→测试集评估→保存 bin 模型 + meta 元数据。

# 和 02-RF 关键区别（一句话）

- RF：**代码层面手动循环训练 9 个独立 RF 模型**
- FastText(ova)：**只训练 1 个模型，模型内部自动完成 9 个独立 sigmoid 二分类**