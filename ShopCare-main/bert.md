# ShopCare BERT‑LoRA 多标签 主流程 + 代码片段 + 通俗例子

> 任务：工单多标签分类，9 个标签，单条工单可同时拥有多个标签；核心方案：**BERT 主干 + LoRA 低秩微调，输出头用 BCEWithLogitsLoss，每个标签独立 sigmoid，不使用 softmax** 输出 N×C 概率矩阵，与 02‑RF、03‑FastText 口径完全对齐，便于横向对比

## 流程 1：读取数据集，构造 multi‑hot 0/1 二元标签矩阵 Y

**作用**：读取工单文本 + 标签，生成文本列表与 N 行 9 列的二元标签矩阵

```
#代码片段
train_dataset = TicketDataset(config.train_path, tokenizer, config.class2id, config.max_len)
X_train_text = train_dataset.texts
Y_train = train_dataset.label_matrix()
```

✅例子： 工单`{"text":"商家一直不发货，我要求退款","labels":["退款","物流"]}` `class_list = ["退款","物流","发票","质量","售后","活动","态度","配件","其他"]` 得到该行标签向量：`[1,1,0,0,0,0,0,0,0]`，多条样本堆叠得到`Y_train(N,9)`。

> 区别 RF/FastText：BERT 需要**子词 tokenizer**，文本转为`input_ids、attention_mask`张量，不是 TF‑IDF 也不是空格分词字符串。

## 流程 2：构建 BERT‑LoRA 多标签模型

**作用**：加载本地 BERT 主干，注入 LoRA，冻结主干大部分参数，只训练 LoRA 旁路 + 多标签分类头；输出 logits（未 sigmoid）

```
#代码片段
model = BertMultiLabelClassifier(cfg)
# 前向输出logits；predict_proba内部执行sigmoid得到各标签独立概率
logits = model(input_ids, attention_mask)
probs = model.predict_proba(input_ids, attention_mask)
```

✅例子： 输入工单文本：`"商家一直不发货，我要求退款"` 文本被 tokenizer 转为`input_ids`张量送入 BERT；经过池化得到句向量，经过分类头输出 9 维 logits；`predict_proba`做 sigmoid，输出概率向量： `[0.91,0.86,0.04,0.06,0.23,0.23,0.23,0.23,0.23]`

> 9 个标签各自独立概率，**没有 softmax，概率之和不强制等于 1**。

## 流程 3：损失函数，适配多标签长尾

**作用**：使用`BCEWithLogitsLoss`，搭配`pos_weight`处理标签样本不均衡。每个标签当做独立二分类任务。

```
#代码片段
pos_weight = compute_pos_weight(train_dataset)
criterion = build_loss(cfg, pos_weight)
loss = criterion(logits, labels)
```

✅例子： “发票” 标签训练样本很少，`pos_weight`会变大，该标签预测错误会产生更大损失，强迫模型学习少样本标签。

## 流程 4（创新点）：难例动态加权采样

**作用**：每轮训练结束，用当前模型对训练集打分，给多诉求、容易漏判的难例更高采样权重，下一轮多抽难例训练。

```
#代码片段
sampler = HardExampleSampler(cfg)
# epoch>1时刷新权重，生成新采样器
sampler, weights = hard_sampler.refresh(model, train_dataset, cfg)
train_loader = DataLoader(train_dataset, sampler=sampler,...)
```

✅例子：

- 简单样本：`"商品开裂要退货"`（仅质量标签）权重 = 1.0
- 难例样本：`"不发货要退款，客服还态度差"`（退款 + 物流 + 态度）权重被拉高到 3.2，训练时更容易被抽到。

## 流程 5：验证集网格搜索双阈值

**作用**：在 dev 集搜索最优单标签激活阈值、全局拒识阈值，和 RF/FastText 复用同一套`grid_search_thresholds`。

```
#代码片段
grid = grid_search_thresholds(probs_dev, Y_dev, class_list, target_reject_rate=cfg.target_reject_rate)
```

✅例子：得到`label_threshold=0.5`，`global_threshold=0.8`； 单标签概率≥0.5 视为激活标签；激活标签平均置信度低于 0.8 则工单拒识，交给人工或 LLM 兜底。

## 流程 6：测试集评估

**作用**：测试集推理得到 N×9 概率矩阵，计算学术指标 (Micro‑F1、Macro‑F1) + 业务指标（自动分流率、拒识率）。

```
#代码片段
probs_test, y_true_test, _ = run_inference(best_model, test_loader, cfg)
metrics = compute_metrics(y_true_test, probs_test, cfg.class_list, threshold=cfg.label_threshold)
biz = compute_business_metrics(probs_test, y_true_test, cfg, cfg.class_list)
```

## 流程 7：保存 LoRA 增量权重 + 元信息

**作用**：不保存完整 BERT 主干，仅保存几十 KB 的 LoRA 适配器 + 分类头；元信息记录标签列表、阈值、分词配置。

```
#代码片段
model.save(cfg.model_save_path,extra=meta)
# meta内保存class_list、threshold、max_len等推理必需信息
```

✅说明：推理时重新加载本地原始 BERT，再把 LoRA 增量加载进去，节省存储空间。

# 一句话完整链路

读取工单 →构造 multi‑hot 二元标签矩阵 →文本转为 bert 的 token 张量 →BERT+LoRA 模型训练，BCE‑logits 损失做多标签独立二分类 →难例动态加权采样提升复杂样本学习 →dev 集网格搜索双阈值 →测试集学术 + 业务指标评估 →保存 LoRA 增量权重与元数据。

# 02‑RF /03‑FastText /04‑BERT 关键差异简表

表格

| 模型         | 多标签实现方式                        | 特征输入             | 训练特点                                                     |
| ------------ | ------------------------------------- | -------------------- | ------------------------------------------------------------ |
| RF OvR       | **代码层面手动训练 9 个独立随机森林** | TF‑IDF 稀疏矩阵      | 树模型，可输出单标签特征重要性                               |
| FastText OVA | **1 个模型内部实现 9 个独立 sigmoid** | 分词空格分隔文本文件 | 浅层词向量，训练速度快                                       |
| BERT‑LoRA    | **1 个模型内部实现 9 个独立 sigmoid** | sub‑word token 序列  | 预训练语言模型，LoRA 微调；增加难例采样创新点；可接入 LLM 拒识兜底 |

> 三者**推理输出都是 (N,9) 独立概率矩阵，共用同一套双阈值拒识与评估工具**，保证可以横向公平对比。