# 数据格式规范（01-data）

## 1. 工单分类数据（train.txt / dev.txt / test.txt）

每一行一条工单，`文本` 与 `标签` 之间用**制表符 `\t`** 分隔，多个标签用**英文逗号 `,`** 连接：

```
快递到广州十天了还没动静，客服也不回复	logistics,service
收到的电饭锅是坏的，我要退款，发票也一直没开	quality,after_sale,invoice
```

约定：

- 标签名一律使用 `class.txt` 中的英文名（小写 + 下划线），按 `class.txt` 行号升序排列；
- 标签数量 1~N 个（本项目 9 类多标签，真实工单集中在 1~3 个）；
- 空行、纯空白行会被 `tools/check_dataset.py` 判为错误；
- 文件编码 UTF-8，换行符 `\n`（Windows 下勿用 `\r\n`，否则最后一个标签会带上 `\r`）。

## 2. 类别定义（class.txt）

每行一个类别名，**行号即类别索引**（第 1 行索引为 0）。

| 索引 | 标签 | 中文名 |
|------|------|--------|
| 0 | logistics | 物流配送 |
| 1 | quality | 商品质量 |
| 2 | after_sale | 退款退货 |
| 3 | invoice | 发票问题 |
| 4 | price_promo | 价格与优惠 |
| 5 | payment_account | 支付与账号 |
| 6 | consult | 售前与使用咨询 |
| 7 | service | 服务态度 |
| 8 | invalid | 无效与恶意 |

各类别的责任部门、优先级权重与示例见 `docs/标签规范.md`。

## 3. 情感极性数据（sentiment_class.txt）

辅助任务标签，顺序固定为 `positive / neutral / negative`（索引 0/1/2）。

当前版本的情感分析由 `08-sentiment/` 的词典 + 规则引擎完成，**不依赖训练数据**；
若后续要训练情感分类头（方案文档中的双任务联合训练），再按 `文本\tpositive`
的单标签格式补充数据即可。

## 4. 如何替换成你自己的真实数据

1. 把原始数据整理为 `文本\t标签1,标签2` 格式（UTF-8，`\t` 分隔）；
2. 若只有**单标签**（绝大多数公开电商评论数据集都是单标签），用
   `python tools/check_dataset.py --upgrade 原始文件` 做「单标签 + 关键词规则 → 多标签」升维，
   它会打印升维前后的标签分布与共现矩阵，便于人工校验；
3. `python tools/check_dataset.py` 做体检（格式 / 标签合法性 / 分布 / 长度）；
4. `python 01-data/data_eda.py` 看标签分布、共现关系与文本长度分布。

## 5. 数据规模参考

| 数据集 | 当前规模 | 用途 |
|--------|----------|------|
| train.txt | 100000 条 | 训练 |
| dev.txt | 8000 条 | 阈值标定 / 早停 / 拒识阈值调参 |
| test.txt | 8000 条 | 最终指标汇报 |

当前数据由 `tools/build_dataset_v2.py` 生成，**只用于把链路跑通**；
真实效果请以你自己的业务数据为准。真实落地建议训练集不少于 1 万条。

### 5.1 这份语料是怎么造的（以及为什么不是 `generate_ticket_data.py`）

早期版本用 `tools/generate_ticket_data.py` 生成（123 条模板，各 split 独立随机采样）。
那份数据有个致命问题：**整句查重为 0，但子句查重有 96.6% 的 test 子句能在 train 里
原样找到** —— 每条 test 文本只是那几千条模板子句的重新排列组合，模型学到的是
「子句 → 标签」的查表规则，所以 Micro-F1 必然是 1.000，指标没有任何区分度。

现在改为 `tools/build_dataset_v2.py`，三层修复：

1. **扩池**：模板从 123 条扩到 540 条（每个标签 60 条，见 `tools/_tpl_part{1,2,3}.py`）；
2. **家族分组切分**：一条模板 + 它的同义变体 = 一个 family，整体只进一个 split，
   避免「同一模板的不同变体」分落 train/test；
3. **槽位隔离**：`tools/ticket_vocab_v2.py` 把商品/城市/天数按 split 切开，
   test 用的商品名和城市名 train 里一个都没有。

结果：test 子句在 train 中原样出现的比例 **0.0000%**（修复前 96.57%）。
这些口径都能用 `python tools/build_dataset_v2.py` 重跑复现，
最新的切分清单与泄漏检测报告见 `01-data/split_manifest.json`。
