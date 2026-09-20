# ShopCare · 电商客服工单智能处理平台

把"用户反馈"变成"可派单的结构化决策": 一条工单进来, 一次性给出
**多标签分类 + 情感极性 + 优先级/SLA + 责任部门 + 建议回复草稿 + 是否需要人工复核**。

技术路线上做了三套**可自由切换的对照模型**(随机森林 / FastText / BERT+LoRA),
外加两档升级路径(自动升级 LLM 兜底 / LLM 直连), 便于直接对比"传统方法到什么程度、
深度模型值不值得上"。这也是本项目作为教学/答辩材料时最有说服力的部分。

> 项目名与包名(ShopCare)是占位用的, 可以整体替换, 不影响功能。

---

## 一、快速开始

```bash
# 1) 依赖(服务侧最小集; 模型侧按需装)
pip install -r backend/requirements.txt

# 2) 数据(仓库里已含一份小规模演示语料, 需要重新生成时执行)
python tools/generate_ticket_data.py
python tools/check_dataset.py

# 3) 训练对照组(任选; 仓库里 02/03 的模型产物已经训练好)
python 02-rf/rf_train.py
python 03-fasttext/ft_train.py
# BERT 需要先把 bert-base-chinese 放到 04-bert/bert-base-chinese/, 再:
python 04-bert/train_bert.py

# 4) 启动后端 + 前端
python backend/main.py
# 浏览器打开 http://127.0.0.1:8000/app   接口文档 /docs
```

**没有任何外部服务也能跑起来**: MySQL / Redis 连不上时, 后端会自动降级到内存存储与
进程内限流, 接口和页面全部可用(数据重启即丢), 降级情况会如实显示在 `/health` 与页面的
「系统状态」里。**唯一需要你自备的**是 BERT 的本地预训练模型目录(见下文)。

登录: 首次启动自动创建管理员 `admin / admin123`(改 `backend/.env` 可覆盖), 也可以直接在
登录页注册普通账号。

---

## 二、目录结构

```
ShopCare-main/
├── 01-data/                    数据层: 9 类标签定义 / 停用词 / train·dev·test / EDA
│   ├── class.txt               9 类标签(顺序即索引)
│   ├── label_meta.json         标签元数据唯一来源(中文名/部门/基础优先级/SLA)
│   └── data_format.md          数据格式与"换成真实数据"的说明
├── 02-rf/                      对照组 1: TF-IDF + OneVsRest 随机森林
├── 03-fasttext/                对照组 2: 多标签 FastText(loss=ova)
├── 04-bert/                    对照组 3: BERT + 手写 LoRA + 难例加权采样 + 拒识 + LLM 兜底
├── 08-sentiment/               情感极性(词典+规则, 支撑优先级)
├── 09-priority/                优先级与人工复核规则引擎
├── 10-reply-recommend/         回复话术推荐(模板 + 审批闸门)
├── 11-ticket-kg/               工单根因知识图谱(标签共现 + 风险组合 + 根因方向)
├── backend/                    FastAPI 后端
│   ├── main.py                 入口: 中间件 → 路由 → 异常处理 → 静态前端
│   ├── api/                    user / model / classify / ticket / dashboard 五组路由
│   ├── common/                 config·security·shop_utils·store·model_registry·pipeline
│   ├── middleware/             请求 ID / 限流 / 访问日志
│   ├── schema.sql              MySQL 建库建表
│   └── .env.example            环境变量模板
├── frontend/web/               前端: index.html + css/style.css + js/app.js(零构建零依赖)
├── tools/                      数据生成 / 数据体检 / 依赖自检 / 一键自检
└── docs/                       环境安装指南 / 标签规范 / 改造说明
```

延伸阅读(都在 `docs/` 下):

| 文档 | 什么时候看 |
|---|---|
| `docs/环境安装指南.md` | 第一次把项目跑起来: Python 环境、MySQL/Redis、BERT 预训练模型放哪、自检命令 |
| `docs/标签规范.md` | 想知道 9 个标签的边界与反例、或者要增删标签体系时 |
| `docs/改造说明.md` | 想知道这个项目相对 `szai8_tmf_project` 改了什么、为什么改 |

---

## 三、9 类标签

`invalid`(无效/骚扰)是**独占标签**, 不与其它标签共现; 其余 8 类可多标签共存。

| 标签 | 中文名 | 责任部门 | 基础优先级 | 响应 SLA |
|---|---|---|---|---|
| `logistics` | 物流配送 | 物流仓储部 | P2 | 24h |
| `quality` | 商品质量 | 品控/供应商 | P1 | 8h |
| `after_sale` | 退款退货 | 售后部 | P1 | 8h |
| `invoice` | 发票问题 | 财务部 | P2 | 24h |
| `price_promo` | 价格与优惠 | 运营部 | P2 | 24h |
| `payment_account` | 支付与账号 | 技术/风控 | P1 | 4h |
| `consult` | 售前与使用咨询 | 客服中心 | P2 | 24h |
| `service` | 服务态度 | 客服质检 | P1 | 4h |
| `invalid` | 无效与恶意 | 风控/归档 | P2 | 72h |

以 `01-data/label_meta.json` 为准(表里只是便于阅读)。改标签体系时**只需要改这一个文件 +
`class.txt`**, 后端、优先级引擎、话术模板、前端图例都会跟着变。

---

## 四、三套对照模型: 实测对比

在同一份演示语料(train 4000 / dev 600 / test 600)上的结果:

| | 随机森林 (TF-IDF) | FastText | BERT + LoRA |
|---|---|---|---|
| 产物 | `02-rf/save_models/rf_tfidf.joblib` | `03-fasttext/save_models/*.bin` | `04-bert/save_models/*.pt` |
| 体积 | ~12.3 MB | ~19.3 MB | 视 LoRA 配置, 通常 10~60 MB |
| 训练耗时 | 秒级 | 3.7 s | 分钟级(CPU 更久) |
| 测试集 Micro-F1 | **1.000** | **1.000** | 未训练(见下) |
| 平均置信度 | 0.969 | 0.999 | - |
| 语料外新说法 | 容易漏判/拒识 | 较好 | 最好 |

> **这两组 1.000 不是"模型很强", 而是"语料太规整"**。演示语料是模板生成的, 同分布切分后
> 训练集与测试集的措辞高度重合, 所以字面特征就足以满分。换成真实工单后指标一定会掉下来,
> 那时这张表才真正有比较意义 —— 这一点已写进各模型脚本的注释里, 避免被误读。

**BERT 为什么是"未训练"状态**: 本项目不联网下载模型。请自行把 `bert-base-chinese`
(需含 `config.json` 与权重文件)放到 `04-bert/bert-base-chinese/`, 或用环境变量
`BERT_MODEL_DIR` 指向已有目录, 然后运行 `python 04-bert/train_bert.py`。
在它就绪之前, 后端会自动把它标为"未就绪", 请求 BERT 时按顺序降级到其它可用模型,
并在响应里用 `fallback_from` 说明 —— 不静默替换。

**页面上的模型选择**: `RF` / `FastText` / `BERT` / `自动` / `LLM 直连`。
「自动」= 先用默认模型, 拒识时升级 LLM 兜底; 「LLM 直连」= 完全不走本地模型(需 API Key)。
工作台的「三模型对比」会用**同一条工单**依次跑各可用模型并排展示, 这是做对照实验最直观的入口。

---

## 五、核心机制

**1. 双阈值拒识(不硬猜)**
9 个标签各自独立 sigmoid(不做 softmax —— 多标签不是互斥的), 然后两道闸门:
单标签激活阈值 `LABEL_THRESHOLD`(默认 0.5)与全局平均置信度 `GLOBAL_THRESHOLD`(默认 0.8)。
都不满足就**拒识**, 并给出机器可读的原因码(`no_label_activated` / `low_confidence` 等)。

> 阈值提示: 02/03 两个阶段在训练时做过阈值搜索, 最优点在 `0.3 / 0.6` 附近;
> 后端默认的 `0.5 / 0.8` 更保守(拒识更多, 更安全)。想复现训练期最优口径, 在
> `backend/.env` 里设 `LABEL_THRESHOLD=0.3`、`GLOBAL_THRESHOLD=0.6` 即可。

**2. 三级兜底链路**
本地模型 → LLM 兜底(默认**关闭**, 避免不知情地产生 API 费用) → 人工复核队列。
每一级都在响应里可见(`resolved_by` = `model` / `llm` / `human` / `none`)。

**3. 难例动态加权采样(04-bert)**
样本权重 `w = 1 + α·(标签数-1)/2 + β·漏判率 + γ·(1-平均真实标签概率)`, 默认
α=0.5 / β=1.0 / γ=0.8, 上限 5.0 —— 标签越多、越容易被漏判、模型越没把握的样本, 权重越高。

**4. 手写 LoRA(不依赖 peft)**
只注入 `query` / `value` 两个投影矩阵, `B` 全零初始化(保证训练起点等价于原模型),
冻结主干参数, 只训练 LoRA 与多标签头。参数量报告在训练日志里。

**5. 优先级是规则算出来的, 不是训出来的**
`base_priority(最急标签) × 情感系数 + 升级信号(消协/起诉/曝光等) + 多标签加分`,
≥6.0 判 P0 / ≥3.0 判 P1 / 否则 P2。规则可解释、可审计、随时可调, 比再训一个模型更划算。

**6. 话术有审批闸门**
模板生成的回复草稿在以下情况强制标记 `needs_approval=true`: P0 工单、涉及资金类标签、
命中投诉/曝光关键词、或模型结果被拒识。**模型没把握时, 话术绝不标记为可直接发送。**

---

## 六、接口一览(统一前缀 `/api/v1`)

| 方法 | 路径 | 认证 | 说明 |
|---|---|:--:|---|
| GET | `/health` | 否 | 健康检查; 存储降级、模型就绪、业务模块、配置告警都在这里 |
| POST | `/user/register` | 否 | 注册(可用 `ALLOW_REGISTER=false` 关闭) |
| POST | `/user/login` | 否 | 登录, 返回 JWT |
| POST | `/user/logout` | 是 | 登出(jti 写入黑名单) |
| GET | `/user/profile` | 是 | 当前用户 |
| GET | `/models` | 是 | 模型清单与可用性 |
| GET | `/labels` | 是 | 9 类标签元数据 |
| POST | `/classify` | 是 | **核心**: 工单 → 全部决策字段 |
| POST | `/classify/batch` | 是 | 批量分类(≤50 条) |
| POST | `/sentiment` | 是 | 单独情感分析 |
| POST | `/priority` | 是 | 单独优先级判定 |
| POST | `/reply` | 是 | 单独话术推荐 |
| GET/POST/PATCH | `/tickets`… | 是 | 工单查询/建单/更新 |
| POST | `/feedback` | 是 | 人工复核回写(修正标签) |
| GET | `/reviews` | 是 | 复核记录 |
| GET | `/dashboard/overview` | 是 | 看板聚合 |
| GET | `/dashboard/review_queue` | 是 | 待复核队列 |
| GET | `/dashboard/audit` | 管理员 | 审计日志 |

统一响应结构: `{"code": 0, "message": "ok", "data": {...}}`;
失败时 `code` 为业务码(40001 参数 / 40101 未登录 / 40301 无权限 / 40401 不存在 /
42901 限流 / 50002 模型不可用), HTTP 状态码同时照常使用。

`POST /classify` 请求:

```json
{ "text": "快递到广州十天了还没动静, 客服也不回复, 我要退款",
  "model": "fasttext", "use_llm_fallback": false, "top_k": 3, "save": true }
```

响应(节选, 真实响应还含 `confidences` / `top_k_scores` / `suggested_reply` / `trace` 等):

```json
{ "code": 0, "message": "建单并分类完成",
  "data": { "ticket_id": "SC20260920-7127", "model_used": "fasttext",
            "labels": [{"label": "logistics", "cn": "物流配送", "dept": "物流仓储部",
                        "base_priority": "P1", "score": 0.9864}],
            "rejected": false, "avg_confidence": 0.8182, "resolved_by": "model",
            "sentiment": {"label": "negative", "cn": "负面", "score": -0.8483},
            "priority": {"priority": "P0", "cn": "紧急", "sla_hours": 1},
            "dept": "售后部", "latency_ms": 1151.22 } }
```

---

## 七、数据库

**MySQL**(`backend/schema.sql`, 4 张表: `users` / `tickets` / `reviews` / `audit_logs`)。
`DB_AUTO_CREATE=true`(默认)时后端启动会自动建库建表; 手动执行也可以:

```bash
mysql -u root -p < backend/schema.sql
```

**Redis** 用于两件事(连不上就退化为进程内实现):

| Key | 用途 | TTL |
|---|---|---|
| `shopcare:ratelimit:{身份}:{窗口}` | 固定窗口限流 | 60s |
| `shopcare:cache:classify:{md5(文本+模型+阈值+top_k)}` | 分类结果缓存 | 60s |
| `shopcare:jwt:blacklist:{jti}` | 登出令牌黑名单 | 令牌剩余有效期 |

缓存 key 里带上了**所有影响结果的参数**: 只带文本的话, 改了阈值或换了模型就会读到旧结果。

**环境变量**: 全部走环境变量, 模板见 `backend/.env.example`。后端启动时会读
`backend/.env`(已存在的真实环境变量优先), 也可以直接 `export`。生产环境务必替换
`JWT_SECRET`(≥32 字节)与 `ADMIN_PASSWORD` —— 仍是默认值时 `/health` 会给出告警。

---

## 八、自检

```bash
python tools/verify_all_phases.py            # 全量: 各阶段自测 + 端到端接口自测
python tools/verify_all_phases.py --api      # 只跑接口自测(最快, 不需要 MySQL/Redis)
python tools/verify_all_phases.py --fast     # 跳过需要加载模型的慢阶段
python tools/check_env.py                    # 依赖与模型产物自检
```

每个阶段都在**独立子进程**里跑(因为 02/03/04 三个目录都叫 `config.py`, 同进程必然串味),
端到端自测在进程内用 TestClient 走一遍真实接口, 覆盖 23 个步骤: 登录鉴权、参数校验、
分类落库、缓存命中、批量、情感/优先级/话术、工单 CRUD、复核回写、看板、前端页面、登出黑名单。

---

## 九、常见问题

**Q: 没装 MySQL / Redis 能跑吗?**
能。会自动降级到内存存储 + 进程内限流, 页面与接口全部可用, 只是重启后数据丢失。
`/health` 和页面「系统状态」会明确标出 `storage_mode: memory` 与 `degraded: true`。

**Q: 请求 BERT 但没训练, 会报错吗?**
不会。`/models` 里它显示为"未就绪"; 真的请求它时会**按顺序降级**到可用模型, 响应里带
`fallback_from: "bert"` 说明发生了降级。三个模型都不可用时才返回 503 + 业务码 50002。

**Q: 前端为什么不用 Chart.js?**
设计文档里写的是 CDN 引入, 但这个项目要求**完全离线可用**。所以图表用 canvas 手绘
(横向柱 / 环形 / 折线, 约 120 行), 整个前端零外部依赖、零构建。

**Q: 怎么换成我自己的真实数据?**
按 `01-data/data_format.md` 的格式(`文本\t标签1,标签2`, UTF-8, `\n` 分隔)替换
`train/dev/test.txt`, 然后 `python tools/check_dataset.py` 体检, 再重训模型即可。
注意 `invalid` 必须独占, 不要与其它标签共现。

**Q: 指标为什么这么高(1.000)?**
因为演示语料是模板生成的, 措辞高度重复。见上文"三套对照模型"的说明 —— 换成真实数据后
才有比较价值。

**Q: 页面上改了模型/阈值, 会影响已落库的工单吗?**
不会。工单落库的是**当次决策的快照**(模型名、置信度、阈值结果都在里面), 之后改配置
不会回溯修改历史工单。

---

## 十、已知局限(诚实清单)

- **演示语料是生成的**: 指标虚高, 不代表真实场景表现; 真实数据上需要重新调阈值与采样权重。
- **没有做单元测试框架**: 每个模块自带 `__main__` 自测 + 一键自检脚本, 够用但不够正式。
- **限流是固定窗口**: 实现简单可预测, 但对窗口边界的突发流量不友好(滑动窗口更准, 也更复杂)。
- **限流与缓存依赖单进程**: Redis 不可用时退化为进程内实现, 多 worker 部署时计数不共享。
- **审计日志只写不查重**: `audit_logs` 记录齐全, 但没有做定期归档与脱敏清理。
- **知识图谱只用了共现**: 目前是 9 节点的共现图 + 规则风险组合, 没有做多跳路径推理。
- **`11-ticket-kg` 的共现强度受生成语料影响**: 当前最强的共现是"售前咨询 + 其它"这类
  数据生成带来的结构, 换成真实数据后会自然变化。
- **前端未做权限分级**: 任何人都能看全部工单与看板(后端已预留 `admin_user` 依赖, 只差分工)。
- **BERT 阶段未实测**: 代码与配置自检通过, 但本机没有预训练模型, 尚未跑通端到端训练。

---

## 十一、下一步可以做什么

1. 用真实工单替换演示语料, 重训三套模型并重跑阈值搜索, 得到真正有对比意义的指标表;
2. 把 04-bert 跑通, 用 `model2dev_utils.py` 产出学术指标 + 业务指标(拒识率/兜底率/复核率);
3. 消融实验矩阵: 有/无难例采样、有/无 LoRA、有/无拒识, 量化每个设计的收益;
4. 第二轮优化: INT8 量化 / 蒸馏 BiLSTM / 剪枝(对应设计文档的 05/06/07);
5. 生产化: 换 `DBUtils.PooledDB` 或 asyncmy、限流改滑动窗口、审计日志归档、前端权限分级。