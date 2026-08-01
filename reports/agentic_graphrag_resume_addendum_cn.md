# Agentic GraphRAG 简历成果增补

> 本文件仅记录 `codex/agentic-graphrag-mvp` 分支新增成果，供主项目后续合并
> 到统一简历文档。状态以本分支代码、测试和实测为准。

## 状态边界

### 已完成

- 基于四本真实 Ford 2026 手册的确定性领域图谱构建与 SQLite/FTS5 存储；
- 带车型硬过滤、实体匹配、1–2 hop 扩展、路径评分和证据引用的 Graph
  Retriever；
- TypedDict 显式状态图：Planner/Router、并行 Text/Visual/Table/Graph
  Retrieval、Evidence Critic、一次有界 replan、Answer/Synthesis 和最终
  Citation/Metadata Guard；
- 独立 Agentic CLI、FastAPI、Gradio tab 和逐节点 trace；
- 12 题 Gold Evidence/Graph Path 对照评测；
- 66 项全量测试通过。

### 正在开发

- 无。当前 MVP 分支功能和评测已闭环，等待主项目 review/merge。

### 规划项

- remote-control battery 与 12 V battery 的实体消歧；
- 跨 chunk 的组合路径检索，提升同页 Symbol + Warning 覆盖；
- graph-specific abstention 校准；
- 独立于开发集的更多车型/年份盲测；
- Agentic Qwen 路径的批量付费生成评测与按区域价格计算。现有 Qwen adapter
  已复用，但本次 Agentic 对照没有调用远程模型。

## 业务/工程问题

现有多模态 RAG 在单跳开发集上已经有较高准确率，直接增加多个 Agent 有
画蛇添足风险；同时 BM25 Evidence Pack 不能显式表达“操作步骤—安全警告—
部件—图片—来源页”之间的多跳关系，也无法审计每个 Agent 如何选择检索器。

工程目标因此定义为：

1. 不降低唯一车型隔离、引用和拒答安全边界；
2. 不重写已有 BM25、视觉、表格和 Qwen provider；
3. 增加可复现图谱、可引用 graph path 和有界状态编排；
4. 用 baseline 对照证明收益或代价，不因项目名称虚构提升。

## 技术决策与路线

- **确定性构图而非 LLM 抽取**：直接复用 30,131 个规范化元素和 12,236 个
  chunks，保证稳定 ID、重建一致性和 evidence provenance；
- **SQLite + FTS5 而非引入图数据库服务**：以最短时间完成可运行 MVP，
  避免部署和数据迁移成本；
- **semantic adjacency cache**：完整保存 APPLIES_TO/REFERENCES，但在线
  扩展只遍历语义关系，首个车型冷加载后缓存邻接表；
- **显式 TypedDict state graph**：状态、条件路由、并行检索、critic、
  retry 和 trace 都可测试；最多重规划一次，避免无限 Agent 循环；
- **Baseline-first safety**：Agentic Critic 必须同时满足必要模态支持和
  metadata 检查；失败回退抽取式回答或拒答；
- **离线对照评测**：同一 Gold 集比较 Baseline RAG、GraphRAG、
  Agentic GraphRAG，并保留所有失败记录。

## 个人落地动作

- 设计并实现 12 类领域节点、10 类关系及其证据保留规则；
- 实现唯一手册解析、FTS 实体匹配、双向二跳扩展、关系/节点/覆盖联合评分；
- 将 Text、Graph、Visual、Table 专业检索器接入并行执行节点；
- 实现 Evidence Critic、统一 Evidence Pack、引用/metadata guard 和一次
  replan；
- 扩展 Qwen/Responses adapter usage 记录，并在视觉路由中传递用户图片和
  evidence images；
- 新增 build/search/answer/evaluate/launch 脚本、独立 API 和 Gradio tab；
- 构造 Gold Evidence/Graph Path/route 数据集，运行三系统对照；
- 支持从独立 worktree 通过 `--data-root` 只读使用主项目 ignored data；
- 增加 Graph、Agentic、API、UI 和 pipeline 测试并运行完整 66-test suite。

## 可复现实验结果

### 图规模

- 29,797 nodes；
- 125,699 evidence-provenance edges；
- 4 个 Vehicle 分区；
- 3,792 Step、1,650 Warning/Caution、1,092 Specification、
  3,287 Image、516 Table；
- 0 跨车型路径。

### 12 题多跳开发集

| System | Evidence recall | Path accuracy | Route accuracy | Decision accuracy | Refusal accuracy | Metadata violations | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline RAG | 0.9167 | N/A | N/A | 1.0000 | 1.0000 | 0 | 17.6 ms |
| GraphRAG | 0.9167 | 0.6250 | N/A | 0.6667 | 0.0000 | 0 | 154.7 ms |
| Agentic GraphRAG | 0.9167 | 0.5714 | 1.0000 | 1.0000 | 1.0000 | 0 | 135.5 ms |

三者 citation faithfulness 均为 `1.0000`。Agentic retry rate 为
`0.3333`。本次使用离线抽取式生成，token usage 不可用，API cost 为
USD 0.00。

### 结果解释

GraphRAG 没有提高 Evidence recall；standalone GraphRAG 对四个无答案问题
全部误答。Agentic Critic 恢复了 baseline 的拒答安全性，但付出了约
118 ms 平均延迟，且 Gold path accuracy 只有 `0.5714`。因此简历只能描述
“实现、量化和守住安全边界”，不能描述为“显著提升准确率”。

### 失败案例

- F-150 charging：Symbol 路径未在同一二跳 path 覆盖 Warning；
- F-150 child-restraint：结构关系正确但页码未命中 Gold；
- Maverick remote battery：被 12 V battery 章节干扰；
- standalone GraphRAG 无拒答阈值，四个 no-answer case 全部误答。

## 精炼简历 bullet

- 基于 4 本 Ford 2026 真实车主手册的 30K+ 规范化元素，确定性构建
  **29,797 节点 / 125,699 边**的汽车手册知识图谱，覆盖 Vehicle、
  Component、Procedure/Step、Warning、Specification、Image/Table 和
  EvidencePage，并为全部节点/边保留车型、页码及源 element provenance。
- 实现带唯一车型 Metadata 硬过滤的 Graph Retriever，支持 FTS 实体匹配、
  1–2 hop 路径扩展、关系/实体联合评分和可引用 `graph_paths`；通过语义
  邻接缓存将 12 题 GraphRAG 平均延迟控制在 **154.7 ms**，保持
  **0 metadata violation**。
- 落地 TypedDict 显式 Agentic 状态图，按条件并行调用 Text/Visual/Table/
  Graph 专家，加入 Evidence Critic、最多一次 replan、Citation/Metadata
  Guard、Qwen/Responses 可选生成及完整执行 trace；新增独立 CLI、FastAPI
  和 Gradio 入口，完整 **66 tests passing**。
- 建立 12 题 Gold Evidence/Graph Path 对照：Agentic route accuracy、
  decision accuracy、refusal accuracy 和 citation faithfulness 均为
  **1.0000**，同时如实记录 Evidence recall 未超过 baseline、path accuracy
  为 **0.5714**及平均延迟从 **17.6 ms** 增至 **135.5 ms**。

## 面试 STAR / 技术路线

### Situation

原系统已经具备较高单跳检索与拒答指标。如果仅把检索流程拆成多个 Agent，
不仅缺少业务价值，还可能引入路由随机性、跨车型证据污染和额外延迟；但复杂
手册问题确实存在 Procedure、Step、Warning、Symbol、Specification 和图片
之间的多跳关系，现有 Evidence Pack 无法直接表达。

### Task

在不使用本地模型、不重写已有 RAG、不泄露本地 PDF/索引/API key 的约束下，
实现一个真实可运行、可追踪、可量化的 Agentic GraphRAG MVP，并用 baseline
对照判断是否真正提升。

### Action

先从规范化元素和 chunks 确定性构图，避免 LLM 抽取不可复现；用 SQLite/FTS5
完成实体检索和一至二跳扩展，并在检索前解析唯一手册。随后用 TypedDict
实现显式状态图，把现有 BM25、视觉、表格和 Graph Retriever 作为专业工具
并行调用；Evidence Critic 决定接受、一次扩大检索或拒答，最终 Guard 校验
所有引用和 metadata。最后建立包含 Gold Evidence、Gold path、route 和
no-answer 的 12 题开发集，分别运行 Baseline、GraphRAG 和 Agentic。

### Result

完成 29,797-node / 125,699-edge 图谱、独立 API/CLI/Gradio、逐节点 trace 和
66 项通过测试。Agentic 在 12 题上保持 1.0000 route/decision/refusal/
citation faithfulness 和零 metadata violation；但 Evidence recall 与
baseline 同为 0.9167，Gold path accuracy 为 0.5714，平均延迟为 135.5 ms。
这证明 Agentic Critic 有效阻止了 standalone GraphRAG 的过度回答，但当前
图谱还没有带来召回提升，后续优化方向应是实体消歧和组合路径，而非继续增加
Agent 数量。
