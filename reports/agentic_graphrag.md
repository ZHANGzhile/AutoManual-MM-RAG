# Agentic GraphRAG 技术报告

## 目标与边界

本扩展的目标不是用更多 Agent 替换已经较准确的 BM25 RAG，而是在保留车型
隔离、Evidence Pack、引用检查和拒答机制的前提下，增加可审计的汽车手册
领域图谱、多跳路径检索和显式状态编排。

实现不使用本地大模型。默认评测完全离线；可选生成继续复用现有 Responses
和 Qwen3-VL-Flash provider adapter，密钥只从忽略的 `.env` 读取。原始 PDF、
MinerU 大型产物、运行时索引和虚拟环境均未提交。

## 确定性领域图谱

`src/automanual_rag/graphrag.py` 从现有 `elements.jsonl` 和
`chunks.jsonl` 确定性构图，不调用 LLM。稳定 ID 由文档、节点类型和源证据
生成；相同输入可重复构建相同节点和边。

真实四手册构建结果：

| 节点类型 | 数量 |
|---|---:|
| Vehicle | 4 |
| Section | 6,233 |
| Component | 6,067 |
| Symbol | 721 |
| Procedure | 4,299 |
| Step | 3,792 |
| Warning | 1,648 |
| Caution | 2 |
| Specification | 1,092 |
| Image | 3,287 |
| Table | 516 |
| EvidencePage | 2,136 |
| **合计** | **29,797** |

| 边类型 | 数量 |
|---|---:|
| APPLIES_TO | 29,793 |
| LOCATED_IN | 22,307 |
| SYMBOL_MEANS | 721 |
| EXPLAINED_BY | 10,366 |
| REQUIRES_STEP | 3,792 |
| NEXT_STEP | 2,392 |
| HAS_WARNING | 5,322 |
| HAS_SPECIFICATION | 2,700 |
| ILLUSTRATED_BY | 5,076 |
| REFERENCES | 43,230 |
| **合计** | **125,699** |

每个节点和边均保存 `doc_id`、品牌、车型、年份、地区、语言、手册类型、
物理 PDF 页码和源 element/chunk ID。图索引使用 SQLite + FTS5，生成文件
`data/indexes/manual_graph.sqlite3` 约 94 MB，受 `.gitignore` 保护。

构图规则包括：

- section path 生成层级 Section，并通过 `LOCATED_IN` 保留层级；
- action section 或 `steps` chunk 生成 Procedure；
- 编号内容生成 Step，并通过 `REQUIRES_STEP`、`NEXT_STEP` 保留顺序；
- Warning/Caution chunk 生成安全节点并连接 Section、Component、Procedure；
- specification/capacity/torque/pressure 等证据生成 Specification；
- Symbols/Indicators 上下文生成 Symbol 和 `SYMBOL_MEANS`；
- 原始 image/table 元素生成 Image/Table，并保留资产路径；
- 所有语义节点通过 `REFERENCES` 回指 EvidencePage，通过
  `APPLIES_TO` 回指唯一 Vehicle。

## Graph Retriever

`GraphRetriever` 的在线流程为：

1. 根据 `doc_id`，或 model + year 等字段解析唯一手册；
2. 在该手册分区内使用 FTS5 匹配实体；
3. 按节点类型、关键词覆盖和 FTS 分数计算实体分；
4. 在语义边子图中进行一跳或二跳双向扩展；
5. 使用关系权重、边置信度、目标节点类型和查询覆盖计算路径分；
6. 返回带节点、边、页码、element ID、来源文本和引用标签的
   `graph_paths`。

`APPLIES_TO` 和 `REFERENCES` 仍保留在完整图中供审计，但不会在线扩展，
避免通过 Vehicle/EvidencePage 扩散到整本手册。每个文档的语义邻接表在
首次访问时缓存；真实查询首个车型为冷加载，后续同车型查询复用缓存。

Retriever 在路径扩展前执行 metadata 硬过滤，且路径中的所有节点和边必须
属于同一个 `doc_id`。缺少唯一车型上下文时直接报错。

## 显式 Agentic 状态图

`src/automanual_rag/agentic.py` 使用 `AgenticState(TypedDict)` 保存：

- query、固定 filters 和可选 image path；
- initial route、当前 route 和 retry count；
- text/visual/table/graph 专家结果；
- 统一 Evidence Pack、critic decision 和 graph paths；
- 最终回答、citation/metadata guard；
- 逐节点 trace、latency、token usage 和 cost 字段。

状态节点及条件流转：

1. **Planner/Router**：确定 text、visual、table、graph 路由和子查询；
2. **Text/Visual/Table/Graph Retrieval**：被选中的专家通过线程池并行运行；
3. **Parallel Retrieval Join**：汇总结果和 graph path IDs；
4. **Evidence Critic**：检查必要模态是否有支持、精确表格是否来自核验行、
   metadata 是否违规；
5. **Conditional Replan**：证据或最终验证失败时最多进行一次扩大路由；
6. **Answer/Synthesis**：仅使用统一 Evidence Pack，离线抽取或调用现有
   Qwen/Responses adapter；
7. **Citation/Metadata Guard**：验证引用 ID 和唯一手册隔离，失败两次则拒答。

执行 trace 至少记录 route、检索器、结果数、graph path IDs、critic decision、
retry、逐节点和总 latency。Responses/Qwen 返回 usage 时记录 token 字段；
当前 adapter 未配置价格表，因此 cost 保持 `null` 并说明原因。

## CLI、API 与 Gradio

- `scripts/build_graph_index.py`：构建图索引；
- `scripts/search_graph.py`：独立实体/路径检索；
- `scripts/answer_agentic_question.py`：Agentic CLI 和完整 JSON trace；
- `scripts/launch_agentic_api.py`：独立 FastAPI，提供 `/health` 和
  `/v1/agentic`；
- 现有 Gradio 新增 Agentic GraphRAG tab，展示答案、统一证据和状态 trace；
- `scripts/run_full_pipeline.py` 将 graph build 和 Agentic comparison 纳入
  indexes/evaluate 阶段。

所有入口都支持或可组合使用显式 `--project-root`、`--data-root` 和
`--graph-index`。本次评测只读访问主项目的 ignored data，没有复制 620 MB
MinerU 产物或 48 MB 原有索引。

## 多跳对照评测

评测集为
`data/eval/agentic_multihop_questions.jsonl`，包含 8 个可回答问题和
4 个拒答案例。每个可回答问题包含 Gold Evidence、Gold 页码、必需节点类型
和必需关系；每个问题还包含预期路由。

最终离线结果：

| System | Evidence recall | Path accuracy | Citation faithfulness | Route accuracy | Decision accuracy | Refusal accuracy | Metadata violations | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline RAG | 0.9167 | N/A | 1.0000 | N/A | 1.0000 | 1.0000 | 0 | 17.6 ms | 27.8 ms |
| GraphRAG | 0.9167 | 0.6250 | 1.0000 | N/A | 0.6667 | 0.0000 | 0 | 154.7 ms | 504.4 ms |
| Agentic GraphRAG | 0.9167 | 0.5714 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 135.5 ms | 490.1 ms |

Agentic retry rate为 `0.3333`，即 12 题中 4 题触发一次扩大检索。评测未调用
远程模型，因此 token usage 不可用，API cost 为 USD 0.00。

结论没有宣称 GraphRAG 带来准确率提升：

- 三个系统的 multi-hop Evidence recall 都是 `0.9167`；
- standalone GraphRAG 能返回显式路径，但将四个无答案问题全部错误回答；
- Agentic Critic 保住 baseline 的 decision/refusal 指标和零 metadata
  violation，但增加约 118 ms 平均延迟；
- Agentic path accuracy 只有 `0.5714`，低于 standalone GraphRAG 的
  `0.6250`，说明条件路由和严格 Gold path 同时带来覆盖损失。

## 已知失败

1. F-150 充电组合问题：图检索找到 `SYMBOL_MEANS`，但一条二跳路径没有同时
   覆盖 Gold Warning 和 charging indicator；
2. F-150 child-restraint 问题：返回正确的 Procedure/Step 结构，但首选路径
   落在错误页，未通过 Gold page 条件；
3. Maverick remote battery 问题：`battery` 被 12 V battery 章节吸引，路径
   页码偏离 remote-control Gold Evidence；
4. standalone GraphRAG 缺少足够的无答案判定，四个拒答案例均误答；
5. 评测集仍来自开发语料，不是独立生产测试集。

这些失败说明下一步应优先改进实体消歧、跨 chunk 同 section 合并和
graph-specific abstention，而不是增加更多 Agent。

## 验证

新增 GraphRAG/Agentic/API/UI 测试与原有测试合计 **66 项全部通过**。包含：

- 所有要求节点/边类型及边证据来源；
- 一至二跳路径和跨车型隔离；
- 并行路由、critic、最多一次 replan；
- 污染 graph evidence 的拒答；
- 独立 Agentic API；
- Gradio trace/evidence 输出；
- 完整主项目 MinerU 数据完整性测试。

机器可读指标位于
`outputs/metrics/agentic_graphrag_comparison.json`，逐问题失败记录未被删除。
