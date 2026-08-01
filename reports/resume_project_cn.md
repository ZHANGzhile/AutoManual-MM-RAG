# AutoManual-MM-RAG 中文简历与面试表述

> 本文只使用已有代码、机器可读评测和真实 API 验证支持的事实。
> Agentic GraphRAG MVP 已完成并合入主分支；未提升的指标和未完成的验证会明确说明。

## 一、简历可直接使用版

### AutoManual-MM-RAG｜汽车手册多模态 Agentic GraphRAG

**技术栈：** Python、MinerU、SQLite/FTS5、BM25、NumPy、Pillow、Qwen3-VL、FastAPI、Gradio、Docker Compose

- 针对汽车手册篇幅长、车型内容易串用、操作步骤与安全警告不能被普通切块拆散、图片及参数表难以统一检索的问题，搭建从官方 PDF 下载与 SHA-256 校验、MinerU 解析、结构化清洗到索引和评测的一体化流水线；处理 4 本 Ford 2026 车主手册，沉淀 30,131 个可追溯元素、12,236 个文本块、3,287 张图片和 516 个表格裁剪。
- 围绕“车型不能串、步骤不能乱、警告不能丢、参数不能猜、答案必须能回到原页”设计证据 Schema，保留车型/年份/地区、章节、物理页码、相邻元素、原始资源路径及 Warning/Caution；实现 BM25、Hash TF-IDF + 128 维 LSA、RRF、1,296 维传统视觉特征和人工核验表格行检索，基于对照实验选择 BM25 作为当前文本默认方案（Recall@10 0.9615），而非为追求架构复杂度强行使用 Dense/RRF。
- 基于规范化证据确定性构建包含 Vehicle、Component、Procedure/Step、Warning、Specification、Image/Table 和 Evidence Page 等 12 类节点、10 类关系的汽车手册图谱，共 29,797 个节点、125,699 条证据溯源边；使用 SQLite/FTS5 完成实体匹配、1–2 跳路径扩展与联合评分，所有路径可回溯到车型、物理页和原始 element，评测中跨车型路径为 0。
- 实现 TypedDict 显式 Agentic 状态图：Planner/Router 按问题条件并行调用 Text、Visual、Table、Graph 专业检索器，Evidence Critic 检查必要模态、车型一致性和证据覆盖度，最多触发一次受控重规划，最终由 Citation/Metadata Guard 校验引用并输出逐节点执行 Trace；复用 Evidence Pack 和 Qwen/Responses 生成适配层，提供独立 CLI、FastAPI `/v1/agentic` 与 Gradio Agentic 页面。
- 建立 Baseline RAG、GraphRAG、Agentic GraphRAG 的 12 题多跳开发集对照：Agentic 路由、回答/拒答决策、拒答准确率和引用忠实度均为 1.0000，Evidence Recall 为 0.9167，Metadata Violation 为 0；同时如实定位 standalone GraphRAG 对 4 个无答案问题全部过度回答、Agentic Gold Path Accuracy 仅 0.5714、平均延迟由 Baseline 17.6 ms 增至 135.5 ms等问题，并将实体消歧和组合路径作为后续优化方向。
- 通过严格导入、可断点重跑、机器可读指标和 66 项自动化测试保证可复现；完成一次法兰克福区域 Qwen3-VL 真实图片问答闭环，检索 5 条 Bronco 专属证据并引用正确物理页。大文件、图索引、虚拟环境及密钥不进入 Git，Docker Compose 仅挂载只读数据目录。

**项目链接：** <https://github.com/ZHANGzhile/AutoManual-MM-RAG>

## 二、篇幅较短时使用的三条版

- 独立落地汽车手册多模态 RAG 数据与检索底座：完成 4 本 Ford 2026 官方手册的 MinerU 解析和可追溯证据建模，构建 30K+ 元素、12K+ 文本块、3K+ 图片及 516 个表格的数据集；保留车型、页码、步骤和 Warning，并通过 Metadata 硬过滤实现 0 跨车型证据违规。
- 基于真实证据确定性构建 29,797 节点、125,699 边的汽车手册图谱，实现 SQLite/FTS5 实体检索和 1–2 跳路径扩展；落地包含 Planner、Text/Visual/Table/Graph 并行检索、Evidence Critic、单次重规划及 Citation/Metadata Guard 的显式 Agentic 工作流，并输出完整执行 Trace。
- 建立 Baseline RAG、GraphRAG、Agentic GraphRAG 对照评测：Agentic 路由、决策、拒答和引用忠实度均为 1.0000，Evidence Recall 0.9167、Metadata Violation 为 0；同时记录路径准确率 0.5714 和 135.5 ms 平均延迟等局限，交付 FastAPI/Gradio、Qwen3-VL 真实调用验证及 66 项通过测试。

## 三、一句话项目定位

这不是把 PDF 切块后堆进向量库，而是针对汽车手册中“车型不能串、步骤不能乱、警告不能丢、参数不能猜、答案必须能回到原页”的实际约束，完成从多模态证据建模、领域图谱、专业检索器编排到引用与拒答审计的一套可重建、可评测、可追踪 Agentic GraphRAG 系统。

## 四、面试时的技术路线

### 1. 先解决数据是否可信

项目从 4 本厂商官方 PDF 开始，对下载文件做大小和 SHA-256 锁定。MinerU 运行在隔离环境中，避免解析依赖与应用环境冲突；成功标记支持断点恢复，避免重复处理大文件。

MinerU 输出没有直接进入向量库，而是先转换为统一元素 Schema。每条文本、图片或表格都带有稳定 ID、车型元数据、章节路径、物理 PDF 页码、坐标、相邻元素和源文件位置；严格导入检查缺页、缺资源和异常引用，当前 30,131 个元素的导入异常数为 0。

### 2. 用实验选择检索器，而不是预设 Dense 更先进

- 文本：实现 BM25、Hash TF-IDF + 128 维随机 LSA Dense 和 RRF。
- 图片：用缩略图强度、边缘方向、颜色分布、投影和低频频谱组成确定性的 1,296 维特征，再与文本提示融合。
- 表格：对扭矩、容量、充电参数等高风险数值，从带源图哈希和适用条件的人工核验行回答，不让模型自由猜测。

在 26 个可回答文本问题上，BM25 Recall@10 为 0.9615，Dense 为 0.7692，RRF 为 0.8846，因此当前默认文本链路保留 BM25。12 个图片查询的图文融合 Recall@1 为 1.0000；23 条人工核验表格行的精确值覆盖率为 1.0000，但该结论不外推到全部 516 个表格裁剪。

### 3. 确定性构图，保证每条关系可复现

图谱没有使用 LLM 抽取，而是复用规范化元素和 chunks 的稳定 ID、章节、邻接关系与车型元数据，确定性生成 12 类节点和 10 类关系。这样相同输入可以稳定重建，节点和边都能回到原始 element、车型和物理页。

为了在短时间内完成可运行 MVP，图存储选择 SQLite/FTS5，而不是额外部署图数据库服务。在线检索先解析唯一手册，再做 FTS 实体匹配、语义邻接缓存、双向 1–2 跳扩展及关系/节点/覆盖联合评分，最终返回可引用的 `graph_paths`。当前图包含 29,797 个节点、125,699 条边和 4 个车型分区，评测中没有跨车型路径。

### 4. Agent 只负责编排和审计，不替代安全规则

工作流使用 TypedDict 定义显式共享状态，Planner/Router 判断需要哪些模态，Text、Visual、Table 和 Graph 检索节点按条件并行执行。Evidence Critic 同时检查必要模态、车型一致性、引用完整性和证据覆盖度；不满足时最多扩大检索一次，避免无限 Agent 循环。

最终答案继续复用已有 Evidence Pack、引用校验和抽取式拒答规则。所有路由、检索结果、Critic 决策、重试次数和节点延迟都会进入 Trace，因此面试时可以展示系统“为什么调用某个检索器、为什么接受或拒绝回答”，而不是只能展示最终文本。

### 5. 用对照实验判断复杂架构是否真正有效

同一 12 题多跳开发集分别运行 Baseline RAG、standalone GraphRAG 和 Agentic GraphRAG：

| System | Evidence Recall | Path Accuracy | Route Accuracy | Decision Accuracy | Refusal Accuracy | Metadata Violations | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline RAG | 0.9167 | N/A | N/A | 1.0000 | 1.0000 | 0 | 17.6 ms |
| GraphRAG | 0.9167 | 0.6250 | N/A | 0.6667 | 0.0000 | 0 | 154.7 ms |
| Agentic GraphRAG | 0.9167 | 0.5714 | 1.0000 | 1.0000 | 1.0000 | 0 | 135.5 ms |

三组系统的 Citation Faithfulness 均为 1.0000。GraphRAG 没有提高 Evidence Recall，而且 standalone GraphRAG 对 4 个无答案问题全部误答；Agentic Critic 恢复了 Baseline 的决策和拒答安全性，但增加了约 118 ms 平均延迟，Gold Path Accuracy 也只有 0.5714。

这个结果证明当前 Agentic 层的价值主要是可控路由、拒答保护和执行可观察性，而不是准确率提升。下一步应优化实体消歧和组合路径，不应继续盲目增加 Agent 数量。

### 6. 用工程闭环证明项目可以复现

完整流水线现已覆盖下载、解析、导入、切块、BM25、Dense、表格、表格行、视觉及图谱索引、评测、审计和测试。系统提供普通与 Agentic CLI、FastAPI 文本/图片/表格及 `/v1/agentic` 接口、Gradio 多页面演示，并通过 66 项自动化测试和真实本地 API smoke。

远程模型不是离线评测的必要条件：默认回答可以完全离线运行；Qwen3-VL 只在证据范围内组织自然语言答案。此前一次法兰克福 API 付费验证成功识别 Bronco 安全带提醒符号并引用物理第 29 页，但 Agentic 三系统对照没有调用远程模型，因此其 API Cost 为 USD 0.00，不能用该对照声称远程生成质量提升。

Docker Compose 配置已验证可解析，并将本地索引和解析目录只读挂载；由于当前开发终端没有 Docker CLI，尚未实际构建镜像，面试时不表述为已验证容器部署。

## 五、面试官追问时的回答要点

### 为什么不用一个向量数据库解决所有问题？

文本、图标和参数表的错误模式不同：文本需要关键词和步骤匹配，图标需要视觉相似度，数值表需要精确字段与适用条件；分开设计才能分别评测和控制风险。当前实验还显示 BM25 比本地 Dense/RRF 更适合这组文本问题，所以没有为了技术名词强行替换。

### 为什么已经有 RAG，还要增加图谱？

普通 Top-K 检索可以找到页面，却不能显式表达 Component、Procedure、Step、Warning、Specification、Image 与 Evidence Page 的关系。图谱的目标是返回可审计路径，并为跨章节问题组织相关证据；它不是为了替换已有检索，也不是默认假设一定提高召回率。

### 为什么需要 Agentic 工作流，而不是固定链式调用？

并非所有问题都需要四种检索器。显式路由可以让文本问题只走必要路径，图片或参数问题再调用对应专业检索器；Critic 则负责发现必要模态缺失、跨车型证据或引用不完整，并进行最多一次重规划。状态和 Trace 都可测试，避免不可控的自由 Agent 循环。

### GraphRAG 最终提高准确率了吗？

没有。在当前 12 题开发集上，三种系统的 Evidence Recall 都是 0.9167。standalone GraphRAG 反而对所有无答案问题过度回答；Agentic Critic 的实际价值是把决策和拒答准确率恢复到 1.0000，并提供路由及证据路径审计。路径准确率和延迟仍需要继续优化。

### 为什么表格只做了 23 条人工核验行？

扭矩、容量等数值答错的风险高，现阶段宁愿对 9 张代表性表格建立可审计的小型 Gold Set，也不把 516 个表格裁剪都宣称为可靠 OCR。1.0000 精确值覆盖率仅指这 23 条核验行。

### 为什么还需要 VLM？

本地系统负责找对车型、页面和证据，VLM 只负责理解用户上传的局部图片并在 Evidence Pack 内组织答案。这样远程模型可以替换，检索、引用和拒答规则不依赖某一家 API，也不需要在本地部署模型。

## 六、当前边界与下一步

- 文本 Gold Set 为 30 题，视觉集为 12 题，Agentic 多跳集为 12 题，均属于开发集规模，需要扩充车型、年份和独立盲测。
- `remote-control battery` 与 `12 V battery` 仍存在实体消歧问题；F-150 的 Symbol 与 Warning 尚不能稳定被同一二跳路径覆盖。
- 当前单条 Graph Path 需要同时覆盖全部 Gold 节点类型、关系和页面，Agentic Path Accuracy 为 0.5714；后续需要组合多条互补路径，而不是单纯加深 hop。
- Qwen3-VL 已完成一次普通多模态 RAG 的付费调用验证，但尚未完成 Agentic 路径的批量远程生成评测。
- Docker Compose 配置可解析，但本机尚未执行镜像构建。

## 七、事实依据

- `reports/project_summary.md`
- `reports/full_pipeline.md`
- `reports/qwen_vlm_validation.md`
- `reports/agentic_graphrag.md`
- `reports/agentic_graphrag_evaluation.md`
- `outputs/metrics/retrieval_comparison.json`
- `outputs/metrics/answering_baseline.json`
- `outputs/metrics/visual_comparison.json`
- `outputs/metrics/table_row_answering.json`
- `outputs/metrics/agentic_graphrag_comparison.json`
- `outputs/metrics/full_pipeline_run.json`
