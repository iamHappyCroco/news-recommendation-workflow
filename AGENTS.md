# 项目概述
- **名称**: 资讯推荐位出词工作流
- **功能**: 对单条资讯（标题+正文）执行完整生词链路：专题召回→专题判断→**东方财富热点题材必走流程（召回→判断→适合则生成热点题材相关搜索词）**；随后统一交给模型主分类+LLM校验，使用原生 82 个二级主题及其四象限区域(A/B/C/D)，按象限走不同生词策略；低分或模型不可用时先经LLM兜底提议与独立校验，最终无法归类时直接进入AI生词。专题词(topic_hit_result)与热点题材词(hot_theme_hit_result)均为“额外输出”，不阻塞所有资讯走模型分类→象限生词；最终所有生词终点统一汇入 `final_summary` ——输出一份汇总：`topic_related`(专题相关，含专题跳转链接)/`hot_theme_related`(热点题材相关，含热点题材跳转链接)/`category_quadrant`(分类+按象限生词)三部分，各部分给出推荐词与**首选词 `best_word` 及首选理由 `best_word_reason`**，专题与热点另附跳转链接；并在专题与热点均命中时给出**优先出词判断** `out_priority`（topic/hot_theme/none）及理由。
- **生词节点输出结构（与提示词对齐）**: `recommended_words` 按推荐优先级排列；专题／热点另含 `display_variants`，分类各分支另含 `ai_queries`。所有分支的中间状态也不输出逐词排序理由。

**最终输出精简（2026-11-06）**：所有方向的最终结果不输出 `ranking_reasons`、`best_word_reason`、`generation_reason`，覆盖专题／热点命中结果、分类象限／AI 汇总、双归类完整结果及接口顶层。保留按优先级排序的推荐词与首项 `best_word`，专题／热点另含同词同序展示版。生成节点不再解析逐词排序理由；分类判断、LLM兜底和专题／热点优先方向的理由继续输出。新 CSV 和续跑导出不包含已移除的生词理由列及候选 JSON 字段。

**最终归类状态展示（2026-11-06）**：最终 JSON 不输出 `classification_mode`，CSV 不输出“归类模式”。最终主层与接口顶层的 `region` 单归类为 A/B/C/D 中的一个代码，双归类按主次顺序连接如 A/B（同象限保留 A/A），未归类为 none；`quadrant` 对应单个象限说明、“双归类：主象限；次象限”或“未归类（AI生词）”。候选自身象限保持单值，内部模式及路由不变。CSV 保留“象限”“象限说明”，续跑复用同步迁移该展示格式。

**最终分类途径（2026-11-06）**：最终 JSON 只保留 `classification_source`，固定为“实际分类器分类”“LLM兜底分类”“未归类”。取消 `classification_source_label`，接口顶层与汇总一致。CSV 仅保留“分类来源”，取消“分类路径”；续跑同步迁移旧值。英文路径标识仅用于内部过程状态，最终汇总节点转成中文。

**分类校验理由精简（2026-11-06）**：最终 JSON（分类汇总、归类候选完整结果、接口顶层）不输出 `classification_reason`，CSV 不输出“归类校验理由”。续跑同步清理旧字段；内部分类校验仍保留该理由。

**模型诊断字段精简（2026-11-06）**：最终输出不返回 `model_cosine_gap`、`model_training_support`，CSV 取消“模型首二差距”“模型主题训练样本量”，续跑同步清理旧列。内部复核与双归类判断继续使用这些数值。

`model_review_reasons` 同样不进入最终 JSON 或 CSV（“模型需复核原因”列），续跑同步清理旧列；内部分类校验继续使用复核原因。

**当前去重输出（2026-11-06，覆盖前述兼容字段）**：接口顶层仅返回 `final_summary`。分类部分保留分类途径、3 个 LLM 兜底字段、整体 `region`／`quadrant` 与统一 `generation_results` 列表，共 7 个字段。单归类 1 项、双归类 2 项，首项为主分类，不输出 `is_primary`；每组推荐词的首项为首选，不输出分类 `best_word`。结果项仅保留分类 ID／名称、相似度、候选依据、生词来源、推荐词，整体象限不重复存入结果项。未归类有 1 项 AI 结果（来源、推荐词、延伸问句），不伪造分类。CSV 统一为“分类生词结果”列，迁移旧行及旧候选字段，保留两组结果与 AI 问句。专题／热点自身结构保持原样；内部状态和路由不变。

## 主图节点清单（`graphs/graph.py` 单条链路，平铺展示）
| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| topic_recall | `nodes/topic_recall_node.py` | task | 基于资讯内容在专题知识库做语义检索召回候选专题（知识库数据源为 `专题清单.xlsx` 每个专题的「专题名+导语」前两列；召回命中专题时同步输出该专题的「专题地址」链接） | - | - |
| topic_fit_judge | `nodes/topic_fit_judge_node.py` | agent | 判断资讯是否存在可跳转的适配专题 | - | `config/topic_fit_judge_llm_cfg.json` |
| topic_block | `nodes/topic_block_judge_node.py` | task | **专题嵌套跳转屏蔽（纯规则，非大模型）**：输入资讯标题命中 `专题清单.xlsx`「文章标题」列对应标题、且适配专题正是该标题所关联专题时，将 has_fit_topic 置 false 屏蔽该专题生词 | has_fit_topic=false（屏蔽）→route_topic_fit（进热点题材流程）；否则→route_topic_fit | - |
| route_topic_fit | `graph.py` | condition | 专题适配判定路由（所有资讯最终都进模型分类→象限生词；专题适配时额外先输出专题生词） | 命中→topic_word_generate（随后进入热点题材必走流程）；未命中→hot_theme_recall | - |
| topic_word_generate | `nodes/topic_word_generate_node.py` | agent | 命中专题后，以目标页面主要内容或专题导语为核心生成3个延伸推荐词及6字／9字展示版，原资讯用于识别已知内容，结果写入topic_hit_result | - | `config/topic_word_generate_llm_cfg.json` |
| hot_theme_recall | `nodes/hot_theme_recall_node.py` | task | **热点题材必走**：读取 `东方财富热点题材_2026-10-22.xlsx`「题材汇总」表的热点题材+热点释义，通过本地词面检索召回最多5个候选，与专题路径一致；过滤分数低于0.05的项 | - | - |
| hot_theme_fit_judge | `nodes/hot_theme_fit_judge_node.py` | agent | 判断资讯是否适合跳转到召回的热点题材；仅核心主题高度契合且可承接资讯内容时，返回最适配候选的名称/代码/真实链接；弱相关或无适配候选时不命中 | - | `config/hot_theme_fit_judge_llm_cfg.json` |
| hot_theme_block | `nodes/hot_theme_block_judge_node.py` | task | **嵌套跳转屏蔽（纯规则，非大模型）**：输入资讯标题命中屏蔽清单 `热点题材_需屏蔽嵌套跳转_2026-10-22.xlsx` 的「资讯标题」列、且适配热点也在其「热点题材」列时，将 has_fit_hot_theme 置 false 屏蔽该热点生词 | has_fit_hot_theme=false（屏蔽）→priority_judge；否则→route_hot_theme_fit | - |
| route_hot_theme_fit | `graph.py` | condition | 热点题材路由（经 hot_theme_block 屏蔽后判定；无论是否适合/是否被屏蔽，最终都经 priority_judge 进入模型分类） | 适合且未屏蔽→hot_theme_word_generate；不适合或被屏蔽→priority_judge | - |
| hot_theme_word_generate | `nodes/hot_theme_word_generate_node.py` | agent | 命中热点后，以目标页面主要内容或热点释义为核心生成3个延伸推荐词及6字／9字展示版，原资讯用于识别已知内容，结果写入hot_theme_hit_result | - | `config/hot_theme_word_generate_llm_cfg.json` |
| priority_judge | `nodes/priority_judge_node.py` | agent | 优先出词判断：专题与热点均命中时由 LLM 判定优先出专题词还是热点词并给理由，写 out_priority/out_priority_reason；仅命中其一则直接按唯一命中方向输出 | 双命中→LLM；仅专题→topic；仅热点→hot_theme；都未命中→none | `config/priority_llm_cfg.json` |
| category_model_classify | `nodes/category_model_classify_node.py` | task | 模型主分类：用搜索意图两级分类器（`分类器/` 82 主题 Qwen 修订版，MiniLM 中心向量）对资讯做主分类，产出预测主题/相似度/噪声判定，并输出向量最近 Top-K 相邻主题 `nearby_topics` | - | - |
| category_llm_fallback | `nodes/category_llm_fallback_node.py` | agent | 模型低分或不可用时，从精简原生82主题目录提出一个分类供独立校验；高分路径不调用兜底LLM | 高分→category_llm_verify；低分/不可用→LLM提议→category_llm_verify | `config/category_fallback_llm_cfg.json` |
| category_llm_verify | `nodes/category_llm_verify_node.py` | agent | LLM 校验：正常路径仅在原生82主题的MiniLM Top-5中确认或改类；兜底路径独立审核提议与备选，第一阶段未提出类目时审核完整目录；接受类目按原生主题ID映射四象限，复核标记不直接否决分类 | - | 正常：`config/category_classify_llm_cfg.json`；兜底审核：`config/category_fallback_verify_llm_cfg.json` |
| route_region | `graph.py` | condition | 类目→象限/AI 路由 | A→quadrant_a；C→quadrant_c；**B、D→hot_topic_select**；无法归类/异常→AI 生词 | - |
| hot_topic_select | `nodes/hot_topic_select_node.py` | agent | 相邻高曝光主题拣选：对 B/D 低曝光资讯，结合模型向量相邻主题与目录中 region=A（高曝光高点击）主题清单，语义拣选一个真正相邻的高曝光主题作为下游生词参照；拣不到则 has_hot_topic=false 优雅降级。选中主题的**模型向量余弦相似度回填 hot_similarity 并作为相邻性门槛**（<0.5 视为不相邻→has_hot_topic=false 走降级） | - | `config/hot_topic_select_llm_cfg.json` |
| route_after_hot | `graph.py` | condition | 热主题拣选完成后按原区域回象限 | B→quadrant_b；D→quadrant_d；异常→ai_generate | - |
| quadrant_a | `nodes/quadrant_a_node.py` | agent | A·高曝光高点击类目生词(热点热搜) | - | `config/quadrant_a_llm_cfg.json` |
| quadrant_b | `nodes/quadrant_b_node.py` | agent | B·高曝光低点击类目生词(驱动点击转化)；将选中A类主题的类别含义、关键词与搜索需求作为重要内容参考，结合原资讯生词并解释结合依据；保留原B归类 | - | `config/quadrant_b_llm_cfg.json` |
| quadrant_c | `nodes/quadrant_c_node.py` | agent | C·低曝光高点击类目生词(深度精准) | - | `config/quadrant_c_llm_cfg.json` |
| quadrant_d | `nodes/quadrant_d_node.py` | agent | D·低曝光低点击类目生词(关联探索)；将选中A类主题作为重要内容参考，结合原资讯寻找关联搜索需求，最多2个关联词并解释结合依据；保留原D归类 | - | `config/quadrant_d_llm_cfg.json` |
| ai_generate | `nodes/ai_generate_node.py` | agent | 分类校验未接受类目或区域异常时，基于资讯事实执行AI生词 | - | `config/ai_generate_llm_cfg.json` |
| final_summary | `nodes/final_summary_node.py` | task | **最终输出汇总**：聚合专题相关/热点题材相关/分类+按象限生词三部分，各部分给出推荐词、跳转链接与首选词 best_word + 理由；并透出 out_priority 优先出词判断 | 所有生词终点(quadrant_a~d、ai_generate)→final_summary→END | - |

**类型说明**: task(task节点) / agent(大模型) / condition(条件分支)
**路由说明(B/D 重要参考类别策略)**: A/C直接生词；B/D先经 `hot_topic_select`，从向量相邻主题与region=A清单中拣选语义相邻的A类主题。选中类别的一级/二级含义、关键词和搜索需求作为重要内容参考，与原资讯主体和事实共同形成推荐词，不仅模仿表达风格；拣选理由随 `hot_reference_text` 传入。概念可超出原文字面表达，但结合关系须能由资讯解释，不引入无依据的事件、数字或因果；原分类、0.5相邻性门槛及B/D出词数量保持不变。无参考时按原资讯生词；有参考但无合理结合角度时不强行结合，并说明原因。

## 状态定义要点（`graphs/state.py`）
- **当前分类校验策略**：统一使用原生82主题及 `assets/category_quadrants_82.json`。`model_is_noise` 保留为“需复核”诊断标记，不再硬性拒绝高相似度资讯。最高相似度达到训练门槛（约0.6543，可用 `CATEGORY_MIN_SIMILARITY` 覆盖）时，候选差距小/样本少仍交由LLM在Top-5内校验；低于门槛或模型不可用时先进入 `category_llm_fallback`，在精简82主题目录中提出一个分类，再由独立LLM调用校验（提议+Top-5备选，可通过/改类/拒绝；兜底拒绝则复核完整目录）。只有校验确认无适合类目才走AI。原始模型分数/噪声标记保留，汇总和CSV透出分类来源、兜底提议及两阶段理由。前两候选都达门槛、差距≤`CATEGORY_DUAL_MAX_GAP`（默认0.005）且都经LLM确认有正文依据时允许双归类，低分LLM兜底仅单归类。非法LLM输出/配置错误作为运行错误重试，不伪装成无适配类别。
- `ClassificationDiagnostics`：保留次优相似度/首二差距/训练样本数/复核原因；`ClassificationDecision`：含 `classification_mode`（single/dual/none）、校验理由与已接受归类候选。
- `category_candidate_generate`（`src/graphs/nodes/category_candidate_generate_node.py`）：四象限生词后追加的归类候选生词汇总节点，复用主分类已生成词，仅对第二个已接受候选按其自身象限独立生词；第二候选B/D仍先执行热主题拣选。汇入final_summary，`category_quadrant.classification_candidates`包含各类、象限、推荐词及首选理由，旧类目字段保留主分类。CSV把两套结果放在同一行。
- `NearbyTopic`：模型向量最近的相邻主题（leaf_id/leaf_name/parent/similarity/representative_searches）
- `ModelClassifyOutput`：含 `nearby_topics`（Top-K 相邻主题候选）
- `HotTopicSelectInput/Output`：热主题拣选出入参（has_hot_topic、hot_leaf_id/name/region、hot_similarity(向量相邻门槛)、hot_rep_searches、hot_keywords、hot_reference_text）
- `QuadrantInput`：含可选热主题参照字段（has_hot_topic、hot_leaf_name、hot_rep_searches、hot_keywords、hot_reference_text），B/D 使用

## 业务配置数据
- `assets/专题清单.xlsx`: 专题清单数据（前两列为「专题名」「导语」，另有「专题地址/专题类型/文章标题/文章地址」列）。topic_recall读取「专题名+导语」进行本地召回，命中时同步输出「专题地址」链接；文章标题用于规则屏蔽
- `assets/东方财富热点题材_2026-10-22.xlsx`: 东方财富热点题材数据（`题材汇总` 表：热点题材名/题材代码/热点释义/详情页链接），hot_theme_recall读取题材名与释义进行本地召回
- `分类器/topic_catalog.json`: **原生82主题两级分类目录**（主题ID、一级/二级名称及代表搜索词）；模型分类、LLM校验、兜底及热主题拣选均使用该目录
- `assets/category_quadrants_82.json`: 原生82主题ID对应的四象限配置，供分类校验及热主题拣选使用
- `src/tools/llm_utils.py`: LLM 调用与 JSON 解析共用工具（invoke_llm/extract_json/ensure_workspace_path）
- `src/tools/search_classifier.py`: 分类器运行时封装（`get_classifier` 单例懒加载 + `classify_article(title, content)`，返回含 `nearby_topics` Top-K 相邻主题）

## 分类模型运行时依赖
- 模型文件：`分类器/prototypes_82.npz`、`分类器/topic_catalog.json`（原生82主题，与四象限配置按主题ID对应）、`分类器/model_metadata.json`
- 预测口径：`src/tools/search_classifier.py`（Qwen 修订 82 主题最近质心模型，MiniLM 384 维向量与归一化主题中心余弦相似度分配；**不是分词器**）
- 依赖包：`sentence-transformers`、`numpy`、`torch`
- embedding 模型：`paraphrase-multilingual-MiniLM-L12-v2`，本地 `/space/models/minilm`
- 环境变量：`SEARCH_INTENT_EMBEDDING_MODEL=/space/models/minilm` 等（search_classifier.py 设置）
- **架构约定**：正常分类主体由MiniLM模型完成（encode+classify），LLM在原生Top-5内校验或改类；低于门槛或模型不可用时执行LLM提议+独立校验。类目与象限始终按原生82主题ID对应

## 模型分类性能（P0：预热 + 缓存）
- **本质**：中心向量已持久化在 `prototypes_82.npz`（加载毫秒级，非耗时来源）；真正耗时是 embedding 模型(MiniLM+torch)一次性冷加载与词向量化。模型加载后以进程内单例 `get_classifier()` 常驻，逐词分类仅 ~0.1s（实测）。
- **`search_classifier.warmup(encode=True)`**：服务启动后就绪、接流量前调用一次，提前加载并常驻模型（实测冷加载约 3~8 分钟因环境波动，此后每词分类 ~0.11s）。
- **结果缓存**：`classify_article` 已按 `(标题,正文)` 做 LRU(512) 缓存，重复/相似资讯命中缓存（实测 ~0.0001s）。
- **接入点**：长驻服务启动时应调用一次 `from tools.search_classifier import warmup; warmup()`（部署侧挂到启动流程/入口，避免首个请求冷启动卡顿）。

## 节点实现
- A 象限生词目标数量为 3–5 个（2026-11-09）；合格延伸不足时允许少出或不出，仍按优先级排序。
- LLM 运行约定（2026-11-09）：所有 LLM 节点统一使用东财接口，凭据为 `EM_API_KEY`，地址默认 `https://dd-ai-api.eastmoney.com/coding/v1`（可通过 `EM_API_BASE_URL` 配置），全局模型覆盖为 `EM_MODEL`。本地 Qwen 权重、加载／推理代码和运行模式切换已移除；离线测试通过 mock HTTP 或节点函数实现，不提供生产 mock 分支。`.models/minilm`、Torch 和 Transformers 是本地分类依赖，继续保留。
- topic_recall / hot_theme_recall读取本地Excel，并通过 `src/tools/local_retrieval.py` 召回候选
- 各LLM节点通过 `src/tools/llm_utils.py` 调用真实模型，Coze部署与存储工具不属于当前运行依赖
- 跑批入口为 `scripts/batch_run.py`（CSV）和 `scripts/batch_final_summary.py`（原样最终 JSON）；本地单条与HTTP入口为 `src/main.py`。保留 `.venv`、`.models`、全部 `config`、原生82主题模型、业务清单、测试及结果文件

**项目清理（2026-11-09）**：删除单节点模拟 case／运行脚本、单独分类与训练脚本、象限配置构建脚本、未接入主图的历史偏好资料和 IDE 配置；分类运行时统一使用 `src/tools/search_classifier.py` 及现有模型产物。完整工作流、跑批入口、依赖环境、验证测试、工作流／提示词说明及成功跑批结果保留。


**分类生词统一问答 query（2026-11-09）**：A/B/C/D 象限和未归类 AI 分支都为每个推荐词同时生成完整问句，最终统一放在 `category_quadrant.generation_results[].ai_queries`，结构为 `{word, query}`，条数、推荐词和顺序与 `recommended_words` 一致。分类途径、象限及生词来源不影响问句输出；双归类两组结果独立生成。专题／热点跳转生词仍保留原结构。本条覆盖此前“仅 AI 分支输出”的约定。旧 CSV 缺少任一分类结果问句时不能直接续跑复用，需要重跑补齐；迁移只保留已有问句，不伪造问句。


**生词中间输出精简（2026-11-09）**：专题、热点、A/B/C/D 和未归类 AI 的所有生词提示词、节点输出及全局过程状态均不再包含逐词排序理由 `ranking_reasons`，移除 `WordRanking` 类型和解析代码。`recommended_words` 仍按优先级排序；专题／热点保留展示版，分类生词保留逐词 query。分类校验理由、专题／热点优先判断理由及已有整体生成策略不受此项调整影响。本条覆盖此前“内部仍输出逐词理由”的约定。


**延伸内容硬规则（2026-11-09）**：A/B/C/D 与未归类 AI 的 system/user prompt 要求推荐词本身及其对应 query 均指向原文未回答的延伸需求；禁止摘词、缩写标题、同义复述、把已披露事实改成问题，不能只在 query 中补充新意图。允许复用实体作锚点、引入与资讯有关且可核实的知识或方法，禁止编造新事实和确定性预测。延伸质量优先于目标数量，无合格方向时允许两数组为空；仍保留推荐词优先级、query 同词同序及原输出结构。专题／热点不适用本次新增规则。这是提示词硬约束及模型自检要求，现有运行时结构校验不等同于语义验证。


**专题／热点目标内容生词（2026-11-09）**：专题和热点生词重点依据跳转目标页面主要内容，原资讯只作阅读起点及已知信息边界，不要求逐事实强关联；推荐词必须提供原文之外且能由目标内容承接的延伸阅读。资料优先级为按URL传入的 `page_contents`、清单或单节点传入的页面主要内容、专题导语／热点释义。清单可选列名为“页面主要内容”。API／CLI输入可选 `page_contents` 对象，键为目标URL，值为内容字符串。专题导语和热点释义由候选命中结果传至生成节点，均为内部过程字段，不增加最终字段。当前不抓取网页，不把摘要冒称为完整页面内容；无依据时不得虚构。推荐词数量仍为3个，保留6字／9字版和最终输出结构。

**D 象限空结果（2026-11-09）**：完整已归类链路允许 D 类目保留分类成功但不出词，结果项为 `word_source="quadrant_D"`、`recommended_words=[]`、`ai_queries=[]`，分类 ID、名称与依据仍保留，不转 AI。单归类和双归类各组独立适用；A/B/C 仍要求非空推荐词。CSV／JSON 跑批校验、续跑及策略刷新均接受合法 D 空结果，不因空词重试。未归类 AI 的现有跑批校验不变。
