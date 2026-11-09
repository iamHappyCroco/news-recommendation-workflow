# 资讯推荐位出词工作流（本地版）

工作流已从 Coze 运行时解耦：LangGraph、Excel 召回、LLM 调用、CLI 和 HTTP 服务均可直接在 VS Code/本机运行。

## 1. 初始化

需要 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
```

MiniLM 负责本地主分类，所有 LLM 节点统一调用东财接口。完整跑批入口为 `scripts/batch_run.py`（CSV）和 `scripts/batch_final_summary.py`（原样 final_summary JSON）。

## 2. 配置东财模型接口

在系统环境变量或 `.env` 中配置：

```dotenv
EM_API_KEY=your-key
EM_API_BASE_URL=https://dd-ai-api.eastmoney.com/coding/v1
EM_MODEL=DeepSeek-V4-Flash-0731
```

只配置 `EM_API_KEY` 也可运行；接口地址默认为上述东财地址，各节点默认模型为 `DeepSeek-V4-Flash-0731`。`EM_MODEL` 可统一覆盖节点模型配置。
工作流必填输入为资讯标题和正文，可选 `page_contents` 按跳转URL提供目标页面主要内容；正文在进入 LangGraph 前统一截取前 5000 个字符，不读取“标题下蓝字”。分类主体与校验目录统一使用 `分类器/topic_catalog.json` 的 82 个主题；高相似度资讯的 LLM 校验仅接收 MiniLM Top-5 的原始主题 ID，低相似度走下述独立兜底路径。四象限读取 `assets/category_quadrants_82.json`，按原生主题ID对应。

分类象限及 AI 生词的最终汇总、归类候选完整结果和接口顶层均不输出 `ranking_reasons`、`best_word_reason`、`generation_reason`。推荐词继续按优先级排列，首项为 `best_word`。生成节点不再解析逐词排序理由；分类判断与优先方向的理由继续返回。CSV 不输出已移除的生词理由列，续跑时也会清理旧行及其候选 JSON 中的这些字段。

最终分类汇总、归类候选完整结果和接口顶层也不输出 `classification_reason`，CSV 取消“归类校验理由”列，续跑同步清理旧字段。内部分类校验仍使用该理由，不改变校验规则。

最终输出不返回 `model_cosine_gap`、`model_training_support`，批处理导出取消“模型首二差距”“模型主题训练样本量”列并在续跑时清理旧列。内部模型诊断与分类校验继续使用这些数值。

`model_review_reasons` 也仅保留在内部模型诊断与分类校验中，不进入最终 JSON；CSV 取消“模型需复核原因”，续跑同步清理旧列。
最终分类途径只使用 `final_summary.category_quadrant.classification_source` 一个字段，固定返回 `实际分类器分类` / `LLM兜底分类` / `未归类`，不再输出 `classification_source_label`。两种有效分类均经过 LLM 校验；最终未接受类目时统一为“未归类”，即使此前尝试过分类器或 LLM 兜底。CSV 只保留“分类来源”一列，取消“分类路径”；续跑自动迁移旧值并保留原推荐词。内部过程状态仍使用英文路径标识，最终节点转换为中文。
模型“需复核/噪声”标记不再一律阻止分类：最高相似度达到训练门槛（默认约 0.6543）时，候选差距小或样本量少仍交给 LLM 在 Top-5 内校验。前两类差距不超过 `CATEGORY_DUAL_MAX_GAP=0.005`、两类相似度均达到门槛且 LLM 确认各自有正文依据时，输出双归类；否则保留一个适配类目。低于门槛（或模型不可用）时，先执行 `category_llm_fallback`，从精简的原生82主题目录提出一个分类，再由独立第二次 LLM 调用校验提议与MiniLM Top-5备选；可通过、改类或拒绝。兜底分类拒绝时，审核员重新检查完整目录。只有审核确认无合适类目才走 AI 生词；接口错误/非法分类输出按运行错误重试，不冒充“无法分类”。CSV保留原始模型分数及噪声标记，并新增分类来源、兜底提议及理由，最终校验理由单列。可用 `CATEGORY_MIN_SIMILARITY` 覆盖训练门槛。

最终 JSON 和 CSV 不单独输出 `classification_mode`／“归类模式”。`region`、`quadrant` 位于 `final_summary.category_quadrant`，统一表达归类情况：单归类为 `A/B/C/D` 中的一个代码及其象限说明；双归类如 `region="A/B"`、`quadrant="双归类：象限A·高曝光·高点击；象限B·高曝光·低点击"`，按主分类、第二分类顺序排列，同象限双归类保留重复代码（如 `C/C`）；未归类为 `region="none"`、`quadrant="未归类（AI生词）"`。内部分类决策仍使用归类模式及单个象限进行路由，最终节点完成展示汇总。CSV 保留“象限”“象限说明”，续跑时同步更新旧行格式。

### 当前去重输出结构

接口只返回一份 `final_summary`，不在顶层重复分类、专题、热点及内部过程字段。分类部分 `category_quadrant` 保留 `classification_source`、`llm_fallback_used`、`llm_fallback_proposals`、`llm_fallback_reason`、`region`、`quadrant`、`generation_results` 共 7 个字段。

类目与推荐词统一存放在 `generation_results`：单归类 1 项、双归类 2 项，按主分类在前排列，不输出 `is_primary`。每项包含 `leaf_id`、`category`、`sub_category`、`similarity`、`reason`、`word_source`、`recommended_words`、`ai_queries`；首选词取词列表第 1 项，不输出 `best_word`。整体 `region` 按同一顺序列出象限，结果项不再重复 `region`、`quadrant`。未归类时该列表有 1 项 AI 生词结果，包含 `word_source="ai"`、`recommended_words`、`ai_queries`，不伪造分类元数据。

CSV 对应使用“分类生词结果”一列保存该列表，删除旧的分类候选 JSON 列及主分类、推荐词、首选词等重复列；续跑自动迁移旧 CSV 并保留各组词及 AI 延伸问句。专题／热点字段保持原业务结构。

前端展示与问答字段：`final_summary.topic_related.display_variants` 和 `final_summary.hot_theme_related.display_variants` 按推荐词顺序提供 `{word, display_6, display_9}`，两个展示版分别不超过 6、9 个字符（汉字、英文字母、数字各算 1 个字符）。`final_summary.category_quadrant.generation_results[].ai_queries` 在 A/B/C/D 象限及未归类 AI 的所有分类生词结果中提供，与推荐词同序，结构为 `{word, query}`；前端展示 `word`，点击后将完整 `query` 发给 AI 问答。每个推荐词都必须有一条非空完整问句，双归类两组结果各自生成。CSV 的问句保存在“分类生词结果”列表中，专题／热点仍有各自展示版列。

专题与热点的最终输出已精简：`topic_related`、`hot_theme_related` 及对应的 `topic_hit_result`、`hot_theme_hit_result` 不输出 `ranking_reasons` 和 `best_word_reason`。`recommended_words` 按推荐优先级降序排列（第 1 项最优），`display_variants` 同词同序，`best_word` 为第 1 项。新生成的 CSV 不再包含专题／热点的逐词排序理由、首选理由列。

Excel/CSV 批处理仅保存一个可续跑的 CSV：

```powershell
$env:LLM_REQUESTS_PER_MINUTE="27"
.\.venv\Scripts\python.exe -u scripts\batch_run.py input.xlsx --limit 100 --workers 4 --output-csv outputs\results.csv
# 中断后在同一命令末尾加 --resume，复用已成功行；旧策略结果不会自动重新计算。
```

DeepSeek Flash 默认通过 `LLM_THINKING_MODE=disabled` 关闭思考，避免推理 token 挤占 JSON 输出并提升跑批速度。

Excel 前 10 条完整工作流的 JSON 跑批：

```powershell
.\.venv\Scripts\python.exe scripts\batch_final_summary.py assets\要闻资讯拉取_2026-10-27.xlsx --limit 10 --workers 2 --output-dir outputs\first_10
# 中断后加 --resume，核对来源并复用已成功的结果。
```

输出目录中 `final_summaries.json` 保留资讯序号、ID、标题和每条原样 `final_summary`，`run_metadata.json` 记录本次参数及成功／失败条数。该入口每完成一条即保存一次。

完整分类生词链路允许 D 象限不出词：仍保留分类结果，在该组输出 `word_source="quadrant_D"`、`recommended_words=[]`、`ai_queries=[]`，不转为未归类，也不因空词报错或重试。双归类按每组自身象限判断，A/B/C 仍要求非空词列表；CSV／JSON 跑批、续跑及策略刷新采用同一规则。

接口需兼容 `POST /coding/v1/chat/completions`。`EM_MODEL` 会覆盖各 `config/*_llm_cfg.json` 中原有的平台模型名。

运行单条资讯（先准备包含 `title`、`content` 的 JSON 对象文件 `article_input.json`）：

```powershell
$env:PYTHONPATH="$PWD\src"
.\.venv\Scripts\python.exe src\main.py --mode flow --input-file article_input.json
```

## 3. HTTP 服务

```powershell
$env:PYTHONPATH="$PWD\src"
.\.venv\Scripts\python.exe src\main.py --mode http --port 5000
```

调用：

```powershell
Invoke-RestMethod http://127.0.0.1:5000/run -Method Post -ContentType application/json -Body (@{
  title = "资讯标题"
  content = "资讯正文"
} | ConvertTo-Json)
```

健康检查为 `GET /health`，交互文档为 `http://127.0.0.1:5000/docs`。

## 分类模型

基础安装会同时安装 `sentence-transformers` 与 Torch。首次运行会下载
`paraphrase-multilingual-MiniLM-L12-v2`；生产环境建议预先下载到本地，并在 `.env` 设置：

```dotenv
SEARCH_INTENT_EMBEDDING_MODEL=C:/models/paraphrase-multilingual-MiniLM-L12-v2
```

分类器加载后由进程单例常驻，并对 `(标题, 正文)` 结果做 LRU 缓存。模型异常时工作流仍会自动由 LLM 分类兜底。
若项目下存在 `.models/minilm` 会自动优先使用，无需再配置路径；HTTP 服务启动时默认执行一次分类器预热。


**分类生词统一问答 query（2026-11-09）**：A/B/C/D 象限和未归类 AI 分支都为每个推荐词同时生成完整问句，最终统一放在 `category_quadrant.generation_results[].ai_queries`，结构为 `{word, query}`，条数、推荐词和顺序与 `recommended_words` 一致。分类途径、象限及生词来源不影响问句输出；双归类两组结果独立生成。专题／热点跳转生词仍保留原结构。本条覆盖此前“仅 AI 分支输出”的约定。旧 CSV 缺少任一分类结果问句时不能直接续跑复用，需要重跑补齐；迁移只保留已有问句，不伪造问句。


**生词中间输出精简（2026-11-09）**：专题、热点、A/B/C/D 和未归类 AI 的所有生词提示词、节点输出及全局过程状态均不再包含逐词排序理由 `ranking_reasons`，移除 `WordRanking` 类型和解析代码。`recommended_words` 仍按优先级排序；专题／热点保留展示版，分类生词保留逐词 query。分类校验理由、专题／热点优先判断理由及已有整体生成策略不受此项调整影响。本条覆盖此前“内部仍输出逐词理由”的约定。


**延伸内容硬规则（2026-11-09）**：A/B/C/D 与未归类 AI 的 system/user prompt 要求推荐词本身及其对应 query 均指向原文未回答的延伸需求；禁止摘词、缩写标题、同义复述、把已披露事实改成问题，不能只在 query 中补充新意图。允许复用实体作锚点、引入与资讯有关且可核实的知识或方法，禁止编造新事实和确定性预测。延伸质量优先于目标数量，无合格方向时允许两数组为空；仍保留推荐词优先级、query 同词同序及原输出结构。专题／热点不适用本次新增规则。这是提示词硬约束及模型自检要求，现有运行时结构校验不等同于语义验证。


**专题／热点目标内容生词（2026-11-09）**：专题和热点生词重点依据跳转目标页面主要内容，原资讯只作阅读起点及已知信息边界，不要求逐事实强关联；推荐词必须提供原文之外且能由目标内容承接的延伸阅读。资料优先级为按URL传入的 `page_contents`、清单或单节点传入的页面主要内容、专题导语／热点释义。清单可选列名为“页面主要内容”。API／CLI输入可选 `page_contents` 对象，键为目标URL，值为内容字符串。专题导语和热点释义由候选命中结果传至生成节点，均为内部过程字段，不增加最终字段。当前不抓取网页，不把摘要冒称为完整页面内容；无依据时不得虚构。推荐词数量仍为3个，保留6字／9字版和最终输出结构。
