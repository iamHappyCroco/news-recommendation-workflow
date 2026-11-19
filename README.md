# 资讯推荐位出词工作流

**当前短词规则（2026-11-19，覆盖此前10字上限）**：所有生词分支的word最多12字，short最多9字且不长于word，空格计1字。并列实体统一用一个普通空格分隔，word和short保留同一并列主体片段及比较／关联关系；生成阶段subject原样填写该片段。空格只用于并列实体，不用在其他位置。专题和热点的word不超过9字时short直接使用原词。续跑版本更新为2026-11-19-short9-parallel-v5，旧结果重新生成。

**当前专题／热点跳转输出（2026-11-19，覆盖此前数组约定）**：两条跳转路径仅生成1个首选词，生成阶段使用word、short、subject、link_type；最终topic_related和hot_theme_related仅保留word、short及原有命中和链接字段，不再输出recommended_words或best_word_short。hot_theme_related沿用现有name字段。word最多12字、short最多9字，空格计1字；word不超过9字时short直接使用原词。无合格词时四个生成字段均为空字符串，matched仍表示适配判断，out_priority只在有可用词的方向间选择。subject和link_type用于内部校验，不进入最终输出。续跑版本为2026-11-19-short9-parallel-v5，旧版结果重新生成。分类与AI的逐词四字段列表保持原样。

**当前分类与AI逐词输出（2026-11-19，覆盖此前分离数组约定）**：最终分类A/B/C/D及未归类AI的 `recommended_words` 为按推荐位排序的对象列表，第一项为首选，每项仅包含 `word` 原词、`short` 短版、`query` 完整AI提问、`query_summary` 用户侧提问概述。原词最多12字、短版最多9字（含空格），短版与概述都保留对应原词主体；概述准确表达query提问意图，简洁但不受12／9字限制，不输出答案或新事实。生成过程内部仍使用同词同序short_words和ai_queries数组，query_summary由模型与query一起生成，最终合并且不再输出两数组。最多3词，双分类合计最多3词，保留质量优先及合法D空结果。专题／热点输出使用word和short。续跑版本为2026-11-19-short9-parallel-v5，缺少概述的旧结果重新生成，不伪造概述。

**当前逐词短版与去重输出（2026-11-18，覆盖此前首选短版和best_word约定）**：所有最终输出不再包含重复的 `best_word`，推荐词首项即首选。专题／热点各仅1词，保留 `best_word_short`。分类A/B/C/D及未归类AI为每个推荐词生成短版，统一输出 `short_words=[{"word":"原词","short":"短版"}]`，与recommended_words同词同序；不再输出分类best_word_short。每个短版仍不超过10字（含空格）、保留对应原词主体与意图。模型在每条short_words中生成内部subject用于校验，最终不输出subject。空结果三数组均为空。双分类裁减词数时同步裁减短版与query，总词数仍最多3个。CSV移除重复“专题首选词”“热点首选词”列；分类短版位于分类生词结果JSON中。续跑规则版本为2026-11-18-all-shorts-v2，旧记录缺少逐词短版时重新生成。

续跑记录使用主体保留规则版本 `2026-11-18-subject-v1`：CSV 记录“生词规则版本”，JSON 跑批记录 `word_policy_version`（位于记录元数据，不在 `final_summary` 中）。缺少当前版本标识的旧结果重新生成，避免复用无主体短版。

**短版主体保留（2026-11-18）**：所有分支的 `best_word_short` 必须保留原首选词主体，单独阅读即可明确对象；不以旁边标题、链接或query补全主体。优先保留公司、人物、机构、产品、技术或板块等原有实体，不能仅输出“财报超预期”“CEO遴选标准”等无主体短词。原词有并列主体时保留各主体及比较／关联关系。模型生成内部字段 `best_word_subject` 标出从原词保留到短版的主体，程序检查原词与短版均包含该主体，缺失则重新生成一次；该字段不进入最终JSON／CSV。主体识别仍由模型按语义完成，不能把事件、谓词或泛称当作主体。10字上限（含空格）保持不变。

**当前生词输出规则（2026-11-18，覆盖此前数量和展示版约定）**：专题与热点跳转各仅输出1个最好的首选词；A/B/C/D及未归类AI生词最多3个词，双分类两组合计最多3个（主分类优先，为次分类保留一个名额）。所有推荐词不超过12个字符，首选词短版不超过10个字符且不长于原词。多个名词并列可用普通空格分隔，每个空格计1个字符，汉字、字母、数字、标点同样各计1个字符。短版统一使用 `best_word_short`，替换原6字／9字展示版；只为每组首选词生成短版，保留原主体和延伸意图，不机械截断。首选词已足够简短时可使用原词。分类所有分支继续提供与推荐词同词同序的完整 `ai_queries`，query不受12字限制；空结果短版为 `""`。节点校验数量、长度和短版，不合规时要求模型重新生成一次。CSV改为“专题首选词短版”“热点首选词短版”，分类短版位于“分类生词结果”JSON中。旧结果缺少短版或不符合当前数量／长度限制时不能直接续跑复用，需要重跑。

输入资讯标题和正文，依次处理专题、热点题材和分类生词，生成推荐词、页面跳转信息及分类推荐词对应的 AI 问答 query。

## 本地环境准备

使用 Python 3.12 或更高版本。以下命令在 Windows PowerShell 中执行。

```powershell
git clone https://github.com/iamHappyCroco/news-recommendation-workflow.git
cd news-recommendation-workflow
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
```

在项目根目录运行后续命令。已有 `.env` 时直接编辑现有文件。

## 配置规则

### 东财模型接口

编辑项目根目录的 `.env`：

```dotenv
EM_API_KEY=你的东财接口密钥
EM_API_BASE_URL=https://dd-ai-api.eastmoney.com/coding/v1
EM_MODEL=DeepSeek-V4-Flash-0731
NEWS_CONTENT_MAX_CHARS=5000
```

`EM_API_KEY` 必填，所有 LLM 节点调用该东财接口。接口地址、模型和正文上限可以使用默认值。

配置既可放在 `.env`，也可设置为系统或当前终端的环境变量；已有环境变量优先于 `.env`。`EM_MODEL` 统一覆盖 `config/*_llm_cfg.json` 中的模型设置，其他节点提示词和采样参数由各自配置文件控制。

| 配置项 | 默认值或示例值 | 作用 |
|---|---|---|
| `EM_API_KEY` | 必填 | 东财接口密钥 |
| `EM_API_BASE_URL` | `https://dd-ai-api.eastmoney.com/coding/v1` | 模型接口地址，使用 `/chat/completions` |
| `EM_MODEL` | 各节点默认 `DeepSeek-V4-Flash-0731` | 统一指定 LLM 模型 |
| `NEWS_CONTENT_MAX_CHARS` | `5000` | 正文保留的最大字符数 |
| `LLM_TIMEOUT_SECONDS` | `120` | 单次模型请求超时秒数 |
| `LLM_MAX_RETRIES` | `3` | 模型 HTTP 请求最大尝试次数 |
| `LLM_REQUESTS_PER_MINUTE` | `.env.example` 设置为 `27` | 同进程模型请求频率；设为 `0` 时不限制 |
| `LLM_THINKING_MODE` | `disabled` | 向模型请求关闭思考模式 |
| `CATEGORY_MIN_SIMILARITY` | 模型元数据，约 `0.6543` | 正常分类与 LLM 兜底分类的分流门槛 |
| `CATEGORY_DUAL_MAX_GAP` | `0.005` | 允许双归类的前两候选最大相似度差 |
| `CLASSIFIER_WARMUP` | `true` | HTTP 服务启动时预热本地分类器 |

### 本地分类模型

MiniLM 负责将资讯编码后与 82 个主题中心比较。首次运行需要下载 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，已下载的模型可从本地加载。

模型位置按以下顺序确定：

1. 环境变量 `SEARCH_INTENT_EMBEDDING_MODEL` 指定的路径或模型名。
2. 项目目录 `.models/minilm`。
3. 模型元数据中的配置；使用默认模型时由 Hugging Face 下载或读取缓存。

指定本地模型目录的示例：

```dotenv
SEARCH_INTENT_EMBEDDING_MODEL=C:/models/paraphrase-multilingual-MiniLM-L12-v2
```

仓库包含主题中心、分类目录和象限映射。虚拟环境与 MiniLM 权重在本地安装或下载。跑批入口会先预热分类器，首次加载可能需要等待。

### 业务数据

| 文件 | 用途 |
|---|---|
| `assets/专题清单.xlsx` | 专题名称、导语、地址及专题嵌套跳转屏蔽依据 |
| `assets/东方财富热点题材_2026-10-22.xlsx` | 热点题材名称、释义、代码和详情页链接 |
| `assets/热点题材_需屏蔽嵌套跳转_2026-10-22.xlsx` | 热点嵌套跳转屏蔽清单 |
| `assets/category_quadrants_82.json` | 分类主题对应的 A/B/C/D 象限 |
| `分类器/topic_catalog.json` | 原生 82 个主题的一级/二级名称及代表搜索词 |
| `分类器/prototypes_82.npz`、`分类器/model_metadata.json` | 主题中心向量及分类参数 |

专题和热点清单可增加“页面主要内容”列。单条输入也可通过 `page_contents` 提供目标页面内容，键为页面 URL，值为内容字符串。

## 本地运行

### Excel/CSV 跑批

输入表需包含“资讯标题”和“资讯正文”列；Excel 中的“资讯全文”也可作为正文列。按原表顺序选取资讯，标题和正文需非空。

将要闻文件前 10 条资讯的结果保存为 CSV：

```powershell
.\.venv\Scripts\python.exe scripts\batch_run.py assets\要闻资讯拉取_2026-10-27.xlsx --limit 10 --workers 2 --attempts 3 --output-csv outputs\results.csv
```

`--limit` 控制条数，`--offset` 控制跳过的行数，`--workers` 控制并发资讯数，`--attempts` 控制每条资讯的最大尝试次数，默认10次，可设为1至10次。每完成一条即保存结果。校验失败后，下一次尝试会向生成节点反馈具体错误；反馈按资讯隔离，不改变原有校验规则和最终JSON结构。

中断后，在同一命令末尾加 `--resume`，核对来源并复用已成功且配套字段完整的结果。模型请求频率可通过 `LLM_REQUESTS_PER_MINUTE` 调整。

需要逐条原样 `final_summary` JSON 时：

```powershell
.\.venv\Scripts\python.exe scripts\batch_final_summary.py assets\要闻资讯拉取_2026-10-27.xlsx --limit 10 --workers 2 --output-dir outputs\first_10
```

该入口接受 Excel 输入，`--attempts` 默认为10次、允许1至10次，支持 `--resume`。输出目录包含 `final_summaries.json`（资讯序号、ID、标题、运行状态和每条最终汇总）及 `run_metadata.json`（运行参数与成功/失败条数）。

### 单条资讯

准备 UTF-8 编码的 `article_input.json`：

```json
{
  "title": "多家机器人企业发布人形机器人新品",
  "content": "多家企业发布人形机器人新品，核心零部件和产业链商业化进展受到关注。"
}
```

```powershell
.\.venv\Scripts\python.exe src\main.py --mode flow --input-file article_input.json
```

输出为包含 `final_summary` 的 JSON 对象。标题和正文去除首尾空白，正文默认保留前 5000 个字符；“蓝字”、资讯 ID、生产时间用于原始数据对照，不拼入模型输入。

### HTTP 服务

```powershell
.\.venv\Scripts\python.exe src\main.py --mode http --host 127.0.0.1 --port 5000
```

调用示例：

```powershell
$requestBody = @{
  title = "资讯标题"
  content = "资讯正文"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:5000/run -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($requestBody))
```

健康检查：`GET /health`。接口文档：`http://127.0.0.1:5000/docs`。单条 CLI 和 HTTP 输入均支持可选的 `page_contents` 对象。

## 工作流流程

各阶段按顺序执行，专题和热点命中与否都会继续进入分类主链路。

1. **专题处理**：本地词面召回最多 5 个候选，过滤低于 0.05 的召回分数；LLM 判断适配，规则检查嵌套跳转屏蔽。有效命中时仅生成1个首选词及10字以内短版。
2. **热点题材处理**：专题处理结束后，执行热点召回、LLM 适配判断和屏蔽检查；有效命中时仅生成1个首选词及10字以内短版。
3. **优先方向判断**：只有专题命中时选择 `topic`，只有热点命中时选择 `hot_theme`，均未命中时为 `none`；双命中时由 LLM 选择专题或热点。
4. **分类与校验**：MiniLM 在 82 个主题中给出预测。最高相似度达到门槛时，LLM 在 Top-5 内校验；低于门槛或模型不可用时，LLM 先从完整目录提议分类，再独立审核提议和备选。无提议时审核完整目录，确认没有适合类目则进入 AI 生词。
5. **按象限生词**：接受的类目按固定映射进入 A/B/C/D；B/D 先尝试拣选相邻 A 类主题作为内容参照，参照的模型相似度至少为 0.5，无有效参照时仍按原象限生词。
6. **最终汇总**：返回 `final_summary`，包含专题、热点、分类生词三部分及专题/热点的优先方向。

正常分类路径中，原始 Top-1 和 Top-2 都达到门槛、差距不超过 `CATEGORY_DUAL_MAX_GAP`，且 LLM 确认两类各有正文依据时，允许双归类。两组按各自象限独立生词，主分类结果在前。

| 生词分支 | 象限含义 | 目标词数 | 完整链路规则 |
|---|---|---|---|
| A | 高曝光·高点击 | 最多3 | 可少出，需至少一个合格词 |
| B | 高曝光·低点击 | 最多3 | 可少出，需至少一个合格词 |
| C | 低曝光·高点击 | 最多3 | 可少出，需至少一个合格词 |
| D | 低曝光·低点击 | 0–3 | 允许推荐词和 query 同时为空，保留成功分类 |
| 未归类 AI | 无适配类目 | 最多3 | 每个输出词都配一条完整 query；跑批要求非空词列表 |

### 生词依据与延伸要求

专题/热点生词以跳转目标内容为重点：按 URL 传入的 `page_contents` 优先，其次为清单页面主要内容，再使用专题导语/热点释义。原资讯提供阅读起点和已知信息边界，推荐词指向原文之外且目标内容可以承接的延伸阅读。

分类和 AI 生词的推荐词及对应 query 必须体现原文尚未回答的需求，例如机制、比较、应用条件或验证方法；推荐词本身就要有延伸意图，避免摘词、缩写标题、同义复述或把已披露事实改成问句。延伸需与资讯有可解释的联系，不能编造事实或确定性预测。

推荐词按优先级排列。专题/热点提供跳转链接和展示文案；分类结果在 `category_quadrant.generation_results` 中逐组保存，`ai_queries` 按同词同序提供 `{word, query}`。D 象限空结果同时使用 `recommended_words=[]`、`ai_queries=[]`，不因空词触发重试。

## 项目目录

```text
config/       各 LLM 节点的提示词与模型参数
assets/       业务清单、象限映射和资讯输入文件
分类器/       分类目录、主题中心与模型元数据
src/          主图、节点、公共工具及单条/HTTP 入口
scripts/      CSV 与 JSON 跑批入口
tests/        工作流和输出校验测试
outputs/      本地跑批结果
.env.example  环境变量配置模板
```

本地验证：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```
