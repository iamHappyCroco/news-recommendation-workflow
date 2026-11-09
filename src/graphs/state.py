"""全局状态、图出入参、节点出入参定义。"""
from typing import List, Literal
from pydantic import BaseModel, Field


class TopicCandidate(BaseModel):
    """RAG 召回的候选专题。"""
    topic_title: str = Field(..., description="专题标题")
    intro: str = Field(default="", description="专题导语，作为目标内容摘要")
    page_content: str = Field(default="", description="可选的专题页面主要内容")
    topic_url: str = Field(default="", description="专题跳转链接")
    score: float = Field(default=0.0, description="RAG 相似度得分")


class NearbyTopic(BaseModel):
    """模型向量层面与目标资讯相近的相邻主题候选（用于低曝光象限找高曝光高点击参考）。"""
    leaf_id: int = Field(..., description="相邻主题的二级主题 id")
    leaf_name: str = Field(..., description="相邻主题的二级主题名称")
    parent_id: int = Field(default=-1, description="相邻主题所属一级主题 id")
    parent_name: str = Field(default="", description="相邻主题所属一级主题名称")
    similarity: float = Field(default=0.0, description="与目标资讯质心的余弦相似度")
    representative_searches: List[str] = Field(default=[], description="该相邻主题的代表搜索词")


class ClassificationCandidate(BaseModel):
    """A native MiniLM topic accepted by the Top-5 semantic verifier."""
    leaf_id: int
    category: str
    sub_category: str
    region: str
    quadrant: str
    similarity: float = 0.0
    reason: str = ""
    keywords: str = ""
    rep_searches: str = ""


class ClassificationDiagnostics(BaseModel):
    """Review flags are diagnostics, not an unconditional classification veto."""
    model_second_similarity: float = 0.0
    model_cosine_gap: float = 0.0
    model_training_support: int = 0
    model_similarity_cutoff: float = 0.6542854905128479
    model_review_reasons: List[str] = Field(default_factory=list)


class ClassificationFallback(BaseModel):
    llm_fallback_used: bool = False
    llm_fallback_proposals: List[dict] = Field(default_factory=list)
    llm_fallback_reason: str = ""


class ClassificationDecision(ClassificationFallback):
    classification_source: str = ""
    classification_mode: str = "none"
    classification_reason: str = ""
    classification_candidates: List[ClassificationCandidate] = Field(default_factory=list)


class HotThemeCandidate(BaseModel):
    """从东方财富热点题材文件中召回的热点题材候选。"""
    theme_code: int = Field(default=-1, description="热点题材代码")
    theme_name: str = Field(..., description="热点题材名称")
    interpretation: str = Field(default="", description="热点释义")
    page_content: str = Field(default="", description="可选的热点页面主要内容")
    url: str = Field(default="", description="热点题材详情页链接")
    score: float = Field(default=0.0, description="与资讯的向量相似度")


class GlobalState(ClassificationDiagnostics, ClassificationDecision):
    """全局过程状态，LangGraph 会自动将节点输出合并到此。"""
    title: str = Field(default="", description="资讯标题")
    content: str = Field(default="", description="资讯正文")
    page_contents: dict[str, str] = Field(default_factory=dict, description="按目标URL提供的页面主要内容，优先于清单摘要")
    fit_topic_intro: str = Field(default="", description="命中专题导语")
    fit_topic_page_content: str = Field(default="", description="命中专题的页面主要内容")
    interpretation: str = Field(default="", description="命中热点题材释义")
    fit_hot_theme_page_content: str = Field(default="", description="命中热点的页面主要内容")
    topic_candidates: List[TopicCandidate] = Field(default=[], description="RAG 召回的候选专题列表")
    has_fit_topic: bool = Field(default=False, description="是否存在适配专题")
    fit_topic_title: str = Field(default="", description="适配专题标题")
    fit_topic_url: str = Field(default="", description="适配专题跳转链接")
    topic_blocked: bool = Field(default=False, description="是否因资讯标题命中专题清单「文章标题」列而屏蔽用该专题生词（规则匹配，非大模型）")
    topic_hit_result: dict = Field(default={}, description="专题适配时额外生成的专题生词结果（推荐词按优先级排序，含展示版与首选词，不含理由），未适配为空")
    hot_theme_candidates: List[HotThemeCandidate] = Field(default=[], description="从东方财富热点题材文件召回的候选热点题材")
    has_fit_hot_theme: bool = Field(default=False, description="是否存在适合跳转的热点题材")
    fit_hot_theme_code: int = Field(default=-1, description="适配热点题材代码")
    fit_hot_theme_name: str = Field(default="", description="适配热点题材名称")
    fit_hot_theme_url: str = Field(default="", description="适配热点题材详情页链接")
    hot_theme_hit_result: dict = Field(default={}, description="热点题材适配时额外生成的热点题材搜索词结果，未适配为空")
    out_priority: str = Field(default="none", description="优先出词判断：topic/hot_theme/none（专题与热点双命中时由 LLM 判定优先出专题词还是热点词）")
    out_priority_reason: str = Field(default="", description="优先出词判断理由")
    hot_theme_blocked: bool = Field(default=False, description="是否因命中嵌套跳转屏蔽清单而屏蔽用该热点生词（规则匹配，非大模型）")
    final_summary: dict = Field(default={}, description="最终汇总输出：专题相关/热点题材相关/分类+按象限生词三部分，各部分含跳转链接")
    category_candidate_results: List[dict] = Field(default_factory=list)
    category: str = Field(default="", description="资讯归属一级类目")
    sub_category: str = Field(default="", description="资讯归属二级类目")
    keywords: str = Field(default="", description="该类目代表关键词")
    rep_searches: str = Field(default="", description="该类目代表搜索词")
    region: str = Field(default="", description="类目所属四象限区域(A/B/C/D)")
    quadrant: str = Field(default="", description="分类到的四象限含义(如 象限D·低曝光·低点击)，无法归类/专题命中时为空")
    can_classify: bool = Field(default=False, description="资讯是否能归入现有类目")
    # 模型分类中间结果（供 LLM 校验节点使用）
    model_available: bool = Field(default=False, description="分类模型是否可用并返回了预测结果")
    model_is_noise: bool = Field(default=False, description="模型是否判定该资讯为噪声")
    model_leaf_id: int = Field(default=-1, description="模型预测的二级主题 leaf_id")
    model_leaf_name: str = Field(default="", description="模型预测的二级主题名称")
    model_parent_name: str = Field(default="", description="模型预测的所属一级主题名称")
    model_similarity: float = Field(default=0.0, description="模型预测与质心的 cosine 相似度")
    model_top_terms: List[str] = Field(default=[], description="模型预测二级主题的代表关键词")
    model_rep_searches: List[str] = Field(default=[], description="模型预测二级主题的代表搜索词")
    model_pred_text: str = Field(default="", description="拼装好用于给 LLM 校验的模型预测描述文本")
    nearby_topics: List[NearbyTopic] = Field(default=[], description="向量最近 Top-K 相邻主题候选")
    recommended_words: List[str] = Field(default=[], description="最终推荐出词")
    ai_queries: List[dict] = Field(default_factory=list, description="分类生词所有分支逐词对应的问答 query，与推荐词同词同序")
    word_source: str = Field(default="", description="分类象限出词来源")
    generation_reason: str = Field(default="", description="分类象限或AI兜底推荐词的整体生成理由")
    has_hot_topic: bool = Field(default=False, description="是否找到相邻高曝光参考主题")
    hot_leaf_id: int = Field(default=-1, description="参考高曝光主题 id")
    hot_leaf_name: str = Field(default="", description="参考高曝光主题名称")
    hot_parent_name: str = Field(default="", description="参考高曝光主题一级类目")
    hot_region: str = Field(default="", description="参考主题区域")
    hot_similarity: float = Field(default=0.0, description="参考主题相似度")
    hot_rep_searches: List[str] = Field(default=[], description="参考主题代表搜索词")
    hot_keywords: str = Field(default="", description="参考主题关键词")
    hot_reference_text: str = Field(default="", description="参考主题提示文本")


class HotTopicSelectInput(BaseModel):
    """低曝光象限参考主题拣选节点输入（在向量 Top-K 相邻主题中挑高曝光高点击者）。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")
    nearby_topics: List[NearbyTopic] = Field(default=[], description="向量最近 Top-K 相邻主题候选")
    region: str = Field(default="", description="资讯当前所属区域(应为 B 或 D)")


class HotTopicSelectOutput(BaseModel):
    """低曝光象限参考主题拣选节点输出。"""
    has_hot_topic: bool = Field(..., description="是否拣选到相邻高曝光高点击参考主题")
    hot_leaf_id: int = Field(default=-1, description="参考高曝光主题的二级主题 id")
    hot_leaf_name: str = Field(default="", description="参考高曝光主题名称")
    hot_parent_name: str = Field(default="", description="参考高曝光主题所属一级主题名称")
    hot_region: str = Field(default="", description="参考高曝光主题所属区域")
    hot_similarity: float = Field(default=0.0, description="参考高曝光主题与资讯的余弦相似度")
    hot_rep_searches: List[str] = Field(default=[], description="参考高曝光主题的代表搜索词")
    hot_keywords: str = Field(default="", description="参考高曝光主题代表关键词")
    hot_reference_text: str = Field(default="", description="拼装好的参考主题描述文本，供生词提示词注入")


class GraphInput(BaseModel):
    """工作流输入：单条资讯。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")
    page_contents: dict[str, str] = Field(default_factory=dict, description="可选：按跳转URL提供页面主要内容，不抓取网页")


class GraphOutput(BaseModel):
    """Return the business summary once; process state remains inside the graph."""
    final_summary: dict = Field(default={}, description="最终汇总：专题相关/热点题材相关/分类+按象限生词，各部分含跳转链接")


class TopicRecallInput(BaseModel):
    """专题RAG召回节点输入。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")


class TopicRecallOutput(BaseModel):
    """专题RAG召回节点输出。"""
    topic_candidates: List[TopicCandidate] = Field(default=[], description="RAG 召回的候选专题列表")


class TopicFitJudgeInput(BaseModel):
    """专题适配判断节点输入。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")
    topic_candidates: List[TopicCandidate] = Field(default=[], description="RAG 召回的候选专题列表")


class TopicFitJudgeOutput(BaseModel):
    """专题适配判断节点输出。"""
    has_fit_topic: bool = Field(..., description="是否存在适配专题")
    fit_topic_title: str = Field(default="", description="适配专题标题")
    fit_topic_url: str = Field(default="", description="适配专题跳转链接")
    fit_topic_intro: str = Field(default="", description="命中候选的专题导语")
    fit_topic_page_content: str = Field(default="", description="命中候选的专题页面主要内容")
    recommended_words: List[str] = Field(default=[], description="命中专题时为[专题标题]，否则为空")
    word_source: str = Field(default="", description="出词来源，命中时固定为 topic")


class TopicWordGenerateInput(BaseModel):
    """专题相关搜索词生成节点输入。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")
    fit_topic_title: str = Field(..., description="适配专题标题")
    fit_topic_url: str = Field(default="", description="适配专题跳转链接")
    fit_topic_intro: str = Field(default="", description="专题导语，页面内容未提供时作为摘要依据")
    fit_topic_page_content: str = Field(default="", description="可直接传入的专题页面主要内容")
    page_contents: dict[str, str] = Field(default_factory=dict, description="按目标URL提供的页面主要内容")


class TopicWordGenerateOutput(BaseModel):
    """专题相关搜索词生成节点输出。"""
    recommended_words: List[str] = Field(default=[], description="基于专题主题生成的3个候选词，第1个为首选词")
    word_source: str = Field(default="topic", description="出词来源，固定为 topic")
    topic_hit_result: dict = Field(default={}, description="专题搜索词额外输出")


class ModelClassifyInput(BaseModel):
    """模型分类节点输入。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")


class ModelClassifyOutput(ClassificationDiagnostics):
    """模型分类节点输出（供 LLM 校验节点使用）。"""
    model_available: bool = Field(..., description="分类模型是否可用并返回了预测结果")
    model_is_noise: bool = Field(default=False, description="模型是否判定该资讯为噪声")
    model_leaf_id: int = Field(default=-1, description="模型预测的二级主题 leaf_id")
    model_leaf_name: str = Field(default="", description="模型预测的二级主题名称")
    model_parent_name: str = Field(default="", description="模型预测的所属一级主题名称")
    model_similarity: float = Field(default=0.0, description="模型预测与质心的 cosine 相似度")
    model_top_terms: List[str] = Field(default=[], description="模型预测二级主题的代表关键词")
    model_rep_searches: List[str] = Field(default=[], description="模型预测二级主题的代表搜索词")
    model_pred_text: str = Field(default="", description="拼装好用于给 LLM 校验的模型预测描述文本")
    nearby_topics: List[NearbyTopic] = Field(default=[], description="向量最近 Top-K 相邻主题候选")


class LlmVerifyInput(ClassificationDiagnostics, ClassificationFallback):
    """LLM 校验节点输入。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")
    model_available: bool = Field(default=False, description="分类模型是否可用并返回了预测结果")
    model_is_noise: bool = Field(default=False, description="模型是否判定该资讯为噪声")
    model_leaf_id: int = Field(default=-1, description="模型预测的二级主题 leaf_id")
    model_leaf_name: str = Field(default="", description="模型预测的二级主题名称")
    model_rep_searches: List[str] = Field(default=[], description="模型预测二级主题的代表搜索词")
    model_pred_text: str = Field(default="", description="用于给 LLM 校验的模型预测描述文本")
    model_similarity: float = Field(default=0.0, description="模型首选主题相似度")
    nearby_topics: List[NearbyTopic] = Field(default=[], description="分类器 Top-K 相邻主题，用于缩小 LLM 校验候选集")


class LlmVerifyOutput(ClassificationDecision):
    """LLM 校验节点输出。"""
    can_classify: bool = Field(..., description="资讯是否能归入现有类目")
    category: str = Field(default="", description="资讯归属一级类目")
    sub_category: str = Field(default="", description="资讯归属二级类目")
    region: str = Field(default="", description="类目所属四象限区域")
    quadrant: str = Field(default="", description="分类到的四象限含义(如 象限D·低曝光·低点击)")
    keywords: str = Field(default="", description="该类目代表关键词")
    rep_searches: str = Field(default="", description="该类目代表搜索词")
    extracted_search: str = Field(default="", description="从资讯提炼的核心搜索意图词")


class QuadrantInput(BaseModel):
    """四象限生词节点输入（四个象限节点复用）。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")
    category: str = Field(default="", description="归属一级类目")
    sub_category: str = Field(default="", description="归属二级类目")
    region: str = Field(default="", description="四象限区域")
    keywords: str = Field(default="", description="该类目代表关键词")
    rep_searches: str = Field(default="", description="该类目代表搜索词")
    # 低曝光象限（B/D）的相邻高曝光高点击参考主题（由热主题拣选节点产出，缺失时为空）
    has_hot_topic: bool = Field(default=False, description="是否存在相邻高曝光参考主题")
    hot_leaf_name: str = Field(default="", description="参考高曝光主题名称")
    hot_rep_searches: List[str] = Field(default=[], description="参考高曝光主题的代表搜索词")
    hot_keywords: str = Field(default="", description="参考高曝光主题代表关键词")
    hot_reference_text: str = Field(default="", description="参考主题描述文本")


class QuadrantOutput(BaseModel):
    """四象限生词节点输出（四个象限节点复用）。"""
    recommended_words: List[str] = Field(default=[], description="本象限策略生成的推荐出词")
    ai_queries: List[dict] = Field(default_factory=list, description="与推荐词同词同序的完整问答 query")
    word_source: str = Field(default="", description="出词来源")
    generation_reason: str = Field(default="", description="本次推荐词的整体生成理由")


class AiGenerateInput(BaseModel):
    """AI 生词节点输入。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")


class AiGenerateOutput(BaseModel):
    """AI 生词节点输出。"""
    recommended_words: List[str] = Field(default=[], description="AI 生成的推荐出词")
    ai_queries: List[dict] = Field(default_factory=list, description="与推荐词同词同序的延伸问句")
    word_source: str = Field(default="", description="出词来源，固定为 ai")
    generation_reason: str = Field(default="", description="AI兜底推荐词的整体生成理由")


class TopicFitRoute(BaseModel):
    """专题适配判定分支的入参。"""
    has_fit_topic: bool = Field(..., description="是否存在适配专题")


class TopicBlockInput(BaseModel):
    """专题屏蔽规则节点输入：资讯标题命中专题清单「文章标题」列对应标题时屏蔽该专题生词。"""
    title: str = Field(..., description="资讯标题")
    has_fit_topic: bool = Field(default=False, description="是否存在适配专题（由上游专题适配判断输出）")
    fit_topic_title: str = Field(default="", description="适配专题标题")


class TopicBlockOutput(BaseModel):
    """专题屏蔽规则节点输出。"""
    has_fit_topic: bool = Field(default=False, description="经屏蔽判断后是否仍用专题生词（命中屏蔽时置为 false）")
    topic_blocked: bool = Field(default=False, description="是否因资讯标题命中专题清单「文章标题」列而屏蔽该专题")


class RegionRoute(BaseModel):
    """四象限/AI生词路由分支的入参。"""
    can_classify: bool = Field(..., description="是否能归入现有类目")
    region: str = Field(default="", description="类目所属四象限区域")


class HotThemeRecallInput(BaseModel):
    """热点题材召回节点输入。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")


class HotThemeRecallOutput(BaseModel):
    """热点题材召回节点输出。"""
    hot_theme_candidates: List[HotThemeCandidate] = Field(default=[], description="召回的热点题材候选列表")


class HotThemeFitJudgeInput(BaseModel):
    """热点题材适配判断节点输入。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")
    hot_theme_candidates: List[HotThemeCandidate] = Field(default=[], description="召回的热点题材候选")


class HotThemeFitJudgeOutput(BaseModel):
    """热点题材适配判断节点输出。"""
    has_fit_hot_theme: bool = Field(..., description="是否存在适合跳转的热点题材")
    fit_hot_theme_code: int = Field(default=-1, description="适配热点题材代码")
    fit_hot_theme_name: str = Field(default="", description="适配热点题材名称")
    fit_hot_theme_url: str = Field(default="", description="适配热点题材详情页链接")
    interpretation: str = Field(default="", description="命中候选的热点释义")
    fit_hot_theme_page_content: str = Field(default="", description="命中候选的热点页面主要内容")


class FinalSummaryInput(ClassificationDiagnostics, ClassificationDecision):
    """最终汇总节点输入：聚合专题相关、热点题材相关与分类+按象限生词结果。"""
    category_candidate_results: List[dict] = Field(default_factory=list)
    has_fit_topic: bool = Field(default=False, description="是否存在适配专题")
    fit_topic_title: str = Field(default="", description="适配专题标题")
    fit_topic_url: str = Field(default="", description="适配专题跳转链接")
    topic_hit_result: dict = Field(default={}, description="专题适配时的额外专题生词结果")
    has_fit_hot_theme: bool = Field(default=False, description="是否存在适合跳转的热点题材")
    fit_hot_theme_name: str = Field(default="", description="适配热点题材名称")
    fit_hot_theme_url: str = Field(default="", description="适配热点题材详情页链接")
    hot_theme_hit_result: dict = Field(default={}, description="热点题材适配时的额外热点题材搜索词结果")
    category: str = Field(default="", description="归属一级类目")
    sub_category: str = Field(default="", description="归属二级类目")
    region: str = Field(default="", description="四象限区域")
    quadrant: str = Field(default="", description="四象限含义")
    word_source: str = Field(default="", description="主口径出词来源")
    recommended_words: List[str] = Field(default=[], description="主口径推荐出词")
    ai_queries: List[dict] = Field(default_factory=list, description="分类生词所有分支逐词对应的问答 query")
    generation_reason: str = Field(default="", description="主口径推荐词的整体生成理由")
    out_priority: str = Field(default="none", description="优先出词判断：topic/hot_theme/none（专题与热点双命中时为 LLM 判定结果）")
    out_priority_reason: str = Field(default="", description="优先出词判断理由")
    hot_theme_blocked: bool = Field(default=False, description="是否因命中嵌套跳转屏蔽清单而屏蔽用该热点生词（规则匹配，非大模型）")


class CategoryCandidateGenerateOutput(BaseModel):
    category_candidate_results: List[dict] = Field(default_factory=list)


class PriorityDecisionInput(BaseModel):
    """优先出词判断节点输入：在专题与热点题材均命中时，判断应优先出专题词还是热点词。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")
    has_fit_topic: bool = Field(default=False, description="是否存在适配专题")
    fit_topic_title: str = Field(default="", description="适配专题标题")
    topic_hit_result: dict = Field(default={}, description="专题生词结果（推荐词按优先级排序，不含理由）")
    has_fit_hot_theme: bool = Field(default=False, description="是否存在适合跳转的热点题材")
    fit_hot_theme_name: str = Field(default="", description="适配热点题材名称")
    hot_theme_hit_result: dict = Field(default={}, description="热点题材搜索结果（推荐词按优先级排序，不含理由）")


class PriorityDecisionOutput(BaseModel):
    """优先出词判断节点输出。"""
    out_priority: str = Field(default="none", description="优先出词判断：topic/hot_theme/none（none 表示无需优先判断）")
    out_priority_reason: str = Field(default="", description="优先出词判断理由")


class HotThemeBlockInput(BaseModel):
    """热点题材屏蔽规则节点输入：命中嵌套跳转屏蔽清单时屏蔽用该热点生词。"""
    title: str = Field(..., description="资讯标题")
    has_fit_hot_theme: bool = Field(default=False, description="是否存在适合跳转的热点题材（由上游热主题判断输出）")
    fit_hot_theme_name: str = Field(default="", description="适配热点题材名称")


class HotThemeBlockOutput(BaseModel):
    """热点题材屏蔽规则节点输出。"""
    has_fit_hot_theme: bool = Field(default=False, description="经屏蔽判断后是否仍用热点生词（命中屏蔽清单时置为 false）")
    hot_theme_blocked: bool = Field(default=False, description="是否命中嵌套跳转屏蔽清单并屏蔽该热点")


class FinalSummaryOutput(BaseModel):
    """最终汇总节点输出。"""
    final_summary: dict = Field(default={}, description="最终汇总：专题相关/热点题材相关/分类+按象限生词")
    classification_source: Literal["实际分类器分类", "LLM兜底分类", "未归类"] = Field(default="未归类", description="最终分类途径，与汇总一致")
    region: str = Field(default="none", description="最终归类象限展示，与 final_summary 一致")
    quadrant: str = Field(default="未归类（AI生词）", description="最终归类象限说明，与 final_summary 一致")
    fit_hot_theme_code: int = Field(default=-1, description="适配热点题材代码")
    fit_hot_theme_name: str = Field(default="", description="适配热点题材名称")
    fit_hot_theme_url: str = Field(default="", description="适配热点题材详情页链接")


class HotThemeWordGenerateInput(BaseModel):
    """热点题材相关搜索词生成节点输入。"""
    title: str = Field(..., description="资讯标题")
    content: str = Field(..., description="资讯正文")
    fit_hot_theme_name: str = Field(default="", description="适配热点题材名称")
    fit_hot_theme_url: str = Field(default="", description="适配热点题材详情页链接")
    interpretation: str = Field(default="", description="适配热点题材的热点释义")
    fit_hot_theme_page_content: str = Field(default="", description="可直接传入的热点页面主要内容")
    page_contents: dict[str, str] = Field(default_factory=dict, description="按目标URL提供的页面主要内容")


class HotThemeWordGenerateOutput(BaseModel):
    """热点题材相关搜索词生成节点输出。"""
    recommended_words: List[str] = Field(default=[], description="基于热点题材生成的3个候选词，第1个为首选词")
    word_source: str = Field(default="", description="出词来源，固定为 hot_theme")
    hot_theme_hit_result: dict = Field(default={}, description="热点题材搜索结果（额外输出，推荐词按优先级排序，不含理由）")


class HasHotThemeFit(BaseModel):
    """热点题材适配判定分支的入参。"""
    has_fit_hot_theme: bool = Field(..., description="是否存在适合跳转的热点题材")
