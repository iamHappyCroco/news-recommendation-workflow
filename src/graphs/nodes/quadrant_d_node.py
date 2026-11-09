"""象限D生词节点：低曝光低点击类目，生成长尾探索词。"""

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import QuadrantInput, QuadrantOutput
from tools.llm_utils import invoke_llm, ensure_workspace_path
from tools.word_extras import invoke_words_with_queries


def quadrant_d_node(
    state: QuadrantInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> QuadrantOutput:
    """
    title: 象限D生词(低曝光低点击)
    desc: 面向长尾冷门内容，结合一级/二级类目与代表词，生成长尾探索推荐出词
    integrations: 大语言模型
    """
    ctx = runtime.context
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])

    hot_reference_text: str = state.hot_reference_text or ""
    data, words, queries = invoke_words_with_queries(
        invoke_llm, ctx,
        cfg_path,
        {
            "title": state.title,
            "content": state.content,
            "category": state.category,
            "sub_category": state.sub_category,
            "region": state.region,
            "keywords": state.keywords,
            "rep_searches": state.rep_searches,
            "has_hot_topic": bool(state.has_hot_topic),
            "hot_reference_text": hot_reference_text,
            "hot_leaf_name": state.hot_leaf_name or "",
            "hot_rep_searches": "、".join(state.hot_rep_searches or []),
            "hot_keywords": state.hot_keywords or "",
        },
    )
    return QuadrantOutput(
        recommended_words=words,
        ai_queries=queries,
        word_source="quadrant_D",
        generation_reason=str(
            data.get("generation_reason")
            or (
                f"采用关联探索策略：以相邻A类主题“{state.hot_leaf_name}”为重要内容参考，结合其类别含义与原资讯主体、事实及相邻搜索需求生成少量关联词。"
                if state.has_hot_topic
                else "采用长尾探索策略：从事件主体、进展、原因和影响中探索正文可支撑的细分长尾搜索词。"
            )
        ).strip(),
    )
