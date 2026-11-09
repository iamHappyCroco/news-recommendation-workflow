"""象限A生词节点：高曝光高点击类目，生成热点热搜词。"""

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import QuadrantInput, QuadrantOutput
from tools.llm_utils import invoke_llm, ensure_workspace_path
from tools.word_extras import invoke_words_with_queries


def quadrant_a_node(
    state: QuadrantInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> QuadrantOutput:
    """
    title: 象限A生词(高曝光高点击)
    desc: 面向曝光点击双高的头部热点，结合一级/二级类目与代表词，生成热点热搜推荐出词
    integrations: 大语言模型
    """
    ctx = runtime.context
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])

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
        },
    )
    return QuadrantOutput(
        recommended_words=words,
        ai_queries=queries,
        word_source="quadrant_A",
        generation_reason=str(
            data.get("generation_reason")
            or "采用热点热搜策略：围绕高曝光高点击主题，优先生成强时效、强相关、适合继续搜索的热点词。"
        ).strip(),
    )
