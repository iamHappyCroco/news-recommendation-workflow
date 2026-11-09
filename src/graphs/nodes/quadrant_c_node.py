"""象限C生词节点：低曝光高点击类目，生成深度精准词。"""

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import QuadrantInput, QuadrantOutput
from tools.llm_utils import invoke_llm, ensure_workspace_path
from tools.word_extras import invoke_words_with_queries


def quadrant_c_node(
    state: QuadrantInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> QuadrantOutput:
    """
    title: 象限C生词(低曝光高点击)
    desc: 面向曝光有限但点击转化高的高价值细分，结合一级/二级类目与代表词，生成深度精准推荐出词
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
        word_source="quadrant_C",
        generation_reason=str(
            data.get("generation_reason")
            or "采用深度精准策略：保留高点击主题的专业性，生成颗粒度更细、意图更明确的深度搜索词。"
        ).strip(),
    )
