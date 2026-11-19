"""AI生词节点：无法归入现有类目的资讯，由AI自由发散生成推荐出词。"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import AiGenerateInput, AiGenerateOutput
from tools.llm_utils import invoke_llm, ensure_workspace_path
from tools.word_extras import invoke_words_with_queries


def ai_generate_node(
    state: AiGenerateInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> AiGenerateOutput:
    """
    title: AI生词
    desc: 对无法归入现有类目的新型资讯，由AI基于内容发散生成最多3个推荐词及对应query，合格方向不足时允许少出
    integrations: 大语言模型
    """
    ctx = runtime.context
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])

    data, words, queries = invoke_words_with_queries(
        invoke_llm, ctx, cfg_path, {"title": state.title, "content": state.content},
    )
    return AiGenerateOutput(
        recommended_words=words,
        ai_queries=queries,
        short_words=data["short_words"],
        word_source="ai",
        generation_reason=str(
            data.get("generation_reason")
            or "采用AI兜底策略：不依赖象限类目，仅依据标题和正文，从事件主体、进展、原因、影响等角度生成有事实支撑的搜索词。"
        ).strip(),
    )
