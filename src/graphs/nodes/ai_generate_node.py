"""AI生词节点：无法归入现有类目的资讯，由AI自由发散生成推荐出词。"""
from typing import Any, List

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import AiGenerateInput, AiGenerateOutput
from tools.llm_utils import invoke_llm, extract_json, ensure_workspace_path
from tools.word_extras import ai_queries


def ai_generate_node(
    state: AiGenerateInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> AiGenerateOutput:
    """
    title: AI生词
    desc: 对无法归入现有类目的新型资讯，由AI基于内容发散生成3–5个推荐词及对应query，合格方向不足时允许少出
    integrations: 大语言模型
    """
    ctx = runtime.context
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])

    render_vars = {"title": state.title, "content": state.content, "repair_note": ""}
    for attempt in range(2):
        data: Any = extract_json(invoke_llm(ctx, cfg_path, render_vars)) or {}
        words: List[str] = list(dict.fromkeys(
            str(w).strip() for w in (data.get("recommended_words") or []) if str(w).strip()
        ))
        try:
            queries = ai_queries(data, words, state.title + state.content)
            break
        except ValueError as exc:
            if attempt:
                raise
            render_vars["repair_note"] = f"上次输出不合规：{exc}。请重新输出完整 JSON，每个词配一条完整问句。"
    return AiGenerateOutput(
        recommended_words=words,
        ai_queries=queries,
        word_source="ai",
        generation_reason=str(
            data.get("generation_reason")
            or "采用AI兜底策略：不依赖象限类目，仅依据标题和正文，从事件主体、进展、原因、影响等角度生成有事实支撑的搜索词。"
        ).strip(),
    )
