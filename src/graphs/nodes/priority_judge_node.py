"""优先出词判断节点：当专题与热点题材均命中时，由 LLM 判断应优先出跳转专题词还是跳转热点词，并给出理由。"""

import os
from typing import Any, List

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import PriorityDecisionInput, PriorityDecisionOutput
from tools.llm_utils import invoke_llm, extract_json, ensure_workspace_path


def priority_judge_node(
    state: PriorityDecisionInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> PriorityDecisionOutput:
    """
    title: 专题/热点优先出词判断
    desc: 专题与热点题材均命中时，结合资讯内容判定优先出专题词还是热点词并给出理由；仅命中其一则直接按唯一命中方向输出
    integrations: 大语言模型
    """
    ctx = runtime.context

    if not state.has_fit_topic and not state.has_fit_hot_theme:
        return PriorityDecisionOutput(
            out_priority="none",
            out_priority_reason="专题与热点均未命中，无需优先判断",
        )

    if state.has_fit_topic and not state.has_fit_hot_theme:
        return PriorityDecisionOutput(
            out_priority="topic",
            out_priority_reason="仅命中专题，优先输出专题相关跳转词",
        )

    if not state.has_fit_topic and state.has_fit_hot_theme:
        return PriorityDecisionOutput(
            out_priority="hot_theme",
            out_priority_reason="仅命中热点题材，优先输出热点题材相关跳转词",
        )

    # 专题与热点均命中：交由 LLM 判断
    topic_words: List[str] = []
    t_hit: dict = state.topic_hit_result or {}
    tw = t_hit.get("recommended_words")
    if isinstance(tw, list):
        topic_words = [str(w).strip() for w in tw if str(w).strip()]

    hot_theme_words: List[str] = []
    h_hit: dict = state.hot_theme_hit_result or {}
    hw = h_hit.get("recommended_words")
    if isinstance(hw, list):
        hot_theme_words = [str(w).strip() for w in hw if str(w).strip()]

    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])
    raw: Any = invoke_llm(
        ctx,
        cfg_path,
        {
            "title": state.title,
            "content": state.content,
            "fit_topic_title": state.fit_topic_title,
            "topic_words": "、".join(topic_words),
            "fit_hot_theme_name": state.fit_hot_theme_name,
            "hot_theme_words": "、".join(hot_theme_words),
        },
    )
    data: Any = extract_json(raw) or {}
    out_priority: str = str(data.get("out_priority") or "topic").strip()
    if out_priority not in ("topic", "hot_theme"):
        out_priority = "topic"
    out_priority_reason: str = str(data.get("out_priority_reason") or "").strip()

    return PriorityDecisionOutput(
        out_priority=out_priority,
        out_priority_reason=out_priority_reason,
    )
