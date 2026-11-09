"""专题适配判断节点：判断资讯是否与召回的候选专题高度适配。"""
import os
from typing import Any, List

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import TopicFitJudgeInput, TopicFitJudgeOutput
from tools.llm_utils import invoke_llm, extract_json, ensure_workspace_path


def topic_fit_judge_node(
    state: TopicFitJudgeInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> TopicFitJudgeOutput:
    """
    title: 专题适配判断
    desc: 结合召回候选专题列表，判断资讯是否存在可直接跳转的适配专题
    integrations: 大语言模型
    """
    ctx = runtime.context
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])

    candidates_text = "\n".join(
        f"- {c.topic_title}（相似度{c.score}，链接：{c.topic_url or '无'}）\n专题导语：{c.intro}" for c in state.topic_candidates
    )
    if not candidates_text:
        candidates_text = "（无召回候选）"

    raw = invoke_llm(
        ctx,
        cfg_path,
        {"title": state.title, "content": state.content, "candidates": candidates_text},
    )
    data: Any = extract_json(raw) or {}

    has_fit = bool(data.get("has_fit_topic", False))
    fit_title = str(data.get("fit_topic_title", "")).strip() if has_fit else ""
    fit_url = str(data.get("fit_topic_url", "")).strip() if has_fit else ""
    matched = next((candidate for candidate in state.topic_candidates
                    if candidate.topic_title == fit_title), None) if has_fit else None
    if matched is not None and not fit_url:
        fit_url = matched.topic_url

    words: List[str] = []
    if has_fit and fit_title:
        words = [fit_title]

    return TopicFitJudgeOutput(
        has_fit_topic=has_fit,
        fit_topic_title=fit_title,
        fit_topic_url=fit_url,
        fit_topic_intro=matched.intro if matched is not None else "",
        fit_topic_page_content=matched.page_content if matched is not None else "",
        recommended_words=words,
        word_source="topic" if has_fit else "",
    )
