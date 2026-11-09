"""热点题材适配判断节点：结合召回的热点题材候选，判断资讯是否存在适合跳转的热点题材。"""
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import HotThemeFitJudgeInput, HotThemeFitJudgeOutput
from tools.llm_utils import invoke_llm, extract_json, ensure_workspace_path


def hot_theme_fit_judge_node(
    state: HotThemeFitJudgeInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> HotThemeFitJudgeOutput:
    """
    title: 热点题材适配判断
    desc: 结合召回候选热点题材，判断资讯是否存在适合跳转的热点题材
    integrations: 大语言模型
    """
    ctx = runtime.context
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])

    candidates_text = "\n".join(
        f"- {c.theme_name}（代码{c.theme_code}，释义：{c.interpretation}，链接：{c.url or '无'}）"
        for c in state.hot_theme_candidates
    )
    if not candidates_text:
        candidates_text = "（无召回候选）"

    raw = invoke_llm(
        ctx,
        cfg_path,
        {"title": state.title, "content": state.content, "candidates": candidates_text},
    )
    data: Any = extract_json(raw) or {}

    has_fit = bool(data.get("has_fit_hot_theme", False))
    code = int(data.get("fit_hot_theme_code", -1) or -1) if has_fit else -1
    name = str(data.get("fit_hot_theme_name", "")).strip() if has_fit else ""
    url = str(data.get("fit_hot_theme_url", "")).strip() if has_fit else ""

    # 反查兜底：LLM 若未回填名称/链接，用命中的候选题材信息补齐（候选自带真实名称与链接）
    matched = next(
        (c for c in state.hot_theme_candidates if str(c.theme_code) == str(code)),
        None,
    )
    if has_fit and matched is not None:
        if not name:
            name = matched.theme_name
        if not url:
            url = matched.url or ""

    return HotThemeFitJudgeOutput(
        has_fit_hot_theme=has_fit,
        fit_hot_theme_code=code,
        fit_hot_theme_name=name,
        fit_hot_theme_url=url,
        interpretation=matched.interpretation if has_fit and matched is not None else "",
        fit_hot_theme_page_content=matched.page_content if has_fit and matched is not None else "",
    )
