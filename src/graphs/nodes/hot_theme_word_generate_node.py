"""热点题材生词节点：命中后生成3个候选词，第1个为首选词。"""
from typing import Any, List

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import HotThemeWordGenerateInput, HotThemeWordGenerateOutput
from tools.llm_utils import invoke_llm, extract_json, ensure_workspace_path
from tools.word_extras import display_variants


def hot_theme_word_generate_node(
    state: HotThemeWordGenerateInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> HotThemeWordGenerateOutput:
    """
    title: 热点题材相关搜索词生成
    desc: 命中热点题材后生成3个候选词并按优先级排序，第1个为首选词
    integrations: 大语言模型
    """
    ctx = runtime.context
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])

    render_vars = {
        "title": state.title,
        "content": state.content,
        "fit_hot_theme_name": state.fit_hot_theme_name,
        "fit_hot_theme_url": state.fit_hot_theme_url,
        "interpretation": state.interpretation,
        "target_page_content": (state.page_contents.get(state.fit_hot_theme_url, "").strip()
                                or state.fit_hot_theme_page_content.strip() or state.interpretation.strip()),
        "target_content_source": ("传入的页面主要内容" if state.page_contents.get(state.fit_hot_theme_url, "").strip()
                                  or state.fit_hot_theme_page_content.strip() else "热点释义摘要"),
        "repair_note": "",
    }
    for attempt in range(2):
        data: Any = extract_json(invoke_llm(ctx, cfg_path, render_vars)) or {}
        words: List[str] = list(dict.fromkeys(
            str(w).strip() for w in (data.get("recommended_words") or []) if str(w).strip()
        ))[:3]
        try:
            if len(words) != 3:
                raise ValueError("热点题材必须生成 3 个不重复推荐词")
            variants = display_variants(data, words)
            break
        except ValueError as exc:
            if attempt:
                raise
            render_vars["repair_note"] = f"上次输出不合规：{exc}。请重新输出完整 JSON，并逐字核对展示版。"

    hot_theme_hit: dict = {
        "word_source": "hot_theme",
        "fit_hot_theme_name": state.fit_hot_theme_name,
        "fit_hot_theme_url": state.fit_hot_theme_url,
        "recommended_words": words,
        "display_variants": variants,
        "best_word": words[0] if words else "",
    }

    return HotThemeWordGenerateOutput(
        recommended_words=words,
        word_source="hot_theme",
        hot_theme_hit_result=hot_theme_hit,
    )
