"""热点题材生词节点：命中后仅生成1个首选词及短版。"""
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import HotThemeWordGenerateInput, HotThemeWordGenerateOutput
from tools.llm_utils import invoke_llm, extract_json, ensure_workspace_path
from tools.word_extras import validate_jump_word


def hot_theme_word_generate_node(
    state: HotThemeWordGenerateInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> HotThemeWordGenerateOutput:
    """
    title: 热点题材相关搜索词生成
    desc: 命中热点题材后仅生成1个首选词及9字以内短版
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
        render_vars["_temperature_offset"] = 0.2 if attempt else 0.0
        data: Any = extract_json(invoke_llm(ctx, cfg_path, render_vars)) or {}
        try:
            word, short = validate_jump_word(data, require_metadata=True)
            break
        except ValueError as exc:
            if attempt:
                raise
            render_vars["repair_note"] = f"上次输出不合规：{exc}。请按系统格式输出article_link、candidates、word、short、subject五字段JSON；article_link说明资讯与页面的结合点。candidates包含2-3个不同构词类型的候选，默认名词短语、问句最多1个，不以禁止词结尾。word最多12字，short最多9字且保留主体和意图核，word不超过9字时short使用原词，空格计1字。页面承接与资讯关联必须同时满足，无合格词时candidates为空数组，其余字段均为空字符串。"

    logging.getLogger(__name__).debug("热点跳转结合点：%s", data["article_link"])

    hot_theme_hit: dict = {
        "word_source": "hot_theme",
        "fit_hot_theme_name": state.fit_hot_theme_name,
        "fit_hot_theme_url": state.fit_hot_theme_url,
        "word": word,
        "short": short,
    }

    return HotThemeWordGenerateOutput(
        word_source="hot_theme",
        hot_theme_hit_result=hot_theme_hit,
    )
