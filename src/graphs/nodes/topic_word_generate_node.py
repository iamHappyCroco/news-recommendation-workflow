"""专题生词：以目标页面内容或导语为核心，提供超出原资讯的延伸阅读。"""
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import TopicWordGenerateInput, TopicWordGenerateOutput
from tools.llm_utils import invoke_llm, extract_json, ensure_workspace_path
from tools.word_extras import validate_jump_word


def topic_word_generate_node(
    state: TopicWordGenerateInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> TopicWordGenerateOutput:
    """
    title: 专题相关搜索词生成
    desc: 命中适配专题后，仅生成1个首选词及9字以内短版
    integrations: 大语言模型
    """
    ctx = runtime.context
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])

    render_vars = {
        "title": state.title,
        "content": state.content,
        "fit_topic_title": state.fit_topic_title,
        "fit_topic_url": state.fit_topic_url,
        "target_page_content": (state.page_contents.get(state.fit_topic_url, "").strip()
                                or state.fit_topic_page_content.strip() or state.fit_topic_intro.strip()),
        "target_content_source": ("传入的页面主要内容" if state.page_contents.get(state.fit_topic_url, "").strip()
                                  or state.fit_topic_page_content.strip() else "专题导语摘要"),
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
            render_vars["repair_note"] = f"上次输出不合规：{exc}。请重新输出word、short、subject、article_link四字段JSON；article_link用一句话说明资讯与页面的结合点。word最多12字，short最多9字且保留主体和意图核，word不超过9字时short使用原词，空格计1字。页面承接与资讯关联必须同时满足，无合格词时四字段均为空字符串。"

    logging.getLogger(__name__).debug("专题跳转结合点：%s", data["article_link"])

    # 额外输出的专题生词结果：专题适配时会保留这份结果，且流程仍继续进入模型分类->象限生词
    topic_hit: dict = {
        "word_source": "topic",
        "fit_topic_title": state.fit_topic_title,
        "fit_topic_url": state.fit_topic_url,
        "word": word,
        "short": short,
    }

    return TopicWordGenerateOutput(
        word_source="topic",
        topic_hit_result=topic_hit,
    )
