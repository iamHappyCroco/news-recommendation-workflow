"""专题生词：以目标页面内容或导语为核心，提供超出原资讯的延伸阅读。"""
from typing import Any, List

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import TopicWordGenerateInput, TopicWordGenerateOutput
from tools.llm_utils import invoke_llm, extract_json, ensure_workspace_path
from tools.word_extras import display_variants


def topic_word_generate_node(
    state: TopicWordGenerateInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> TopicWordGenerateOutput:
    """
    title: 专题相关搜索词生成
    desc: 命中适配专题后，生成3个候选词并按优先级排序，第1个为首选词
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
        data: Any = extract_json(invoke_llm(ctx, cfg_path, render_vars)) or {}
        words: List[str] = list(dict.fromkeys(
            str(w).strip() for w in (data.get("recommended_words") or []) if str(w).strip()
        ))[:3]
        try:
            if len(words) != 3:
                raise ValueError("专题必须生成 3 个不重复推荐词")
            variants = display_variants(data, words)
            break
        except ValueError as exc:
            if attempt:
                raise
            render_vars["repair_note"] = f"上次输出不合规：{exc}。请重新输出完整 JSON，并逐字核对展示版。"

    # 额外输出的专题生词结果：专题适配时会保留这份结果，且流程仍继续进入模型分类->象限生词
    topic_hit: dict = {
        "word_source": "topic",
        "fit_topic_title": state.fit_topic_title,
        "fit_topic_url": state.fit_topic_url,
        "recommended_words": words,
        "display_variants": variants,
        "best_word": words[0] if words else "",
    }

    return TopicWordGenerateOutput(
        recommended_words=words,
        word_source="topic",
        topic_hit_result=topic_hit,
    )
