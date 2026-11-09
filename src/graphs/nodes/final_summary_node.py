"""最终汇总节点：将专题相关、热点题材相关与「分类+按象限生词」三部分聚合为一份最终输出，各部分给出推荐首选词及理由，并给出「专题 vs 热点」优先出词判断。"""

from typing import List

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import FinalSummaryInput, FinalSummaryOutput


def classification_source_display(source: str, classified: bool, fallback_used: bool = False) -> str:
    """Describe the origin of an accepted category, not just an attempted route."""
    if not classified:
        return "未归类"
    labels = {
        "minilm_verified": "实际分类器分类",
        "llm_fallback_verified": "LLM兜底分类",
        "实际分类器分类": "实际分类器分类",
        "LLM兜底分类": "LLM兜底分类",
    }
    return labels.get(source, "LLM兜底分类" if fallback_used else "实际分类器分类")


def classification_region_display(
    candidates: List[dict], region: str, quadrant: str, classified: bool,
) -> tuple[str, str]:
    """Summarize accepted regions for output after per-category routing is complete."""
    if not classified:
        return "none", "未归类（AI生词）"
    accepted = [item for item in candidates if item.get("region") in {"A", "B", "C", "D"}]
    labels = {
        "A": "象限A·高曝光·高点击", "B": "象限B·高曝光·低点击",
        "C": "象限C·低曝光·高点击", "D": "象限D·低曝光·低点击",
    }
    if accepted:
        regions = [item["region"] for item in accepted]
        quadrants = [item.get("quadrant") or labels[item["region"]] for item in accepted]
        return "/".join(regions), ("双归类：" if len(accepted) > 1 else "") + "；".join(quadrants)
    regions = region.split("/")
    if 1 <= len(regions) <= 2 and all(item in labels for item in regions):
        fallback_label = ("双归类：" if len(regions) > 1 else "") + "；".join(labels[item] for item in regions)
        return region, quadrant or fallback_label
    return "none", "未归类（AI生词）"


def compact_generation_results(results: List[dict]) -> List[dict]:
    """Keep each category and its ranked words once, in primary-first order."""
    compact = []
    for item in results:
        result = {key: item[key] for key in ("leaf_id", "category", "sub_category", "similarity", "reason") if key in item}
        source = item.get("word_source") or (f"quadrant_{item['region']}" if item.get("region") in {"A", "B", "C", "D"} else "")
        result.update(word_source=source, recommended_words=[str(word).strip() for word in (item.get("recommended_words") or []) if str(word).strip()])
        result["ai_queries"] = item.get("ai_queries") or []
        compact.append(result)
    return compact


def final_summary_node(
    state: FinalSummaryInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> FinalSummaryOutput:
    """
    title: 最终输出汇总
    desc: 聚合专题相关、热点题材相关与「分类+按象限生词」三部分，输出各自推荐首选词、理由与页面跳转链接，并给出「优先出专题词还是热点词」的判断及理由
    """
    topic_hit: dict = state.topic_hit_result or {}
    hot_theme_hit: dict = state.hot_theme_hit_result or {}

    topic_words: List[str] = []
    tw = topic_hit.get("recommended_words")
    if isinstance(tw, list):
        topic_words = [str(w).strip() for w in tw if str(w).strip()]
    hot_words: List[str] = []
    hw = hot_theme_hit.get("recommended_words")
    if isinstance(hw, list):
        hot_words = [str(w).strip() for w in hw if str(w).strip()]
    category_words = [str(w).strip() for w in (state.recommended_words or []) if str(w).strip()]
    region, quadrant = classification_region_display(
        state.category_candidate_results, state.region, state.quadrant,
        bool(state.sub_category) and state.word_source != "ai",
    )
    classification_source = classification_source_display(
        state.classification_source, region != "none", state.llm_fallback_used,
    )
    if region != "none":
        generated = state.category_candidate_results or [{
            "category": state.category, "sub_category": state.sub_category,
            "word_source": state.word_source, "recommended_words": category_words, "ai_queries": state.ai_queries,
        }]
    else:
        generated = [{"word_source": "ai", "recommended_words": category_words, "ai_queries": state.ai_queries}]

    final_summary: dict = {
        "out_priority": state.out_priority,
        "out_priority_reason": state.out_priority_reason,
        "hot_theme_blocked": state.hot_theme_blocked,
        "topic_related": {
            "matched": state.has_fit_topic,
            "title": state.fit_topic_title,
            "jump_url": state.fit_topic_url,
            "word_source": topic_hit.get("word_source", "topic"),
            "recommended_words": topic_words,
            "display_variants": topic_hit.get("display_variants", []),
            "best_word": topic_words[0] if topic_words else "",
        },
        "hot_theme_related": {
            "matched": state.has_fit_hot_theme,
            "name": state.fit_hot_theme_name,
            "jump_url": state.fit_hot_theme_url,
            "word_source": hot_theme_hit.get("word_source", "hot_theme"),
            "recommended_words": hot_words,
            "display_variants": hot_theme_hit.get("display_variants", []),
            "best_word": hot_words[0] if hot_words else "",
        },
        "category_quadrant": {
            "classification_source": classification_source,
            "llm_fallback_used": state.llm_fallback_used,
            "llm_fallback_proposals": state.llm_fallback_proposals,
            "llm_fallback_reason": state.llm_fallback_reason,
            "region": region,
            "quadrant": quadrant,
            "generation_results": compact_generation_results(generated),
        },
    }

    return FinalSummaryOutput(
        final_summary=final_summary,
        classification_source=classification_source,
        region=region,
        quadrant=quadrant,
        fit_hot_theme_name=state.fit_hot_theme_name,
        fit_hot_theme_url=state.fit_hot_theme_url,
    )
