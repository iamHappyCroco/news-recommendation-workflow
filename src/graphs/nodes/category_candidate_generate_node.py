"""Generate words separately for a verified alternate category, using its own quadrant."""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from graphs.state import (
    CategoryCandidateGenerateOutput, GlobalState, HotTopicSelectInput, QuadrantInput,
)
from graphs.nodes.hot_topic_select_node import hot_topic_select_node
from graphs.nodes.quadrant_a_node import quadrant_a_node
from graphs.nodes.quadrant_b_node import quadrant_b_node
from graphs.nodes.quadrant_c_node import quadrant_c_node
from graphs.nodes.quadrant_d_node import quadrant_d_node
from tools.runtime import Context
from tools.word_extras import ai_queries


GENERATORS = {
    "A": quadrant_a_node, "B": quadrant_b_node,
    "C": quadrant_c_node, "D": quadrant_d_node,
}


def _node_config(config: RunnableConfig, cfg: str) -> RunnableConfig:
    return {**config, "metadata": {**config.get("metadata", {}), "llm_cfg": cfg}}


def category_candidate_generate_node(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[Context],
) -> CategoryCandidateGenerateOutput:
    """Primary words are already generated; at most one alternate requires calls."""
    results = []
    for index, candidate in enumerate(state.classification_candidates):
        if index == 0:
            words = state.recommended_words
            source = state.word_source
            queries = state.ai_queries
        else:
            reference = {}
            if candidate.region in {"B", "D"}:
                selected = hot_topic_select_node(
                    HotTopicSelectInput(
                        title=state.title, content=state.content,
                        nearby_topics=state.nearby_topics, region=candidate.region,
                    ),
                    _node_config(config, "config/hot_topic_select_llm_cfg.json"), runtime,
                )
                reference = {
                    key: value for key, value in selected.model_dump().items()
                    if key in QuadrantInput.model_fields
                }
            generated = GENERATORS[candidate.region](
                QuadrantInput(
                    title=state.title, content=state.content, category=candidate.category,
                    sub_category=candidate.sub_category, region=candidate.region,
                    keywords=candidate.keywords, rep_searches=candidate.rep_searches,
                    **reference,
                ),
                _node_config(config, f"config/quadrant_{candidate.region.lower()}_llm_cfg.json"), runtime,
            )
            words = generated.recommended_words
            source = generated.word_source
            queries = generated.ai_queries
        if not words and candidate.region != "D":
            raise ValueError(f"分类候选ID{candidate.leaf_id}未生成推荐词，需重试")
        queries = ai_queries({"ai_queries": queries}, list(words), state.title + state.content)
        results.append({
            **candidate.model_dump(exclude={"keywords", "rep_searches"}),
            "word_source": source, "recommended_words": list(words), "ai_queries": queries,
        })
    return CategoryCandidateGenerateOutput(category_candidate_results=results)
