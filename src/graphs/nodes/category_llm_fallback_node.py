"""Propose a native category for low-confidence articles; never finalize it here."""
from __future__ import annotations

import os
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from graphs.state import ClassificationFallback, LlmVerifyInput
from graphs.nodes.category_llm_verify_node import _load_json, _find_topic_by_id, _topic_id
from tools.llm_utils import ensure_workspace_path, extract_json, invoke_llm
from tools.runtime import Context


def needs_llm_fallback(state: LlmVerifyInput) -> bool:
    cutoff = float(os.getenv("CATEGORY_MIN_SIMILARITY", str(state.model_similarity_cutoff)))
    if not 0 <= cutoff <= 1:
        raise ValueError("Invalid category similarity threshold")
    return not state.model_available or state.model_similarity < cutoff


def route_category_confidence(state: LlmVerifyInput) -> str:
    return "fallback" if needs_llm_fallback(state) else "verify"


def category_llm_fallback_node(
    state: LlmVerifyInput, config: RunnableConfig, runtime: Runtime[Context],
) -> ClassificationFallback:
    if not needs_llm_fallback(state):
        return ClassificationFallback()
    topics = _load_json(ensure_workspace_path("分类器/topic_catalog.json")).get("leaf_topics", {})
    if not topics:
        raise ValueError("Fallback classifier requires the native topic catalog")
    # Broad but compact taxonomy: low-confidence Top-5 may omit the correct category.
    catalog = "\n".join(
        f"[ID{_topic_id(t)}] {t.get('parent_topic_name', '')} / {t.get('leaf_topic_name', '')}"
        f" | 例词：{'、'.join(map(str, (t.get('representative_searches') or [])[:3]))}"
        for t in topics.values()
    )
    data = extract_json(invoke_llm(runtime.context, ensure_workspace_path(config["metadata"]["llm_cfg"]), {
        "title": state.title, "content": state.content, "catalog": catalog,
    }))
    if not isinstance(data, dict) or not isinstance(data.get("can_classify"), bool):
        raise ValueError("Fallback classifier returned invalid can_classify")
    reason = str(data.get("classification_reason") or "").strip()
    proposals = data.get("accepted_candidates")
    if not isinstance(proposals, list) or len(proposals) != int(data["can_classify"]) or not reason:
        raise ValueError("Fallback classifier must return one supported proposal or an explicit rejection")
    mapped = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Invalid fallback proposal")
        topic = _find_topic_by_id(topics, proposal.get("leaf_id"))
        evidence = str(proposal.get("reason") or "").strip()
        if topic is None or not evidence:
            raise ValueError("Fallback proposal needs a valid native category and article evidence")
        mapped.append({"leaf_id": _topic_id(topic), "category": topic.get("parent_topic_name", ""),
                       "sub_category": topic.get("leaf_topic_name", ""), "reason": evidence})
    return ClassificationFallback(llm_fallback_used=True, llm_fallback_proposals=mapped,
                                  llm_fallback_reason=reason)
