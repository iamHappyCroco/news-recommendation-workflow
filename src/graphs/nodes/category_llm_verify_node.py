"""Validate the MiniLM 82-topic prediction without remapping it to another taxonomy."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from graphs.state import ClassificationCandidate, LlmVerifyInput, LlmVerifyOutput
from tools.llm_utils import ensure_workspace_path, extract_json, invoke_llm
from tools.local_retrieval import similarity
from tools.runtime import Context

logger = logging.getLogger(__name__)


def _load_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_topic_by_id(topics: dict[str, dict], topic_id: Any) -> dict | None:
    try:
        return topics.get(str(int(topic_id)))
    except (TypeError, ValueError):
        return None


def _topic_id(topic: dict) -> int:
    return int(topic.get("leaf_topic_id", -1))


def _build_catalog_text(topics: list[dict], quadrant_topics: dict[str, dict]) -> str:
    lines: list[str] = []
    for topic in topics:
        topic_id = _topic_id(topic)
        quadrant = quadrant_topics.get(str(topic_id), {})
        searches = "、".join(topic.get("representative_searches", []) or [])
        lines.append(
            f"[ID{topic_id}] {topic.get('leaf_topic_name', '')}"
            f"（一级：{topic.get('parent_topic_name', '')}｜"
            f"象限{quadrant.get('region', '')}：{quadrant.get('quadrant_label', '')}｜"
            f"代表搜索词：{searches}）"
        )
    return "\n".join(lines) or "（无分类候选）"


def _shortlist_topics(state: LlmVerifyInput, topics: dict[str, dict]) -> list[dict]:
    """Use the classifier's exact Top-5 topic IDs; never bridge by characters."""
    selected: list[dict] = []
    seen: set[int] = set()
    for nearby in state.nearby_topics[:5]:
        topic = _find_topic_by_id(topics, nearby.leaf_id)
        if topic is None:
            continue
        topic_id = _topic_id(topic)
        if topic_id not in seen:
            seen.add(topic_id)
            selected.append(topic)

    if state.model_available and not selected:
        topic = _find_topic_by_id(topics, state.model_leaf_id)
        if topic is not None:
            selected.append(topic)

    if not state.model_available:
        query = f"{state.title}\n{state.content}"
        selected = sorted(
            topics.values(),
            key=lambda topic: similarity(
                query,
                " ".join(
                    [
                        str(topic.get("leaf_topic_name", "")),
                        str(topic.get("review_note", "")),
                        *map(str, topic.get("representative_searches", []) or []),
                    ]
                ),
            ),
            reverse=True,
        )[:5]
    return selected[:5]


def category_llm_verify_node(
    state: LlmVerifyInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> LlmVerifyOutput:
    """Validate MiniLM inside its native 82-topic catalog and attach its exact quadrant."""
    if not state.title or not str(state.title).strip():
        return LlmVerifyOutput(can_classify=False)

    cutoff = float(os.getenv("CATEGORY_MIN_SIMILARITY", str(state.model_similarity_cutoff)))
    dual_gap = float(os.getenv("CATEGORY_DUAL_MAX_GAP", "0.005"))
    if not 0 <= cutoff <= 1 or not 0 <= dual_gap <= 0.02:
        raise ValueError("Invalid category similarity/dual-gap threshold")
    fallback = not state.model_available or state.model_similarity < cutoff
    if fallback and not state.llm_fallback_used:
        raise ValueError("Low-confidence classification must first pass the LLM fallback proposal stage")
    audit = {
        "classification_source": "llm_fallback_verified" if fallback else "minilm_verified",
        "llm_fallback_used": fallback,
        "llm_fallback_proposals": state.llm_fallback_proposals if fallback else [],
        "llm_fallback_reason": state.llm_fallback_reason if fallback else "",
    }

    catalog = _load_json(ensure_workspace_path("分类器/topic_catalog.json"))
    topics: dict[str, dict] = catalog.get("leaf_topics", {}) or {}
    quadrant_config = _load_json(ensure_workspace_path("assets/category_quadrants_82.json"))
    quadrant_topics: dict[str, dict] = quadrant_config.get("topics", {}) or {}
    regions_meta: dict[str, str] = quadrant_config.get("regions_meta", {}) or {}

    shortlist = _shortlist_topics(state, topics)
    if fallback:
        proposals = []
        for item in state.llm_fallback_proposals:
            topic = _find_topic_by_id(topics, item.get("leaf_id"))
            if topic is None:
                raise ValueError("Invalid fallback proposal passed to verifier")
            proposals.append(topic)
        if proposals:
            # Proposed class plus independent MiniLM alternatives, deduplicated.
            shortlist = list({str(_topic_id(t)): t for t in proposals + shortlist}.values())
        else:
            # Independently audit rejection across the taxonomy, not just weak Top-5.
            shortlist = list(topics.values())
    shortlist_ids = {_topic_id(topic) for topic in shortlist}
    if not shortlist:
        return LlmVerifyOutput(can_classify=False, classification_reason="未召回可用分类候选")
    similarities = {nb.leaf_id: nb.similarity for nb in state.nearby_topics[:5]}
    if state.model_available:
        similarities.setdefault(state.model_leaf_id, state.model_similarity)
    top_two = state.nearby_topics[:2]
    dual_ids = {nb.leaf_id for nb in top_two}
    dual_allowed = (
        not fallback and state.model_available and len(dual_ids) == 2
        and dual_ids.issubset(shortlist_ids)
        and all(nb.similarity >= cutoff for nb in top_two)
        and 0 <= top_two[0].similarity - top_two[1].similarity <= dual_gap
    )
    catalog_text = _build_catalog_text(shortlist, quadrant_topics)
    catalog_text += "\n模型候选相似度：" + "、".join(
        f"ID{topic_id}={score:.5f}" for topic_id, score in similarities.items()
    )
    model_prediction = (
        str(state.model_pred_text or "")
        if state.model_available
        else "（此处无需模型预测，由LLM在候选目录内自主分类）"
    )
    if fallback:
        model_prediction = json.dumps({"proposals": state.llm_fallback_proposals,
                                       "reason": state.llm_fallback_reason}, ensure_ascii=False)
    render_vars = {
            "title": state.title,
            "content": state.content,
            "model_prediction": model_prediction,
            "catalog": catalog_text,
            "review_context": (
                f"需复核原因：{state.model_review_reasons}；首二差距：{state.model_cosine_gap:.5f}；"
                f"训练样本量：{state.model_training_support}；"
                + (f"允许双归类，仅限ID{top_two[0].leaf_id}和ID{top_two[1].leaf_id}，须两类都有正文依据。"
                   if dual_allowed else "只允许输出一个最适合的归类。")
            ),
        }
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])
    if fallback:
        cfg_path = ensure_workspace_path("config/category_fallback_verify_llm_cfg.json")
        render_vars["review_context"] += (
            "\n这是独立第二次调用：审核LLM兜底提议。MiniLM相似度只作诊断，不作为否决门槛。"
            "可以通过、改为审核目录中更合适的类目，或明确拒绝；只能输出一个有事实依据的类目。"
        )
    raw = invoke_llm(
        runtime.context, cfg_path, render_vars,
    )
    data: Any = extract_json(raw)
    if isinstance(data, dict) and data.get("can_classify") is True and data.get("accepted_candidates") == []:
        # Ask the same verifier to resolve a contradictory flag, rather than guessing
        # or rerunning topic/hot-theme generation for a format-only problem.
        repair_vars = dict(render_vars)
        repair_vars["review_context"] += (
            "\n上次输出存在结构矛盾：can_classify=true但accepted_candidates为空。"
            "请重新确认：若所有审核候选均不合适，必须返回can_classify=false；"
            "若可归类，必须返回非空且有正文依据的候选。不得凭空补候选。"
            f"\n上次判断依据：{data.get('classification_reason', '')}"
        )
        data = extract_json(invoke_llm(runtime.context, cfg_path, repair_vars))
    if not isinstance(data, dict) or not isinstance(data.get("can_classify"), bool):
        raise ValueError("分类校验未返回合法 can_classify，需重试而非当作无适合类目")
    reason = str(data.get("classification_reason") or "").strip()
    if fallback and not reason:
        raise ValueError("Independent fallback verification requires an explicit reason")
    extracted_search = str(data.get("extracted_search") or state.title).strip()
    if not data["can_classify"]:
        return LlmVerifyOutput(
            **audit,
            can_classify=False, extracted_search=extracted_search,
            classification_reason=reason or "LLM确认审核候选均缺少资讯语义依据",
        )
    accepted = data.get("accepted_candidates")
    if accepted is None:  # Compatibility with existing single-topic responses.
        chosen_id = data.get("suggested_leaf_id") if data.get("verdict") == "建议改类" else data.get("leaf_id")
        accepted = [{"leaf_id": chosen_id, "reason": reason}]
    if not isinstance(accepted, list) or not accepted or len(accepted) > 2:
        raise ValueError("分类校验必须返回1或2个合法候选")
    if fallback and len(accepted) != 1:
        raise ValueError("Fallback verification permits exactly one category")
    decisions: list[ClassificationCandidate] = []
    seen: set[int] = set()
    for item in accepted:
        if not isinstance(item, dict):
            raise ValueError("Invalid classification candidate")
        if (len(accepted) == 2 or fallback) and not str(item.get("reason") or "").strip():
            raise ValueError("双归类的每个候选必须提供独立语义依据，需重试")
        topic = _find_topic_by_id(topics, item.get("leaf_id"))
        if topic is None or _topic_id(topic) not in shortlist_ids:
            raise ValueError("LLM分类候选超出给定审核目录，需重试")
        topic_id = _topic_id(topic)
        if topic_id in seen:
            continue
        seen.add(topic_id)
        quadrant = quadrant_topics.get(str(topic_id), {})
        name = str(topic.get("leaf_topic_name", "")).strip()
        region = str(quadrant.get("region", "")).strip()
        if quadrant.get("leaf_topic_name") != name or region not in {"A", "B", "C", "D"}:
            raise ValueError(f"分类ID{topic_id}缺少一致的象限配置")
        decisions.append(ClassificationCandidate(
            leaf_id=topic_id, category=str(topic.get("parent_topic_name", "")).strip(),
            sub_category=name, region=region,
            quadrant=f"象限{region}·{regions_meta.get(region, quadrant.get('quadrant_label', ''))}",
            similarity=similarities.get(topic_id, 0.0),
            reason=str(item.get("reason") or reason).strip(), keywords=name,
            rep_searches="、".join(map(str, topic.get("representative_searches", []) or [])),
        ))
    if len(decisions) == 2 and not (dual_allowed and {d.leaf_id for d in decisions} == dual_ids):
        decisions = decisions[:1]
    primary = decisions[0]
    return LlmVerifyOutput(
        **audit,
        can_classify=True,
        category=primary.category, sub_category=primary.sub_category,
        region=primary.region, quadrant=primary.quadrant,
        keywords=primary.keywords, rep_searches=primary.rep_searches,
        extracted_search=extracted_search,
        classification_candidates=decisions,
        classification_mode="dual" if len(decisions) == 2 else "single",
        classification_reason=reason or primary.reason,
    )
