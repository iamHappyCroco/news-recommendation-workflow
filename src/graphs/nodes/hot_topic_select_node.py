""""热主题拣选"节点：对低曝光象限（B/D）资讯，结合「模型向量相邻主题」与「目录中高曝光高点击(region=A)主题清单」，
通过语义判断拣选一个真正相邻的高曝光高点击主题作为出词参照。

说明：分类器、候选主题和象限配置统一使用同一套 82 主题 ID。
"""
import json
import logging
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import HotTopicSelectInput, HotTopicSelectOutput
from tools.llm_utils import invoke_llm, extract_json, ensure_workspace_path

logger = logging.getLogger(__name__)

# 相邻性门槛：拣选出的高曝光(A)主题与资讯的模型向量余弦相似度低于该值时，
# 视为"其实并不相邻"，放弃注入参考（降级为无参考生词）。
HOT_SIM_THRESHOLD: float = 0.5


def _load_catalog(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as fd:
        return json.load(fd)


def hot_topic_select_node(
    state: HotTopicSelectInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> HotTopicSelectOutput:
    """
    title: 相邻高曝光主题拣选
    desc: 对落入低曝光象限(B/D)的资讯，结合模型向量相邻主题与目录中高曝光高点击(region=A)主题清单，语义拣选一个真正相邻的高曝光主题作为下游生词的参照
    integrations: 大语言模型
    """
    ctx = runtime.context
    cfg_path = ensure_workspace_path(config["metadata"]["llm_cfg"])
    catalog_file = ensure_workspace_path("分类器/topic_catalog.json")
    catalog = _load_catalog(catalog_file)
    leaf_topics: List[dict] = list((catalog.get("leaf_topics", {}) or {}).values())
    quadrant_config = _load_catalog(ensure_workspace_path("assets/category_quadrants_82.json"))
    quadrant_topics: Dict[str, dict] = quadrant_config.get("topics", {}) or {}

    # region=A：高曝光·高点击主题清单（作为可选的"热门参照"词汇池）
    hot_leaves: List[dict] = [
        leaf
        for leaf in leaf_topics
        if str(quadrant_topics.get(str(leaf.get("leaf_topic_id", -1)), {}).get("region", "")).strip() == "A"
    ]
    hot_lines: List[str] = []
    hot_by_id: Dict[str, dict] = {}
    for i, leaf in enumerate(hot_leaves, start=1):
        lid = int(leaf.get("leaf_topic_id", -1))
        searches = "、".join(leaf.get("representative_searches", []) or [])
        kw = str(leaf.get("leaf_topic_name", "") or "").strip()
        hot_lines.append(
            f"[{i}] ID{lid}｜主题：{leaf.get('leaf_topic_name', '')}（一级：{leaf.get('parent_topic_name', '')}）｜"
            f"代表搜索词：{searches}｜代表关键词：{kw}"
        )
        hot_by_id[str(lid)] = leaf
    hot_vocab_text: str = "\n".join(hot_lines) or "（无）"

    # 模型向量相近的相邻主题（向量维度的候选，供 LLM 判断语义相邻性）
    nearby_lines: List[str] = []
    for j, nb in enumerate(state.nearby_topics or [], start=1):
        searches = "、".join(nb.representative_searches or [])
        nearby_lines.append(
            f"[{j}] 主题：{nb.leaf_name}（一级：{nb.parent_name}｜向量相似度 {nb.similarity:.4f}）｜代表搜索词：{searches}"
        )
    nearby_text: str = "\n".join(nearby_lines) or "（无）"

    raw = invoke_llm(
        ctx,
        cfg_path,
        {
            "title": state.title,
            "content": state.content,
            "region": state.region,
            "nearby_text": nearby_text,
            "hot_vocab_text": hot_vocab_text,
        },
    )
    data: Any = extract_json(raw) or {}
    hot_leaf_id: int = int(data.get("hot_leaf_id", -1) or -1)
    leaf: dict = hot_by_id.get(str(hot_leaf_id))

    # 由分类模型+向量给出选中主题与资讯的余弦相似度（相邻性依据，非空值）
    hot_similarity: float = 0.0
    for nb in state.nearby_topics or []:
        try:
            if int(nb.leaf_id) == hot_leaf_id:
                hot_similarity = float(nb.similarity or 0.0)
                break
        except (TypeError, ValueError):
            continue

    # 相邻性门槛：未命中，或向量相似度过低（不够相邻）→ 降级，不注入参考
    if leaf is None or hot_similarity < HOT_SIM_THRESHOLD:
        return HotTopicSelectOutput(
            has_hot_topic=False,
            hot_leaf_id=int(hot_leaf_id) if leaf else -1,
            hot_leaf_name="",
            hot_parent_name="",
            hot_region="A" if leaf else "",
            hot_similarity=hot_similarity,
            hot_rep_searches=[],
            hot_keywords="",
            hot_reference_text="",
        )

    rep_searches: List[str] = list(leaf.get("representative_searches", []) or [])
    keywords: str = str(leaf.get("leaf_topic_name", "") or "").strip()
    reference_text: str = (
        f"相邻高曝光高点击重要参考类别：{leaf.get('leaf_topic_name', '')}"
        f"（一级：{leaf.get('parent_topic_name', '')}｜region=A｜向量相似度 {hot_similarity:.4f}）\n"
        f"代表搜索词：{'、'.join(rep_searches)}\n代表关键词：{keywords}\n"
        f"与原资讯的结合依据及搜索方向：{str(data.get('reason') or '').strip()}"
    )
    return HotTopicSelectOutput(
        has_hot_topic=True,
        hot_leaf_id=int(leaf.get("leaf_topic_id", -1)),
        hot_leaf_name=str(leaf.get("leaf_topic_name", "")),
        hot_parent_name=str(leaf.get("parent_topic_name", "")),
        hot_region="A",
        hot_similarity=hot_similarity,
        hot_rep_searches=rep_searches,
        hot_keywords=keywords,
        hot_reference_text=reference_text,
    )
