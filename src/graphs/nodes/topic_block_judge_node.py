"""专题屏蔽规则节点：当输入资讯标题命中「专题清单.xlsx」的「文章标题」列对应标题、
且适配专题正是该文章标题所关联的专题时，屏蔽用该专题生词（纯规则匹配，非大模型）。

数据源：`assets/专题清单.xlsx`（Sheet1）的「文章标题」+「专题名」两列，建立 文章标题→相关专题名 的映射。
"""

import re
from typing import Any, Dict, List, Set

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import TopicBlockInput, TopicBlockOutput
from tools.llm_utils import ensure_workspace_path

_BLOCK_XLSX = "assets/专题清单.xlsx"
_SHEET_NAME = "Sheet1"
# 全局缓存：{ 规范化文章标题 -> {规范化专题名} } 与 规范化文章标题列表
_BLOCK_TITLE_TO_TOPICS: Dict[str, Set[str]] = {}
_BLOCK_TITLES: List[str] = []
_BLOCK_LOADED: bool = False


def _norm(s: str) -> str:
    """去空白归一化，用于标题与专题名称的对照。"""
    return re.sub(r"\s+", "", s or "")


def _load_block_list() -> Dict[str, Set[str]]:
    """加载专题清单：返回 规范化文章标题→规范化专题名集合 的映射，并缓存标题列表。"""
    global _BLOCK_TITLE_TO_TOPICS, _BLOCK_TITLES, _BLOCK_LOADED
    if _BLOCK_LOADED:
        return _BLOCK_TITLE_TO_TOPICS

    title_to_topics: Dict[str, Set[str]] = {}
    titles: List[str] = []
    try:
        import pandas as pd  # type: ignore

        xlsx_path = ensure_workspace_path(_BLOCK_XLSX)
        df: Any = pd.read_excel(xlsx_path, sheet_name=_SHEET_NAME)
        columns = [str(c).strip() for c in df.columns]
        name_idx = columns.index("专题名") if "专题名" in columns else -1
        title_idx = columns.index("文章标题") if "文章标题" in columns else -1
        rows: Any = df.values.tolist()
        for row in rows:
            if title_idx < 0 or name_idx < 0 or title_idx >= len(row) or name_idx >= len(row):
                continue
            t = _norm(str(row[title_idx]))
            n = _norm(str(row[name_idx]))
            if not t or not n:
                continue
            if t not in title_to_topics:
                title_to_topics[t] = set()
                titles.append(t)
            title_to_topics[t].add(n)
    except Exception:
        title_to_topics, titles = {}, []

    _BLOCK_TITLE_TO_TOPICS, _BLOCK_TITLES = title_to_topics, titles
    _BLOCK_LOADED = True
    return _BLOCK_TITLE_TO_TOPICS


def _title_hit(title_norm: str, titles: List[str]) -> List[str]:
    """标题命中判定：返回命中的规范化文章标题列表（完全相等，或存在长度足够的一方包含另一方）。"""
    hits: List[str] = []
    if not title_norm or not titles:
        return hits
    for t in titles:
        if t == title_norm:
            hits.append(t)
    for t in titles:
        if len(t) >= 6 and (t in title_norm or title_norm in t) and t not in hits:
            hits.append(t)
    return hits


def topic_block_judge_node(
    state: TopicBlockInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> TopicBlockOutput:
    """
    title: 专题屏蔽
    desc: 纯规则匹配：若输入资讯标题命中「专题清单.xlsx」文章标题列对应标题、且适配专题正是该标题所关联的专题，则屏蔽用该专题生词；否则沿用专题生词
    """
    if not state.has_fit_topic:
        return TopicBlockOutput(has_fit_topic=False, topic_blocked=False)

    title_to_topics = _load_block_list()
    title_norm = _norm(state.title)
    fit_title_norm = _norm(state.fit_topic_title)
    if not title_norm or not fit_title_norm:
        return TopicBlockOutput(has_fit_topic=state.has_fit_topic, topic_blocked=False)

    matched_topics: Set[str] = set()
    for hit_title in _title_hit(title_norm, list(title_to_topics.keys())):
        topic_set = title_to_topics.get(hit_title)
        if topic_set:
            matched_topics.update(topic_set)

    if fit_title_norm in matched_topics:
        return TopicBlockOutput(has_fit_topic=False, topic_blocked=True)

    return TopicBlockOutput(has_fit_topic=True, topic_blocked=False)
