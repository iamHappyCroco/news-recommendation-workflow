"""热点题材召回节点：从东方财富热点题材文件中读取「热点题材+热点释义」，通过知识库向量检索召回与资讯主题相关的候选热点题材。"""
import pandas as pd
from typing import List

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import HotThemeRecallInput, HotThemeRecallOutput, HotThemeCandidate
from tools.llm_utils import ensure_workspace_path
from tools.local_retrieval import search

HOT_THEME_DATASET = "hot_theme_list"
HOT_THEME_XLSX = "assets/东方财富热点题材_2026-10-22.xlsx"
SHEET_NAME = "题材汇总"


def _load_theme_rows(xlsx_path: str) -> List[dict]:
    """读取 xlsx「题材汇总」表，提取 热点题材名称 + 热点释义 + 代码 + 链接。"""
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
    rows: List[dict] = []
    for _, r in df.iterrows():
        name = str(r.get("热点题材") or "").strip()
        if not name:
            continue
        raw_code = r.get("题材代码")
        try:
            code = int(raw_code)
        except (TypeError, ValueError):
            code = -1
        rows.append(
            {
                "code": code,
                "name": name,
                "interpretation": str(r.get("热点释义") or "").strip(),
                "url": str(r.get("详情页链接") or "").strip(),
                "page_content": "" if pd.isna(r.get("页面主要内容", "")) else str(r.get("页面主要内容", "")).strip(),
            }
        )
    return rows


def hot_theme_recall_node(
    state: HotThemeRecallInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> HotThemeRecallOutput:
    """
    title: 热点题材召回
    desc: 从东方财富热点题材文件的「热点题材+热点释义」中，基于资讯内容在知识库做向量检索，召回与资讯主题相关的候选热点题材
    integrations: 知识库
    """
    xlsx_path = ensure_workspace_path(HOT_THEME_XLSX)
    rows = _load_theme_rows(xlsx_path)

    query = ((state.title or "") + "\n" + (state.content or "")).strip()
    if not query:
        query = state.title or ""

    candidates: List[HotThemeCandidate] = []
    docs = [f"热点题材：{item['name']}\n{item['interpretation']}" for item in rows]
    for index, score in search(query, docs, top_k=5):
        if score < 0.05:
            continue
        matched = rows[index]
        candidates.append(HotThemeCandidate(
            theme_code=matched["code"],
            theme_name=matched["name"],
            interpretation=matched["interpretation"],
            page_content=matched.get("page_content", ""),
            url=matched["url"],
            score=round(score, 6),
        ))

    return HotThemeRecallOutput(hot_theme_candidates=candidates)
