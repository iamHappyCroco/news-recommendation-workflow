"""专题RAG召回节点：基于资讯内容在专题清单知识库中做向量检索。

知识库数据源：`assets/专题清单.xlsx` 每个专题的「专题名 + 导语」（前两列）；
召回命中专题时同步输出该专题的「专题地址」链接。其余逻辑保持不变。
"""
import math

import pandas as pd
from typing import Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import TopicRecallInput, TopicRecallOutput, TopicCandidate
from tools.llm_utils import ensure_workspace_path
from tools.local_retrieval import search

# 知识库数据集名（数据源变更，使用新数据集加载「专题名+导语」）
TOPIC_DATASET = "topic_relation_list_v2"
# 专题清单 xlsx 文件（相对于项目根）
TOPIC_XLSX = "assets/专题清单.xlsx"
SHEET_NAME = "Sheet1"

# 使用前两列作为知识库内容（专题名、导语），并同步输出专题地址链接
COL_TOPIC_NAME = "专题名"
COL_TOPIC_INTRO = "导语"
COL_TOPIC_URL = "专题地址"


def _load_topic_map(xlsx_path: str) -> Dict[str, dict]:
    """读取 xlsx 前两列（专题名、导语）作为知识库内容，并建立 专题名→{导语, 地址} 映射（去重，保持首次出现顺序）。

    知识库只依赖「专题名 + 导语」两列；「专题地址」列仅用于在命中时同步输出该专题的链接。
    """
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
    topic_map: Dict[str, dict] = {}
    if len(df) == 0:
        return topic_map
    name_col = COL_TOPIC_NAME if COL_TOPIC_NAME in df.columns else df.columns[0]
    intro_col = COL_TOPIC_INTRO if COL_TOPIC_INTRO in df.columns else None
    url_col = COL_TOPIC_URL if COL_TOPIC_URL in df.columns else None
    for _, r in df.iterrows():
        name_val = r.get(name_col)
        if _is_na(name_val):
            continue
        name = str(name_val).strip()
        if not name or name in topic_map:
            continue
        intro = ""
        if intro_col is not None and not _is_na(r.get(intro_col)):
            intro = str(r.get(intro_col)).strip()
        url = ""
        if url_col is not None and not _is_na(r.get(url_col)):
            url = str(r.get(url_col)).strip()
        page_content = r.get("页面主要内容", "")
        topic_map[name] = {"intro": intro, "url": url,
                           "page_content": "" if _is_na(page_content) else str(page_content).strip()}
    return topic_map


def _is_na(value: object) -> bool:
    """判断单元格标量是否为空（None / NaN / 空字符串 / 文本"nan"）。兼容 pyright 标量类型推断。"""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == "nan"
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _compose_doc(name: str, info: dict) -> str:
    """拼装知识库文档内容：专题名 + 导语（仅前两列）。"""
    text = name
    intro = str(info.get("intro") or "").strip()
    if intro:
        text = f"{text}\n{intro}"
    return text


def topic_recall_node(
    state: TopicRecallInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> TopicRecallOutput:
    """
    title: 专题RAG召回
    desc: 基于资讯内容在专题清单知识库（取自「专题清单.xlsx」每个专题的专题名+导语）做向量检索，召回可能适配的专题，并同步输出命中专题的地址链接
    integrations: 知识库
    """
    xlsx_path = ensure_workspace_path(TOPIC_XLSX)
    topic_map = _load_topic_map(xlsx_path)

    query = ((state.title or "") + "\n" + (state.content or "")).strip()
    if not query:
        query = state.title or ""

    candidates: List[TopicCandidate] = []
    items = list(topic_map.items())
    docs = [_compose_doc(name, info) for name, info in items]
    for index, score in search(query, docs, top_k=5):
        if score < 0.05:
            continue
        name, info = items[index]
        candidates.append(TopicCandidate(
            topic_title=name,
            intro=str(info.get("intro") or "").strip(),
            page_content=str(info.get("page_content") or "").strip(),
            topic_url=str(info.get("url") or "").strip(),
            score=round(score, 6),
        ))

    return TopicRecallOutput(topic_candidates=candidates)
