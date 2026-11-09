"""热点题材屏蔽规则节点：当输入资讯命中「需屏蔽嵌套跳转」清单中的资讯标题，且返回的适配热点也在该清单的「热点题材」列中时，屏蔽用该热点生词（纯规则匹配，非大模型）。"""

import os
import re
from typing import Any, List, Set, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context
from tools.llm_utils import ensure_workspace_path

from graphs.state import HotThemeBlockInput, HotThemeBlockOutput

_BLOCK_XLSX = ensure_workspace_path("assets/热点题材_需屏蔽嵌套跳转_2026-10-22.xlsx")
_BLOCK_THEMES: Set[str] = set()
_BLOCK_TITLES: List[str] = []
_BLOCK_LOADED: bool = False


def _norm(s: str) -> str:
    """去空白归一化，用于标题与题材名称的对照。"""
    return re.sub(r"\s+", "", s or "")


def _load_block_list() -> Tuple[Set[str], List[str]]:
    """加载屏蔽清单：返回（需要屏蔽的热点题材集合，需屏蔽的资讯标题列表）。"""
    global _BLOCK_THEMES, _BLOCK_TITLES, _BLOCK_LOADED
    if _BLOCK_LOADED:
        return _BLOCK_THEMES, _BLOCK_TITLES

    themes: Set[str] = set()
    titles: List[str] = []
    try:
        import pandas as pd  # type: ignore

        df: Any = pd.read_excel(_BLOCK_XLSX, sheet_name="屏蔽清单")
        columns = [str(c).strip() for c in df.columns]
        th_idx = columns.index("热点题材") if "热点题材" in columns else -1
        ti_idx = columns.index("资讯标题") if "资讯标题" in columns else -1
        rows: Any = df.values.tolist()
        for row in rows:
            if th_idx >= 0 and th_idx < len(row):
                s = _norm(str(row[th_idx]))
                if s:
                    themes.add(s)
            if ti_idx >= 0 and ti_idx < len(row):
                s = _norm(str(row[ti_idx]))
                if s:
                    titles.append(s)
    except Exception:
        themes, titles = set(), []

    _BLOCK_THEMES, _BLOCK_TITLES = themes, titles
    _BLOCK_LOADED = True
    return _BLOCK_THEMES, _BLOCK_TITLES


def _title_hit(title_norm: str, titles: List[str]) -> bool:
    """标题命中判定：完全相等，或存在长度足够的一方包含另一方的包含关系。"""
    if not title_norm or not titles:
        return False
    for t in titles:
        if t == title_norm:
            return True
    for t in titles:
        if len(t) >= 6 and (t in title_norm or title_norm in t):
            return True
    return False


def hot_theme_block_judge_node(
    state: HotThemeBlockInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> HotThemeBlockOutput:
    """
    title: 热点题材嵌套跳转屏蔽
    desc: 纯规则匹配：若输入资讯命中需屏蔽嵌套跳转清单的标题、且适配热点也在该清单「热点题材」列中，则屏蔽用该热点生词；否则沿用热点生词
    """
    if not state.has_fit_hot_theme:
        return HotThemeBlockOutput(has_fit_hot_theme=False, hot_theme_blocked=False)

    themes, titles = _load_block_list()
    title_norm = _norm(state.title)
    theme_norm = _norm(state.fit_hot_theme_name)

    if theme_norm and theme_norm in themes and _title_hit(title_norm, titles):
        return HotThemeBlockOutput(has_fit_hot_theme=False, hot_theme_blocked=True)

    return HotThemeBlockOutput(has_fit_hot_theme=True, hot_theme_blocked=False)
