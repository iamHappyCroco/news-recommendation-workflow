"""Validate per-word display variants and AI question extensions."""

from typing import Any


def display_variants(data: Any, words: list[str]) -> list[dict[str, str]]:
    raw = data.get("display_variants") if isinstance(data, dict) else None
    if not isinstance(raw, list) or len(raw) != len(words):
        raise ValueError("每个推荐词都必须对应一组 6 字版和 9 字版")
    result: list[dict[str, str]] = []
    for word, item in zip(words, raw):
        if not isinstance(item, dict) or str(item.get("word") or "").strip() != word:
            raise ValueError("展示版必须与推荐词同词同序")
        short = str(item.get("display_6") or "").strip()
        long = str(item.get("display_9") or "").strip()
        if not short or len(short) > 6 or not long or len(long) > 9:
            raise ValueError(f"{word} 的展示版超过 6/9 字上限或为空")
        if any(char.isspace() for char in short + long):
            raise ValueError(f"{word} 的展示版不能包含空白字符")
        result.append({"word": word, "display_6": short, "display_9": long})
    return result


def ai_queries(data: Any, words: list[str], source_text: str = "") -> list[dict[str, str]]:
    raw = data.get("ai_queries") if isinstance(data, dict) else None
    if not isinstance(raw, list) or len(raw) != len(words):
        raise ValueError("每个分类生词推荐词都必须对应一条延伸问句")
    result: list[dict[str, str]] = []
    for word, item in zip(words, raw):
        if not isinstance(item, dict) or str(item.get("word") or "").strip() != word:
            raise ValueError("AI 问句必须与推荐词同词同序")
        query = str(item.get("query") or "").strip()
        if query and not query.endswith(("？", "?")):
            if any(mark in query for mark in ("如何", "哪些", "什么", "为何", "为什么", "是否", "有何", "能否", "怎样", "多少", "吗")):
                query = query.rstrip("。.!！?？") + "？"
        if not query or not query.endswith(("？", "?")):
            raise ValueError(f"{word} 的 AI query 必须是一条完整问句：{query[:80]}")
        if "业绩比较基准" in source_text and "跟踪" not in source_text and "跟踪" in query:
            raise ValueError(f"{word} 的 AI query 把业绩比较基准误写为跟踪关系")
        result.append({"word": word, "query": query})
    return result


def invoke_words_with_queries(invoke, ctx, cfg_path, values):
    """Generate aligned words/questions together, repairing invalid output once."""
    from tools.llm_utils import extract_json

    render_vars = {**values, "repair_note": ""}
    for attempt in range(2):
        data = extract_json(invoke(ctx, cfg_path, render_vars)) or {}
        words = list(dict.fromkeys(
            str(word).strip() for word in (data.get("recommended_words") or []) if str(word).strip()
        ))
        try:
            queries = ai_queries(data, words, str(values.get("title", "")) + str(values.get("content", "")))
            return data, words, queries
        except ValueError as exc:
            if attempt:
                raise
            render_vars = {**values, "repair_note": f"上次输出不合规：{exc}。请重新输出完整 JSON，每个推荐词配一条完整问句，保持同词同序。"}
