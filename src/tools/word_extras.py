"""Validate ranked recommendation lengths, first-choice short forms and AI queries."""

from typing import Any

SHORT_SUBJECT_POLICY_VERSION = "2026-11-19-jump-v2.2-v6"


def validate_jump_word(data: Any, *, require_metadata: bool = False) -> tuple[str, str]:
    """Validate a single destination word; empty output is a legitimate abstention."""
    if not isinstance(data, dict) or any(key in data for key in ("recommended_words", "best_word", "best_word_short")):
        raise ValueError("跳转生词只使用word和short，不再使用推荐词数组或best_word字段")
    word, short = data.get("word"), data.get("short")
    if not isinstance(word, str) or not isinstance(short, str):
        raise ValueError("跳转word和short必须是字符串")
    if not word:
        if short or (require_metadata and any(data.get(key) != "" for key in ("subject", "article_link"))):
            raise ValueError("无合格跳转词时word、short、subject和article_link均为空字符串")
        return word, short
    validate_words([word], maximum=1)
    values = {"best_word_short": short, "best_word_subject": data.get("subject")}
    if require_metadata:
        generated_best_word_short(values, [word])
        article_link = data.get("article_link")
        if not isinstance(article_link, str) or not article_link.strip():
            raise ValueError("article_link必须用非空字符串说明原资讯与目标页面的结合点")
    else:
        best_word_short(values, [word])
    if len(word) <= 9 and short != word:
        raise ValueError("word不超过9字时short必须直接使用原词")
    return word, short


def validate_words(words: list[str], maximum: int = 3, *, exact: bool = False) -> None:
    """Count every character, including spaces, without truncating search intent."""
    if len(words) > maximum or (exact and len(words) != maximum):
        raise ValueError(f"推荐词数量必须{'恰好' if exact else '不超过'} {maximum} 个")
    for word in words:
        if not isinstance(word, str) or not word.strip() or len(word) > 12:
            raise ValueError(f"推荐词必须非空且不超过 12 字（含空格）：{word}")
        if word != word.strip() or any(c.isspace() and c != ' ' for c in word):
            raise ValueError("推荐词仅允许名词之间使用普通空格，不允许首尾空白、换行或制表符")
    if len(set(words)) != len(words):
        raise ValueError("推荐词不得重复")


def best_word_short(data: Any, words: list[str]) -> str:
    short = data.get("best_word_short", "") if isinstance(data, dict) else ""
    if not isinstance(short, str):
        raise ValueError("首选词短版必须是字符串")
    if not words:
        if short:
            raise ValueError("无推荐词时首选词短版必须为空")
        return ""
    if not short.strip() or len(short) > min(9, len(words[0])):
        raise ValueError("首选词短版必须非空、不超过 9 字且不能长于首选词（含空格）")
    validate_words([short], maximum=1)
    return short


def generated_best_word_short(data: Any, words: list[str]) -> str:
    """Require the model's identified subject in both original and short form."""
    short = best_word_short(data, words)
    subject = data.get("best_word_subject", "") if isinstance(data, dict) else ""
    if not words:
        if subject:
            raise ValueError("无推荐词时首选词主体必须为空")
        return short
    if not isinstance(subject, str) or not subject.strip() or subject != subject.strip():
        raise ValueError("必须提供首选词主体best_word_subject，不能只填事件或评价词")
    if subject in {"财报", "CEO", "公司", "其", "该公司"}:
        raise ValueError("首选词主体不能用财报、CEO、公司或代词等泛称替代原词对象")
    if subject not in words[0]:
        raise ValueError("首选词主体必须取自原首选词，不能添加新主体")
    if subject not in short:
        raise ValueError(f"首选词短版必须保留原词主体“{subject}”，不能仅留下事件或评价")
    return short


def short_words(data: Any, words: list[str], *, require_subject: bool = False) -> list[dict[str, str]]:
    """Validate one aligned short form per recommendation; hide subject metadata."""
    raw = data.get("short_words") if isinstance(data, dict) else None
    if not isinstance(raw, list) or len(raw) != len(words):
        raise ValueError("每个推荐词都必须对应一个short_words短版，条数必须一致")
    result = []
    for word, item in zip(words, raw):
        if not isinstance(item, dict) or item.get("word") != word:
            raise ValueError("短版必须与推荐词同词同序")
        values = {"best_word_short": item.get("short"), "best_word_subject": item.get("subject", "")}
        short = generated_best_word_short(values, [word]) if require_subject else best_word_short(values, [word])
        result.append({"word": word, "short": short})
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
        summary = item.get("query_summary")
        if not isinstance(summary, str) or not summary.strip() or summary != summary.strip():
            raise ValueError(f"{word} 必须提供用户侧query_summary概述")
        if any(c in summary for c in '\n\r\t') or summary.endswith(('?', '？')):
            raise ValueError(f"{word} 的query_summary必须是简洁、独立可理解的概述短语")
        result.append({"word": word, "query": query, "query_summary": summary})
    return result


def recommendation_details(item: dict) -> list[dict[str, str]]:
    """Join internal aligned arrays, or whitelist already consolidated exports."""
    words = item.get("recommended_words") or []
    fields = ("word", "short", "query", "query_summary")
    if words and isinstance(words[0], dict):
        return [{key: entry.get(key, "") for key in fields} for entry in words]
    queries = {v.get("word"): v for v in (item.get("ai_queries") or []) if isinstance(v, dict)}
    shorts = {v.get("word"): v.get("short", "") for v in (item.get("short_words") or []) if isinstance(v, dict)}
    return [{"word": word, "short": shorts.get(word, ""),
             "query": queries.get(word, {}).get("query", ""),
             "query_summary": queries.get(word, {}).get("query_summary", "")} for word in words]


def invoke_words_with_queries(invoke, ctx, cfg_path, values):
    """Generate aligned words/questions together, repairing invalid output once."""
    from tools.llm_utils import extract_json

    render_vars = {**values, "repair_note": ""}
    for attempt in range(2):
        render_vars["_temperature_offset"] = 0.2 if attempt else 0.0
        data = extract_json(invoke(ctx, cfg_path, render_vars)) or {}
        try:
            words = data.get("recommended_words")
            if not isinstance(words, list):
                raise ValueError("recommended_words 必须是数组")
            validate_words(words)
            subjects = {v.get("word"): v.get("subject", "") for v in (data.get("short_words") or []) if isinstance(v, dict)}
            data["short_words"] = short_words(data, words, require_subject=True)
            queries = ai_queries(data, words, str(values.get("title", "")) + str(values.get("content", "")))
            for item in queries:
                if subjects[item["word"]] not in item["query_summary"]:
                    raise ValueError(f"{item['word']} 的query_summary必须保留原词主体“{subjects[item['word']]}”")
            return data, words, queries
        except ValueError as exc:
            if attempt:
                raise
            render_vars = {**values, "repair_note": f"上次输出不合规：{exc}。请重新输出完整 JSON，最多3个词，每词不超过12字，每词配short_words中的word/short/subject，short必须保留该词subject且不超过9字（空格计1字），每词配同词同序query及query_summary；概述须保留主体并表达与query一致的提问意图。"}
