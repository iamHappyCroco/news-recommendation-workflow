"""Canonical workflow input preparation: title plus bounded article body only."""
from __future__ import annotations

import os
from typing import Any


def prepare_article_input(title: str, content: str, page_contents: dict[str, str] | None = None) -> dict[str, Any]:
    """Normalize one article and truncate its body before it enters LangGraph."""
    limit = max(1, int(os.getenv("NEWS_CONTENT_MAX_CHARS", "5000")))
    prepared: dict[str, Any] = {
        "title": str(title or "").strip(),
        "content": str(content or "").strip()[:limit],
    }
    if page_contents:
        prepared["page_contents"] = dict(page_contents)
    return prepared
