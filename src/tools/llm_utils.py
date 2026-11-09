"""Platform-independent LLM invocation and JSON parsing helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
from urllib import error, request

from jinja2 import Template

from tools.runtime import Context


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REQUEST_RATE_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


def _wait_for_request_slot() -> None:
    """Pace remote requests across workflow threads, including retries."""
    global _NEXT_REQUEST_AT
    rpm = float(os.getenv("LLM_REQUESTS_PER_MINUTE", "0"))
    if rpm <= 0:
        return
    while True:
        with _REQUEST_RATE_LOCK:
            delay = _NEXT_REQUEST_AT - time.monotonic()
            if delay <= 0:
                _NEXT_REQUEST_AT = time.monotonic() + 60.0 / rpm
                return
        # Sleep outside the lock so HTTP 429 can defer all pending requests promptly.
        time.sleep(min(delay, 5.0))


def _defer_remote_requests(seconds: float) -> None:
    """Respect provider cooldown across all workflow workers, without retry storms."""
    global _NEXT_REQUEST_AT
    with _REQUEST_RATE_LOCK:
        _NEXT_REQUEST_AT = max(_NEXT_REQUEST_AT, time.monotonic() + seconds)


def load_llm_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as fd:
        return json.load(fd)


def invoke_llm(
    ctx: Context,
    cfg_path: str,
    render_vars: Optional[Dict[str, Any]] = None,
) -> str:
    """Invoke the configured Eastmoney chat endpoint for every LLM node."""
    cfg = load_llm_config(cfg_path)
    llm_config: Dict[str, Any] = cfg.get("config", {})
    render_vars = _prepare_render_vars(render_vars or {}, preserve_catalog=Path(cfg_path).name in {
        "category_fallback_llm_cfg.json", "category_fallback_verify_llm_cfg.json",
    })
    api_key = os.getenv("EM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未配置东财模型接口：请设置 EM_API_KEY。"
        )
    base_url = (os.getenv("EM_API_BASE_URL") or "https://dd-ai-api.eastmoney.com/coding/v1").rstrip("/")
    model = (
        os.getenv("EM_MODEL")
        or llm_config.get("model")
        or "DeepSeek-V4-Flash-0731"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": cfg.get("sp", "")},
            {"role": "user", "content": Template(cfg.get("up", "")).render(render_vars)},
        ],
        "temperature": llm_config.get("temperature", 0.3),
        "max_tokens": llm_config.get("max_completion_tokens", 1000),
    }
    top_p = float(llm_config.get("top_p", 1.0) or 0.0)
    if 0.0 < top_p <= 1.0:
        payload["top_p"] = top_p
    if os.getenv("LLM_THINKING_MODE", "disabled").strip().lower() == "disabled":
        payload["thinking"] = {"type": "disabled"}
    req = request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    attempts = max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))
    body = None
    for attempt in range(attempts):
        try:
            _wait_for_request_slot()
            with request.urlopen(req, timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                try:
                    retry_after = float(exc.headers.get("Retry-After", "60"))
                except (TypeError, ValueError):
                    retry_after = 60.0
                _defer_remote_requests(max(60.0, retry_after))
            if exc.code not in {403, 408, 429, 500, 502, 503, 504} or attempt + 1 >= attempts:
                raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:1000]}") from exc
        except error.URLError as exc:
            if attempt + 1 >= attempts:
                raise RuntimeError(f"LLM connection failed: {exc.reason}") from exc
        time.sleep(2 ** attempt)
    if body is None:
        raise RuntimeError("LLM request failed without a response")
    return extract_text(body["choices"][0]["message"]["content"])


def _prepare_render_vars(render_vars: Dict[str, Any], *, preserve_catalog: bool = False) -> Dict[str, Any]:
    """Bound article/context size for the Eastmoney LLM request."""
    prepared = dict(render_vars)
    content_limit = int(os.getenv("NEWS_CONTENT_MAX_CHARS", "5000"))
    field_limit = int(os.getenv("LLM_CONTEXT_FIELD_MAX_CHARS", "7000"))
    if "content" in prepared:
        prepared["content"] = str(prepared.get("content") or "")[:content_limit]
    for key in ("catalog", "candidates", "nearby_text", "hot_vocab_text", "target_page_content"):
        if key == "catalog" and preserve_catalog:
            # A fixed 82-topic taxonomy must not be silently cut midway through an ID.
            continue
        if key in prepared:
            prepared[key] = str(prepared.get(key) or "")[:field_limit]
    return prepared


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(parts).strip()
    return str(content)


def extract_json(text: str) -> Any:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
        t = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(t):
        if char in "{[":
            try:
                obj, _ = decoder.raw_decode(t, idx)
                if isinstance(obj, list):
                    words: List[str] = []
                    for item in obj:
                        if isinstance(item, str) and item.strip():
                            words.append(item.strip())
                        elif isinstance(item, dict):
                            word = str(item.get("word") or item.get("keyword") or "").strip()
                            if word:
                                words.append(word)
                    return {"recommended_words": words}
                return obj
            except json.JSONDecodeError:
                pass
    return None


def ensure_workspace_path(rel_path: str) -> str:
    path = Path(rel_path)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def parse_word_result(data: Any) -> Dict[str, Any]:
    data = data or {}
    words: List[str] = [str(w).strip() for w in (data.get("recommended_words") or []) if str(w).strip()]
    return {"recommended_words": words}
