"""Run the local workflow for a sample of articles from a CSV file."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from graphs.graph import main_graph  # noqa: E402
from graphs.nodes.final_summary_node import classification_source_display, classification_region_display, compact_generation_results  # noqa: E402
from tools.article_input import prepare_article_input  # noqa: E402
from tools.runtime import Context  # noqa: E402
from tools.search_classifier import warmup  # noqa: E402
from tools.word_extras import ai_queries  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run the news workflow")
    parser.add_argument("input_csv")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many valid articles before running")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "batch_run"))
    parser.add_argument("--output-csv", help="Write only this CSV, with incremental progress; accepts Excel input")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Reuse successful rows in --output-csv")
    parser.add_argument("--rerun-indices", default="", help="With --resume, retry only these comma-separated article positions")
    parser.add_argument("--reuse-from", help="Reuse a previous CSV and rerun only categories affected by the new policy")
    parser.add_argument("--seed-results", help="Reuse already validated results from this same input before running remaining rows")
    return parser.parse_args()


async def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    graph_payload = prepare_article_input(payload["title"], payload["content"], payload.get("page_contents"))
    updates: dict[str, Any] = {}
    async for event in main_graph.astream(graph_payload, context=Context(), stream_mode="updates"):
        updates.update(event)
    model = updates.get("category_model_classify", {}) or {}
    final_node = updates.get("final_summary", {}) or {}
    final_summary = final_node.get("final_summary", {}) or {}
    return {
        "model_classification": {key: value for key, value in _jsonable(model).items()
                                 if key not in {"model_cosine_gap", "model_training_support", "model_review_reasons"}},
        "final_summary": _jsonable(final_summary),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _summary_row(source: dict[str, Any], result: dict[str, Any], index: int) -> dict[str, Any]:
    model = result.get("model_classification", {})
    final = result.get("final_summary", {})
    topic = final.get("topic_related", {})
    theme = final.get("hot_theme_related", {})
    category = final.get("category_quadrant", {})
    return {
        "序号": index,
        "资讯ID": source.get("资讯ID", ""),
        "资讯源EID": source.get("资讯源EID", ""),
        "资讯code": source.get("资讯code", ""),
        "生产时间": source.get("生产时间", ""),
        "资讯标题": source.get("资讯标题", ""),
        "运行状态": "成功",
        "耗时秒": result.get("elapsed_seconds", 0),
        "模型可用": model.get("model_available", False),
        "模型主题": model.get("model_leaf_name", ""),
        "模型相似度": model.get("model_similarity", 0),
        "模型噪声": model.get("model_is_noise", False),
        "模型次优相似度": model.get("model_second_similarity", 0),
        "分类来源": classification_source_display(
            category.get("classification_source", ""), category.get("region") != "none" and bool(category.get("generation_results") or category.get("sub_category")),
            bool(category.get("llm_fallback_used")),
        ),
        "LLM兜底分类": category.get("llm_fallback_used", False),
        "LLM兜底分类提议": json.dumps(category.get("llm_fallback_proposals", []), ensure_ascii=False),
        "LLM兜底分类理由": category.get("llm_fallback_reason", ""),
        "分类生词结果": json.dumps(category.get("generation_results", []), ensure_ascii=False),
        "象限": category.get("region", ""),
        "象限说明": category.get("quadrant", ""),
        "专题命中": topic.get("matched", False),
        "专题": topic.get("title", ""),
        "专题推荐词": "｜".join(topic.get("recommended_words", []) or []),
        "专题展示版": json.dumps(topic.get("display_variants", []), ensure_ascii=False),
        "专题首选词": topic.get("best_word", ""),
        "专题跳转链接": topic.get("jump_url", ""),
        "热点题材命中": theme.get("matched", False),
        "热点题材": theme.get("name", ""),
        "热点题材推荐词": "｜".join(theme.get("recommended_words", []) or []),
        "热点展示版": json.dumps(theme.get("display_variants", []), ensure_ascii=False),
        "热点首选词": theme.get("best_word", ""),
        "热点跳转链接": theme.get("jump_url", ""),
        "优先出词": final.get("out_priority", "none"),
        "优先理由": final.get("out_priority_reason", ""),
    }


def _has_word_extras(row: dict[str, Any]) -> bool:
    """Do not reuse a successful row that predates the new per-word fields."""
    def aligned(field: str, words_field: str) -> bool:
        try:
            items = json.loads(row.get(field) or "[]")
        except (TypeError, ValueError):
            return False
        words = [word for word in str(row.get(words_field) or "").split("｜") if word]
        return isinstance(items, list) and len(items) == len(words) and all(
            isinstance(item, dict) and item.get("word") == word
            for item, word in zip(items, words)
        )

    try:
        _validate_generation_output({"generation_results": _row_generation_results(row)})
        ai_aligned = True
    except (TypeError, ValueError, AttributeError):
        ai_aligned = False
    return (
        (str(row.get("专题命中", "")).lower() != "true" or aligned("专题展示版", "专题推荐词"))
        and (str(row.get("热点题材命中", "")).lower() != "true" or aligned("热点展示版", "热点题材推荐词"))
        and ai_aligned
    )


def _compact_export_row(row: dict[str, Any]) -> dict[str, Any]:
    """Migrate historical CSV rows to one ordered list of generation results."""
    retired_columns = {
        "专题首选理由", "专题逐词排序理由", "热点首选理由", "热点逐词排序理由",
        "分类象限首选理由", "分类象限逐词排序理由", "分类象限推荐词生成策略及理由",
        "归类模式", "分类路径", "归类校验理由", "模型首二差距", "模型主题训练样本量", "模型需复核原因",
        "归类候选及各自推荐词", "最终一级类目", "最终二级类目", "分类象限推荐词",
        "AI延伸问句", "分类象限首选词", "生词来源",
    }
    compact = {key: value for key, value in row.items() if key not in retired_columns}
    candidates = json.loads(row.get("分类生词结果") or row.get("归类候选及各自推荐词") or "[]")
    if not isinstance(candidates, list):
        raise ValueError("Invalid prior generation results")
    source = str(row.get("生词来源") or "")
    if not candidates and row.get("分类象限推荐词"):
        primary = {"word_source": source,
                   "recommended_words": [word for word in str(row["分类象限推荐词"]).split("｜") if word]}
        if source == "ai":
            primary["ai_queries"] = json.loads(row.get("AI延伸问句") or "[]")
        else:
            primary.update(category=row.get("最终一级类目", ""), sub_category=row.get("最终二级类目", ""))
        candidates = [primary]
    if candidates:
        primary = candidates[0]
        for key, column in (("category", "最终一级类目"), ("sub_category", "最终二级类目")):
            if key not in primary and row.get(column):
                primary[key] = row[column]
        if not primary.get("word_source") and source:
            primary["word_source"] = source
    primary = candidates[0] if candidates else {}
    classified = bool(primary.get("sub_category") or primary.get("leaf_id") or row.get("最终二级类目") or row.get("分类来源") in {"实际分类器分类", "LLM兜底分类"}) and primary.get("word_source") != "ai" and source != "ai"
    compact["象限"], compact["象限说明"] = classification_region_display(
        candidates, str(compact.get("象限") or ""), str(compact.get("象限说明") or ""),
        classified,
    )
    compact["分类来源"] = classification_source_display(
        str(row.get("分类路径") or row.get("分类来源") or ""),
        classified,
        str(compact.get("LLM兜底分类", "")).lower() == "true",
    )
    compact["分类生词结果"] = json.dumps(compact_generation_results(candidates), ensure_ascii=False)
    return compact


def _row_generation_results(row: dict[str, Any]) -> list[dict]:
    return json.loads(_compact_export_row(row)["分类生词结果"])


def _validate_generation_output(category: dict[str, Any]) -> None:
    generated = category.get("generation_results") or []
    if not isinstance(generated, list) or not generated:
        raise ValueError("Workflow returned missing generation results or ranked words")
    for item in generated:
        if not isinstance(item, dict):
            raise ValueError("Workflow returned invalid generation result")
        words = item.get("recommended_words")
        if not isinstance(words, list) or (not words and item.get("word_source") != "quadrant_D"):
            raise ValueError("Workflow returned missing ranked words outside quadrant D")
        ai_queries(item, words)


def _upgrade_provenance_row(row: dict[str, Any]) -> dict[str, Any]:
    """Migrate existing successful rows without redoing any LLM work."""
    row = _compact_export_row(row)
    return {
        **row,
        "本次处理": "复用", "本次处理原因": "复用已完成结果，补充分类来源并精简生词理由字段",
        "本次耗时秒": 0,
    }


def _validate_resume_source(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Restore Excel-rounded IDs only after exact title/position and rounding checks."""
    if row.get("资讯标题") != source.get("资讯标题"):
        raise ValueError("Resume CSV does not match the input article title")
    actual, expected = str(row.get("资讯ID", "")), str(source.get("资讯ID", ""))
    if actual == expected:
        return row
    # Excel saves long numeric IDs in scientific notation and loses digits.
    # Never forgive arbitrary ID changes: the displayed rounded number must agree.
    if expected.isdigit() and len(expected) >= 16 and re.fullmatch(r"\d+(?:\.\d+)?[eE]\+?\d+", actual):
        try:
            displayed = Decimal(actual)
            quantum = Decimal(1).scaleb(displayed.as_tuple().exponent)
            if quantum > 1 and Decimal(expected).quantize(quantum, rounding=ROUND_HALF_EVEN) == displayed:
                print(f"Restored source ID for article {row.get('序号')} from Excel scientific notation", flush=True)
                return {**row, "资讯ID": expected}
        except InvalidOperation:
            pass
    raise ValueError("Resume CSV does not match the input article ID")


async def _run_csv_only(args: argparse.Namespace) -> None:
    """Run original-order rows, checkpointing only the single deliverable CSV."""
    input_path = Path(args.input_csv).resolve()
    output_path = Path(args.output_csv).resolve()
    if output_path == input_path:
        raise ValueError("Output must differ from input")
    if min(args.limit, args.workers, args.attempts) < 1 or args.offset < 0:
        raise ValueError("Invalid batch size, concurrency, attempts or offset")
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(input_path, dtype=str, keep_default_na=False)
    else:
        frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    frame = frame.rename(columns={"资讯全文": "资讯正文"})
    if not {"资讯标题", "资讯正文"}.issubset(frame.columns):
        raise ValueError("Missing article title/body columns")
    selected = frame.iloc[args.offset : args.offset + args.limit]
    if len(selected) != args.limit:
        raise ValueError(f"Requested {args.limit} articles, found {len(selected)}")
    if selected[["资讯标题", "资讯正文"]].apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Selected articles contain empty titles/bodies")
    sources = [row.to_dict() for _, row in selected.iterrows()]
    rerun_positions = {int(value.strip()) for value in getattr(args, "rerun_indices", "").split(",") if value.strip()}
    available_positions = set(range(args.offset + 1, args.offset + len(sources) + 1))
    if rerun_positions and (not args.resume or not rerun_positions.issubset(available_positions)):
        raise ValueError("Rerun positions must belong to the selected range and require --resume")
    rows: dict[int, dict[str, Any]] = {}
    seed_path = Path(args.seed_results).resolve() if args.seed_results else None
    if seed_path is not None:
        if seed_path in {input_path, output_path}:
            raise ValueError("Seed CSV must differ from input and output")
        seeded = pd.read_csv(seed_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if seeded["序号"].duplicated().any():
            raise ValueError("Duplicate source positions in seed results")
        for row in seeded.to_dict("records"):
            position = int(row["序号"])
            source_index = position - args.offset - 1
            if not 0 <= source_index < len(sources):
                raise ValueError("Seed results belong to another source range")
            source = sources[source_index]
            if row["资讯标题"] != source["资讯标题"] or row.get("资讯ID", "") != source.get("资讯ID", ""):
                raise ValueError(f"Seed row {position} does not match the source article")
            if row.get("运行状态") == "成功" and _has_word_extras(row):
                rows[position] = {
                    **row, "本次处理": "复用", "本次处理原因": "复用已按当前策略完成的结果",
                    "本次耗时秒": 0,
                }
    if output_path.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists; use --resume: {output_path}")
        previous = pd.read_csv(output_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        for row in previous.to_dict("records"):
            position = int(row["序号"])
            source_index = position - args.offset - 1
            if not 0 <= source_index < len(sources):
                raise ValueError("Resume CSV belongs to another input range")
            source = sources[source_index]
            row = _validate_resume_source(row, source)
            if row.get("运行状态") == "成功" and _has_word_extras(row):
                rows[position] = row
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        rows = {position: _upgrade_provenance_row(row) for position, row in rows.items()}
        pd.DataFrame([rows[key] for key in sorted(rows)]).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Prepared {len(sources)} articles; resumed {len(rows)}; workers={args.workers}; warming classifier", flush=True)
    await asyncio.to_thread(warmup)
    print("Classifier ready", flush=True)
    semaphore = asyncio.Semaphore(args.workers)

    async def run_article(position: int, source: dict[str, Any]) -> None:
        if position in rows and position not in rerun_positions:
            return
        async with semaphore:
            payload = prepare_article_input(source["资讯标题"], source["资讯正文"])
            started = time.perf_counter()
            for attempt in range(1, args.attempts + 1):
                try:
                    result = await _run_one(payload)
                    final = result.get("final_summary", {})
                    _validate_generation_output(final.get("category_quadrant", {}))
                    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                    row = _summary_row(source, result, position)
                    row["输入正文字符数"] = len(payload["content"])
                    row["运行次数"] = attempt
                    row["错误"] = ""
                    row["本次处理"] = "新跑"
                    row["本次处理原因"] = "新增资讯，执行完整工作流"
                    row["本次耗时秒"] = result["elapsed_seconds"]
                    break
                except Exception as exc:
                    print(f"Article {position} attempt {attempt} failed: {exc}", flush=True)
                    if attempt < args.attempts:
                        await asyncio.sleep(min(2 ** attempt, 8))
                    else:
                        row = {"序号": position, "资讯ID": source.get("资讯ID", ""),
                               "资讯标题": source["资讯标题"], "运行状态": "失败",
                               "错误": str(exc), "耗时秒": round(time.perf_counter() - started, 3)}
            rows[position] = row
            pd.DataFrame([rows[key] for key in sorted(rows)]).to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"[{len(rows)}/{len(sources)}] article={position} status={row['运行状态']} elapsed={row['耗时秒']}s", flush=True)

    await asyncio.gather(*(run_article(args.offset + index, source) for index, source in enumerate(sources, 1)))
    failed = [position for position, row in rows.items() if row.get("运行状态") != "成功"]
    print(json.dumps({"csv": str(output_path), "rows": len(rows), "failed": failed}, ensure_ascii=False), flush=True)
    if failed:
        raise RuntimeError(f"Incomplete articles: {failed}; rerun with --resume")


def _old_final_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Restore the previous outputs without consulting the LLM or blue-label column."""
    row = _compact_export_row(row)
    def words(key: str) -> list[str]:
        return [word for word in str(row.get(key, "")).split("｜") if word]

    return {
        "out_priority": row.get("优先出词", "none"),
        "out_priority_reason": row.get("优先理由", ""),
        "topic_related": {
            "matched": str(row.get("专题命中", "")).lower() == "true",
            "title": row.get("专题", ""), "jump_url": row.get("专题跳转链接", ""),
            "word_source": "topic", "recommended_words": words("专题推荐词"),
            "display_variants": json.loads(row.get("专题展示版") or "[]"),
            "best_word": row.get("专题首选词", ""),
        },
        "hot_theme_related": {
            "matched": str(row.get("热点题材命中", "")).lower() == "true",
            "name": row.get("热点题材", ""), "jump_url": row.get("热点跳转链接", ""),
            "word_source": "hot_theme", "recommended_words": words("热点题材推荐词"),
            "display_variants": json.loads(row.get("热点展示版") or "[]"),
            "best_word": row.get("热点首选词", ""),
        },
        "category_quadrant": {
            "classification_source": row.get("分类来源", "未归类"),
            "llm_fallback_used": str(row.get("LLM兜底分类", "")).lower() == "true",
            "llm_fallback_proposals": json.loads(row.get("LLM兜底分类提议") or "[]"),
            "llm_fallback_reason": row.get("LLM兜底分类理由", ""),
            "region": row.get("象限", ""),
            "quadrant": row.get("象限说明", ""),
            "generation_results": json.loads(row.get("分类生词结果") or "[]"),
        },
    }


def _category_refresh_graph():
    """Reuse the original category nodes/routing with the pre-classification outputs seeded."""
    from langgraph.graph import StateGraph, END
    from graphs import graph as graph_module
    from graphs.state import GlobalState, GraphOutput
    builder = StateGraph(GlobalState, output_schema=GraphOutput)
    names = ["category_llm_fallback", "category_llm_verify", "hot_topic_select", "quadrant_a", "quadrant_b",
             "quadrant_c", "quadrant_d", "ai_generate", "category_candidate_generate", "final_summary"]
    for name in names:
        original = main_graph.builder.nodes[name]
        builder.add_node(name, getattr(graph_module, f"{name}_node"), metadata=original.metadata)
    builder.set_entry_point("category_llm_fallback")
    builder.add_edge("category_llm_fallback", "category_llm_verify")
    builder.add_conditional_edges("category_llm_verify", graph_module.route_region, {
        "象限A": "quadrant_a", "象限C": "quadrant_c", "热主题拣选": "hot_topic_select", "AI生词": "ai_generate",
    })
    builder.add_conditional_edges("hot_topic_select", graph_module.route_after_hot, {
        "象限B": "quadrant_b", "象限D": "quadrant_d", "AI生词": "ai_generate",
    })
    for region in "abcd":
        builder.add_edge(f"quadrant_{region}", "category_candidate_generate")
    builder.add_edge("category_candidate_generate", "final_summary")
    builder.add_edge("ai_generate", "final_summary")
    builder.add_edge("final_summary", END)
    return builder.compile()


async def _run_policy_refresh(args: argparse.Namespace) -> None:
    """Refresh affected classification branches, preserving other outputs in one CSV."""
    from types import SimpleNamespace
    from graphs.nodes.category_model_classify_node import category_model_classify_node
    from graphs.state import ModelClassifyInput

    input_path = Path(args.input_csv).resolve()
    old_path = Path(args.reuse_from).resolve()
    output_path = Path(args.output_csv).resolve()
    if output_path in {input_path, old_path}:
        raise ValueError("Refresh output must differ from the original input and prior CSV")
    if min(args.limit, args.workers, args.attempts) < 1 or args.offset < 0:
        raise ValueError("Invalid refresh arguments")
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(input_path, dtype=str, keep_default_na=False)
    else:
        frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    frame = frame.rename(columns={"资讯全文": "资讯正文"})
    selected = frame.iloc[args.offset : args.offset + args.limit]
    if len(selected) != args.limit:
        raise ValueError("Insufficient source articles")
    if selected[["资讯标题", "资讯正文"]].apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Empty source titles/bodies")
    previous = pd.read_csv(old_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if previous["序号"].duplicated().any():
        raise ValueError("Duplicate article positions in prior results")
    old_rows = {int(row["序号"]): row for row in previous.to_dict("records")}
    sources = selected.to_dict("records")
    for position, source in enumerate(sources, args.offset + 1):
        if position not in old_rows or old_rows[position]["资讯标题"] != source["资讯标题"]:
            raise ValueError(f"Prior result does not match source article {position}")
    completed = {}
    if output_path.exists():
        if not args.resume:
            raise FileExistsError("Refresh output already exists; use --resume")
        saved = pd.read_csv(output_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        for row in saved.to_dict("records"):
            position = int(row["序号"])
            if position not in old_rows or row["资讯标题"] != old_rows[position]["资讯标题"]:
                raise ValueError("Resume output does not match this batch")
            if row.get("运行状态") == "成功" and _has_word_extras(row):
                completed[position] = row
    print("Preparing classifier diagnostics for selective refresh", flush=True)
    await asyncio.to_thread(warmup)

    def diagnose():
        result = []
        for source in sources:
            payload = prepare_article_input(source["资讯标题"], source["资讯正文"])
            model = category_model_classify_node(ModelClassifyInput(**payload), {}, SimpleNamespace(context=Context()))
            result.append((payload, model.model_dump()))
        return result

    diagnosed = await asyncio.to_thread(diagnose)
    category_graph = _category_refresh_graph()
    plans = []
    for position, (source, (payload, model)) in enumerate(zip(sources, diagnosed), args.offset + 1):
        old = old_rows[position]
        old_generated = _row_generation_results(old)
        old_primary = old_generated[0] if old_generated else {}
        cutoff = float(os.getenv("CATEGORY_MIN_SIMILARITY", str(model["model_similarity_cutoff"])))
        near = model["nearby_topics"][:2]
        tiny_gap = (len(near) == 2 and near[1]["similarity"] >= cutoff
                    and 0 <= near[0]["similarity"] - near[1]["similarity"]
                    <= float(os.getenv("CATEGORY_DUAL_MAX_GAP", "0.005")))
        high_review = model["model_is_noise"] and model["model_similarity"] >= cutoff
        score_changed = abs(float(old.get("模型相似度") or 0) - model["model_similarity"]) > 0.0001
        reasons = []
        try:
            _validate_generation_output({"generation_results": old_generated})
            generation_missing = False
        except (ValueError, TypeError, AttributeError):
            generation_missing = True
        if old.get("运行状态") != "成功" or generation_missing:
            reasons.append("原结果缺失或失败")
        extras_missing = not _has_word_extras(old)
        if extras_missing:
            reasons.append("缺少新增展示版或AI问答问句，需完整重跑")
        if score_changed or not model["model_available"]:
            reasons.append("输入或模型预测变化")
        if high_review:
            reasons.append("高相似度需复核，不再直接AI兜底")
        if model["model_similarity"] < cutoff and str(old.get("LLM兜底分类", "")).lower() != "true":
            reasons.append("低相似度需新增LLM兜底分类及独立校验")
        if tiny_gap:
            reasons.append("前两候选近乎并列，需检查双归类")
        if not old_primary.get("sub_category") and model["model_similarity"] >= cutoff:
            reasons.append("高相似度但原类目为空")
        plans.append((position, source, payload, model, reasons, score_changed or extras_missing))
    changed = [position for position, _, _, _, reasons, _ in plans if reasons]
    print(json.dumps({"total": len(plans), "reuse": len(plans)-len(changed), "rerun": len(changed), "rerun_indices": changed}, ensure_ascii=False), flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        pd.DataFrame([_compact_export_row(completed[key]) for key in sorted(completed)]).to_csv(output_path, index=False, encoding="utf-8-sig")

    async def refresh(plan):
        position, source, payload, model, reasons, force_full = plan
        if position in completed:
            return
        old = old_rows[position]
        prior = _old_final_summary(old)
        if not reasons:
            category = prior["category_quadrant"]
            reason = "复用原批次校验与生词结果；未触发本次策略调整"
            primary = category["generation_results"][0] if category["generation_results"] else {}
            if primary.get("sub_category") and "leaf_id" not in primary:
                catalog = json.loads((ROOT / "分类器" / "topic_catalog.json").read_text(encoding="utf-8"))["leaf_topics"]
                topic = next((topic for topic in catalog.values() if topic["leaf_topic_name"] == primary["sub_category"]
                              and topic["parent_topic_name"] == primary.get("category")), None)
                if topic:
                    topic_id = int(topic["leaf_topic_id"])
                    primary.update(leaf_id=topic_id, similarity=next((nb["similarity"] for nb in model["nearby_topics"] if nb["leaf_id"] == topic_id), 0))
            row = _summary_row(source, {"model_classification": model, "final_summary": prior,
                                       "elapsed_seconds": old.get("耗时秒", 0)}, position)
            row.update({"输入正文字符数": len(payload["content"]), "运行次数": 0, "错误": "",
                        "本次处理": "复用", "本次处理原因": reason, "本次耗时秒": 0})
            completed[position] = row
            return
        async with semaphore:
            started = time.perf_counter()
            for attempt in range(1, args.attempts + 1):
                try:
                    if force_full or old.get("运行状态") != "成功":
                        result = await _run_one(payload)
                    else:
                        topic, theme = prior["topic_related"], prior["hot_theme_related"]
                        seed = {
                            **payload, **model,
                            "has_fit_topic": topic["matched"], "fit_topic_title": topic["title"], "fit_topic_url": topic["jump_url"],
                            "topic_hit_result": topic, "has_fit_hot_theme": theme["matched"],
                            "fit_hot_theme_name": theme["name"], "fit_hot_theme_url": theme["jump_url"],
                            "hot_theme_hit_result": theme, "out_priority": prior["out_priority"],
                            "out_priority_reason": prior["out_priority_reason"],
                        }
                        output = await category_graph.ainvoke(seed, context=Context())
                        final = _jsonable(output["final_summary"])
                        # Keep prior extra outputs byte-for-byte at the business-field level.
                        final.update(topic_related=topic, hot_theme_related=theme,
                                     out_priority=prior["out_priority"], out_priority_reason=prior["out_priority_reason"])
                        result = {"model_classification": model, "final_summary": final}
                    _validate_generation_output(result["final_summary"]["category_quadrant"])
                    result["elapsed_seconds"] = round(time.perf_counter()-started, 3)
                    row = _summary_row(source, result, position)
                    row.update({"输入正文字符数": len(payload["content"]), "运行次数": attempt, "错误": "",
                                "本次处理": "重跑", "本次处理原因": "；".join(reasons),
                                "本次耗时秒": result["elapsed_seconds"]})
                    break
                except Exception as exc:
                    print(f"Refresh {position}, attempt {attempt} failed: {exc}", flush=True)
                    if attempt < args.attempts:
                        await asyncio.sleep(min(2 ** attempt, 8))
                    else:
                        row = {**old, "运行状态": "失败", "错误": str(exc), "本次处理": "重跑失败"}
            completed[position] = row
            save()
            print(f"[{len(completed)}/{len(plans)}] refreshed article={position} status={row['运行状态']}", flush=True)

    semaphore = asyncio.Semaphore(args.workers)
    # Populate all reusable rows before starting any API calls.
    for plan in plans:
        if not plan[4]:
            await refresh(plan)
    save()
    await asyncio.gather(*(refresh(plan) for plan in plans if plan[4]))
    failed = [key for key, row in completed.items() if row.get("运行状态") != "成功"]
    print(json.dumps({"csv": str(output_path), "rows": len(completed), "reuse": len(plans)-len(changed),
                      "refreshed": len(changed), "failed": failed}, ensure_ascii=False), flush=True)
    if failed:
        raise RuntimeError(f"Incomplete refresh rows: {failed}; use --resume")


async def main() -> None:
    args = _args()
    if args.reuse_from:
        if not args.output_csv:
            raise ValueError("--reuse-from requires --output-csv")
        await _run_policy_refresh(args)
        return
    if args.output_csv:
        await _run_csv_only(args)
        return
    input_path = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {"资讯标题", "资讯正文"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {sorted(missing)}")
    eligible = frame[
        frame["资讯标题"].str.strip().ne("") & frame["资讯正文"].str.strip().ne("")
    ]
    selected = eligible.iloc[args.offset : args.offset + args.limit]

    await asyncio.to_thread(warmup)
    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for batch_position, (_, row) in enumerate(selected.iterrows(), start=1):
        position = args.offset + batch_position
        source = row.to_dict()
        payload = {
            "title": source["资讯标题"],
            "content": source["资讯正文"],
        }
        started = time.perf_counter()
        try:
            result = await _run_one(payload)
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            details.append({"index": position, "source": source, **result})
            summaries.append(_summary_row(source, result, position))
            print(f"[{batch_position}/{len(selected)}] 成功 {payload['title']} ({result['elapsed_seconds']}s)", flush=True)
        except Exception as exc:
            elapsed = round(time.perf_counter() - started, 3)
            details.append({"index": position, "source": source, "error": str(exc), "elapsed_seconds": elapsed})
            summaries.append({"序号": position, "资讯标题": source["资讯标题"], "运行状态": "失败", "错误": str(exc), "耗时秒": elapsed})
            print(f"[{batch_position}/{len(selected)}] 失败 {payload['title']}: {exc}", flush=True)

    if args.offset:
        range_label = f"{args.offset + 1}_{args.offset + len(selected)}"
    else:
        range_label = str(args.limit)
    detail_path = output_dir / f"workflow_results_{range_label}.json"
    summary_path = output_dir / f"workflow_summary_{range_label}.csv"
    detail_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"detail": str(detail_path), "summary": str(summary_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
