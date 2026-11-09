"""Run original-order Excel articles and checkpoint their exact final summaries."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd

from batch_run import _run_one, _validate_generation_output, prepare_article_input, warmup


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.workers < 1:
        raise ValueError("limit and workers must be positive")
    frame = pd.read_excel(args.input_file, dtype=str, keep_default_na=False)
    frame = frame.rename(columns={"资讯全文": "资讯正文"})
    selected = frame.iloc[:args.limit]
    if len(selected) != args.limit:
        raise ValueError("Insufficient source rows")
    if selected[["资讯标题", "资讯正文"]].apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Selected rows contain empty titles/bodies")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "final_summaries.json"
    if target.exists() and not args.resume:
        raise FileExistsError(target)
    records = {}
    if target.exists():
        for record in json.loads(target.read_text(encoding="utf-8")):
            index = record["index"]
            if not 1 <= index <= len(selected):
                raise ValueError("Resume record outside selected range")
            source = selected.iloc[index - 1]
            if record["title"] != source["资讯标题"] or record["article_id"] != source.get("资讯ID", ""):
                raise ValueError("Resume record does not match source")
            if record.get("status") == "success":
                _validate_generation_output(record["final_summary"].get("category_quadrant", {}))
                records[index] = record
    semaphore = asyncio.Semaphore(args.workers)
    print(f"Prepared {len(selected)} articles; warming local classifier", flush=True)
    await asyncio.to_thread(warmup)
    print("Classifier ready", flush=True)

    def save() -> None:
        data = [records[i] for i in sorted(records)]
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)

    async def run(index, source) -> None:
        if index in records:
            return
        async with semaphore:
            payload = prepare_article_input(source["资讯标题"], source["资讯正文"])
            record = {
                "index": index,
                "article_id": source.get("资讯ID", ""),
                "title": source["资讯标题"],
                "original_content_chars": len(source["资讯正文"].strip()),
                "input_content_chars": len(payload["content"]),
            }
            started = time.perf_counter()
            for attempt in range(1, 4):
                try:
                    result = await _run_one(payload)
                    final = result["final_summary"]
                    if not final:
                        raise ValueError("Empty final_summary")
                    _validate_generation_output(final.get("category_quadrant", {}))
                    record.update(status="success", final_summary=final)
                    break
                except Exception as exc:
                    print(f"Article {index} attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
                    if attempt == 3:
                        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    else:
                        await asyncio.sleep(2 ** attempt)
            record.update(attempts=attempt, elapsed_seconds=round(time.perf_counter() - started, 3))
            records[index] = record
            save()
            print(f"[{len(records)}/{len(selected)}] Article {index}: {record['status']} ({record['elapsed_seconds']}s)", flush=True)

    await asyncio.gather(*(run(i, row.to_dict()) for i, (_, row) in enumerate(selected.iterrows(), 1)))
    metadata = {
        "source": str(args.input_file.resolve()),
        "body_max_chars": int(os.getenv("NEWS_CONTENT_MAX_CHARS", "5000")),
        "llm_base_url": os.getenv("EM_API_BASE_URL", "https://dd-ai-api.eastmoney.com/coding/v1"),
        "model": os.getenv("EM_MODEL", "DeepSeek-V4-Flash-0731"),
        "success": sum(r["status"] == "success" for r in records.values()),
        "failed": sum(r["status"] == "failed" for r in records.values()),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    if metadata["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
