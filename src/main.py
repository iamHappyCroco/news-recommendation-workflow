"""Local CLI and HTTP entrypoint for the recommendation-word workflow."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from graphs.graph import main_graph
from graphs.state import GraphInput, GraphOutput
from tools.article_input import prepare_article_input
from tools.runtime import Context
from tools.search_classifier import warmup


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("CLASSIFIER_WARMUP", "true").strip().lower() not in {"0", "false", "no"}:
        logger.info("warming up MiniLM classifier")
        await asyncio.to_thread(warmup)
    yield


app = FastAPI(title="资讯推荐位出词工作流", version="1.0.0", lifespan=lifespan)


async def run_workflow(payload: Dict[str, Any]) -> Dict[str, Any]:
    validated = GraphInput.model_validate(payload)
    graph_payload = prepare_article_input(validated.title, validated.content, validated.page_contents)
    result = await main_graph.ainvoke(graph_payload, context=Context())
    return GraphOutput.model_validate(result).model_dump()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "llm_mode": "eastmoney"}


@app.post("/run")
async def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return await run_workflow(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("workflow failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the workflow locally")
    parser.add_argument("--mode", choices=("flow", "http"), default="flow")
    parser.add_argument("--input", help="JSON object containing title and content")
    parser.add_argument("--input-file", help="UTF-8 JSON input file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def _load_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.input_file:
        return json.loads(Path(args.input_file).read_text(encoding="utf-8-sig"))
    if args.input:
        return json.loads(args.input)
    raise SystemExit("flow 模式必须提供 --input 或 --input-file")


if __name__ == "__main__":
    cli_args = _parse_args()
    if cli_args.mode == "http":
        uvicorn.run(app, host=cli_args.host, port=cli_args.port)
    else:
        output = asyncio.run(run_workflow(_load_payload(cli_args)))
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
