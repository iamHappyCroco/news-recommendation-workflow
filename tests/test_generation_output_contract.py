"""Generation output never contains per-word ranking reasons."""
import json
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from graphs.nodes.topic_word_generate_node import topic_word_generate_node
from graphs.nodes.hot_theme_word_generate_node import hot_theme_word_generate_node
from graphs.state import (GlobalState, FinalSummaryInput, TopicWordGenerateInput,
                          HotThemeWordGenerateInput)
from tools.llm_utils import extract_json, parse_word_result
from tools.runtime import Context
from batch_run import ResumableGraphRun
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


class RetryState(TypedDict, total=False):
    title: str
    prefix: str
    stable: str
    flaky: str
    result: str


class FailedNodeRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_prefix_is_reused_and_completed_result_is_cached(self):
        calls = {"prefix": 0, "flaky": 0}
        def prefix(state):
            calls["prefix"] += 1
            return {"prefix": state["title"]}
        def flaky(state):
            calls["flaky"] += 1
            if calls["flaky"] < 3:
                raise ValueError("invalid generated word")
            return {"result": state["prefix"]}
        builder = StateGraph(RetryState)
        builder.add_node("prefix", prefix)
        builder.add_node("flaky", flaky)
        builder.set_entry_point("prefix")
        builder.add_edge("prefix", "flaky")
        builder.add_edge("flaky", END)
        run = ResumableGraphRun(builder.compile(), {"title": "article"})
        for _ in range(2):
            with self.assertRaises(ValueError):
                await run.run()
            self.assertEqual(run.pending_nodes, ("flaky",))
        output = await run.run(retry_feedback="invalid generated word")
        self.assertEqual(output["flaky"]["result"], "article")
        self.assertEqual(await run.run(), output)
        self.assertEqual(calls, {"prefix": 1, "flaky": 3})

    async def test_successful_parallel_branch_and_article_isolation(self):
        calls = {name: {"prefix": 0, "stable": 0, "flaky": 0} for name in ("one", "two")}
        def prefix(state):
            calls[state["title"]]["prefix"] += 1
            return {"prefix": state["title"]}
        def stable(state):
            calls[state["title"]]["stable"] += 1
            return {"stable": state["prefix"] + "-stable"}
        def flaky(state):
            calls[state["title"]]["flaky"] += 1
            if calls[state["title"]]["flaky"] == 1:
                raise ValueError("retry this branch")
            return {"flaky": state["prefix"] + "-flaky"}
        def finish(state):
            return {"result": state["stable"] + "/" + state["flaky"]}
        builder = StateGraph(RetryState)
        for name, node in (("prefix", prefix), ("stable", stable), ("flaky", flaky), ("finish", finish)):
            builder.add_node(name, node)
        builder.set_entry_point("prefix")
        builder.add_edge("prefix", "stable")
        builder.add_edge("prefix", "flaky")
        builder.add_edge(["stable", "flaky"], "finish")
        builder.add_edge("finish", END)
        async def article(name):
            run = ResumableGraphRun(builder.compile(), {"title": name})
            with self.assertRaises(ValueError):
                await run.run()
            output = await run.run()
            return output["finish"]["result"]
        self.assertEqual(await asyncio.gather(article("one"), article("two")),
                         ["one-stable/one-flaky", "two-stable/two-flaky"])
        for item in calls.values():
            self.assertEqual(item, {"prefix": 1, "stable": 1, "flaky": 2})


class GenerationOutputContractTests(unittest.TestCase):
    def test_topic_and_hot_theme_outputs_keep_order_and_display_without_reasons(self):
        words = ["robots"]
        payload = {"word": "robots", "short": "robots", "subject": "robots", "article_link": "The article and destination both cover robots."}
        cases = [
            ("topic_word_generate", topic_word_generate_node,
             TopicWordGenerateInput(title="robots", content="body", fit_topic_title="robot topic"), "topic_hit_result"),
            ("hot_theme_word_generate", hot_theme_word_generate_node,
             HotThemeWordGenerateInput(title="robots", content="body", fit_hot_theme_name="robots"), "hot_theme_hit_result")]
        for name, generate, state, result_key in cases:
            for legacy in (False, True):
                with self.subTest(name=name, legacy=legacy):
                    data = {**payload, **({"ranking_reasons": [{"word": "robots", "reason": "old reason"}]} if legacy else {})}
                    with patch(f"graphs.nodes.{name}_node.invoke_llm", return_value=json.dumps(data)):
                        output = generate(state, {"metadata": {"llm_cfg": f"config/{name}_llm_cfg.json"}},
                                          SimpleNamespace(context=Context())).model_dump()
                    self.assertNotIn("ranking_reasons", output)
                    self.assertNotIn("ranking_reasons", output[result_key])
                    self.assertNotIn("recommended_words", output)
                    self.assertEqual(output[result_key]["word"], "robots")
                    self.assertEqual(output[result_key]["short"], "robots")
                    self.assertNotIn("subject", output[result_key])
                    self.assertNotIn("article_link", output[result_key])
                    self.assertNotIn("display_variants", output[result_key])
                    self.assertNotIn("best_word", output[result_key])

    def test_prompt_json_examples_and_process_models_have_no_ranking_field(self):
        names = ["topic_word_generate", "hot_theme_word_generate", "quadrant_a", "quadrant_b",
                 "quadrant_c", "quadrant_d", "ai_generate"]
        for name in names:
            cfg = json.loads((ROOT / f"config/{name}_llm_cfg.json").read_text(encoding="utf-8"))
            self.assertNotIn("ranking_reasons", cfg["sp"] + cfg["up"])
            sample = cfg["sp"].split("# \u8f93\u51fa\u683c\u5f0f", 1)[1]
            example, _ = json.JSONDecoder().raw_decode(sample[sample.index("{"):])
            if name.startswith("quadrant_") or name == "ai_generate":
                self.assertEqual([q["word"] for q in example["ai_queries"]], example["recommended_words"])
        for model in (GlobalState, FinalSummaryInput):
            self.assertNotIn("ranking_reasons", model.model_fields)

    def test_legacy_word_parsers_do_not_recreate_ranking_fields(self):
        self.assertEqual(extract_json('[{"word":"robots","reason":"legacy"}]'), {"recommended_words": ["robots"]})
        self.assertEqual(parse_word_result({"recommended_words": ["robots"], "ranking_reasons": []}),
                         {"recommended_words": ["robots"]})
