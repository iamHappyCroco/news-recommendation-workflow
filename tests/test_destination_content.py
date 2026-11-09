"""Destination summaries reach generation; supplied page content takes priority."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphs.nodes.topic_recall_node import topic_recall_node
from graphs.nodes.topic_fit_judge_node import topic_fit_judge_node
from graphs.nodes.topic_word_generate_node import topic_word_generate_node
from graphs.nodes.hot_theme_recall_node import hot_theme_recall_node
from graphs.nodes.hot_theme_fit_judge_node import hot_theme_fit_judge_node
from graphs.nodes.hot_theme_word_generate_node import hot_theme_word_generate_node
from graphs.state import (TopicRecallInput, TopicFitJudgeInput, TopicWordGenerateInput,
                          HotThemeRecallInput, HotThemeFitJudgeInput, HotThemeWordGenerateInput, GraphInput)
from tools.article_input import prepare_article_input
from tools.runtime import Context

RUNTIME = SimpleNamespace(context=Context())
WORDS = ["routes", "chains", "limits"]
OUTPUT = json.dumps({"recommended_words": WORDS, "display_variants": [
    {"word": word, "display_6": word, "display_9": word} for word in WORDS]})


def config(name):
    return {"metadata": {"llm_cfg": f"config/{name}_llm_cfg.json"}}


class DestinationContentTests(unittest.TestCase):
    def test_topic_intro_is_preserved_from_recall_through_generation(self):
        with patch("graphs.nodes.topic_recall_node._load_topic_map", return_value={
            "robots": {"intro": "technology routes and industrial chains", "url": "https://example.test/topic"}}), \
             patch("graphs.nodes.topic_recall_node.search", return_value=[(0, 0.8)]):
            recalled = topic_recall_node(TopicRecallInput(title="robots", content="new robot released"), {}, RUNTIME)
        with patch("graphs.nodes.topic_fit_judge_node.invoke_llm", return_value=json.dumps(
            {"has_fit_topic": True, "fit_topic_title": "robots"})):
            fitted = topic_fit_judge_node(TopicFitJudgeInput(title="robots", content="new robot released",
                topic_candidates=recalled.topic_candidates), config("topic_fit_judge"), RUNTIME)
        self.assertEqual(fitted.fit_topic_intro, "technology routes and industrial chains")
        self.assertEqual(fitted.fit_topic_url, "https://example.test/topic")
        with patch("graphs.nodes.topic_word_generate_node.invoke_llm", return_value=OUTPUT) as llm:
            output = topic_word_generate_node(TopicWordGenerateInput(title="robots", content="new robot released",
                **{key: value for key, value in fitted.model_dump().items() if key in TopicWordGenerateInput.model_fields}),
                config("topic_word_generate"), RUNTIME)
        self.assertEqual(llm.call_args.args[2]["target_page_content"], fitted.fit_topic_intro)
        self.assertEqual(llm.call_args.args[2]["content"], "new robot released")
        self.assertEqual(output.topic_hit_result["recommended_words"], WORDS)
        self.assertNotIn("target_page_content", output.topic_hit_result)

    def test_hot_theme_interpretation_is_preserved_from_recall_through_generation(self):
        with patch("graphs.nodes.hot_theme_recall_node._load_theme_rows", return_value=[
            {"code": 9, "name": "robots", "interpretation": "routes, chains and application limits",
             "url": "https://example.test/theme"}]), \
             patch("graphs.nodes.hot_theme_recall_node.search", return_value=[(0, 0.8)]):
            recalled = hot_theme_recall_node(HotThemeRecallInput(title="robots", content="new robot released"), {}, RUNTIME)
        with patch("graphs.nodes.hot_theme_fit_judge_node.invoke_llm", return_value=json.dumps(
            {"has_fit_hot_theme": True, "fit_hot_theme_code": 9})):
            fitted = hot_theme_fit_judge_node(HotThemeFitJudgeInput(title="robots", content="new robot released",
                hot_theme_candidates=recalled.hot_theme_candidates), config("hot_theme_fit_judge"), RUNTIME)
        self.assertEqual(fitted.interpretation, "routes, chains and application limits")
        with patch("graphs.nodes.hot_theme_word_generate_node.invoke_llm", return_value=OUTPUT) as llm:
            output = hot_theme_word_generate_node(HotThemeWordGenerateInput(title="robots", content="new robot released",
                **{key: value for key, value in fitted.model_dump().items() if key in HotThemeWordGenerateInput.model_fields}),
                config("hot_theme_word_generate"), RUNTIME)
        self.assertEqual(llm.call_args.args[2]["target_page_content"], fitted.interpretation)
        self.assertEqual(output.hot_theme_hit_result["recommended_words"], WORDS)

    def test_only_matching_url_content_overrides_page_text_and_summary(self):
        cases = [
            ("topic_word_generate", topic_word_generate_node, TopicWordGenerateInput(
                title="robots", content="new robot released", fit_topic_title="robots",
                fit_topic_url="https://example.test/topic", fit_topic_intro="summary",
                fit_topic_page_content="sheet page", page_contents={
                    "https://example.test/topic": "supplied page", "https://example.test/other": "wrong page"})),
            ("hot_theme_word_generate", hot_theme_word_generate_node, HotThemeWordGenerateInput(
                title="robots", content="new robot released", fit_hot_theme_name="robots",
                fit_hot_theme_url="https://example.test/theme", interpretation="summary",
                fit_hot_theme_page_content="sheet page", page_contents={
                    "https://example.test/theme": "supplied page", "https://example.test/other": "wrong page"}))]
        for name, generate, state in cases:
            for expected, override in [("supplied page", state.page_contents), ("sheet page", {"https://example.test/other": "wrong page"})]:
                with self.subTest(name=name, expected=expected), \
                     patch(f"graphs.nodes.{name}_node.invoke_llm", return_value=OUTPUT) as llm:
                    generate(state.model_copy(update={"page_contents": override}), config(name), RUNTIME)
                    self.assertEqual(llm.call_args.args[2]["target_page_content"], expected)

    def test_optional_workflow_input_preserves_page_content_mapping(self):
        supplied = {"https://example.test/topic": "technology comparisons"}
        article = GraphInput(title="robots", content="news", page_contents=supplied)
        prepared = prepare_article_input(article.title, article.content, article.page_contents)
        self.assertEqual(prepared["page_contents"], supplied)
        self.assertEqual(prepare_article_input("robots", "news"), {"title": "robots", "content": "news"})
