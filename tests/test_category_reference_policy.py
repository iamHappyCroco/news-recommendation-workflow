"""Offline checks for selected-category plus article generation in B/D."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphs.nodes.hot_topic_select_node import hot_topic_select_node
from graphs.nodes.quadrant_b_node import quadrant_b_node
from graphs.nodes.quadrant_d_node import quadrant_d_node
from graphs.nodes.category_candidate_generate_node import category_candidate_generate_node
from graphs.state import (ClassificationCandidate, GlobalState, HotTopicSelectInput,
                          HotTopicSelectOutput, NearbyTopic, QuadrantInput, QuadrantOutput)
from tools.runtime import Context

RUNTIME = SimpleNamespace(context=Context())
REFERENCE_NAME = "computing infrastructure"
CONNECTION = "The article's server orders relate to computing infrastructure demand."


def catalogs(path):
    if "quadrants" in path:
        return {"topics": {"1": {"region": "A"}}}
    return {"leaf_topics": {"1": {"leaf_topic_id": 1, "leaf_topic_name": REFERENCE_NAME,
                                  "parent_topic_name": "technology",
                                  "representative_searches": ["computing servers"]}}}


class CategoryReferencePolicyTests(unittest.TestCase):
    def select(self, similarity):
        module = "graphs.nodes.hot_topic_select_node"
        with patch(f"{module}._load_catalog", side_effect=catalogs), \
             patch(f"{module}.invoke_llm", return_value=json.dumps({"hot_leaf_id": 1, "reason": CONNECTION})):
            return hot_topic_select_node(
                HotTopicSelectInput(title="server orders", content="company wins server orders", region="B",
                                    nearby_topics=[NearbyTopic(leaf_id=1, leaf_name=REFERENCE_NAME,
                                                               similarity=similarity)]),
                {"metadata": {"llm_cfg": "config/hot_topic_select_llm_cfg.json"}}, RUNTIME,
            )

    def test_selection_passes_category_and_combination_reason(self):
        selected = self.select(0.7)
        self.assertTrue(selected.has_hot_topic)
        self.assertEqual(selected.hot_leaf_name, REFERENCE_NAME)
        self.assertIn(REFERENCE_NAME, selected.hot_reference_text)
        self.assertIn("technology", selected.hot_reference_text)
        self.assertIn(CONNECTION, selected.hot_reference_text)
        self.assertEqual(selected.hot_rep_searches, ["computing servers"])

    def test_existing_similarity_gate_remains(self):
        selected = self.select(0.499)
        self.assertFalse(selected.has_hot_topic)
        self.assertEqual(selected.hot_reference_text, "")
        self.assertEqual(selected.hot_rep_searches, [])

    def test_b_and_d_receive_important_category_without_changing_region(self):
        for region, generate in [("B", quadrant_b_node), ("D", quadrant_d_node)]:
            with self.subTest(region=region):
                response = {"recommended_words": ["company computing servers"],
                            "ai_queries": [{"word": "company computing servers", "query": "How do server orders affect computing demand?"}],
                            "generation_reason": CONNECTION}
                module = f"graphs.nodes.quadrant_{region.lower()}_node"
                with patch(f"{module}.invoke_llm", return_value=json.dumps(response)) as llm:
                    output = generate(
                        QuadrantInput(title="server orders", content="company wins server orders",
                                      category="company", sub_category="orders", region=region,
                                      has_hot_topic=True, hot_leaf_name=REFERENCE_NAME,
                                      hot_keywords=REFERENCE_NAME, hot_rep_searches=["computing servers"],
                                      hot_reference_text=CONNECTION),
                        {"metadata": {"llm_cfg": f"config/quadrant_{region.lower()}_llm_cfg.json"}}, RUNTIME,
                    )
                values = llm.call_args.args[2]
                self.assertEqual(values["region"], region)
                self.assertEqual(values["category"], "company")
                self.assertEqual(values["hot_leaf_name"], REFERENCE_NAME)
                self.assertEqual(values["hot_reference_text"], CONNECTION)
                self.assertEqual(values["content"], "company wins server orders")
                self.assertEqual(output.word_source, f"quadrant_{region}")
                self.assertEqual(output.generation_reason, CONNECTION)

    def test_no_reference_retains_original_article_and_fallback_reason(self):
        for region, generate in [("B", quadrant_b_node), ("D", quadrant_d_node)]:
            with self.subTest(region=region):
                module = f"graphs.nodes.quadrant_{region.lower()}_node"
                with patch(f"{module}.invoke_llm", return_value=json.dumps(
                    {"recommended_words": ["server orders"],
                     "ai_queries": [{"word": "server orders", "query": "What changed in server orders?"}]},
                )) as llm:
                    output = generate(
                        QuadrantInput(title="server orders", content="body", region=region, has_hot_topic=False),
                        {"metadata": {"llm_cfg": f"config/quadrant_{region.lower()}_llm_cfg.json"}}, RUNTIME,
                    )
                self.assertFalse(llm.call_args.args[2]["has_hot_topic"])
                self.assertEqual(llm.call_args.args[2]["content"], "body")
                self.assertEqual(output.word_source, f"quadrant_{region}")
                self.assertTrue(output.generation_reason)

    def test_prompts_require_category_article_combination_and_keep_counts(self):
        for region, count in [("b", "3-5"), ("d", "1-2")]:
            with self.subTest(region=region):
                cfg = json.loads((ROOT / f"config/quadrant_{region}_llm_cfg.json").read_text(encoding="utf-8"))
                self.assertIn(count, cfg["sp"])
                self.assertIn("generation_reason", cfg["sp"])
                self.assertIn("\u91cd\u8981\u53c2\u8003\u7c7b\u522b", cfg["sp"])
                self.assertIn("\u539f\u8d44\u8baf\u4e3b\u4f53\u4e0e\u4e8b\u5b9e", cfg["up"])
                self.assertNotIn("\u4f9b\u98ce\u683c\u53c2\u7167", cfg["up"])

    def test_second_d_candidate_uses_its_selected_category_independently(self):
        module = "graphs.nodes.category_candidate_generate_node"
        reference = HotTopicSelectOutput(has_hot_topic=True, hot_leaf_name=REFERENCE_NAME,
                                         hot_reference_text=CONNECTION, hot_rep_searches=["computing servers"])
        generated = QuadrantOutput(recommended_words=["company computing servers"], word_source="quadrant_D",
                                   ai_queries=[{"word": "company computing servers", "query": "How do server orders affect computing demand?"}],
                                   generation_reason=CONNECTION)
        article = GlobalState(title="server orders", content="company wins server orders",
                              ai_queries=[{"word": "primary word", "query": "What changed in server orders?"}],
                              recommended_words=["primary word"], word_source="quadrant_A",
                              classification_candidates=[
                                  ClassificationCandidate(leaf_id=1, category="technology", sub_category="primary",
                                                          region="A", quadrant="A"),
                                  ClassificationCandidate(leaf_id=2, category="company", sub_category="orders",
                                                          region="D", quadrant="D"),
                              ])
        with patch(f"{module}.hot_topic_select_node", return_value=reference), \
             patch.dict(f"{module}.GENERATORS", {"D": lambda state, config, runtime: self.capture(state, generated)}):
            output = category_candidate_generate_node(article, {"metadata": {}}, RUNTIME)
        self.assertTrue(self.captured.has_hot_topic)
        self.assertEqual(self.captured.hot_leaf_name, REFERENCE_NAME)
        self.assertEqual(self.captured.hot_reference_text, CONNECTION)
        self.assertEqual(self.captured.region, "D")
        self.assertEqual(output.category_candidate_results[0]["recommended_words"], ["primary word"])
        self.assertEqual(output.category_candidate_results[1]["recommended_words"], ["company computing servers"])
        self.assertEqual(output.category_candidate_results[0]["ai_queries"], article.ai_queries)
        self.assertEqual(output.category_candidate_results[1]["ai_queries"], generated.ai_queries)
        for result in output.category_candidate_results:
            for field in ("ranking_reasons", "best_word_reason", "generation_reason"):
                self.assertNotIn(field, result)

    def capture(self, state, output):
        self.captured = state
        return output


if __name__ == "__main__":
    unittest.main()
