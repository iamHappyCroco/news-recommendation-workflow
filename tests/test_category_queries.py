"""Category queries must survive every branch and export/resume."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from graphs.nodes.ai_generate_node import ai_generate_node
from graphs.nodes.quadrant_a_node import quadrant_a_node
from graphs.nodes.quadrant_b_node import quadrant_b_node
from graphs.nodes.quadrant_c_node import quadrant_c_node
from graphs.nodes.quadrant_d_node import quadrant_d_node
from graphs.nodes.final_summary_node import final_summary_node
from graphs.nodes.category_candidate_generate_node import category_candidate_generate_node
from graphs.state import (AiGenerateInput, FinalSummaryInput, QuadrantInput, GlobalState,
                          ClassificationCandidate, QuadrantOutput, HotTopicSelectOutput)
from tools.runtime import Context
from batch_run import _has_word_extras, _summary_row, _old_final_summary, _validate_generation_output

RUNTIME = SimpleNamespace(context=Context())
WORDS = ["robot orders", "robot components"]
QUERIES = [{"word": WORDS[0], "query": "What is the commercialization progress of the newly released robots?"},
           {"word": WORDS[1], "query": "Which components are used in the newly released robots?"}]


class CategoryQueryTests(unittest.TestCase):
    def test_all_five_branches_preserve_questions_through_summary_and_csv(self):
        for region, generate in [("A", quadrant_a_node), ("B", quadrant_b_node),
                                 ("C", quadrant_c_node), ("D", quadrant_d_node), ("ai", ai_generate_node)]:
            with self.subTest(region=region):
                node = "ai_generate" if region == "ai" else f"quadrant_{region.lower()}"
                article = AiGenerateInput(title="new robots", content="robot component and order progress") if region == "ai" else QuadrantInput(
                    title="new robots", content="robot component and order progress", region=region, category="concept", sub_category="robots")
                with patch(f"graphs.nodes.{node}_node.invoke_llm", return_value=json.dumps(
                    {"recommended_words": WORDS, "ai_queries": QUERIES})):
                    generated = generate(article, {"metadata": {"llm_cfg": f"config/{node}_llm_cfg.json"}}, RUNTIME)
                self.assertEqual(generated.ai_queries, QUERIES)
                self.assertNotIn("ranking_reasons", generated.model_dump())
                result = final_summary_node(FinalSummaryInput(
                    region="" if region == "ai" else region,
                    category="" if region == "ai" else "concept", sub_category="" if region == "ai" else "robots",
                    recommended_words=generated.recommended_words, ai_queries=generated.ai_queries,
                    word_source=generated.word_source), {}, RUNTIME).final_summary
                category = result["category_quadrant"]
                self.assertEqual(category["generation_results"][0]["ai_queries"], QUERIES)
                _validate_generation_output(category)
                row = _summary_row({}, {"final_summary": result}, 1)
                self.assertTrue(_has_word_extras(row))
                self.assertEqual(_old_final_summary(row)["category_quadrant"], category)

    def test_invalid_questions_are_repaired_once_for_each_quadrant(self):
        for region, generate in [("A", quadrant_a_node), ("B", quadrant_b_node),
                                 ("C", quadrant_c_node), ("D", quadrant_d_node)]:
            with self.subTest(region=region):
                module = f"graphs.nodes.quadrant_{region.lower()}_node"
                with patch(f"{module}.invoke_llm", side_effect=[
                    json.dumps({"recommended_words": WORDS}),
                    json.dumps({"recommended_words": WORDS, "ai_queries": QUERIES})]) as llm:
                    result = generate(QuadrantInput(title="new robots", content="body", region=region),
                        {"metadata": {"llm_cfg": f"config/quadrant_{region.lower()}_llm_cfg.json"}}, RUNTIME)
                self.assertEqual(result.ai_queries, QUERIES)
                self.assertEqual(llm.call_count, 2)
                self.assertTrue(llm.call_args.args[2]["repair_note"])

    def test_missing_reordered_or_blank_queries_prevent_resume(self):
        for queries in [None, [], QUERIES[::-1], [{"word": w, "query": ""} for w in WORDS]]:
            with self.subTest(queries=queries):
                group = {"word_source": "quadrant_C", "recommended_words": WORDS, "ai_queries": queries}
                self.assertFalse(_has_word_extras({"分类生词结果": json.dumps([group])}))
                with self.assertRaises(ValueError):
                    _validate_generation_output({"generation_results": [group]})
        self.assertFalse(_has_word_extras({"分类生词结果": json.dumps([
            {"word_source": "quadrant_A", "recommended_words": WORDS, "ai_queries": QUERIES},
            {"word_source": "quadrant_D", "recommended_words": WORDS}])}))

    def test_persistently_invalid_questions_fail_generation(self):
        with patch("graphs.nodes.quadrant_c_node.invoke_llm", return_value=json.dumps(
            {"recommended_words": WORDS, "ai_queries": QUERIES[::-1]})) as llm:
            with self.assertRaises(ValueError):
                quadrant_c_node(QuadrantInput(title="robots", content="body", region="C"),
                    {"metadata": {"llm_cfg": "config/quadrant_c_llm_cfg.json"}}, RUNTIME)
            self.assertEqual(llm.call_count, 2)

    def test_empty_recommendations_have_empty_queries(self):
        with patch("graphs.nodes.quadrant_d_node.invoke_llm", return_value=json.dumps(
            {"recommended_words": [], "ai_queries": []})):
            result = quadrant_d_node(QuadrantInput(title="robots", content="body", region="D"),
                {"metadata": {"llm_cfg": "config/quadrant_d_llm_cfg.json"}}, RUNTIME)
        self.assertEqual(result.recommended_words, [])
        self.assertEqual(result.ai_queries, [])

    def test_empty_d_survives_single_and_dual_summary_export_and_resume(self):
        for regions in [("D",), ("A", "D"), ("D", "D")]:
            with self.subTest(regions=regions):
                candidates = [ClassificationCandidate(leaf_id=i, category="concept", sub_category=f"topic {i}",
                    region=region, quadrant=f"quadrant {region}", reason="article evidence")
                    for i, region in enumerate(regions, 1)]
                words, queries = ([], []) if regions[0] == "D" else (WORDS, QUERIES)
                state = GlobalState(title="article", content="body", category="concept", sub_category="topic 1",
                    can_classify=True, region=regions[0], word_source=f"quadrant_{regions[0]}",
                    classification_source="minilm_verified", classification_candidates=candidates,
                    recommended_words=words, ai_queries=queries)
                with patch.dict("graphs.nodes.category_candidate_generate_node.GENERATORS", {
                    "D": lambda *args: QuadrantOutput(recommended_words=[], ai_queries=[], word_source="quadrant_D")
                }), patch("graphs.nodes.category_candidate_generate_node.hot_topic_select_node",
                           return_value=HotTopicSelectOutput(has_hot_topic=False)):
                    generated = category_candidate_generate_node(state, {}, RUNTIME)
                final = final_summary_node(FinalSummaryInput(**{
                    **state.model_dump(), **generated.model_dump()}), {}, RUNTIME).final_summary
                category = final["category_quadrant"]
                self.assertEqual(category["region"], "/".join(regions))
                self.assertEqual(len(category["generation_results"]), len(regions))
                for candidate, result in zip(candidates, category["generation_results"]):
                    self.assertEqual(result["leaf_id"], candidate.leaf_id)
                    if candidate.region == "D":
                        self.assertEqual(result["recommended_words"], [])
                        self.assertEqual(result["ai_queries"], [])
                _validate_generation_output(category)
                row = _summary_row({}, {"final_summary": final}, 1)
                self.assertTrue(_has_word_extras(row))
                self.assertEqual(_old_final_summary(row)["category_quadrant"], category)

    def test_empty_a_b_c_still_fail_candidates_and_batch_validation(self):
        for region in ["A", "B", "C"]:
            with self.subTest(region=region):
                candidate = ClassificationCandidate(leaf_id=1, category="concept", sub_category="topic",
                    region=region, quadrant=f"quadrant {region}")
                state = GlobalState(title="article", content="body", classification_candidates=[candidate],
                    word_source=f"quadrant_{region}", recommended_words=[], ai_queries=[])
                with self.assertRaises(ValueError):
                    category_candidate_generate_node(state, {}, RUNTIME)
                group = {"word_source": f"quadrant_{region}", "recommended_words": [], "ai_queries": []}
                with self.assertRaises(ValueError):
                    _validate_generation_output({"generation_results": [group]})
                self.assertFalse(_has_word_extras({"分类生词结果": json.dumps([group])}))

    def test_empty_d_requires_explicit_aligned_empty_arrays(self):
        for group in [
            {"word_source": "quadrant_D", "ai_queries": []},
            {"word_source": "quadrant_D", "recommended_words": []},
            {"word_source": "quadrant_D", "recommended_words": [], "ai_queries": QUERIES},
            {"word_source": "ai", "recommended_words": [], "ai_queries": []},
        ]:
            with self.subTest(group=group), self.assertRaises(ValueError):
                _validate_generation_output({"generation_results": [group]})


if __name__ == "__main__":
    unittest.main()
