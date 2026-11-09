"""Regression checks for review gating, dual category routing and single-file export."""
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from graphs.nodes.category_llm_verify_node import category_llm_verify_node
from graphs.nodes.category_llm_fallback_node import category_llm_fallback_node, route_category_confidence
from graphs.state import LlmVerifyInput, NearbyTopic
from tools.runtime import Context


TOPICS = {str(i): {
    "leaf_topic_id": i, "leaf_topic_name": f"topic-{i}",
    "parent_topic_name": f"parent-{i}", "representative_searches": [f"term-{i}"],
} for i in range(1, 7)}
QUADRANTS = {str(i): {
    "leaf_topic_name": f"topic-{i}", "region": "A" if i == 1 else "B",
    "quadrant_label": "test",
} for i in range(1, 7)}
CONFIG = {"metadata": {"llm_cfg": "config/category_classify_llm_cfg.json"}}
RUNTIME = SimpleNamespace(context=Context())
VERIFY_MODULE = "graphs.nodes.category_llm_verify_node"
FALLBACK_MODULE = "graphs.nodes.category_llm_fallback_node"
FALLBACK_CONFIG = {"metadata": {"llm_cfg": "config/category_fallback_llm_cfg.json"}}


def load_catalog(path):
    return {"topics": QUADRANTS} if "quadrants" in path else {"leaf_topics": TOPICS}


def state(gap=0.001, score=0.73, support=10):
    return LlmVerifyInput(
        title="article", content="body", model_available=True, model_is_noise=True,
        model_leaf_id=1, model_similarity=score, model_cosine_gap=gap,
        model_second_similarity=score-gap, model_training_support=support,
        model_review_reasons=["close_candidates"],
        nearby_topics=[NearbyTopic(leaf_id=i, leaf_name=f"topic-{i}", similarity=s)
                       for i, s in enumerate([score, score-gap, 0.63, 0.62, 0.61], 1)],
    )


def response(ids):
    return {"can_classify": bool(ids), "classification_reason": "article evidence",
            "accepted_candidates": [{"leaf_id": i, "reason": f"evidence-{i}"} for i in ids]}


class CategoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"CATEGORY_DUAL_MAX_GAP": "0.005",
                                          "CATEGORY_MIN_SIMILARITY": "0.6542854905128479"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def verify(self, article, output):
        with patch(f"{VERIFY_MODULE}._load_json", side_effect=load_catalog), \
             patch(f"{VERIFY_MODULE}.invoke_llm", return_value=json.dumps(output)) as llm:
            result = category_llm_verify_node(article, CONFIG, RUNTIME)
            return result, llm

    def test_high_similarity_review_reaches_llm_and_keeps_two_accepted_topics(self):
        result, llm = self.verify(state(), response([2, 1]))
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(result.classification_mode, "dual")
        self.assertEqual(result.sub_category, "topic-2")
        self.assertEqual([c.leaf_id for c in result.classification_candidates], [2, 1])

    def test_close_but_unrelated_topic_is_not_forced_into_output(self):
        result, _ = self.verify(state(), response([2]))
        self.assertEqual(result.classification_mode, "single")
        self.assertEqual(result.sub_category, "topic-2")

    def test_non_tiny_gap_limits_output_to_one(self):
        result, _ = self.verify(state(gap=0.015), response([1, 2]))
        self.assertEqual(result.classification_mode, "single")
        self.assertEqual(len(result.classification_candidates), 1)

    def test_secondary_below_similarity_floor_cannot_be_dual(self):
        result, _ = self.verify(state(score=0.656, gap=0.004), response([1, 2]))
        self.assertEqual(result.classification_mode, "single")

    def test_low_best_similarity_requires_fallback_before_verification(self):
        article = state(score=0.60)
        self.assertEqual(route_category_confidence(article), "fallback")
        with self.assertRaises(ValueError):
            self.verify(article, response([1]))
        self.assertEqual(route_category_confidence(state()), "verify")

    def fallback_state(self, output=None):
        article = state(score=0.60)
        with patch(f"{FALLBACK_MODULE}._load_json", side_effect=load_catalog), \
             patch(f"{FALLBACK_MODULE}.invoke_llm", return_value=json.dumps(response([6]) if output is None else output)) as llm:
            proposal = category_llm_fallback_node(article, FALLBACK_CONFIG, RUNTIME)
        llm.assert_called_once()
        self.assertIn("[ID6]", llm.call_args.args[2]["catalog"])
        return article.model_copy(update=proposal.model_dump())

    def test_low_similarity_proposal_outside_top5_is_independently_verified(self):
        article = self.fallback_state()
        result, llm = self.verify(article, response([6]))
        self.assertEqual(Path(llm.call_args.args[1]).name, "category_fallback_verify_llm_cfg.json")
        self.assertEqual(result.sub_category, "topic-6")
        self.assertEqual(result.classification_source, "llm_fallback_verified")
        self.assertTrue(result.llm_fallback_used)
        self.assertEqual(result.llm_fallback_proposals[0]["leaf_id"], 6)

    def test_independent_verifier_can_modify_or_reject_fallback(self):
        article = self.fallback_state()
        for ids in [[2], []]:
            with self.subTest(ids=ids):
                result, llm = self.verify(article, response(ids))
                llm.assert_called_once()
                self.assertEqual(result.can_classify, bool(ids))
                if ids:
                    self.assertEqual(result.sub_category, "topic-2")

    def test_fallback_rejection_is_independently_rechecked(self):
        article = self.fallback_state(response([]))
        result, llm = self.verify(article, response([6]))
        self.assertTrue(result.can_classify)
        self.assertIn("[ID6]", llm.call_args.args[2]["catalog"])

    def test_fallback_invalid_output_and_api_errors_raise_not_silent_ai(self):
        for output in [{}, response([99]), response([1, 2])]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.fallback_state(output)
        with patch(f"{FALLBACK_MODULE}._load_json", side_effect=load_catalog), \
             patch(f"{FALLBACK_MODULE}.invoke_llm", side_effect=RuntimeError("API unavailable")):
            with self.assertRaises(RuntimeError):
                category_llm_fallback_node(state(score=0.6), FALLBACK_CONFIG, RUNTIME)
        article = self.fallback_state()
        with self.assertRaises(ValueError):
            self.verify(article, response([99]))
        with patch(f"{VERIFY_MODULE}._load_json", side_effect=load_catalog), \
             patch(f"{VERIFY_MODULE}.invoke_llm", side_effect=RuntimeError("API unavailable")):
            with self.assertRaises(RuntimeError):
                category_llm_verify_node(article, CONFIG, RUNTIME)

    def test_high_similarity_fallback_node_does_not_call_llm(self):
        with patch(f"{FALLBACK_MODULE}.invoke_llm") as llm:
            result = category_llm_fallback_node(state(), FALLBACK_CONFIG, RUNTIME)
        llm.assert_not_called()
        self.assertFalse(result.llm_fallback_used)

    def test_high_similarity_small_sample_is_reviewed_not_vetoed(self):
        article = state(support=2)
        article.model_review_reasons = ["low_support"]
        result, llm = self.verify(article, response([1]))
        llm.assert_called_once()
        self.assertTrue(result.can_classify)

    def test_semantic_rejection_is_honored_even_with_high_similarity(self):
        result, llm = self.verify(state(), response([]))
        llm.assert_called_once()
        self.assertFalse(result.can_classify)
        self.assertEqual(result.classification_mode, "none")

    def test_invalid_json_or_outside_top5_is_error_not_silent_ai_fallback(self):
        for output in [{}, response([6]), {"can_classify": "false"}, response([1, 2, 3])]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.verify(state(), output)

    def test_missing_quadrant_is_configuration_error(self):
        with patch.dict(QUADRANTS, {"1": {"leaf_topic_name": "wrong", "region": "A"}}):
            with self.assertRaises(ValueError):
                self.verify(state(), response([1]))

    def test_dual_candidates_require_independent_evidence(self):
        output = response([1, 2])
        del output["accepted_candidates"][1]["reason"]
        with self.assertRaises(ValueError):
            self.verify(state(), output)

    def test_positive_flag_with_empty_candidates_is_repaired_by_verifier(self):
        inconsistent = {"can_classify": True, "accepted_candidates": [],
                        "classification_reason": "none of the candidates fit"}
        with patch(f"{VERIFY_MODULE}._load_json", side_effect=load_catalog), \
             patch(f"{VERIFY_MODULE}.invoke_llm", side_effect=[json.dumps(inconsistent), json.dumps(response([]))]) as llm:
            result = category_llm_verify_node(state(), CONFIG, RUNTIME)
        self.assertFalse(result.can_classify)
        self.assertEqual(llm.call_count, 2)

    def test_full_graph_keeps_two_independent_word_sets_and_csv_one_row(self):
        self.assert_full_graph()

    def test_csv_source_labels_and_resume_migration_do_not_change_words(self):
        from batch_run import _summary_row, _upgrade_provenance_row
        for source, name in [("minilm_verified", "实际分类器分类"),
                             ("llm_fallback_verified", "LLM兜底分类")]:
            final = {"category_quadrant": {"sub_category": "topic", "classification_source": source}}
            row = _summary_row({}, {"final_summary": final}, 1)
            self.assertEqual(row["分类来源"], name)
            self.assertNotIn("分类路径", row)
            old = {"分类来源": source, "最终二级类目": "topic", "分类象限推荐词": "word1｜word2",
                   "分类象限首选理由": "old reason", "专题逐词排序理由": "[]", "归类校验理由": "old classification reason",
                   "模型首二差距": 0.01, "模型主题训练样本量": 27, "模型需复核原因": "low_support",
                   "归类候选及各自推荐词": json.dumps([{"best_word": "word1", "recommended_words": ["word1", "word2"],
                       "ranking_reasons": [], "best_word_reason": "old reason", "generation_reason": "old strategy", "classification_reason": "old classification reason"}])}
            migrated = _upgrade_provenance_row(old)
            self.assertEqual(migrated["分类来源"], name)
            self.assertNotIn("分类路径", migrated)
            self.assertNotIn("分类象限推荐词", migrated)
            self.assertNotIn("分类象限首选理由", migrated)
            self.assertNotIn("专题逐词排序理由", migrated)
            self.assertNotIn("归类校验理由", migrated)
            self.assertNotIn("模型首二差距", migrated)
            self.assertNotIn("模型主题训练样本量", migrated)
            self.assertNotIn("模型需复核原因", migrated)
            self.assertNotIn("归类候选及各自推荐词", migrated)
            candidate = json.loads(migrated["分类生词结果"])[0]
            self.assertEqual(candidate["recommended_words"], ["word1", "word2"])
            for field in ("ranking_reasons", "best_word_reason", "generation_reason", "classification_reason", "model_cosine_gap", "model_training_support", "model_review_reasons"):
                self.assertNotIn(field, candidate)
            self.assertEqual(_upgrade_provenance_row(migrated)["分类来源"], name)
            old["最终二级类目"] = ""
            self.assertEqual(_upgrade_provenance_row(old)["分类来源"], "未归类")

    def test_resume_restores_excel_scientific_id_but_rejects_wrong_articles(self):
        from batch_run import _validate_resume_source
        source = {"资讯标题": "article", "资讯ID": "202610273908970854"}
        row = {"序号": "1", "资讯标题": "article", "资讯ID": "2.0261E+17", "分类象限推荐词": "word"}
        restored = _validate_resume_source(row, source)
        self.assertEqual(restored["资讯ID"], source["资讯ID"])
        self.assertEqual(restored["分类象限推荐词"], "word")
        for wrong in [{**row, "资讯标题": "different"}, {**row, "资讯ID": "2.0262E+17"},
                      {**row, "资讯ID": "202610273908970855"}]:
            with self.assertRaises(ValueError):
                _validate_resume_source(wrong, source)

    def test_full_graph_low_similarity_calls_two_stages_before_quadrant_generation(self):
        self.assert_full_graph(score=0.60)

    def test_final_regions_cover_single_dual_same_region_and_unclassified(self):
        from graphs.nodes.final_summary_node import final_summary_node
        from graphs.state import FinalSummaryInput, GraphOutput
        from batch_run import _summary_row, _old_final_summary, _compact_export_row
        cases = [("A", ["A"], "A"), ("A", ["A", "B"], "A/B"),
                 ("C", ["C", "C"], "C/C"), ("", [], "none")]
        for primary, regions, expected in cases:
            with self.subTest(regions=regions):
                candidates = [{"region": r, "quadrant": f"label-{r}", "is_primary": i == 0,
                               "ai_queries": [{"word": f"word-{i}", "query": f"What changed for category {i}?"}],
                               "recommended_words": [f"word-{i}"], "best_word": f"word-{i}"}
                              for i, r in enumerate(regions)]
                article = FinalSummaryInput(region=primary, quadrant=f"label-{primary}",
                    sub_category="topic" if regions else "", word_source=f"quadrant_{primary}" if regions else "ai",
                    recommended_words=["word-0"], category_candidate_results=candidates,
                    ai_queries=[] if regions else [{"word": "word-0", "query": "What changed?"}])
                final_output = final_summary_node(article, {}, RUNTIME)
                category = final_output.final_summary["category_quadrant"]
                self.assertEqual(category["region"], expected)
                expected_label = ("双归类：" if len(regions) == 2 else "") + "；".join(f"label-{r}" for r in regions)
                self.assertEqual(category["quadrant"], expected_label if regions else "未归类（AI生词）")
                output = GraphOutput.model_validate(final_output.model_dump()).model_dump()
                self.assertNotIn("classification_mode", output)
                self.assertNotIn("classification_mode", category)
                self.assertNotIn("classification_source_label", category)
                self.assertEqual(category["classification_source"], "实际分类器分类" if regions else "未归类")
                self.assertEqual(list(output), ["final_summary"])
                self.assertEqual(len(category["generation_results"]), max(1, len(regions)))
                for item in category["generation_results"]:
                    for field in ("is_primary", "best_word", "region", "quadrant"):
                        self.assertNotIn(field, item)
                row = _summary_row({}, {"final_summary": final_output.final_summary}, 1)
                self.assertNotIn("归类模式", row)
                restored = _old_final_summary(row)["category_quadrant"]
                self.assertEqual(restored["classification_source"], category["classification_source"])
                self.assertEqual(restored["region"], expected)
                self.assertEqual(restored["quadrant"], category["quadrant"])
                self.assertEqual(restored["generation_results"], category["generation_results"])
                if not regions:
                    self.assertEqual(category["generation_results"][0]["ai_queries"], [{"word": "word-0", "query": "What changed?"}])
                migrated = _compact_export_row({**row, "归类模式": "old"})
                self.assertNotIn("归类模式", migrated)
                self.assertEqual(migrated["象限"], expected)

    def assert_full_graph(self, score=0.73):
        from graphs.graph import main_graph
        from batch_run import _summary_row
        generated_regions = []
        classification_calls = []

        def llm(ctx, cfg, values):
            name = Path(cfg).name
            if name.startswith("category_"):
                classification_calls.append(name)
            if name in {"category_fallback_llm_cfg.json", "category_fallback_verify_llm_cfg.json"}:
                return json.dumps(response([6]))
            if name == "topic_fit_judge_llm_cfg.json":
                return json.dumps({"has_fit_topic": True, "fit_topic_title": "test-topic",
                                   "fit_topic_url": "https://example.test/topic"})
            if name == "hot_theme_fit_judge_llm_cfg.json":
                return json.dumps({"has_fit_hot_theme": True, "fit_hot_theme_name": "test-theme",
                                   "fit_hot_theme_code": 1, "fit_hot_theme_url": "https://example.test/theme"})
            if name == "category_classify_llm_cfg.json":
                return json.dumps(response([1, 2]))
            if name == "priority_llm_cfg.json":
                return json.dumps({"out_priority": "topic", "out_priority_reason": "topic evidence"})
            if name == "hot_topic_select_llm_cfg.json":
                return json.dumps({"has_hot_topic": False, "hot_leaf_id": -1})
            if name.startswith("quadrant_"):
                generated_regions.append(values["region"])
            anchor = values.get("sub_category") or "extra"
            words = [f"{anchor}-word-{i}" for i in range(3)]
            return json.dumps({"recommended_words": words,
                               "ai_queries": [{"word": w, "query": f"What changed for {anchor}?"} for w in words],
                               "display_variants": [{"word": w, "display_6": w[:6], "display_9": w[:9]} for w in words]})

        article = state(score=score)
        model_result = {
            "leaf_id": 1, "leaf_name": "topic-1", "parent_name": "parent-1",
            "similarity": article.model_similarity, "second_best_similarity": article.model_second_similarity,
            "cosine_gap": article.model_cosine_gap, "training_support": 10, "is_noise": True,
            "review_reasons": ["close_candidates"],
            "nearby_topics": [nb.model_dump() for nb in article.nearby_topics],
        }
        modules = ["topic_fit_judge", "hot_theme_fit_judge", "topic_word_generate",
                   "hot_theme_word_generate", "priority_judge", "category_llm_verify", "category_llm_fallback",
                   "hot_topic_select", "quadrant_a", "quadrant_b", "quadrant_c", "quadrant_d", "ai_generate"]
        with ExitStack() as stack:
            stack.enter_context(patch(f"{VERIFY_MODULE}._load_json", side_effect=load_catalog))
            stack.enter_context(patch(f"{FALLBACK_MODULE}._load_json", side_effect=load_catalog))
            stack.enter_context(patch("graphs.nodes.category_model_classify_node.classify_article", return_value=model_result))
            stack.enter_context(patch("graphs.nodes.topic_recall_node._load_topic_map", return_value={}))
            stack.enter_context(patch("graphs.nodes.hot_theme_recall_node._load_theme_rows", return_value=[]))
            for module in modules:
                stack.enter_context(patch(f"graphs.nodes.{module}_node.invoke_llm", side_effect=llm))
            output = asyncio.run(main_graph.ainvoke({"title": "article", "content": "body"}, context=Context()))
        final = output["final_summary"]
        category = final["category_quadrant"]
        if score < 0.6542854905128479:
            self.assertEqual(classification_calls, ["category_fallback_llm_cfg.json", "category_fallback_verify_llm_cfg.json"])
            self.assertEqual(generated_regions, ["B"])
            self.assertEqual(category["generation_results"][0]["leaf_id"], 6)
            self.assertEqual(category["classification_source"], "LLM兜底分类")
            self.assertEqual(list(output), ["final_summary"])
            self.assertEqual(category["region"], "B")
            self.assertNotIn("classification_mode", category)
            row = _summary_row({"资讯标题": "article"}, {"final_summary": final}, 1)
            self.assertTrue(row["LLM兜底分类"])
            self.assertEqual(row["分类来源"], "LLM兜底分类")
            self.assertEqual(len(json.loads(row["分类生词结果"])), 1)
            return
        self.assertNotIn("classification_mode", category)
        self.assertNotIn("classification_mode", output)
        self.assertEqual(category["classification_source"], "实际分类器分类")
        self.assertEqual(list(output), ["final_summary"])
        self.assertNotIn("classification_source_label", category)
        self.assertEqual(category["region"], "A/B")
        self.assertEqual(category["quadrant"], "双归类：象限A·test；象限B·test")
        candidates = category["generation_results"]
        self.assertEqual([c["leaf_id"] for c in candidates], [1, 2])
        self.assertEqual(generated_regions, ["A", "B"])
        self.assertNotEqual(candidates[0]["recommended_words"], candidates[1]["recommended_words"])
        for field in ("category", "sub_category", "word_source", "recommended_words", "best_word", "classification_candidates"):
            self.assertNotIn(field, category)
        for candidate in candidates:
            self.assertEqual([item["word"] for item in candidate["ai_queries"]], candidate["recommended_words"])
            self.assertTrue(all(item["query"] for item in candidate["ai_queries"]))
            for field in ("is_primary", "best_word", "region", "quadrant"):
                self.assertNotIn(field, candidate)
        for result in [output, category, *candidates]:
            for field in ("ranking_reasons", "best_word_reason", "generation_reason", "classification_reason", "model_cosine_gap", "model_training_support", "model_review_reasons"):
                self.assertNotIn(field, result)
        self.assertTrue(final["topic_related"]["matched"])
        self.assertTrue(final["hot_theme_related"]["matched"])
        for direction in ("topic_related", "hot_theme_related"):
            for result in (final[direction],):
                self.assertNotIn("ranking_reasons", result)
                self.assertNotIn("best_word_reason", result)
                self.assertEqual(result["recommended_words"], [f"extra-word-{i}" for i in range(3)])
                self.assertEqual(result["best_word"], result["recommended_words"][0])
        row = _summary_row({"资讯标题": "article"}, {"final_summary": final}, 1)
        self.assertEqual(row["分类来源"], "实际分类器分类")
        self.assertEqual(len(json.loads(row["分类生词结果"])), 2)


if __name__ == "__main__":
    unittest.main()
