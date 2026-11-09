"""Offline regression checks for hot-theme recall and jump-fit judgment."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphs.nodes.hot_theme_fit_judge_node import hot_theme_fit_judge_node
from graphs.nodes.hot_theme_recall_node import hot_theme_recall_node
from graphs.state import HotThemeCandidate, HotThemeFitJudgeInput, HotThemeRecallInput
from tools.local_retrieval import search
from tools.runtime import Context

RECALL_MODULE = "graphs.nodes.hot_theme_recall_node"
JUDGE_MODULE = "graphs.nodes.hot_theme_fit_judge_node"
CONFIG = {"metadata": {"llm_cfg": "config/hot_theme_fit_judge_llm_cfg.json"}}
RUNTIME = SimpleNamespace(context=Context())


class HotThemePolicyTests(unittest.TestCase):
    def test_recall_caps_at_five_and_keeps_candidate_metadata(self):
        rows = [{"code": i, "name": f"news-{i}", "interpretation": "news market",
                 "url": f"https://example.test/theme/{i}"} for i in range(10)]
        with patch(f"{RECALL_MODULE}._load_theme_rows", return_value=rows), \
             patch(f"{RECALL_MODULE}.search", wraps=search) as retrieval:
            output = hot_theme_recall_node(
                HotThemeRecallInput(title="news market", content="news market"), {}, RUNTIME,
            )
        self.assertEqual(retrieval.call_args.kwargs["top_k"], 5)
        self.assertEqual(len(output.hot_theme_candidates), 5)
        scores = [item.score for item in output.hot_theme_candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for item in output.hot_theme_candidates:
            row = rows[item.theme_code]
            self.assertEqual(item.theme_name, row["name"])
            self.assertEqual(item.interpretation, row["interpretation"])
            self.assertEqual(item.url, row["url"])

    def test_recall_still_filters_below_existing_threshold(self):
        rows = [{"code": i, "name": f"theme-{i}", "interpretation": "news",
                 "url": "https://example.test/theme"} for i in range(3)]
        with patch(f"{RECALL_MODULE}._load_theme_rows", return_value=rows), \
             patch(f"{RECALL_MODULE}.search", return_value=[(0, 0.8), (1, 0.05), (2, 0.049)]):
            output = hot_theme_recall_node(HotThemeRecallInput(title="news", content="body"), {}, RUNTIME)
        self.assertEqual([item.theme_code for item in output.hot_theme_candidates], [0, 1])

    def test_empty_recall_returns_no_candidates(self):
        with patch(f"{RECALL_MODULE}._load_theme_rows", return_value=[]):
            output = hot_theme_recall_node(HotThemeRecallInput(title="news", content="body"), {}, RUNTIME)
        self.assertEqual(output.hot_theme_candidates, [])

    def test_jump_fit_prompt_allows_rejection_and_requires_real_candidate(self):
        config = json.loads((ROOT / "config/hot_theme_fit_judge_llm_cfg.json").read_text(encoding="utf-8"))
        prompt = config["sp"]
        self.assertIn("\u6700\u591a5\u4e2a", prompt)
        self.assertIn("\u5148\u5224\u65ad\u8df3\u8f6c\u662f\u5426\u9002\u5408", prompt)
        self.assertIn("\u5f31\u76f8\u5173", prompt)
        self.assertIn("\u5019\u9009\u5217\u8868\u4e3a\u7a7a", prompt)
        self.assertIn("\u540c\u4e00\u4e2a\u5019\u9009", prompt)
        self.assertIn("\u662f\u5426\u9002\u5408\u8df3\u8f6c\u5230\u53ec\u56de\u7684\u70ed\u70b9\u9898\u6750", config["up"])

    def test_nonfit_clears_selected_fields_and_preserves_output_contract(self):
        response = {"has_fit_hot_theme": False, "fit_hot_theme_code": 9,
                    "fit_hot_theme_name": "ignored", "fit_hot_theme_url": "ignored"}
        with patch(f"{JUDGE_MODULE}.invoke_llm", return_value=json.dumps(response)) as llm:
            output = hot_theme_fit_judge_node(
                HotThemeFitJudgeInput(title="news", content="body", hot_theme_candidates=[]), CONFIG, RUNTIME,
            )
        self.assertFalse(output.has_fit_hot_theme)
        self.assertEqual(output.fit_hot_theme_code, -1)
        self.assertEqual(output.fit_hot_theme_name, "")
        self.assertEqual(output.fit_hot_theme_url, "")
        self.assertEqual(llm.call_count, 1)

    def test_fit_keeps_candidate_name_url_backfill(self):
        candidate = HotThemeCandidate(theme_code=9, theme_name="theme-9",
                                      interpretation="news", url="https://example.test/theme/9")
        with patch(f"{JUDGE_MODULE}.invoke_llm", return_value=json.dumps(
            {"has_fit_hot_theme": True, "fit_hot_theme_code": 9},
        )):
            output = hot_theme_fit_judge_node(
                HotThemeFitJudgeInput(title="news", content="body", hot_theme_candidates=[candidate]),
                CONFIG, RUNTIME,
            )
        self.assertTrue(output.has_fit_hot_theme)
        self.assertEqual(output.fit_hot_theme_code, candidate.theme_code)
        self.assertEqual(output.fit_hot_theme_name, candidate.theme_name)
        self.assertEqual(output.fit_hot_theme_url, candidate.url)


if __name__ == "__main__":
    unittest.main()
