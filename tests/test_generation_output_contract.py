"""Generation output never contains per-word ranking reasons."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphs.nodes.topic_word_generate_node import topic_word_generate_node
from graphs.nodes.hot_theme_word_generate_node import hot_theme_word_generate_node
from graphs.state import (GlobalState, FinalSummaryInput, TopicWordGenerateInput,
                          HotThemeWordGenerateInput)
from tools.llm_utils import extract_json, parse_word_result
from tools.runtime import Context


class GenerationOutputContractTests(unittest.TestCase):
    def test_topic_and_hot_theme_outputs_keep_order_and_display_without_reasons(self):
        words = ["robots", "orders", "parts"]
        payload = {"recommended_words": words, "display_variants": [
            {"word": word, "display_6": word, "display_9": word} for word in words]}
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
                    self.assertEqual(output["recommended_words"], words)
                    self.assertEqual(output[result_key]["display_variants"], payload["display_variants"])
                    self.assertEqual(output[result_key]["best_word"], words[0])

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
