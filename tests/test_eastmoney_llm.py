"""All LLM calls use Eastmoney, independently of retired provider settings."""
import json
import os
from pathlib import Path
import sys
from unittest import TestCase
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.llm_utils import invoke_llm
from tools.runtime import Context

CFG = str(ROOT / "config/quadrant_c_llm_cfg.json")


class EastmoneyLLMTests(TestCase):
    def response(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": '{"recommended_words": []}'}}]}
        ).encode()
        return response

    def test_eastmoney_request_ignores_retired_provider_and_local_settings(self):
        for mode in ("local", "mock", "transformers", "openai"):
            with self.subTest(mode=mode), patch.dict(os.environ, {
                "EM_API_KEY": "test-em-key", "LOCAL_LLM_MODE": mode,
                "OPENAI_API_KEY": "test-other-key", "OPENAI_BASE_URL": "https://example.invalid/v1",
                "OPENAI_MODEL": "retired-model"}, clear=True), \
                 patch("tools.llm_utils.request.urlopen", return_value=self.response()) as http:
                result = invoke_llm(Context(), CFG, {"title": "robots", "content": "body"})
                req = http.call_args.args[0]
                payload = json.loads(req.data)
                self.assertEqual(req.full_url, "https://dd-ai-api.eastmoney.com/coding/v1/chat/completions")
                self.assertEqual(req.get_header("Authorization"), "Bearer test-em-key")
                self.assertEqual(payload["model"], "DeepSeek-V4-Flash-0731")
                self.assertEqual(payload["max_tokens"], 1800)
                self.assertEqual(result, '{"recommended_words": []}')

    def test_missing_em_key_fails_without_other_provider_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "other", "LOCAL_LLM_MODE": "mock"}, clear=True), \
             patch("tools.llm_utils.request.urlopen") as http:
            with self.assertRaisesRegex(RuntimeError, "EM_API_KEY"):
                invoke_llm(Context(), CFG)
            http.assert_not_called()

    def test_em_configuration_can_override_model_and_base_url(self):
        with patch.dict(os.environ, {
            "EM_API_KEY": "test-em-key", "EM_MODEL": "configured-em-model",
            "EM_API_BASE_URL": "https://dd-ai-api.eastmoney.com/coding/v1/"}, clear=True), \
             patch("tools.llm_utils.request.urlopen", return_value=self.response()) as http:
            invoke_llm(Context(), CFG)
            req = http.call_args.args[0]
            self.assertEqual(json.loads(req.data)["model"], "configured-em-model")
            self.assertEqual(req.full_url, "https://dd-ai-api.eastmoney.com/coding/v1/chat/completions")
