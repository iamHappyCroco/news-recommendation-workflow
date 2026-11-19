"""All LLM calls use Eastmoney, independently of retired provider settings."""
import json
import os
from pathlib import Path
import sys
from unittest import TestCase
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.llm_utils import invoke_llm, generation_retry_feedback
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
                self.assertEqual(payload["temperature"], 0.5)
                self.assertEqual(payload["top_p"], 1.0)
                self.assertEqual(result, '{"recommended_words": []}')

    def test_missing_em_key_fails_without_other_provider_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "other", "LOCAL_LLM_MODE": "mock"}, clear=True), \
             patch("tools.llm_utils.request.urlopen") as http:
            with self.assertRaisesRegex(RuntimeError, "EM_API_KEY"):
                invoke_llm(Context(), CFG)
            http.assert_not_called()

    def test_validation_feedback_only_reaches_generation_and_is_cleared(self):
        with patch.dict(os.environ, {"EM_API_KEY": "test-em-key"}, clear=True), \
             patch("tools.llm_utils.request.urlopen", return_value=self.response()) as http:
            with generation_retry_feedback("invalid subject marker"):
                invoke_llm(Context(), CFG, {"title": "robots", "content": "body"})
                invoke_llm(Context(), str(ROOT / "config/category_classify_llm_cfg.json"), {})
            invoke_llm(Context(), CFG, {"title": "robots", "content": "body"})
            prompts = [json.loads(call.args[0].data)["messages"][1]["content"] for call in http.call_args_list]
            self.assertIn("invalid subject marker", prompts[0])
            self.assertNotIn("invalid subject marker", prompts[1])
            self.assertNotIn("invalid subject marker", prompts[2])

    def test_concurrent_articles_keep_feedback_in_their_own_threads(self):
        import asyncio
        requests = []
        def respond(req, **kwargs):
            requests.append(json.loads(req.data))
            return self.response()
        async def article(marker):
            with generation_retry_feedback(marker):
                await asyncio.to_thread(invoke_llm, Context(), CFG, {"title": marker, "content": "body"})
        async def run():
            await asyncio.gather(article("article-one-marker"), article("article-two-marker"))
        with patch.dict(os.environ, {"EM_API_KEY": "test-em-key"}, clear=True), \
             patch("tools.llm_utils.request.urlopen", side_effect=respond):
            asyncio.run(run())
        self.assertEqual(len(requests), 2)
        for payload in requests:
            prompt = payload["messages"][1]["content"]
            self.assertNotEqual("article-one-marker" in prompt, "article-two-marker" in prompt)

    def test_em_configuration_can_override_model_and_base_url(self):
        with patch.dict(os.environ, {
            "EM_API_KEY": "test-em-key", "EM_MODEL": "configured-em-model",
            "EM_API_BASE_URL": "https://dd-ai-api.eastmoney.com/coding/v1/"}, clear=True), \
             patch("tools.llm_utils.request.urlopen", return_value=self.response()) as http:
            invoke_llm(Context(), CFG)
            req = http.call_args.args[0]
            self.assertEqual(json.loads(req.data)["model"], "configured-em-model")
            self.assertEqual(req.full_url, "https://dd-ai-api.eastmoney.com/coding/v1/chat/completions")

    def test_all_generation_repairs_add_point_two_to_temperature(self):
        from importlib import import_module
        from types import SimpleNamespace
        from graphs.state import QuadrantInput, AiGenerateInput, TopicWordGenerateInput, HotThemeWordGenerateInput

        cases = [(f'quadrant_{r}', QuadrantInput(title='news', content='body', region=r.upper()), 0.5, 1800) for r in 'abcd']
        cases += [('ai_generate', AiGenerateInput(title='news', content='body'), 0.5, 1800),
                  ('topic_word_generate', TopicWordGenerateInput(title='news', content='body', fit_topic_title='topic'), 0.4, 800),
                  ('hot_theme_word_generate', HotThemeWordGenerateInput(title='news', content='body'), 0.4, 800)]
        word = '算力需求验证'
        for name, state, temperature, tokens in cases:
            jump = name in {'topic_word_generate', 'hot_theme_word_generate'}
            data = {'word': word, 'short': word, 'subject': '算力', 'link_type': '主线'} if jump else {
                'recommended_words': [word], 'short_words': [{'word': word, 'short': word, 'subject': '算力'}],
                'ai_queries': [{'word': word, 'query': '如何验证算力需求？', 'query_summary': '算力需求验证方法'}]}
            responses = []
            for output in ({}, data):
                response = MagicMock()
                response.__enter__.return_value.read.return_value = json.dumps({
                    'choices': [{'message': {'content': json.dumps(output)}}]}).encode()
                responses.append(response)
            with self.subTest(name=name), patch.dict(os.environ, {'EM_API_KEY': 'test-em-key'}, clear=True), \
                 patch('tools.llm_utils.request.urlopen', side_effect=responses) as http:
                node = getattr(import_module(f'graphs.nodes.{name}_node'), f'{name}_node')
                node(state, {'metadata': {'llm_cfg': f'config/{name}_llm_cfg.json'}}, SimpleNamespace(context=Context()))
                payloads = [json.loads(call.args[0].data) for call in http.call_args_list]
                self.assertEqual([p['temperature'] for p in payloads], [temperature, round(temperature + 0.2, 2)])
                self.assertEqual([p['max_tokens'] for p in payloads], [tokens, tokens])
                self.assertEqual([p['top_p'] for p in payloads], [1.0, 1.0])
                self.assertTrue(all('_temperature_offset' not in json.dumps(p['messages']) for p in payloads))
