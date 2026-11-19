"""Queries and user-facing summaries travel with their ranked recommendation."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from tools.word_extras import ai_queries, invoke_words_with_queries, recommendation_details
from tools.runtime import Context
from batch_run import _validate_generation_output


def payload():
    word = '华为云具身智能开源'
    return {'recommended_words': [word],
            'short_words': [{'word': word, 'short': '华为云具身智能', 'subject': '华为云'}],
            'ai_queries': [{'word': word, 'query': '华为云开源具身世界模型，释放了什么信号？',
                            'query_summary': '华为云具身智能模型开源影响'}]}


class QuerySummaryTests(unittest.TestCase):
    def test_user_example_is_consolidated_without_truncating_summary(self):
        data = payload()
        rows = recommendation_details(data)
        self.assertEqual(rows, [{
            'word': '华为云具身智能开源', 'short': '华为云具身智能',
            'query': '华为云开源具身世界模型，释放了什么信号？',
            'query_summary': '华为云具身智能模型开源影响'}])
        self.assertGreater(len(rows[0]['query_summary']), 12)
        _validate_generation_output({'generation_results': [{'word_source': 'ai', 'recommended_words': rows}]})

    def test_missing_blank_question_and_multiline_summary_fail(self):
        for summary in [None, '', '  ', '华为云释放什么信号？', '华为云\n开源影响']:
            data = payload()
            data['ai_queries'][0]['query_summary'] = summary
            with self.subTest(summary=summary), self.assertRaises(ValueError):
                ai_queries(data, data['recommended_words'])

    def test_summary_without_subject_is_repaired(self):
        valid = payload()
        bad = payload()
        bad['ai_queries'][0]['query_summary'] = '具身智能模型开源影响'
        calls = []
        def invoke(ctx, cfg, values):
            calls.append(values)
            return json.dumps(bad if len(calls) == 1 else valid)
        data, words, queries = invoke_words_with_queries(invoke, Context(), 'unused', {})
        self.assertEqual(queries[0]['query_summary'], valid['ai_queries'][0]['query_summary'])
        self.assertIn('华为云', calls[1]['repair_note'])
        self.assertEqual(len(calls), 2)

    def test_every_exported_word_requires_all_four_fields(self):
        row = recommendation_details(payload())[0]
        for field in ['word', 'short', 'query', 'query_summary']:
            invalid = {k: v for k, v in row.items() if k != field}
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_generation_output({'generation_results': [{'word_source': 'quadrant_A', 'recommended_words': [invalid]}]})

    def test_old_result_migration_does_not_invent_summary(self):
        old = payload()
        old['ai_queries'][0].pop('query_summary')
        rows = recommendation_details(old)
        self.assertEqual(rows[0]['query_summary'], '')
        with self.assertRaises(ValueError):
            _validate_generation_output({'generation_results': [{'word_source': 'quadrant_A', 'recommended_words': rows}]})

    def test_order_survives_already_consolidated_export(self):
        rows = recommendation_details(payload())
        rows += [{'word': '英伟达CEO遴选标准', 'short': '英伟达CEO遴选',
                  'query': '英伟达CEO遴选需要考虑哪些标准？', 'query_summary': '英伟达CEO遴选评价标准'}]
        self.assertEqual(recommendation_details({'recommended_words': rows}), rows)


if __name__ == '__main__':
    unittest.main()
