"""Every ranked word keeps an aligned subject-preserving short form."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from tools.word_extras import short_words, invoke_words_with_queries
from tools.runtime import Context
from graphs.nodes.final_summary_node import final_summary_node
from graphs.state import FinalSummaryInput
from batch_run import _summary_row, _compact_export_row, _validate_generation_output

WORDS = ['英伟达CEO遴选标准', '英伟达算力需求持续性', '华为昇腾生态验证指标']
SHORTS = [{'word': w, 'short': s, 'subject': subject} for w, s, subject in zip(
    WORDS, ['英伟达CEO遴选', '英伟达算力持续性', '华为昇腾生态验证'], ['英伟达', '英伟达', '华为昇腾'])]
QUERIES = [{'word': w, 'query': f'{w}如何评估？', "query_summary": w + "验证要点"} for w in WORDS]


class AllWordShortTests(unittest.TestCase):
    def test_all_three_shorts_are_aligned_and_subject_metadata_is_hidden(self):
        result = short_words({'short_words': SHORTS}, WORDS, require_subject=True)
        self.assertEqual([v['word'] for v in result], WORDS)
        self.assertTrue(all(set(v) == {'word', 'short'} and len(v['short']) <= 9 for v in result))

    def test_missing_reordered_and_blank_second_short_fail(self):
        for variants in [SHORTS[:1], SHORTS[::-1], [SHORTS[0], {**SHORTS[1], 'short': ''}, SHORTS[2]]]:
            with self.subTest(variants=variants), self.assertRaises(ValueError):
                short_words({'short_words': variants}, WORDS)
        with self.assertRaises(ValueError):
            short_words({'best_word_short': '英伟达CEO遴选'}, WORDS)

    def test_second_word_missing_subject_requests_repair(self):
        valid = {'recommended_words': WORDS, 'short_words': SHORTS, 'ai_queries': QUERIES}
        invalid = {**valid, 'short_words': [SHORTS[0], {**SHORTS[1], 'short': '算力持续性'}, SHORTS[2]]}
        calls = []
        def invoke(ctx, cfg, values):
            calls.append(values)
            return json.dumps(invalid if len(calls) == 1 else valid)
        data, words, queries = invoke_words_with_queries(invoke, Context(), 'unused', {})
        self.assertEqual(len(data['short_words']), 3)
        self.assertEqual(words, WORDS)
        self.assertEqual(queries, QUERIES)
        self.assertIn('必须保留原词主体', calls[1]['repair_note'])
        self.assertEqual(len(calls), 2)

    def test_final_output_and_csv_have_no_duplicate_best_word(self):
        variants = short_words({'short_words': SHORTS}, WORDS)
        final = final_summary_node(FinalSummaryInput(
            recommended_words=WORDS, short_words=variants, ai_queries=QUERIES, word_source='ai',
            has_fit_topic=True, fit_topic_title='专题', topic_hit_result={
                'word': '英伟达财报超预期', 'short': '英伟达财报超预期'},
            hot_theme_hit_result={'word': '', 'short': ''}), {}, SimpleNamespace(context=Context())).final_summary
        self.assertNotIn('best_word', final['topic_related'])
        self.assertNotIn('best_word', final['hot_theme_related'])
        group = final['category_quadrant']['generation_results'][0]
        self.assertNotIn('best_word_short', group)
        self.assertNotIn('best_word', group)
        self.assertEqual([{k: v[k] for k in ('word', 'short')} for v in group['recommended_words']], variants)
        self.assertNotIn('short_words', group)
        self.assertNotIn('ai_queries', group)
        _validate_generation_output(final['category_quadrant'])
        row = _summary_row({}, {'final_summary': final}, 1)
        self.assertNotIn('专题首选词', row)
        self.assertNotIn('热点首选词', row)
        migrated = _compact_export_row({**row, '专题首选词': 'old', '热点首选词': 'old'})
        self.assertNotIn('专题首选词', migrated)
        self.assertNotIn('热点首选词', migrated)

    def test_empty_shorts_and_prompt_examples(self):
        self.assertEqual(short_words({'short_words': []}, [], require_subject=True), [])
        with self.assertRaises(ValueError):
            short_words({}, [])
        for name in ['quadrant_a', 'quadrant_b', 'quadrant_c', 'quadrant_d', 'ai_generate']:
            cfg = json.loads((ROOT / f'config/{name}_llm_cfg.json').read_text(encoding='utf-8'))
            sample = cfg['sp'].split('# 输出格式', 1)[1]
            example, _ = json.JSONDecoder().raw_decode(sample[sample.index('{'):])
            self.assertNotIn('best_word', example)
            self.assertNotIn('best_word_short', example)
            short_words(example, example['recommended_words'], require_subject=True)


if __name__ == '__main__':
    unittest.main()
