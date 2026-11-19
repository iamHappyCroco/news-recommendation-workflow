"""Single jump words survive routing, final output and CSV without legacy arrays."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from graphs.nodes.topic_word_generate_node import topic_word_generate_node
from graphs.nodes.hot_theme_word_generate_node import hot_theme_word_generate_node
from graphs.nodes.priority_judge_node import priority_judge_node
from graphs.nodes.final_summary_node import final_summary_node
from graphs.state import TopicWordGenerateInput, HotThemeWordGenerateInput, PriorityDecisionInput, FinalSummaryInput
from tools.runtime import Context
from tools.word_extras import validate_jump_word
from batch_run import _summary_row, _old_final_summary, _validate_final_summary

RUNTIME = SimpleNamespace(context=Context())
VALID = {'word': 'A股三季报盈利全景', 'short': 'A股三季报盈利全景', 'subject': 'A股', 'article_link': '资讯讲A股三季报，专题提供全市场盈利全景'}
EMPTY = {key: '' for key in VALID}

class JumpSingleWordTests(unittest.TestCase):
    def test_nine_character_shorts_keep_parallel_subjects_and_allow_ten_character_words(self):
        parallel = {'word': '华为 小米智驾方案对比', 'short': '华为 小米智驾对比', 'subject': '华为 小米', 'article_link': '资讯介绍两家智驾方案，页面提供两家路线比较'}
        self.assertEqual(len(parallel['short']), 9)
        self.assertEqual(validate_jump_word(parallel, require_metadata=True), (parallel['word'], parallel['short']))
        with self.assertRaises(ValueError):
            validate_jump_word({**parallel, 'short': '小米 华为智驾对比'}, require_metadata=True)
        ten = {'word': '算力硬件题材驱动逻辑', 'short': '算力硬件驱动逻辑', 'subject': '算力硬件', 'article_link': '资讯讲算力扩张，题材页说明硬件需求驱动逻辑'}
        self.assertEqual(len(ten['word']), 10)
        self.assertEqual(validate_jump_word(ten, require_metadata=True), (ten['word'], ten['short']))
        with self.assertRaises(ValueError):
            validate_jump_word({**ten, 'short': ten['word']}, require_metadata=True)

    def test_empty_destination_output_does_not_interrupt_generation(self):
        for name, node, state, key in [
            ('topic_word_generate', topic_word_generate_node, TopicWordGenerateInput(title='news', content='body', fit_topic_title='topic'), 'topic_hit_result'),
            ('hot_theme_word_generate', hot_theme_word_generate_node, HotThemeWordGenerateInput(title='news', content='body', fit_hot_theme_name='theme'), 'hot_theme_hit_result'),
        ]:
            with self.subTest(name=name), patch(f'graphs.nodes.{name}_node.invoke_llm', return_value=json.dumps(EMPTY)) as llm:
                result = node(state, {'metadata': {'llm_cfg': f'config/{name}_llm_cfg.json'}}, RUNTIME)
                hit = getattr(result, key)
                self.assertEqual(hit['word'], '')
                self.assertEqual(hit['short'], '')
                self.assertEqual(llm.call_count, 1)

    def test_word_short_subject_type_and_boundaries_are_checked(self):
        for invalid in [
            {**VALID, 'short': 'A股三季报盈利'},
            {**VALID, 'word': '一'*13},
            {**VALID, 'short': '一'*11},
            {**VALID, 'subject': 'AMD'},
            {**VALID, 'article_link': None},
            {**VALID, 'article_link': '   '},
            {**VALID, 'article_link': ''},
            {key: value for key, value in VALID.items() if key != 'article_link'},
            {**EMPTY, 'subject': 'A股'},
            {**EMPTY, 'article_link': '资讯与页面有关联'},
            {'recommended_words': [VALID['word']], 'best_word_short': VALID['short']},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_jump_word(invalid, require_metadata=True)
        self.assertEqual(validate_jump_word(VALID, require_metadata=True), (VALID['word'], VALID['short']))
        self.assertEqual(validate_jump_word(EMPTY, require_metadata=True), ('', ''))

    def test_missing_article_link_repairs_only_generation_and_keeps_metadata_internal(self):
        for name, node, state, key in [
            ('topic_word_generate', topic_word_generate_node, TopicWordGenerateInput(title='news', content='body', fit_topic_title='topic'), 'topic_hit_result'),
            ('hot_theme_word_generate', hot_theme_word_generate_node, HotThemeWordGenerateInput(title='news', content='body', fit_hot_theme_name='theme'), 'hot_theme_hit_result'),
        ]:
            incomplete = {key: value for key, value in VALID.items() if key != 'article_link'}
            with self.subTest(name=name), patch(f'graphs.nodes.{name}_node.invoke_llm', side_effect=[json.dumps(incomplete), json.dumps(VALID)]) as llm:
                output = node(state, {'metadata': {'llm_cfg': f'config/{name}_llm_cfg.json'}}, RUNTIME)
                self.assertEqual(llm.call_count, 2)
                self.assertIn('article_link', llm.call_args_list[1].args[2]['repair_note'])
                self.assertNotIn('link_type', llm.call_args_list[1].args[2]['repair_note'])
                self.assertEqual(getattr(output, key)['word'], VALID['word'])
                self.assertNotIn('article_link', getattr(output, key))

    def test_priority_only_chooses_directions_with_available_words(self):
        for topic_word, theme_word, expected in [('', '', 'none'), (VALID['word'], '', 'topic'), ('', '算力硬件驱动逻辑', 'hot_theme')]:
            with self.subTest(expected=expected), patch('graphs.nodes.priority_judge_node.invoke_llm') as llm:
                result = priority_judge_node(PriorityDecisionInput(
                    title='news', content='body', has_fit_topic=True, has_fit_hot_theme=True,
                    topic_hit_result={'word': topic_word}, hot_theme_hit_result={'word': theme_word}), {}, RUNTIME)
                self.assertEqual(result.out_priority, expected)
                llm.assert_not_called()

    def test_dual_priority_receives_single_words(self):
        with patch('graphs.nodes.priority_judge_node.invoke_llm', return_value=json.dumps({'out_priority': 'topic', 'out_priority_reason': 'evidence'})) as llm:
            priority_judge_node(PriorityDecisionInput(title='news', content='body', has_fit_topic=True, has_fit_hot_theme=True,
                topic_hit_result={'word': VALID['word']}, hot_theme_hit_result={'word': '算力硬件驱动逻辑'}),
                {'metadata': {'llm_cfg': 'config/priority_llm_cfg.json'}}, RUNTIME)
        self.assertEqual(llm.call_args.args[2]['topic_words'], VALID['word'])
        self.assertEqual(llm.call_args.args[2]['hot_theme_words'], '算力硬件驱动逻辑')

    def test_final_summary_and_csv_roundtrip_keep_single_words_only(self):
        word = '算力需求验证'
        final = final_summary_node(FinalSummaryInput(title='news', content='body', word_source='ai',
            has_fit_topic=True, has_fit_hot_theme=True, fit_topic_title='三季报', fit_hot_theme_name='算力',
            topic_hit_result=VALID, hot_theme_hit_result=EMPTY,
            recommended_words=[word], short_words=[{'word': word, 'short': word}],
            ai_queries=[{'word': word, 'query': '如何验证算力需求？', 'query_summary': '算力需求验证方法'}]), {}, RUNTIME).final_summary
        for key in ('topic_related', 'hot_theme_related'):
            self.assertEqual(set(final[key]), {'matched', 'title' if key == 'topic_related' else 'name', 'jump_url', 'word_source', 'word', 'short'})
        _validate_final_summary(final)
        row = _summary_row({}, {'final_summary': final}, 1)
        self.assertEqual(row['专题推荐词'], VALID['word'])
        restored = _old_final_summary(row)
        self.assertEqual(restored['topic_related'], final['topic_related'])
        self.assertEqual(restored['hot_theme_related'], final['hot_theme_related'])
        _validate_final_summary(restored)

    def test_final_export_rejects_legacy_jump_arrays_and_internal_metadata(self):
        category = {'generation_results': [{'word_source': 'quadrant_D', 'recommended_words': []}]}
        base = {'topic_related': {'matched': True, 'word': '', 'short': ''},
                'hot_theme_related': {'matched': False, 'word': '', 'short': ''}, 'category_quadrant': category}
        _validate_final_summary(base)
        for extra in ({'recommended_words': []}, {'best_word_short': ''}, {'subject': ''}, {'article_link': ''}, {'link_type': ''}, {'candidates': []}):
            invalid = {**base, 'topic_related': {**base['topic_related'], **extra}}
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                _validate_final_summary(invalid)
