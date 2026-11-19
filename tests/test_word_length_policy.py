"""Recommendation length, short-form, count and repair boundary regressions."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from tools.word_extras import validate_words, best_word_short, generated_best_word_short, invoke_words_with_queries, SHORT_SUBJECT_POLICY_VERSION
from tools.runtime import Context
from graphs.nodes.topic_word_generate_node import topic_word_generate_node
from graphs.nodes.hot_theme_word_generate_node import hot_theme_word_generate_node
from graphs.state import TopicWordGenerateInput, HotThemeWordGenerateInput
from batch_run import _validate_final_summary, _has_word_extras


class WordLengthPolicyTests(unittest.TestCase):
    def test_prompts_use_current_contract(self):
        for name in ['topic_word_generate', 'hot_theme_word_generate', 'quadrant_a', 'quadrant_b',
                     'quadrant_c', 'quadrant_d', 'ai_generate']:
            cfg = json.loads((ROOT / f'config/{name}_llm_cfg.json').read_text(encoding='utf-8'))
            text = cfg['sp'] + cfg['up']
            for retired in ['生成恰好3个', '对应的6字版和9字版', '3–5', '最多2个']:
                self.assertNotIn(retired, text)
            sample = cfg['sp'].split('# 输出格式', 1)[1]
            example, _ = json.JSONDecoder().raw_decode(sample[sample.index('{'):])
            if name in ['topic_word_generate', 'hot_theme_word_generate']:
                expected = {'word', 'short', 'subject', 'article_link'}
                if name == 'hot_theme_word_generate':
                    expected.add('candidates')
                self.assertEqual(set(example), expected)
            else:
                self.assertEqual(set(example), {'recommended_words', 'short_words', 'ai_queries'})

    def test_short_subject_must_remain_visible_for_user_examples(self):
        for word, subject, valid, invalid in [
            ('英伟达财报超预期', '英伟达', '英伟达财报超预期', '财报超预期'),
            ('英伟达CEO遴选标准', '英伟达', '英伟达CEO遴选', 'CEO遴选标准'),
        ]:
            with self.subTest(word=word):
                self.assertEqual(generated_best_word_short(
                    {'best_word_subject': subject, 'best_word_short': valid}, [word]), valid)
                with self.assertRaisesRegex(ValueError, '必须保留原词主体'):
                    generated_best_word_short({'best_word_subject': subject, 'best_word_short': invalid}, [word])

    def test_subject_cannot_be_invented_generic_or_missing(self):
        for data in [
            {'best_word_subject': 'AMD', 'best_word_short': '英伟达财报超预期'},
            {'best_word_subject': '财报', 'best_word_short': '财报超预期'},
            {'best_word_subject': 'CEO', 'best_word_short': 'CEO遴选标准'},
            {'best_word_short': '英伟达财报超预期'},
        ]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                generated_best_word_short(data, ['英伟达财报超预期'])

    def test_parallel_subjects_are_kept_together(self):
        data = {'best_word_subject': '英伟达 AMD', 'best_word_short': '英伟达 AMD比较'}
        word = '英伟达 AMD芯片比较'
        self.assertEqual(generated_best_word_short(data, [word]), data['best_word_short'])
        with self.assertRaises(ValueError):
            generated_best_word_short({**data, 'best_word_short': '英伟达芯片比较'}, [word])

    def test_missing_subject_is_rewritten_before_returning_ai_output(self):
        words = ['英伟达CEO遴选标准']
        valid = {'recommended_words': words, 'short_words': [{'word': words[0], 'short': '英伟达CEO遴选', 'subject': '英伟达'}],
                 'ai_queries': [{'word': words[0], 'query': '英伟达CEO遴选需要考虑哪些标准？', "query_summary": words[0] + "验证要点"}]}
        calls = []
        def invoke(ctx, cfg, values):
            calls.append(values)
            return json.dumps({**valid, 'short_words': [{'word': words[0], 'short': 'CEO遴选标准', 'subject': '英伟达'}]} if len(calls) == 1 else valid)
        data, _, _ = invoke_words_with_queries(invoke, Context(), 'unused', {})
        self.assertEqual(data['short_words'][0]['short'], '英伟达CEO遴选')
        self.assertIn('英伟达', calls[1]['repair_note'])
        self.assertEqual(len(calls), 2)

    def test_subject_is_internal_and_absent_from_jump_result(self):
        payload = {'word': '英伟达财报超预期', 'subject': '英伟达',
                   'short': '英伟达财报超预期', 'article_link': '资讯和目标页面都涉及英伟达财报'}
        with patch('graphs.nodes.topic_word_generate_node.invoke_llm', return_value=json.dumps(payload)):
            result = topic_word_generate_node(
                TopicWordGenerateInput(title='英伟达', content='正文', fit_topic_title='财报专题'),
                {'metadata': {'llm_cfg': 'config/topic_word_generate_llm_cfg.json'}},
                SimpleNamespace(context=Context())).topic_hit_result
        self.assertNotIn('best_word_subject', result)
        self.assertNotIn('subject', result)
        self.assertNotIn('link_type', result)
        self.assertNotIn('article_link', result)
        self.assertEqual(result['short'], payload['short'])

    def test_spaces_letters_and_digits_count_toward_boundaries(self):
        word = '算力 芯片1234567'
        self.assertEqual(len(word), 12)
        validate_words([word])
        short = '算力 芯片1234'
        self.assertEqual(len(short), 9)
        self.assertEqual(best_word_short({'best_word_short': short}, [word]), short)
        with self.assertRaises(ValueError):
            validate_words([word + '8'])
        with self.assertRaises(ValueError):
            best_word_short({'best_word_short': short + '6'}, [word])

    def test_counts_duplicates_empty_strings_and_nontext_fail(self):
        for words in [['词1', '词2', '词3', '词4'], ['重复', '重复'], [''], ['首尾 '],
                      ['词\t词'], ['词\n词'], [123], [{}]]:
            with self.subTest(words=words), self.assertRaises(ValueError):
                validate_words(words)
        with self.assertRaises(ValueError):
            validate_words([], maximum=1, exact=True)

    def test_short_form_is_required_only_for_nonempty_results(self):
        self.assertEqual(best_word_short({}, []), '')
        for data, words in [({}, ['算力需求']), ({'best_word_short': '新增'}, []),
                            ({'best_word_short': '算力需求扩展'}, ['算力需求'])]:
            with self.subTest(data=data, words=words), self.assertRaises(ValueError):
                best_word_short(data, words)

    def test_all_generation_contract_errors_request_rewrite(self):
        valid = {'recommended_words': ['算力需求验证'], 'short_words': [{'word': '算力需求验证', 'short': '算力需求验证', 'subject': '算力'}],
                 'ai_queries': [{'word': '算力需求验证', 'query': '哪些指标可以验证算力需求的持续性？', "query_summary": '算力需求验证' + "验证要点"}]}
        invalids = [
            {**valid, 'recommended_words': ['一' * 13]},
            {**valid, 'short_words': [{'word': '算力需求验证', 'short': '一' * 11, 'subject': '算力'}]},
            {k: v for k, v in valid.items() if k != 'short_words'},
            {**valid, 'recommended_words': ['词1', '词2', '词3', '词4']},
        ]
        for invalid in invalids:
            calls = []
            def invoke(ctx, cfg, values):
                calls.append(values)
                return json.dumps(invalid if len(calls) == 1 else valid)
            with self.subTest(invalid=invalid):
                data, words, queries = invoke_words_with_queries(invoke, Context(), 'unused', {})
                self.assertEqual(words, valid['recommended_words'])
                self.assertEqual(queries, valid['ai_queries'])
                self.assertEqual(data['short_words'], [{'word': '算力需求验证', 'short': '算力需求验证'}])
                self.assertEqual(len(calls), 2)
                self.assertIn('上次输出不合规', calls[1]['repair_note'])

    def test_both_jump_branches_repair_old_three_word_output(self):
        valid = {'word': '算力需求验证', 'short': '算力需求验证', 'subject': '算力', 'article_link': '资讯讲算力扩张，页面提供需求验证维度'}
        runtime = SimpleNamespace(context=Context())
        for name, node, state, key in [
            ('topic_word_generate', topic_word_generate_node,
             TopicWordGenerateInput(title='算力', content='资讯', fit_topic_title='专题'), 'topic_hit_result'),
            ('hot_theme_word_generate', hot_theme_word_generate_node,
             HotThemeWordGenerateInput(title='算力', content='资讯', fit_hot_theme_name='热点'), 'hot_theme_hit_result')]:
            with self.subTest(name=name), patch(f'graphs.nodes.{name}_node.invoke_llm', side_effect=[
                json.dumps({'recommended_words': ['词1', '词2', '词3']}), json.dumps(valid)]) as invoke:
                output = node(state, {'metadata': {'llm_cfg': f'config/{name}_llm_cfg.json'}}, runtime)
                result = getattr(output, key)
                self.assertEqual(result['word'], valid['word'])
                self.assertEqual(result['short'], valid['short'])
                self.assertNotIn('display_variants', result)
                self.assertEqual(invoke.call_count, 2)

    def test_final_validation_and_resume_reject_old_jump_output(self):
        group = {'word_source': 'quadrant_A', 'recommended_words': [
            {'word': '算力需求验证', 'short': '算力需求验证', 'query': '如何验证算力需求？', 'query_summary': '算力需求验证方法'}]}
        final = {'topic_related': {'matched': True, 'word': '专题词', 'short': '专题词'},
                 'hot_theme_related': {'matched': False, 'word': '', 'short': ''},
                 'category_quadrant': {'generation_results': [group]}}
        _validate_final_summary(final)
        final['topic_related'].pop('short')
        with self.assertRaises(ValueError):
            _validate_final_summary(final)
        row = {'专题命中': True, '专题推荐词': '专题词', '生词规则版本': SHORT_SUBJECT_POLICY_VERSION,
               '专题展示版': json.dumps([{'word': '专题词', 'display_6': '专题词', 'display_9': '专题词'}]),
               '分类生词结果': json.dumps([group])}
        self.assertFalse(_has_word_extras(row))
        row['专题首选词短版'] = '专题词'
        self.assertTrue(_has_word_extras(row))
        row.pop('生词规则版本')
        self.assertFalse(_has_word_extras(row))


if __name__ == '__main__':
    unittest.main()
