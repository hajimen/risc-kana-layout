import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


class TestEvaluate(unittest.TestCase):
    def test_eval_ks_doc_example(self):
        from risc_kana_layout.evaluate import eval_ks
        esr, timelines = eval_ks(' jrtopfj ')
        self.assertEqual(timelines[0], '__r___t_______f__')
        self.assertEqual(timelines[1], 'j_______o___p___j')
        self.assertEqual(esr.n_left_hit, 3)
        self.assertEqual(esr.n_right_hit, 4)
        self.assertEqual(esr.left_run, {1: 1, 2: 1})
        self.assertEqual(esr.right_run, {1: 2, 2: 1})
        self.assertEqual(esr.trs_costs, {10: 1})

        esr, timelines = eval_ks(' jrotpfj ')
        self.assertEqual(timelines[0], '__r___t____f__')
        self.assertEqual(timelines[1], 'j___o___p____j')
        self.assertEqual(esr.n_left_hit, 3)
        self.assertEqual(esr.n_right_hit, 4)
        self.assertEqual(esr.left_run, {1: 3})
        self.assertEqual(esr.right_run, {1: 4})
        self.assertEqual(esr.trs_costs, {7: 1})

    def test_eval_ks(self):
        from risc_kana_layout.evaluate import eval_ks
        esr, _ = eval_ks('fj fj fj fj ')
        self.assertEqual(esr.n_trs, 3)
        self.assertEqual(esr.n_left_hit, 3)
        self.assertEqual(esr.n_right_hit, 3)
        self.assertEqual(esr.left_run, {1: 3})
        self.assertEqual(esr.right_run, {1: 3})
        self.assertEqual(esr.trs_costs, {2: 3})

        esr, _ = eval_ks('fj fj fj fj fj ')
        self.assertEqual(esr.n_trs, 4)
        self.assertEqual(esr.n_left_hit, 4)
        self.assertEqual(esr.n_right_hit, 4)
        self.assertEqual(esr.trs_costs, {2: 4})

    def test_roman_mora_table(self):
        from risc_kana_layout.evaluate import separate_by_moras, get_roman_mora_table
        roman_mora_table = get_roman_mora_table()
        moras = list(roman_mora_table.keys())

        for sep, _ in separate_by_moras(['バッファ'], moras):
            self.assertEqual(sep, ['バ', 'ッファ'])
            break

        for sep, _ in separate_by_moras(['バッイ'], moras):
            self.assertEqual(sep, ['バ', 'ッ', 'イ'])
            break

        for sep, _ in separate_by_moras(['「ァ」'], moras):
            self.assertEqual(sep, ['「', 'ァ', '」'])
            break
