import unittest
from pathlib import Path
from itertools import combinations, chain
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np


TEST_MORAS = list('あいうえおか')
TEST_LEFT_KEYS = ['q', 'w']
TEST_LEFT_TKNS = ['q', 'w', 'qw', '_']
TEST_RIGHT_KEYS = ['p', 'o', 'i']
TEST_MARKOV_RAW = {
    'あ': {
        'あ': 1,
        'い': 2,
        'う': 3,
        'え': 4,
        'お': 5,
        'か': 6,
    },
    'い': {
        'あ': 2,
        'い': 3,
        'う': 4,
        'え': 1,
        'お': 5,
        'か': 6,
    },
    'う': {
        'あ': 3,
        'い': 4,
        'う': 1,
        'え': 2,
        'お': 5,
        'か': 6,
    },
    'え': {
        'あ': 1,
        'い': 3,
        'う': 2,
        'え': 4,
        'お': 5,
        'か': 6,
    },
    'お': {
        'あ': 4,
        'い': 2,
        'う': 3,
        'え': 1,
        'お': 5,
        'か': 6,
    },
    'か': {
        'あ': 1,
        'い': 4,
        'う': 3,
        'え': 2,
        'お': 5,
        'か': 6,
    },
}
TEST_MORA_FREQ_RAW = {
    'あ': 10,
    'い': 3,
    'う': 4,
    'え': 5,
    'お': 50,
    'か': 55,
}
TEST_PAIR_SCORE = {frozenset(k): 5 for k in chain(combinations(TEST_LEFT_KEYS, 2), combinations(TEST_RIGHT_KEYS, 2))}
for k in TEST_LEFT_KEYS + TEST_RIGHT_KEYS:
    TEST_PAIR_SCORE[frozenset([k, k])] = 1
for k0, k1, score in [('i', 'p', 4), ('i', 'o', 2), ('o', 'p', 3), ('q', 'w', 2)]:
    TEST_PAIR_SCORE[frozenset([k0, k1])] = score

TEST_MARKOV_FILENAME_SUFFIX = 'test'
TEST_ORACLE_DIR = Path('tests/oracle')


class TestOpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sum_mora_freq = sum(TEST_MORA_FREQ_RAW.values())
        test_mora_freq = {k: v / sum_mora_freq for k, v in TEST_MORA_FREQ_RAW.items()}
        test_markov = {}
        for k0, v0 in TEST_MARKOV_RAW.items():
            s = sum(v0.values())
            test_markov[k0] = {k1: v1 / s for k1, v1 in v0.items()}
        
        with open(f'markov_{TEST_MARKOV_FILENAME_SUFFIX}.pkl', 'wb') as f:
            pickle.dump((test_markov, test_mora_freq), f)

    def test_load_score_freq(self):
        from risc_kana_layout.optimize import load_score_freq
        left_trs_score, right_trs_score, raw_mora_freq, screen_mora_pairs = load_score_freq(
            TEST_MORAS, TEST_MARKOV_FILENAME_SUFFIX, TEST_LEFT_TKNS, TEST_RIGHT_KEYS, TEST_PAIR_SCORE)
        print(left_trs_score)
        print(right_trs_score)
        self.assertTrue(np.all(np.array([
            [2.8, 2.9, 4.8, 0],
            [2.9, 2.8, 4.9, 0],
            [2.9, 2.8, 4.9, 0],
            [2.7, 2.7, 4.7, 0]], dtype=np.float64) == left_trs_score))
        self.assertTrue(np.all(np.array([
            [1, 3, 4],
            [3, 1, 2],
            [4, 2, 1]], dtype=np.int64) == right_trs_score))

        _ = screen_mora_pairs(10, TEST_MORAS[:1])

    def prepare_opt(self):
        from risc_kana_layout.optimize import load_score_freq
        left_trs_score, right_trs_score, raw_mora_freq, screen_mora_pairs = load_score_freq(
            TEST_MORAS, TEST_MARKOV_FILENAME_SUFFIX, TEST_LEFT_TKNS, TEST_RIGHT_KEYS, TEST_PAIR_SCORE)
        _, mora_freq = screen_mora_pairs(20, [])
        moras_sorted = sorted(zip(TEST_MORAS, mora_freq), key=lambda x: x[1], reverse=True)
        return moras_sorted

    def check_result_file(self, result_filename: str, oracle_filenames: list[str]):
        def replace_qw(ss: list[str]):
            return [s.replace('q', 'w') for s in ss]

        for fn in oracle_filenames:
            with open(result_filename, 'r', encoding='utf-8') as f:
                result_txts = f.readlines()
            with open(TEST_ORACLE_DIR / fn, 'r', encoding='utf-8') as f:
                oracle_txts = f.readlines()
            if replace_qw(result_txts) == replace_qw(oracle_txts):
                return True
        return False

    def test_opt_initial(self):
        from risc_kana_layout.optimize import build_opt_model, do_optimize
        moras_sorted = self.prepare_opt()
        moras = list(map(lambda x: x[0], moras_sorted[:5]))
        opt_model = build_opt_model(moras, TEST_MARKOV_FILENAME_SUFFIX, 20, None, [], TEST_LEFT_TKNS, TEST_RIGHT_KEYS, TEST_PAIR_SCORE)
        obj_val, status, result = do_optimize(opt_model, moras, 10, result_filename='test_opt_initial.txt')
        self.assertEqual(status, 'OPTIMAL', '10 seconds should be enough to OPTIMAL.')
        self.assertTrue(self.check_result_file('test_opt_initial.txt', [f'test_opt_initial{i}.txt' for i in range(2)]))

    def test_opt_iteration(self):
        from risc_kana_layout.optimize import build_opt_model, do_optimize
        moras_sorted = self.prepare_opt()
        moras = list(map(lambda x: x[0], moras_sorted))
        opt_model = build_opt_model(moras, TEST_MARKOV_FILENAME_SUFFIX, 100, TEST_ORACLE_DIR / 'test_opt_initial0.txt', moras[:1], TEST_LEFT_TKNS, TEST_RIGHT_KEYS, TEST_PAIR_SCORE)
        obj_val, status, result = do_optimize(opt_model, moras, 10, result_filename='test_opt_iteration.txt')
        self.assertEqual(status, 'OPTIMAL', '10 seconds should be enough to OPTIMAL.')
        self.assertTrue(self.check_result_file('test_opt_iteration.txt', [f'test_opt_iteration{i}.txt' for i in range(2)]))
