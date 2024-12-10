import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


# smoke test
class TestCalc(unittest.TestCase):
    def test_to_markov(self):
        # smoke test
        from risc_kana_layout.calc_stat import to_raw_markov
        from risc_kana_layout import read_corpus, INITIAL_MORAS, SHORTHANDS
        from risc_kana_layout.constant import MEIDAI_DIR
        kana_reader = read_corpus(MEIDAI_DIR)
        _ = to_raw_markov(kana_reader, INITIAL_MORAS + SHORTHANDS)

    def test_calc_save(self):
        # smoke test
        from risc_kana_layout.calc_stat import calc_save_markov, ave_markov
        from risc_kana_layout import read_corpus
        from risc_kana_layout.constant import MEIDAI_DIR
        kana_reader = read_corpus(MEIDAI_DIR)
        calc_save_markov(kana_reader, 'test_calc_save')
        ave_markov('test_ave_temp', 'test_ave_temp', 'test_ave')
