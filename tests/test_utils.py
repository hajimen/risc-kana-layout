import unittest


class TestUtils(unittest.TestCase):
    def test_separate_by_moras(self):
        from risc_kana_layout import separate_by_moras
        t1 = 'タイムズ'
        t1s = list(t1) + ['タイ', 'イムズ']
        parsed1, freq1 = next(separate_by_moras([t1], t1s))
        self.assertEqual(parsed1, ['タ', 'イムズ'])
        self.assertEqual(freq1, {'タ': 1, 'イ': 0, 'ム': 0, 'ズ': 0, 'タイ': 0, 'イムズ': 1})

        t2 = 'タイムズスクエア'
        t2s = list(t2) + ['タイ', 'イムズ', 'ズスクエア', 'ムズスクエア']
        parsed2, freq2 = next(separate_by_moras([t2], t2s))
        self.assertEqual(parsed2, ['タイ', 'ムズスクエア'])
        self.assertEqual(freq2, {'タ': 0, 'イ': 0, 'ム': 0, 'ズ': 0, 'ス': 0, 'ク': 0, 'エ': 0, 'ア': 0, 'タイ': 1, 'イムズ': 0, 'ズスクエア': 0, 'ムズスクエア': 1})

    def test_find(self):  # smoke test
        from risc_kana_layout.constant import ARM_DIR, KANA_TXT
        from risc_kana_layout.enum_shorthand_candidates import find
        with open(ARM_DIR / KANA_TXT, 'r', encoding='utf-8') as f:
            find(f.readlines(), 'test')
