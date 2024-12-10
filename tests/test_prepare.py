import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


# smoke test
class TestPrepare(unittest.TestCase):
    @unittest.skip("it connects to server.")
    def test_prepare_meidai(self):
        from risc_kana_layout.prepare_meidai_dialogue import prepare
        prepare()

    @unittest.skip("it takes quite long time.")
    def test_prepare_cc100(self):
        from risc_kana_layout.prepare_cc100 import prepare
        prepare()

    def test_prepare_arm(self):
        from risc_kana_layout.prepare_arm import prepare
        prepare()

    def test_prepare_wikipedia(self):
        from risc_kana_layout.prepare_wikipedia import prepare
        prepare()
