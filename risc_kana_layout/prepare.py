import shutil
from risc_kana_layout.constant import DATA_DIR, INITIAL_MORA_TXT, SHORTHAND_TXT, SYMBOL_TXT


def prepare_all():
    import random
    from risc_kana_layout.constant import TRS_SCORE_TXT
    from risc_kana_layout.prepare_arm import prepare as prepare_arm
    prepare_arm()
    random.seed(0)
    from risc_kana_layout.prepare_cc100 import prepare as prepare_cc100
    prepare_cc100()
    from risc_kana_layout.prepare_meidai_dialogue import prepare as prepare_meidai
    prepare_meidai()
    random.seed(0)
    from risc_kana_layout.prepare_wikipedia import prepare as prepare_wikipedia
    prepare_wikipedia()
    shutil.copy(DATA_DIR / TRS_SCORE_TXT, TRS_SCORE_TXT)
    shutil.copy(DATA_DIR / INITIAL_MORA_TXT, INITIAL_MORA_TXT)
    shutil.copy(DATA_DIR / SHORTHAND_TXT, SHORTHAND_TXT)
    shutil.copy(DATA_DIR / SYMBOL_TXT, SYMBOL_TXT)
