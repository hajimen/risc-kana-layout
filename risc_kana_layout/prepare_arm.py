import polars as pl
from random import sample
from risc_kana_layout import to_kana
from risc_kana_layout.constant import ARM_SRC, ARM_DIR, KANA_TXT


N_SAMPLE = 100000


def prepare():
    df = pl.read_ndjson(ARM_SRC)
    texts = df['text']
    if not ARM_DIR.exists():
        ARM_DIR.mkdir()
    with open(ARM_DIR / KANA_TXT, 'w', encoding='utf-8') as kf:
        for kana in to_kana([texts[i] for i in sample(range(len(texts)), N_SAMPLE)]):
            kf.write(kana + '\n')
