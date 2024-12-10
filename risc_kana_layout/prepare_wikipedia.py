import pyarrow.parquet as pq
from random import sample
from risc_kana_layout.constant import WIKIPEDIA_SRC, WIKIPEDIA_DIR, KANA_TXT
from risc_kana_layout import to_kana


N_SAMPLE = 100000


def reader(n_item: int = -1):
    dataset = pq.ParquetDataset(WIKIPEDIA_SRC)
    texts = dataset.read(['text'])[0]
    if n_item == -1:
        for i, t in enumerate(texts):
            yield t.as_py()
    else:
        for i in sample(range(len(texts)), n_item):
            yield texts[i].as_py()


def prepare():
    if not WIKIPEDIA_DIR.exists():
        WIKIPEDIA_DIR.mkdir()
    with open(WIKIPEDIA_DIR / KANA_TXT, 'w', encoding='utf-8') as f:
        for kana in to_kana(reader(N_SAMPLE)):
            f.write(kana + '\n')
