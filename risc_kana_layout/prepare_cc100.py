import re
from glob import glob
import pyarrow.parquet as pq
from random import sample
from risc_kana_layout import to_kana
from risc_kana_layout.constant import CC100_SRC, CC100_DIR, KANA_TXT


N_SAMPLE = 1_000_000


def reader(n_item: int):
    n_file = {}
    acc = 0
    m = re.compile('[\r\n]+')
    for fn in glob(CC100_SRC):
        dataset = pq.ParquetDataset(fn)
        before = acc
        acc += dataset.read(['id']).num_rows
        n_file[fn] = (before, acc)
    to_sample = sample(range(acc), n_item)
    for fn in glob(CC100_SRC):
        dataset = pq.ParquetDataset(fn)
        start, end = n_file[fn]
        texts = dataset.read(['text'])[0]
        for i in filter(lambda x: x >= start and x < end, to_sample):
            t = texts[i - start].as_py().strip()
            for tt in m.split(t):
                tt = tt.strip()
                if len(tt) == 0:
                    continue
                yield tt


def prepare():
    if not CC100_DIR.exists():
        CC100_DIR.mkdir()
    with open(CC100_DIR / KANA_TXT, 'w', encoding='utf-8') as cf:
        for kana in to_kana(reader(N_SAMPLE)):
            cf.write(kana + '\n')
