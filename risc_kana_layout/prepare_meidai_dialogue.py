from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import httpx
from stream_unzip import stream_unzip
from risc_kana_layout.constant import MEIDAI_SRC, MEIDAI_DIR, KANA_TXT
from risc_kana_layout import to_kana


def prepare():
    with TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        with httpx.stream('GET', MEIDAI_SRC) as r:
            for file_name, file_size, unzipped_chunks in stream_unzip(r.iter_bytes(chunk_size=65536)):
                # unzipped_chunks must be iterated to completion or UnfinishedIterationError will be raised
                fp: Path = tmp / file_name.decode()
                if file_size == 0:
                    fp.mkdir()
                    continue
                with open(fp, 'wb') as f:
                    for chunk in unzipped_chunks:
                        f.write(chunk)
        from make_meidai_dialogue import mksequence
        mksequence.nuc_dir = str(tmp / 'nucc')
        if not MEIDAI_DIR.exists():
            MEIDAI_DIR.mkdir()
        with open(MEIDAI_DIR / 'raw.txt', 'w', encoding='utf-8') as cf:
            sys.stdout = cf
            mksequence.main()

        with open(MEIDAI_DIR / 'raw.txt', 'r', encoding='utf-8') as cf:
            with open(MEIDAI_DIR / KANA_TXT, 'w', encoding='utf-8') as kf:
                for kana in to_kana([li.split(':')[1] for li in cf.readlines()]):
                    kf.write(kana + '\n')
