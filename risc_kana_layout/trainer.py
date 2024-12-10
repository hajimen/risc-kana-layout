from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict
import io
from random import sample
import os
import getpass
import urllib.request
from pathlib import Path

import MeCab
from jaconv import hankaku2zenkaku
import pyarrow.parquet as pq
from pyarrow.csv import read_csv, ParseOptions
from PIL import Image as PILImage
import numpy as np
from numpy.typing import NDArray
import mip
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

from risc_kana_layout.constant import WIKIPEDIA_SRC
from risc_kana_layout import separate_by_moras, get_sorted_mora, INITIAL_MORAS, SHORTHANDS, SYMBOLS


def reader(n_item: int):
    dataset = pq.ParquetDataset(WIKIPEDIA_SRC)
    texts = dataset.read(['text'])[0]
    for c, i in enumerate(sample(range(len(texts)), n_item)):
        yield hankaku2zenkaku(texts[i].as_py(), ascii=True, digit=True)


class P(Enum):
    Start = auto()
    Continue = auto()
    Unavailable = auto()


@dataclass
class Token:
    original: str
    yomi: str
    category: str
    subcat: str


STARTS = {'名詞', '動詞', '形容詞', '代名詞', '形状詞', '連体詞', '接続詞', '感動詞', '接頭辞', '副詞'}


def classify(token: Token, last_token: Token | None):
    if last_token is not None and last_token.subcat == '句点':
        return P.Start
    if '記号' in token.category:
        if token.original not in SYMBOLS:
            return P.Unavailable
        if '括弧開' == token.subcat:
            return P.Start
        return P.Continue
    if any([token.category.startswith(s) for s in STARTS]):
        if last_token is not None and last_token.category == '名詞' and token.category == '名詞':
            return P.Continue
        return P.Start
    return P.Continue


def to_group(reader: Iterable[str]):
    tagger = MeCab.Tagger("-Odump")
    for t in reader:
        xs: str = '0 原文 カンマ 3 4 5 6 7 8 9 10 11 12 13 14\n' + tagger.parse(t.strip())
        stream = io.BytesIO(xs.encode())
        table = read_csv(stream, parse_options=ParseOptions(delimiter=' ', quote_char=False, invalid_row_handler=lambda x: 'skip'))
        commas = table.to_pylist()
        tokens: list[Token] = []
        for d in commas:
            c = str(d['カンマ'])
            sep = c.split(',')
            if sep[0] == '空白' or sep[0] == 'BOS/EOS':
                continue
            yomi = str(d['原文'])
            if len(sep) >= 17 and len(sep[17]) > 0:
                yomi = sep[17]
            if yomi == 'ワタクシ':
                yomi = 'ワタシ'
            tokens.append(Token(str(d['原文']), yomi, sep[0], sep[1]))

        last_cons = []
        last_token: Token | None = None
        i = 0
        while i < len(tokens):
            w = tokens[i:]
            match classify(w[0], last_token):
                case P.Start:
                    ii = 1
                    while ii < len(w) and classify(w[ii], w[ii - 1]) == P.Continue:
                        ii += 1
                    yield w[:ii]
                    i += 1
                    last_cons = []
                case P.Continue:
                    last_cons.append(w[0])
                    i += 1
                case P.Unavailable:
                    yield []
                    last_cons = []
                    i += 1
            last_token = w[0]


def separate_by_full(h):
    return sum([part_ms for part_ms, _ in separate_by_moras([h], INITIAL_MORAS + SHORTHANDS + SYMBOLS)], [])


def find_example():
    sorted_mora, _ = get_sorted_mora()

    matched: list[list[tuple[str, list[str]]]] = []
    current_matching: list[list[tuple[list[Token], list[str]]]] = [[], [], [], [], []]
    matching_pos = [0, 0]

    def reset_match():
        nonlocal current_matching, matching_pos
        current_matching = [[], [], [], [], []]
        matching_pos = [0, 0]

    def append_current_matching(g: list[Token], h: str):
        nonlocal current_matching
        ms = separate_by_full(h)
        current_matching[matching_pos[0]].append((g, ms))

    TO_MATCH = [5, 7, 5, 7, 7]
    cnt = True
    for i, group in enumerate(to_group(reader(500_000))):
        if len(group) == 0:
            reset_match()
            continue
        while cnt:
            h = ''.join([t.yomi for t in group])
            if h != ''.join(list(separate_by_full(h))):
                reset_match()
                break
            ms = sum([part_ms for part_ms, _ in separate_by_moras([h], INITIAL_MORAS)], [])
            n = len(ms)
            d = TO_MATCH[matching_pos[0]] - matching_pos[1]
            if n > d:
                if matching_pos[0] == 4 and group[0].category.startswith('名詞'):
                    h = group[0].yomi
                    ms = sum([part_ms for part_ms, _ in separate_by_moras([h], INITIAL_MORAS)], [])
                    n = len(ms)
                    if n == d:
                        append_current_matching([group[0]], h)
                        matching_pos[0] += 1
                        matching_pos[1] = 0
                        break
                reset_match()
                break
            if n < d:
                append_current_matching(group, h)
                matching_pos[1] += n
            elif n == d:
                append_current_matching(group, h)
                matching_pos[0] += 1
                matching_pos[1] = 0
            else:
                raise Exception('Never occur')
            break
        if matching_pos[0] == 5:
            k2s = [[(''.join([g1.original for g1 in g]), ''.join(ms)) for g, ms in m] for m in current_matching]
            ks = [(''.join([g for g, _ in ks]), separate_by_full(''.join([h for _, h in ks]))) for ks in k2s]
            matched.append(ks)
            reset_match()

    bin_max: dict[int, list[tuple[list[tuple[str, list[str]]], float]]] = defaultdict(list)
    bins = [20 * i for i in range(1, 9)]
    for m in matched:
        h_all: list[str] = []
        for ii in range(0, 5):
            h_all.extend(m[ii][1])
            print(m[ii][0] + ' ' + ' '.join(m[ii][1]))
        sum_s = 0.
        max_s = 0
        for h in h_all:
            s0 = sorted_mora.index(h)
            sum_s += s0 ** 2
            max_s = max(max_s, s0)
        s = (sum_s / len(ms)) ** 0.5
        for b in bins:
            if max_s < b:
                bin_max[b].append((m, s))
                break
        print(f'score:{s:.2f}  max:{max_s}\n')
    for b, bm in bin_max.items():
        with open(f'bin_{b}.txt', 'w', encoding='utf-8') as f:
            sorted_bm = sorted(bm, key=lambda x: x[1], reverse=True)
            for m, s in sorted_bm:
                ks = [k[0] for k in m]
                hs = [' '.join(k[1]) for k in m]
                f.write(f'{" _ ".join(ks)}\t{" _ ".join(hs)}\t{s:.2f}\n')


def find_deficient(level: str, range_lower: int, range_upper: int, to_fill: int):
    sorted_mora, _ = get_sorted_mora()

    bin_count: dict[int, int] = defaultdict(int)
    with open(f'{level}.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            s, k, _ = line.split('\t')
            k = k.replace(' _ ', ' ')
            ks = k.split(' ')
            for m in ks:
                bin_count[sorted_mora.index(m)] += 1
    for im, c in bin_count.items():
        if im >= range_upper and c > 0:
            print(f'{sorted_mora[im]}が混入')
    for ii in range(1, to_fill + 1):
        ufs = []
        for i in range(range_lower, range_upper):
            if bin_count[i] == to_fill - ii:
                ufs.append(sorted_mora[i])
        print(f'{ii}個足りないモーラ： ' + ' '.join(ufs))


def dump_kana_table_as(filename='trainer/public/assets/layout.json'):
    import json
    from risc_kana_layout import OPT_COMPLETE_TXT
    from risc_kana_layout.evaluate import ROMAN_SUPP_TABLE
    from jaconv import kata2hira
    d = {}
    with open(OPT_COMPLETE_TXT, 'r', encoding='utf-8') as f:
        for li in f.readlines():
            ks, m = li.strip().split('\t')
            d[kata2hira(m)] = ks.replace(' ', '').replace('_', '')
    for m, kps in ROMAN_SUPP_TABLE.items():
        s = ''
        for kp in kps:
            s += ''.join(kp).replace('_', '')
        d[kata2hira(m)] = s
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(d, f)


def read_stages(level: str):
    from jaconv import kata2hira
    from risc_kana_layout.evaluate import eval_sep
    from risc_kana_layout.evaluate import get_mora_table
    from risc_kana_layout import OPT_COMPLETE_TXT
    risc_mora_table = get_mora_table(OPT_COMPLETE_TXT)

    stages: list[tuple[list[str], list[list[str]], float]] = []
    with open(f'{level}.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            s, k, _ = line.split('\t')
            ks = k.split(' _ ')
            fks: list[list[str]] = [kata2hira(k).split(' ') for k in ks]
            s = s.replace(' ', '')
            ss = s.split('_')
            sep = sum([k.split(' ') for k in ks], [])
            esr = eval_sep(sep, risc_mora_table)
            cost = sum(k * v for k, v in esr.trs_costs.items()) + esr.n_left_hit + esr.n_right_hit
            stages.append((ss, fks, cost))
    return stages


def dump_stage_data():
    import json
    stage_data = {}
    for level in ['Basic', 'Advanced', 'Practical', 'Maniac']:
        stage_data[level] = read_stages(level)
    with open('trainer/public/assets/stage-data.json', 'w', encoding='utf-8') as f:
        json.dump(stage_data, f)


def generate_image(generate_levels=['Basic', 'Advanced', 'Practical', 'Maniac']):
    import hashlib
    import time
    import pathlib
    from openai import OpenAI, BadRequestError
    os.environ["OPENAI_API_KEY"] = getpass.getpass('OpenAI API Key:')
    client = OpenAI()
    for level in generate_levels:
        stages = read_stages(level)
        last_time = 0.
        for s, _, _ in stages:
            q = ''.join(s)
            q_hex = hashlib.md5(q.encode('utf-8')).hexdigest()
            image_fp = pathlib.Path(f'trainer/public/assets/illust/{q_hex}.jpg')
            if image_fp.exists():
                continue
            now = time.time()
            gone = now - last_time
            if gone < 0.3 * 60:
                time.sleep(0.3 * 60 - gone)
            last_time = now
            try:
                response = client.images.generate(
                    model="dall-e-2",
                    prompt=q,
                    size="512x512",
                    quality="standard",
                    n=1,
                )
            except BadRequestError as e:
                print(e.message)
                print(q)
                print('\n\n')
                continue
            image_url = response.data[0].url
            assert image_url is not None
            req = urllib.request.Request(image_url)
            with urllib.request.urlopen(req) as res:
                body = res.read()
                image_data = io.BytesIO(body)
                image = PILImage.open(image_data)
            image.save(image_fp)


def fill(sources: list[str], target: str, range_lower: int, range_upper: int, to_fill: int, ignores: list[str] = []):
    sorted_mora, _ = get_sorted_mora()
    lines = []
    n = range_upper - range_lower - len(ignores)

    usage_counts: list[list[int]] = []
    for source in sources:
        with open(f'{source}.txt', 'r', encoding='utf-8') as f:
            for _, line in enumerate(f.readlines()):
                lines.append(line)
                cs = [0] * n
                _, k, _ = line.split('\t')
                for m in k.replace(' _ ', ' ').split(' '):
                    if m in ignores:
                        continue
                    i = sorted_mora.index(m)
                    if range_lower <= i and i < range_upper - len(ignores):
                        cs[i - range_lower] += 1
                usage_counts.append(cs)
    usage_counts_np = np.array(usage_counts).T

    model = mip.Model()
    use_m = model.add_var_tensor((usage_counts_np.shape[1],), name='use_m', var_type=mip.BINARY)
    for c in np.apply_along_axis(np.sum, 1, use_m * usage_counts_np):
        model.add_constr(c >= to_fill)
    model.objective = mip.minimize(mip.xsum(use_m))
    o = model.optimize(max_seconds=30)  # type: ignore
    if o.name == 'INFEASIBLE':
        raise Exception('Infeasible')

    with open(f'{target}.txt', 'w', encoding='utf-8') as f:
        for i, u in enumerate(use_m):
            if u.x > 0.9:
                f.write(lines[i])


def rebin():
    bin_lines: dict[int, list[str]] = defaultdict(list)
    sorted_mora, _ = get_sorted_mora()
    bins = [40, 80, 105, 130]

    for fn in [f'bin_{i}0-1' for i in [4, 6, 8, 10, 12]]:
        with open(fn + '.txt', 'r', encoding='utf-8') as f:
            for _, line in enumerate(f.readlines()):
                _, k, _ = line.split('\t')
                max_o = 0
                for m in k.replace(' _ ', ' ').split(' '):
                    o = sorted_mora.index(m)
                    max_o = max(max_o, o)
                for b in bins:
                    if max_o < b:
                        bin_lines[b].append(line)
                        break

    for b, lines in bin_lines.items():
        with open(f'bin_{b}-2.txt', 'w', encoding='utf-8') as f:
            f.writelines(lines)


PAPER_WIDTH = 297.
PAPER_HEIGHT = 210.
MARGIN = 10.
LIVE_WIDTH = PAPER_WIDTH - MARGIN * 2
LIVE_HEIGHT = PAPER_HEIGHT - MARGIN * 2
pdfmetrics.registerFont(TTFont("MSGothic", "C:\\Windows\\Fonts\\msgothic.ttc"))
pdfmetrics.registerFont(TTFont("OpenSans", str((Path(os.getcwd()) / 'font/OpenSans-Regular.ttf').absolute())))
DEFAULT_PITCH = 19.


class Document:
    def __init__(self, filepath: Path) -> None:
        self.canvas = canvas.Canvas(str(filepath))
        self.i_page = 0

    def init_page(self):
        self.canvas.setPageSize((PAPER_WIDTH * mm, PAPER_HEIGHT * mm))
        self.canvas.translate(MARGIN * mm, MARGIN * mm)
        self.canvas.saveState()
        self.canvas.setFont('OpenSans', 5 * mm)
        self.i_page += 1

    def lines(self, vertices: NDArray):
        for s, e in zip(vertices[:-1], vertices[1:]):
            self.canvas.line(*s, *e)

    def string(self, s: str, pos: NDArray):
        self.canvas.drawString(pos[0], pos[1], s)

    def string_center(self, s: str, pos: NDArray):
        self.canvas.drawCentredString(pos[0], pos[1], s)

    def string_right(self, s: str, pos: NDArray):
        self.canvas.drawRightString(pos[0], pos[1], s)


def generate_table():
    import datetime as dt
    from jaconv import kata2hira
    from risc_kana_layout.evaluate import get_mora_table
    from risc_kana_layout import OPT_COMPLETE_TXT
    from risc_kana_layout.format import Assign, TABLE, MORA_INDEX

    timestamp = dt.datetime.now().strftime("%Y/%m/%d %H:%M")
    sorted_mora, _ = get_sorted_mora()
    mora_table = get_mora_table(OPT_COMPLETE_TXT)

    pitch = 27 * mm
    qwerty_table = [('qwertyuiop', 0.), ('asdfghjkl', 0.25), ('zxcvbnm,', 0.75)]
    qwerty_order = '_' + ''.join(s for s, _ in qwerty_table)
    top = (LIVE_HEIGHT - 40) * mm

    for cn, (lb, ub) in zip(['Basic', 'Advanced', 'Practical', 'Maniac', 'All'], [(0, 40), (40, 80), (80, 105), (105, 130), (0, 130)]):
        moras = sorted_mora[lb:ub]
        assign = Assign()
        rev_table: dict[str, dict[str, str]] = defaultdict(defaultdict)
        for m in moras:
            kps = mora_table[m]
            if len(kps) > 1:
                continue
            left, right = kps[0]
            if left[0] == 'b':
                continue
            m = kata2hira(m)
            rev_table[left][right] = m
            if m in MORA_INDEX:
                idx = MORA_INDEX[m]
                assign.regular[idx[0]][idx[1]] = (left, right)
            else:
                assign.special[m] = (left, right)
        doc = Document(Path(f'{cn}.pdf'))
        doc.canvas.setAuthor("Hajime NAKAZATO")
        title = f'RISCかな配列 {cn}配列表'
        doc.canvas.setTitle(title)
        doc.canvas.setSubject(title)
        n_page = len(rev_table) + 1

        def _init_page(i_page):
            doc.init_page()
            doc.canvas.setFont('MSGothic', 10 * mm)
            doc.string_center(title, np.array([LIVE_WIDTH / 2 * mm, top + 10 * mm]))
            doc.canvas.setFont('MSGothic', 7 * mm)
            doc.string_center(f'{i_page + 1} / {n_page}', np.array([(LIVE_WIDTH - 20) * mm, top + 10 * mm]))
            doc.canvas.setFont('MSGothic', 5 * mm)
            doc.string_right(timestamp + ' DecentKeyboards', np.array([(LIVE_WIDTH - 5) * mm, 10 * mm]))

        # aiueo table
        _init_page(0)
        regular_table = []
        for i, r in enumerate(assign.regular):
            rs = []
            for ii, a in enumerate(r):
                if a is None:
                    rs.extend(('', ''))
                    continue
                left = a[0]
                left = '' if left == '_' else left
                rs.extend((TABLE[i][ii], left + a[1]))
            if rs == ['', ''] * 5:
                continue
            regular_table.append(rs)

        regular_ts_cmds = []
        for ii in range(5):
            i2 = ii * 2
            regular_ts_cmds.extend([
                ('FONT', (i2 + 0, 0), (i2 + 1, -1), 'MSGothic', 3.5 * mm),
                ('ALIGN', (i2 + 0, 0), (i2 + 1, -1), 'RIGHT'),
                ('FONT', (i2 + 1, 0), (i2 + 2, -1), 'OpenSans', 3.5 * mm),
                ('ALIGN', (i2 + 1, 0), (i2 + 2, -1), 'LEFT'),
            ])
        for i in range(len(regular_table)):
            for ii in range(5):
                c = colors.lavender if ((i + ii) % 2) == 0 else colors.white
                regular_ts_cmds.append(('BACKGROUND', (ii * 2, i), (ii * 2 + 2, i), c))

        pdf_regular_table = Table(regular_table, colWidths=12 * mm, rowHeights=6 * mm)
        pdf_regular_table.setStyle(TableStyle(regular_ts_cmds))
        pdf_regular_table.wrapOn(doc.canvas, LIVE_WIDTH / 2 * mm, LIVE_HEIGHT * mm)
        pdf_regular_table.drawOn(doc.canvas, 20 * mm, (LIVE_HEIGHT - len(regular_table) * 6 - 40) * mm)

        # symbol and shorthand table
        special_table = []
        for m, lr in sorted(assign.special.items(), key=lambda x: x[0]):
            left = lr[0]
            left = '' if left == '_' else left
            special_table.append([m, left + lr[1]])

        if len(special_table) > 0:
            special_ts_cmds = [
                ('FONT', (0, 0), (1, -1), 'MSGothic', 3.5 * mm),
                ('ALIGN', (0, 0), (1, -1), 'RIGHT'),
                ('FONT', (1, 0), (2, -1), 'OpenSans', 3.5 * mm),
                ('ALIGN', (1, 0), (2, -1), 'LEFT'),
            ]
            for i in range(len(special_table)):
                c = colors.lavender if (i % 2) == 0 else colors.white
                special_ts_cmds.append(('BACKGROUND', (0, i), (2, i), c))

            pdf_special_table = Table(special_table, colWidths=20 * mm, rowHeights=6 * mm)
            pdf_special_table.setStyle(TableStyle(special_ts_cmds))
            pdf_special_table.wrapOn(doc.canvas, LIVE_WIDTH / 2 * mm, LIVE_HEIGHT * mm)
            pdf_special_table.drawOn(doc.canvas, (LIVE_WIDTH / 2 + 20) * mm, (LIVE_HEIGHT - len(special_table) * 6 - 40) * mm)

        doc.canvas.showPage()

        # layout tables
        for i_page, left in enumerate(sorted(rev_table.keys(), key=lambda x: qwerty_order.index(x) if len(x) == 1 else qwerty_order.index(x[0]) * 100 + qwerty_order.index(x[1]))):
            _init_page(i_page + 1)
            if left == '_':
                doc.canvas.setFont('MSGothic', 7 * mm)
                doc.string_center('右手単独', np.array([LIVE_WIDTH / 2 * mm, 50 * mm]))
            doc.canvas.setFont('OpenSans', 7 * mm)
            for i, (qr_letters, qr_offset) in enumerate(qwerty_table):
                x_offset = qr_offset * pitch + 5 * mm
                y_offset = top - i * pitch
                xyo = np.array([x_offset, y_offset])
                doc.lines(np.array([
                    xyo,
                    xyo + np.array([len(qr_letters) * pitch, 0])
                ]))
                doc.lines(np.array([
                    xyo + np.array([0, -pitch]),
                    xyo + np.array([len(qr_letters) * pitch, -pitch]),
                ]))
                for ii, lt in enumerate(qr_letters + ' '):
                    pos = xyo + np.array([ii * pitch, 0])
                    doc.lines(np.array([
                        pos,
                        pos + np.array([0, -pitch])
                    ]))
                    hit = False
                    center_pos = pos + np.array([pitch, -pitch]) / 2
                    if lt == left:
                        hit = True
                        doc.canvas.circle(center_pos[0], center_pos[1], pitch / 3, fill=1)
                    elif len(left) > 1 and lt in left:
                        hit = True
                        mark = ''
                        if lt == left[0]:
                            mark = '前'
                        else:
                            mark = '後'
                        doc.canvas.setFont('MSGothic', 10 * mm)
                        doc.string_center(mark, center_pos + np.array([0, -4 * mm]))
                        doc.canvas.circle(center_pos[0], center_pos[1], pitch / 3)
                    elif lt in rev_table[left]:
                        hit = True
                        doc.canvas.setFont('MSGothic', 9 * mm)
                        to_print = rev_table[left][lt]
                        doc.string_center(to_print, center_pos + np.array([0, -10 * mm]))
                        if to_print in 'ぁぃぅぇぉゃゅょ':
                            doc.canvas.setDash([2, 2])
                            doc.canvas.roundRect(
                                center_pos[0] - 4 * mm,
                                center_pos[1] - 12 * mm,
                                13 * mm,
                                13 * mm,
                                1 * mm
                            )
                            doc.canvas.setDash([])
                    if hit:
                        doc.canvas.setFont('OpenSans', 7 * mm)
                        doc.string(lt, pos + np.array([2, -7]) * mm)
            doc.canvas.showPage()
        doc.canvas.save()
