from dataclasses import dataclass, field
from jaconv import alphabet2kana, kana2alphabet, kata2hira
from risc_kana_layout import SYMBOLS, get_sorted_mora

normal_mora_rows = list('あか が   ざ た だな は ばぱま  ら  ')
special_rows = [
    ['きゃ', None, 'きゅ', None, 'きょ'],
    ['ぎゃ', None, 'ぎゅ', None, 'ぎょ'],
    ['さ', 'し', 'す', 'せ', 'そ'],
    ['しゃ', None, 'しゅ', 'しぇ', 'しょ'],
    ['じゃ', None, 'じゅ', 'じぇ', 'じょ'],
    ['ちゃ', None, 'ちゅ', 'ちぇ', 'ちょ'],
    ['にゃ', None, 'にゅ', None, 'にょ'],
    ['ひゃ', None, 'ひゅ', None, 'ひょ'],
    ['や', None, 'ゆ', None, 'よ'],
    ['ゃ', None, 'ゅ', None, 'ょ'],
    ['りゃ', None, 'りゅ', None, 'りょ'],
    ['わ', 'ゐ', None, 'ゑ', 'を'],
]
aiueo = 'aiueo'


def _build_table():
    table: list[list[str | None]] = []
    mora_index: dict[str, tuple[int, int]] = {}
    i_special_row = 0
    for nmr in normal_mora_rows:
        if nmr == ' ':
            r = special_rows[i_special_row]
            for i, m in enumerate(r):
                if m is not None:
                    mora_index[m] = (len(table), i)
            table.append(r)
            i_special_row += 1
            continue
        al = kana2alphabet(nmr)[:-1]
        table_row = []
        for i, p in enumerate(aiueo):
            m = alphabet2kana(al + p)
            mora_index[m] = (len(table), i)
            table_row.append(m)
        table.append(table_row)
    return table, mora_index


TABLE, MORA_INDEX = _build_table()


@dataclass
class Assign:
    regular: list[list[tuple[str, str] | None]] = field(default_factory=lambda: [[None,] * 5 for _ in TABLE])
    special: dict[str, tuple[str, str]] = field(default_factory=dict)


def read_solution(filename: str, range_lower=0, range_upper=157):
    assign = Assign()
    k, _ = get_sorted_mora()
    with open(filename, 'r', encoding='utf-8') as f:
        for li in f.readlines():
            ks, m = li.strip().split('\t')
            rank = k.index(m)
            if rank < range_lower or rank >= range_upper:
                continue
            m = kata2hira(m)
            left, right = ks.split(' ')
            if m in MORA_INDEX:
                idx = MORA_INDEX[m]
                assign.regular[idx[0]][idx[1]] = (left, right)
            else:
                assign.special[m] = (left, right)
    return assign


def format_markdown(assign: Assign):
    ret = '''|   |   |   |   |   |
|---|---|---|---|---|
'''
    for i, r in enumerate(assign.regular):
        rs = '|'
        for ii, a in enumerate(r):
            if a is None:
                rs += '   |'
                continue
            rs += f'{TABLE[i][ii]} {a[0]} {a[1]}|'
        ret += (rs + '\n')
    ret += '|   |   |   |   |   |\n'
    for length in range(1, 4):
        ret += '''
|   |   |
|---|---|
'''
        to_sort = []
        for m, lr in assign.special.items():
            if len(m) != length:
                continue
            if m in SYMBOLS:
                continue
            to_sort.append((m, lr))
        for m, lr in sorted(to_sort, key=lambda x: x[0]):
            ret += f'|{m}|{lr[0]} {lr[1]}|\n'
        ret += '|   |   |\n'

    ret += '''
|   |   |
|---|---|
'''
    to_sort = []
    for s in SYMBOLS:
        if s not in assign.special:
            continue
        lr = assign.special[s]
        to_sort.append((s, lr))
    for s, lr in sorted(to_sort, key=lambda x: x[0]):
        ret += f'|{s}|{lr[0]} {lr[1]}|\n'

    ret += '|   |   |\n'
    return ret


def format_markdown_binned(assign: Assign):
    ret = '''|   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|
'''
    for i, r in enumerate(assign.regular):
        rs = '|'
        for ii, a in enumerate(r):
            if a is None:
                rs += '   |   |'
                continue
            left = a[0]
            left = '' if left == '_' else left
            rs += f'{TABLE[i][ii]}|{left}{a[1]}|'
        if rs == '|   |   |   |   |   |   |   |   |   |   |':
            continue
        ret += (rs + '\n')
    ret += '|   |   |   |   |   |   |   |   |   |   |\n'
    for length in range(1, 5):
        ret += '''
|   |   |
|---|---|
'''
        to_sort = []
        for m, lr in assign.special.items():
            if len(m) != length:
                continue
            if m in SYMBOLS:
                continue
            to_sort.append((m, lr))
        for m, lr in sorted(to_sort, key=lambda x: x[0]):
            left = lr[0]
            left = '' if left == '_' else left
            ret += f'|{m}|{left}{lr[1]}|\n'
        ret += '|   |   |\n'

    ret += '''
|   |   |
|---|---|
'''
    to_sort = []
    for s in SYMBOLS:
        if s not in assign.special:
            continue
        lr = assign.special[s]
        to_sort.append((s, lr))
    for s, lr in sorted(to_sort, key=lambda x: x[0]):
        left = lr[0]
        left = '' if left == '_' else left
        ret += f'|{s}|{left}{lr[1]}|\n'

    ret += '|   |   |\n'
    return ret


def format_roman_table(assign: Assign):
    import unicodedata
    from risc_kana_layout import DATA_DIR
    ent = ''
    with open(DATA_DIR / 'google_romantable.txt', 'r', encoding='utf-8') as f:
        for li in f.readlines():
            es = li.strip().split('\t')
            if len(es) == 2:
                ks, kana = es
                if (ks[0] == 'z' and 'HIRAGANA' not in unicodedata.name(kana[0])) or (not ks[0].isalpha()):
                    ent += li
                else:
                    ent += f'b{ks}\t{kana}\n'
                    ent += f'y{ks}\t{kana}\ty\n'
            elif len(es) == 3:
                ks, kana, append = es
                ent += f'b{ks}\t{kana}\tb{append}\n'
                ent += f'y{ks}\t{kana}\ty{append}\n'
            else:
                raise Exception()

    for i, r in enumerate(assign.regular):
        for ii, a in enumerate(r):
            if a is None:
                continue
            left = a[0]
            left = '' if left == '_' else left
            ent += f'{left}{a[1]}\t{TABLE[i][ii]}\n'
    for m, lr in assign.special.items():
        left = lr[0]
        left = '' if left == '_' else left
        ent += f'{left}{lr[1]}\t{m}\n'

    return ent


def format_assign_list_asc(assign: Assign):
    d = {}
    for i, r in enumerate(assign.regular):
        for ii, a in enumerate(r):
            if a is None:
                continue
            left = a[0]
            left = '' if left == '_' else left
            d[TABLE[i][ii]] = f'{left}{a[1]}'
    for m, lr in assign.special.items():
        left = lr[0]
        left = '' if left == '_' else left
        d[m] = f'{left}{lr[1]}'
    ret = '''\n
|   |   |
|---|---|
'''
    for k in sorted(d.keys(), key=lambda x: d[x]):
        ret += f'|{d[k]}|{k}|\n'

    ret += '|   |   |\n'

    return ret
