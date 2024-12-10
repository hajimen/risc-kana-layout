from collections.abc import Iterable
import pathlib
import shutil
from risc_kana_layout.constant import DATA_DIR, INITIAL_MORA_TXT, SHORTHAND_TXT, SYMBOL_TXT, AVE


MAX_SECONDS = 600
OPT_COMPLETE_TXT = 'opt_third_complete.txt'


for txt_file in [INITIAL_MORA_TXT, SHORTHAND_TXT, SYMBOL_TXT]:
    if not pathlib.Path(txt_file).exists():
        shutil.copy(DATA_DIR / txt_file, txt_file)
del txt_file


def _read_initial_moras():
    import jaconv
    initial_moras = []
    with open(INITIAL_MORA_TXT, 'r', encoding='utf-8') as v:
        for li in v.readlines():
            m = jaconv.hira2kata(li.strip())
            initial_moras.append(m)
    return initial_moras


INITIAL_MORAS: list[str] = _read_initial_moras()
del _read_initial_moras


def _read_shorthands():
    import jaconv
    shorthands = []
    with open(SHORTHAND_TXT, 'r', encoding='utf-8') as v:
        for li in v.readlines():
            m = jaconv.hira2kata(li.strip())
            shorthands.append(m)
    return shorthands


SHORTHANDS: list[str] = _read_shorthands()
del _read_shorthands


def _read_symbols():
    symbols = []
    with open(SYMBOL_TXT, 'r', encoding='utf-8') as v:
        for li in v.readlines():
            symbols.append(li.strip())
    return symbols


SYMBOLS: list[str] = _read_symbols()
del _read_symbols

_KKS = None


def to_kana(reader: Iterable[str]):
    global _KKS
    import pykakasi
    if _KKS is None:
        _KKS = pykakasi.kakasi()

    for t in reader:
        kana = ''.join([x['kana'] for x in _KKS.convert(t.strip())])
        yield kana


def separate_by_moras(kana_reader: Iterable[str], moras: list[str]):
    mora_c = {m: 0 for m in moras}
    max_mora_length = max([len(m) for m in moras])
    mora_len = {i + 1: list(filter(lambda x: len(x) == i + 1, moras)) for i in range(max_mora_length)}
    for li in kana_reader:
        nl: list[str] = []
        cpos = 0
        skip_to = 0
        while cpos < len(li):
            rest = len(li) - cpos
            ri = 0
            while ri + 1 < min(max_mora_length, rest):
                i = min(max_mora_length, rest) - ri - 1
                skip = False
                ms = mora_len[i + 1]
                for ii in range(1, min(i, rest)):
                    if li[cpos + ii: cpos + i + ii + 1] in ms:
                        if skip_to < cpos + i + ii + 1:
                            skip_to = cpos + i + ii + 1
                            rest = ii
                            skip = True
                            break
                if skip:
                    break
                ri += 1
            hit = False
            for i in reversed(range(min(max_mora_length, rest))):
                ms = mora_len[i + 1]
                try:
                    im = ms.index(li[cpos: cpos + i + 1])
                    mora_c[ms[im]] += 1
                    nl.append(ms[im])
                    cpos += i + 1
                    hit = True
                    break
                except ValueError:
                    pass
            if not hit:
                if len(nl) > 0:
                    yield nl, mora_c
                    nl = []
                cpos += 1
        if len(nl) > 0:
            yield nl, mora_c


def read_corpus(corpus_dir: pathlib.Path):
    from risc_kana_layout.constant import KANA_TXT
    with open(corpus_dir / KANA_TXT, 'r', encoding='utf-8') as f:
        yield from f.readlines()


def calc_save_markov_all():
    from risc_kana_layout import constant
    from risc_kana_layout.calc_stat import calc_save_markov, ave_markov
    
    for d, s in zip([constant.MEIDAI_DIR, constant.ARM_DIR, constant.CC100_DIR, constant.WIKIPEDIA_DIR],
                    [constant.MEIDAI, constant.ARM, constant.CC100, constant.WIKIPEDIA]):
        kana_reader = read_corpus(d)
        calc_save_markov(kana_reader, s)

    ave_markov(constant.CC100, constant.WIKIPEDIA, constant.AVE)


def _logging_opt(result_filename: str, status: str, moras: list[str], n_fix_mora: int, n_mora_pair: int):
    from risc_kana_layout.constant import OPT_LOG, AVE
    from risc_kana_layout.optimize import build_opt_model
    opt_model = build_opt_model(moras, AVE, n_mora_pair, result_filename, fix_all=True)
    opt_model.model.optimize()
    cost = opt_model.model.objective_value
    assert cost is not None
    with open(OPT_LOG, 'a', encoding='utf-8') as f:
        f.write(f'{result_filename}\t{cost:.4g}\t{status}\t{n_fix_mora}\t{n_mora_pair}\n')


def _prepare_moras(markov_filename_suffix: str, n_moras: int):
    from risc_kana_layout.optimize import load_mora_freq
    moras = INITIAL_MORAS + SHORTHANDS + SYMBOLS
    mora_freq = load_mora_freq(moras, markov_filename_suffix)
    moras_sorted = sorted(zip(moras, mora_freq), key=lambda x: x[1], reverse=True)
    return list(map(lambda x: x[0], moras_sorted[:n_moras]))


def example_opt():
    '''
    上位10モーラで厳密解を得て、線形緩和解と比較する
    '''
    from risc_kana_layout.optimize import build_opt_model, do_optimize
    from risc_kana_layout import constant
    n_mora_pair = 100
    moras = _prepare_moras(constant.AVE, 10)
    left_keys = [k for k in constant.LEFT_KEYS if k not in 'ertgx'] + ['_']
    right_keys = [k for k in constant.RIGHT_KEYS if k not in 'uh,']
    opt_model = build_opt_model(moras, constant.AVE, n_mora_pair, None, [], left_keys, right_keys)
    _, status, _ = do_optimize(
        opt_model,
        moras,
        result_filename='min_opt.txt')
    optimized_cost = opt_model.model.objective_value
    print(f'optimized_cost: {optimized_cost}')
    assert status == 'OPTIMAL'

    opt_model = build_opt_model(moras, constant.AVE, n_mora_pair, None, [], left_keys, right_keys)
    model = opt_model.model
    model.optimize(relax=True)
    relaxed_solution = model.objective_value
    assert relaxed_solution is not None
    print(f'relaxed solution: {relaxed_solution}')
    print(f'Minimize(Q_10) - Relaxed: {optimized_cost - relaxed_solution}')


def initial_opt():
    '''
    最初の初期解を得る
    '''
    from risc_kana_layout.optimize import build_opt_model, do_optimize
    from risc_kana_layout import constant
    n_mora_pair = 400
    n_fix_mora = 3
    moras = _prepare_moras(constant.AVE, 40)
    left_keys = [k for k in constant.LEFT_KEYS if k not in 'ertgx'] + ['_']
    right_keys = [k for k in constant.RIGHT_KEYS if k not in 'h']
    opt_model = build_opt_model(
        moras,
        constant.AVE,
        n_mora_pair,
        DATA_DIR / 'opt_start.txt',
        moras[:n_fix_mora] + ['ッ', '。'],
        left_keys,
        right_keys)
    _, status, _ = do_optimize(
        opt_model,
        moras,
        max_seconds_same_incumbent=1000,
        result_filename='initial_opt.txt')
    _logging_opt('initial_opt.txt', status, moras, n_fix_mora, n_mora_pair)


def _opt_iteration(
        result_filename: str,
        last_result_filename: str,
        n_mora: int,
        n_fix_mora: int,
        n_mora_pair: int,
        left_tkns: list[str] | None = None,
        right_keys: list[str] | None = None,
        max_seconds_same_incumbent=0,
        max_seconds=0,
        exclude_moras: list[str] | None = None):
    '''
    初期解からの改善を試みる
    '''
    from risc_kana_layout.optimize import build_opt_model, do_optimize
    from risc_kana_layout import constant
    moras = _prepare_moras(constant.AVE, n_mora)
    if exclude_moras is not None:
        moras = [m for m in moras if m not in exclude_moras]
    if left_tkns is None:
        left_tkns = constant.LEFT_TKNS
    if right_keys is None:
        right_keys = constant.RIGHT_KEYS
    opt_model = build_opt_model(
        moras,
        constant.AVE,
        n_mora_pair,
        last_result_filename,
        list(set(moras[:n_fix_mora] + ['ッ', '。'])),
        left_tkns,
        right_keys)
    kwargs = {}
    if max_seconds != 0:
        kwargs['max_seconds'] = max_seconds
    if max_seconds_same_incumbent != 0:
        kwargs['max_seconds_same_incumbent'] = max_seconds_same_incumbent
    _, status, result = do_optimize(opt_model, moras, result_filename=result_filename, **kwargs)
    _logging_opt(result_filename, status, moras, n_fix_mora, n_mora_pair)
    return status, result


def opt_first_10(c=0, n_fix_mora=4):
    import numpy as np
    import os
    last_filename = 'initial_opt.txt' if c == 0 else f'opt_first10_{c}.txt'
    from risc_kana_layout import constant
    left_keys = [k for k in constant.LEFT_KEYS if k not in 'rtgx'] + ['_']
    right_keys = [k for k in constant.RIGHT_KEYS if k not in 'h']
    n_mora_pair = 600
    last_result = None
    while n_fix_mora <= 10:
        filename = f'opt_first10_{c + 1}.txt'
        status, result = _opt_iteration(
            filename,
            last_filename,
            40, n_fix_mora, n_mora_pair, left_keys, right_keys,
            max_seconds_same_incumbent=3000)
        if last_result is not None and np.any(last_result != result):
            if n_fix_mora > 3:
                n_fix_mora -= 1
            last_result = result
        elif status == 'OPTIMAL':
            if n_mora_pair >= 1600:
                break
            n_mora_pair += 200
            last_result = None
        else:
            n_fix_mora += 2
            last_result = result
        last_result = result
        last_filename = filename
        c += 1
    p = pathlib.Path('opt_first10.txt')
    if p.exists():
        p.unlink()
    os.rename(f'opt_first10_{c}.txt', p)


def opt_second_80(c=0, n_fix_mora=10, n_mora_pair=800, last_filename='opt_first10.txt'):
    import os
    while n_fix_mora <= 80 and n_mora_pair <= 4_000:
        filename = f'opt_second80_{c + 1}.txt'
        status, _ = _opt_iteration(
            filename,
            last_filename,
            100, n_fix_mora, n_mora_pair,
            max_seconds=MAX_SECONDS * 2)
        if status == 'OPTIMAL':
            n_mora_pair += 500
        else:
            n_fix_mora += 10
        last_filename = filename
        c += 1
    p = pathlib.Path('opt_second80.txt')
    if p.exists():
        p.unlink()
    os.rename(last_filename, p)


def opt_third_complete():
    from jaconv import hira2kata
    exclude_moras = ['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'ふぁ', 'ふぉ', 'ふぃ', 'じぇ', 'うぇ', 'ふぇ', 'ゔぃ', 'うぃ', 'ゔぁ', 'ゔぇ', 'ゔ', 'ゔぉ', 'ぢ']
    exclude_moras = [hira2kata(m) for m in exclude_moras]
    _, _ = _opt_iteration(
        'opt_third_1.txt',
        'opt_second80.txt',
        120, 90, 4_000,
        max_seconds=MAX_SECONDS * 2,
        exclude_moras=exclude_moras)
    _, _ = _opt_iteration(
        OPT_COMPLETE_TXT,
        'opt_third_1.txt',
        200, 100, 4_000,
        max_seconds=MAX_SECONDS * 2,
        exclude_moras=exclude_moras)


def optimize_all():
    initial_opt()
    opt_first_10()
    opt_second_80()
    opt_third_complete()


def calc_relaxed_solution():
    '''
    Minimize(Q)とその線形緩和解を得て比較する
    '''
    import os
    from risc_kana_layout.optimize import build_opt_model
    from risc_kana_layout import constant
    moras = _prepare_moras(constant.AVE, 200)

    opt_model = build_opt_model(moras, constant.AVE, 4_000, OPT_COMPLETE_TXT, fix_all=True)
    model = opt_model.model
    model.optimize(relax=True)
    optimized_cost = model.objective_value
    assert optimized_cost is not None
    print(f'Minimize(Q): {optimized_cost}')

    opt_model = build_opt_model(moras, constant.AVE, 4000)
    model = opt_model.model
    model.threads = os.cpu_count() - 1  # type: ignore
    model.optimize(relax=True)
    relaxed_solution = model.objective_value
    assert relaxed_solution is not None
    print(f'relaxed solution: {relaxed_solution}')
    print(f'Minimize(Q) - Relaxed: {optimized_cost - relaxed_solution}')


def get_sorted_mora(suffix=AVE):
    import pickle
    with open(f'markov_{suffix}.pkl', 'rb') as f:
        _, mora_freq = pickle.load(f)

    mf = [(k, v) for k, v in mora_freq.items()]
    mf = sorted(mf, key=lambda x: x[1], reverse=True)
    sorted_mora: list[str] = [k for k, _ in mf]

    return sorted_mora, mf


def calc_mora_freq(s='ave'):
    import jaconv
    with open(f'mora_freq_{s}.md', 'w', encoding='utf-8') as f:
        _, mf = get_sorted_mora(s)
        ss = 0.
        f.write('|モーラ|使用頻度|累積使用頻度|1 - 累積使用頻度|\n|--|--|--|--|\n')
        for e in mf:
            ss += e[1]
            f.write(f'|{jaconv.kata2hira(e[0])}|{e[1]:.3g}|{ss:.3g}|{1. - ss:.3g}|\n')


def eval_format_all():
    from risc_kana_layout.evaluate import eval_mora_table_and_save, draw_heat_map, get_mora_table, get_roman_mora_table

    risc_mora_table = get_mora_table(OPT_COMPLETE_TXT)
    roman_mora_table = get_roman_mora_table()

    eval_mora_table_and_save(risc_mora_table, 'risc', ['3', '1', '4', '4'], 'risc')

    eval_mora_table_and_save(roman_mora_table, 'roman', ['x', '2', '5', '5'], 'roman')

    draw_heat_map(risc_mora_table)

    from risc_kana_layout.format import format_markdown, read_solution, format_roman_table

    s = format_roman_table(read_solution(OPT_COMPLETE_TXT))
    with open('risc-romantable.txt', 'w', encoding='utf-8') as f:
        f.write(s)

    s = format_markdown(read_solution(OPT_COMPLETE_TXT))
    with open('risc-md.txt', 'w', encoding='utf-8') as f:
        f.write(s)
