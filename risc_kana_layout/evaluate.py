from dataclasses import dataclass
from collections.abc import Iterable
from collections import defaultdict
import pathlib
import io
from jaconv import kata2hira, hira2kata
import numpy as np
from random import sample
from risc_kana_layout import separate_by_moras, INITIAL_MORAS, SHORTHANDS, SYMBOLS


MAX_TRS_COST = 7
MAG_TRS_CLASS = 1
LEFT_KEYS = list('qwertasdfgzxcvb')
RIGHT_KEYS = list('yuiophjklnm,./[]-()')


def sample_corpus(corpus_dir: pathlib.Path, min_length: int):
    from risc_kana_layout.constant import KANA_TXT
    with open(corpus_dir / KANA_TXT, 'r', encoding='utf-8') as f:
        n_line = len(f.readlines())
        f.seek(0)
        while True:
            i_samples = sample(range(n_line), n_line // 10)
            for i, li in enumerate(f.readlines()):
                if i in i_samples:
                    t = li.strip()
                    if len(t) >= min_length:
                        yield t
            f.seek(0)


@dataclass
class EvalSepResult:
    trs_costs: dict[int, int]
    n_trs: int
    n_left_hit: int
    n_right_hit: int
    left_idle_costs: dict[int, int]
    right_idle_costs: dict[int, int]
    left_run: dict[int, int]
    right_run: dict[int, int]
    stroke: str


def get_solution_assign(solution_filename: str):
    assign: dict[str, tuple[str, str]] = {}
    with open(solution_filename, 'r', encoding='utf-8') as f:
        for li in f.readlines():
            ks, m = li.strip().split('\t')
            left, right = ks.split(' ')
            assign[m] = (left, right)
    return assign


ROMAN_SUPP_TABLE: dict[str, list[tuple[str, str]]] = {
    'ァ': [('b', 'l'), ('a', '_')],
    'ィ': [('b', 'li')],
    'ゥ': [('b', 'lu')],
    'ェ': [('b', 'l'), ('e', '_')],
    'ォ': [('b', 'lo')],
    'ファ': [('bfa', '_')],
    'フォ': [('bf', 'o')],
    'フィ': [('bf', 'i')],
    'ジェ': [('b', 'j'), ('e', '_')],
    'ウェ': [('bwe', '_')],
    'フェ': [('bfe', '_')],
    'ヴィ': [('bv', 'i')],
    'ウィ': [('bw', 'i')],
    'ヴァ': [('bva', '_')],
    'ヴェ': [('bve', '_')],
    'ヴ': [('bv', 'u')],
    'ヴォ': [('bv', 'o')],
    'ヂ': [('bd', 'i')],
}


def get_mora_table(solution_filename: str):
    assign = get_solution_assign(solution_filename)
    moras = [hira2kata(k) for k in (INITIAL_MORAS + SHORTHANDS + SYMBOLS)]
    mora_table: dict[str, list[tuple[str, str]]] = {}
    for m in moras:
        if m in assign:
            mora_table[m] = [assign[m]]
        elif m in ROMAN_SUPP_TABLE:
            mora_table[m] = ROMAN_SUPP_TABLE[m]
        else:
            raise Exception('never')
    return mora_table


def to_keypairs(raw_keystroke: str):
    keypairs: list[tuple[str, str]] = []
    kp = ['_', '_']
    for k in raw_keystroke:
        if k in LEFT_KEYS:
            if kp[1] == '_':
                if kp[0] == '_':
                    kp[0] = k
                else:
                    kp[0] += k
            else:
                keypairs.append((kp[0], kp[1]))
                kp = [k, '_']
        else:
            if kp[1] == '_':
                kp[1] = k
            else:
                kp[1] += k
    keypairs.append((kp[0], kp[1]))
    return keypairs


def get_roman_mora_table():
    from risc_kana_layout import DATA_DIR
    mora_table: dict[str, list[tuple[str, str]]] = {}

    with open(DATA_DIR / 'basic_romantable.txt', 'r', encoding='utf-8') as f:
        for li in f.readlines():
            ks, m = li.strip().split('\t')
            m = hira2kata(m)
            if m != kata2hira(m) and m != 'ッ' and ks[0] not in 'lnaiueo':
                mora_table['ッ' + m] = to_keypairs(ks[0] + ks)
            mora_table[m] = to_keypairs(ks)
    return mora_table


def eval_sep(sep: list[str], mora_table: dict[str, list[tuple[str, str]]]):
    '''
    最初のモーラの押し下げはカウントされない
    '''
    keystroke = ''
    for m in sep:
        for left, right in mora_table[m]:
            if left != '_':
                keystroke += left
            if right != '_':
                keystroke += right

        keystroke += ' '  # モーラの切れ目

    return eval_ks(keystroke)[0]


def eval_ks(keystroke: str):
    '''
    $C$を計算する
    '''
    from risc_kana_layout.optimize import load_pair_score
    pair_score = load_pair_score(LEFT_KEYS, RIGHT_KEYS)
    left_run: dict[int, int] = defaultdict(int)
    right_run: dict[int, int] = defaultdict(int)
    n_left_hit = 0
    n_right_hit = 0
    lr = 0
    rr = 0
    is_first_mora = True
    for i, k in enumerate(keystroke):
        if k == ' ':
            if is_first_mora:
                is_first_mora = False
            continue
        elif is_first_mora:
            continue
        elif k in LEFT_KEYS:
            n_left_hit += 1
            lr += 1
            right_run[rr] += 1
            rr = 0
        else:
            n_right_hit += 1
            rr += 1
            left_run[lr] += 1
            lr = 0
    right_run[rr] += 1
    left_run[lr] += 1
    del lr, rr, i
    if 0 in left_run:
        del left_run[0]
    if 0 in right_run:
        del right_run[0]

    timeline_lr = [[], []]
    trs_costs: dict[int, int] = defaultdict(int)
    idle_costs_lr: list[dict[int, int]] = [defaultdict(int), defaultdict(int)]
    trs_cost = 0
    is_first_mora = True
    for i, k in enumerate(keystroke):
        if k == ' ':
            if is_first_mora:
                is_first_mora = False
            else:
                trs_costs[trs_cost] += 1
            trs_cost = 0
            continue

        lr, nlr = (0, 1) if k in LEFT_KEYS else (1, 0)

        wait_sync = False
        if len(timeline_lr[nlr]) > 0 and timeline_lr[nlr][-1] != '_':  # 同期コスト
            wait_sync = True
            timeline_lr[lr].append('_')
            timeline_lr[nlr].append('_')

        last_k = '_'
        paid_cost = 0
        for paid_cost, e in enumerate(timeline_lr[lr][::-1]):
            if e != '_':
                last_k = e
                break

        raw_trs_cost = (0 if last_k == '_' else pair_score[frozenset([last_k, k])])

        mismatch_cost = raw_trs_cost - paid_cost
        for _ in range(mismatch_cost):
            timeline_lr[lr].append('_')
            timeline_lr[nlr].append('_')
        if last_k != '_':
            if mismatch_cost < 0:
                idle_costs_lr[lr][-mismatch_cost] += 1
            elif not all(k == '_' for k in timeline_lr[nlr][-paid_cost - 1]):
                idle_costs_lr[lr][0] += 1

        trs_cost += max(mismatch_cost, 0) + (1 if wait_sync else 0)

        timeline_lr[lr].append(k)
        timeline_lr[nlr].append('_')

    if 0 in trs_costs:
        del trs_costs[0]

    return EvalSepResult(
        trs_costs,
        sum(trs_costs.values()),
        n_left_hit,
        n_right_hit,
        idle_costs_lr[0],
        idle_costs_lr[1],
        left_run,
        right_run,
        ''.join([k for k in keystroke if k != ' '])
    ), [''.join(t) for t in timeline_lr]


@dataclass
class EvalFullResult:
    ave_cm: float
    ave_ck: float
    trs_stat: list[float]
    left_idle_ratio: list[float]
    right_idle_ratio: list[float]
    left_run_ratio: dict[int, float]
    right_run_ratio: dict[int, float]
    left_hit_ratio: float
    n_left_ratio: dict[int, float]
    n_right_ratio: dict[int, float]


def eval_mora_table_by_corpus(kana_reader: Iterable[str], mora_table: dict[str, list[tuple[str, str]]], n_sample: int):
    full_result: list[EvalFullResult] = []
    fragment_cm: list[list[float]] = []
    fragment_ck: list[list[float]] = []
    fragment_stroke_cost_bin: dict[float, set[str]] = {3.7: set(), 4.3: set(), 5.0: set()}
    for kana in kana_reader:
        full_esr: EvalSepResult | None = None
        for raw_sep, _ in separate_by_moras([kana], list(mora_table.keys())):
            sep: list[str] = []
            i = 0
            while len(sep) < 50 and i < len(raw_sep):
                sep.append(raw_sep[i])
                i += 1
            if len(sep) < 20:
                continue
            full_esr = eval_sep(sep, mora_table)

            fragment_cm_li: list[float] = []
            fragment_ck_li: list[float] = []
            for i in range(len(sep) - 4):
                fragment_sep = sep[i: i + 4]
                fragment_esr = eval_sep(fragment_sep, mora_table)
                fragment_trs_cost = sum(
                    i * c for i, c in fragment_esr.trs_costs.items()
                )
                fragment_cost = fragment_trs_cost + fragment_esr.n_left_hit + fragment_esr.n_right_hit
                ave_cm = fragment_cost / fragment_esr.n_trs
                ave_ck = fragment_cost / (len(fragment_sep) - 1)
                fragment_cm_li.append(ave_cm)
                fragment_ck_li.append(ave_ck)
                if fragment_esr.stroke.count('_') <= 2:
                    for k, v in fragment_stroke_cost_bin.items():
                        if k - 0.1 < ave_cm and ave_cm < k + 0.1:
                            v.add(fragment_esr.stroke)
                    del k, v
            del i, ave_cm, ave_ck, fragment_esr, fragment_cost, fragment_sep
            break
        del raw_sep, kana

        if full_esr is None:
            continue

        n_lefts: dict[int, int] = defaultdict(int)
        n_rights: dict[int, int] = defaultdict(int)
        for m in sep:
            if m not in mora_table:
                continue
            for left, right in mora_table[m]:
                n = len(left)
                if left == '_':
                    n = 0
                n_lefts[n] += 1
                n = len(right)
                if right == '_':
                    n = 0
                n_rights[n] += 1
                del n
            del left, right
        n_left_ratios: dict[int, float] = {k: v / sum(n_lefts.values()) for k, v in n_lefts.items()}
        n_right_ratios: dict[int, float] = {k: v / sum(n_rights.values()) for k, v in n_rights.items()}
        del m, n_lefts, n_rights

        full_trs_cost = sum([i * c for i, c in full_esr.trs_costs.items()])
        full_cost = full_trs_cost + full_esr.n_left_hit + full_esr.n_right_hit
        full_ave_cm = full_cost / full_esr.n_trs
        full_ave_ck = full_cost / (len(''.join(sep)) - 1)
        full_trs_stat = [
            full_esr.trs_costs[i] / full_esr.n_trs
            for i in range(1 + max(full_esr.trs_costs.keys())) if i in full_esr.trs_costs
        ]
        full_left_idle_ratios, full_right_idle_ratios = [
            [
                costs[i] / sum(costs.values()) for i in range(MAX_TRS_COST)
            ] + [
                sum([v / sum(costs.values()) for k, v in costs.items() if k >= MAX_TRS_COST])
            ]
            for costs in [full_esr.left_idle_costs, full_esr.right_idle_costs]
        ]
        full_left_run_ratios, full_right_run_ratios = [
            {k: v / sum(run.values()) for k, v in run.items()}
            for run in [full_esr.left_run, full_esr.right_run]
        ]
        full_result.append(EvalFullResult(
            full_ave_cm,
            full_ave_ck,
            full_trs_stat,
            full_left_idle_ratios,
            full_right_idle_ratios,
            full_left_run_ratios,
            full_right_run_ratios,
            full_esr.n_left_hit / (full_esr.n_left_hit + full_esr.n_right_hit),
            n_left_ratios,
            n_right_ratios
        ))

        fragment_cm_li_bin = [0.] * 4
        for i in range(4):
            s = sum(np.array(fragment_cm_li) >= (i + 1) * MAG_TRS_CLASS + full_ave_cm)
            fragment_cm_li_bin[i] = s / len(fragment_cm_li)
        fragment_cm.append(fragment_cm_li_bin)

        fragment_ck_li_bin = [0.] * 4
        for i in range(4):
            s = sum(np.array(fragment_ck_li) >= (i + 1) * MAG_TRS_CLASS + full_ave_ck)
            fragment_ck_li_bin[i] = s / len(fragment_ck_li)
        fragment_ck.append(fragment_ck_li_bin)

        del fragment_cm_li, fragment_ck_li, s

        if len(full_result) == n_sample:
            break

    for an in ['trs_stat', 'left_idle_ratio', 'right_idle_ratio']:
        lv = max(len(getattr(efr, an)) for efr in full_result)
        for efr in full_result:
            v = getattr(efr, an)
            v += [0.] * (lv - len(v))

    return sorted(full_result, key=lambda x: x.ave_cm), sorted(full_result, key=lambda x: x.ave_ck), np.array(fragment_cm), np.array(fragment_ck), fragment_stroke_cost_bin


N_SAMPLE_LARGE = 1000
N_SAMPLE_SMALL = 200


def eval_mora_table_and_save(mora_table: dict[str, list[tuple[str, str]]], filename_suffix: str, n_figs: list[str], layout: str):
    import random
    from risc_kana_layout.constant import WIKIPEDIA_DIR, CC100_DIR, MEIDAI_DIR, ARM_DIR

    random.seed(0)
    cm_wp, ck_wp, fcm_wp, fck_wp, sc_wp = eval_mora_table_by_corpus(
        sample_corpus(WIKIPEDIA_DIR, 50),
        mora_table,
        N_SAMPLE_LARGE
    )
    random.seed(0)
    cm_cc, ck_cc, fcm_cc, fck_cc, _ = eval_mora_table_by_corpus(
        sample_corpus(CC100_DIR, 50),
        mora_table,
        N_SAMPLE_LARGE
    )
    random.seed(0)
    cm_meidai, ck_meidai, fcm_meidai, fck_meidai, _ = eval_mora_table_by_corpus(
        sample_corpus(MEIDAI_DIR, 50),
        mora_table,
        N_SAMPLE_SMALL
    )
    random.seed(0)
    cm_arm, ck_arm, fcm_arm, fck_arm, _ = eval_mora_table_by_corpus(
        sample_corpus(ARM_DIR, 50),
        mora_table,
        N_SAMPLE_SMALL
    )
    cms = [cm_wp, cm_cc, cm_meidai, cm_arm]
    cks = [ck_wp, ck_cc, ck_meidai, ck_arm]
    fcms = [fcm_wp, fcm_cc, fcm_meidai, fcm_arm]
    labels = ["Wikipedia", "CC100", "名大コーパス", "Amazonレビュー"]

    with open(f'eval-md-{filename_suffix}.txt', 'w', encoding='utf-8') as f:
        s = f'$C_m^{{{layout}}}$の値に対応する4モーラのストロークの例：\n'
        for k, v in sc_wp.items():
            s += f'$C \\fallingdotseq {k}$:'
            i = 0
            for st in v:
                s += f' {st} '
                i += 1
                if i > 20:
                    break
            s += '\n'
        f.write(s)
        generate_eval_md(cms, cks, fcms, labels, f, layout)

    draw_eval_chart(cms, cks, fcms, labels, filename_suffix, n_figs, layout)


def generate_eval_md(
        cms: list[list[EvalFullResult]],
        cks: list[list[EvalFullResult]],
        fcms: list[np.ndarray],
        labels: list[str],
        f: io.TextIOWrapper,
        layout: str):

    cm_label = f'C_m^{{{layout}}}'
    ck_label = f'C_k^{{{layout}}}'

    def add_typical_row(acc, stat, lb, ub):
        acc.append(np.array(
            stat[lb * N_SAMPLE_LARGE // 100: ub * N_SAMPLE_LARGE // 100]
        ).mean())

    def add_typical_row_axis0(acc, stat, lb, ub):
        acc.append(np.array(
            stat[lb * N_SAMPLE_LARGE // 100: ub * N_SAMPLE_LARGE // 100]
        ).mean(axis=0))

    cm_wp, cm_cc, _, _ = cms
    header = '|'.join(labels)

    for an, sn, cs in zip(('ave_cm', 'ave_ck'), (cm_label, ck_label), (cms, cks)):
        s = f'''
無作為抽出した20-50モーラ区間の${sn}$の分布:

|パーセンタイル|{header}|
|---|---|---|---|---|
'''
        for section in [50, 65, 95]:
            s += f'|{section}%|'
            for c in cs:
                a = [getattr(efr, an) for efr in c]
                s += f'{a[len(c) * section // 100]:.2f}|'
            del c, a
            s += '\n'
        del section
        s += '|   |   |   |   |   |\n'
        f.write(s)
    del an, sn, cs

    for i in range(1, 3):
        s = f'''
4モーラ区間の${cm_label}$が親の20-50モーラ区間の${cm_label}$より{i * MAG_TRS_CLASS}以上大きい割合（%）の分布:

|パーセンタイル|{header}|
|---|---|---|---|---|
'''
        for section in [50, 65, 95]:
            s += f'|{section}%|'
            for cm in fcms:
                a = np.sort(cm[:, i - 1])
                s += f'{a[len(cm) * section // 100] * 100:.2f}|'
            del cm, a
            s += '\n'
        del section
        s += '|   |   |   |   |   |\n'
        f.write(s)
        del s
    del i

    for an, label in zip(['left_run_ratio', 'right_run_ratio'], ['左手', '右手']):
        s = f'''
連続打鍵数の頻度分布（{label}）（%）:

|連続打鍵数|Wikipedia 60-70%|Wikipedia 90-100%|CC100 60-70%|CC100 90-100%|
|---|---|---|---|---|
'''
        cm_ps = [(cm_wp, 60, 70), (cm_wp, 90, 100), (cm_cc, 60, 70), (cm_cc, 90, 100)]
        for n_hit in range(1, 3):
            acc = []
            for cm, lb, ub in cm_ps:
                stat = [
                    getattr(efr, an)[n_hit] if n_hit in getattr(efr, an) else 0.
                    for efr in cm
                ]
                add_typical_row(acc, stat, lb, ub)
            del cm, lb, ub
            s += f'|{n_hit}打|{acc[0] * 100:.1f}|{acc[1] * 100:.1f}|{acc[2] * 100:.1f}|{acc[3] * 100:.1f}|\n'
        del n_hit
        acc = []
        for cm, lb, ub in cm_ps:
            stat = []
            for efr in cm:
                raw_ratio: dict[int, float] = getattr(efr, an)
                residue = sum(r for n_hit, r in raw_ratio.items() if n_hit >= 3)
                stat.append(residue)
            del efr
            add_typical_row(acc, stat, lb, ub)
        del cm, lb, ub
        s += f'|3打以上|{acc[0] * 100:.1f}|{acc[1] * 100:.1f}|{acc[2] * 100:.1f}|{acc[3] * 100:.1f}|\n'
        del acc

        s += '|   |   |   |   |   |\n'
        f.write(s)
        del s
    del an, label

    s = '''
モーラ間の連接遷移コストの頻度分布（%）:

|遷移コスト|Wikipedia 60-70%|Wikipedia 90-100%|CC100 60-70%|CC100 90-100%|
|---|---|---|---|---|
'''
    cm_ps = [(cm_wp, 60, 70), (cm_wp, 90, 100), (cm_cc, 60, 70), (cm_cc, 90, 100)]
    acc = []
    for cm, lb, ub in cm_ps:
        stat = [efr.trs_stat for efr in cm]
        add_typical_row_axis0(acc, stat, lb, ub)
    del cm, lb, ub
    lv = max(a.shape[0] for a in acc)
    a = np.zeros((4, lv))
    for i, v in enumerate(acc):
        a[i, :v.shape[0]] = v
    for i in range(a.shape[1]):
        s += f'|{i + 1}|{a[0, i] * 100:.1f}|{a[1, i] * 100:.1f}|{a[2, i] * 100:.1f}|{a[3, i] * 100:.1f}|\n'
    del i, v, a, lv, acc
    s += '|   |   |   |   |   |\n'
    f.write(s)
    del s

    for hand, an in zip(['左手', '右手'], ['left_idle_ratio', 'right_idle_ratio']):
        s = f'''
{hand}が相手を待って遊んでいたコストの頻度分布（%）:

|コスト|Wikipedia 60-70%|Wikipedia 90-100%|CC100 60-70%|CC100 90-100%|
|---|---|---|---|---|
'''
        cm_ps = [(cm_wp, 60, 70), (cm_wp, 90, 100), (cm_cc, 60, 70), (cm_cc, 90, 100)]
        acc = []
        for cm, lb, ub in cm_ps:
            stat = [getattr(efr, an) for efr in cm]
            add_typical_row_axis0(acc, stat, lb, ub)
        for i in range(MAX_TRS_COST + 1):
            s += f'|{i if i < MAX_TRS_COST else f"{i}以上"}|{acc[0][i] * 100:.1f}|{acc[1][i] * 100:.1f}|{acc[2][i] * 100:.1f}|{acc[3][i] * 100:.1f}|\n'
        s += '|   |   |   |   |   |\n'
        f.write(s)
        del s


def draw_eval_chart(
        cms: list[list[EvalFullResult]],
        cks: list[list[EvalFullResult]],
        fcms: list[np.ndarray],
        labels: list[str],
        filename_suffix: str,
        n_figs: list[str],
        layout: str):

    import matplotlib.pyplot as plt
    import japanize_matplotlib  # noqa

    i_fig = 0
    for an, sn, cs in zip(('ave_cm', 'ave_ck'), ('C_m', 'C_k'), (cms, cks)):
        fig = plt.figure(
            figsize=(8., 8.),
            facecolor="w",
        )
        ax = fig.add_subplot(111, title=f'図{n_figs[i_fig]} 無作為抽出した20-50モーラ区間の${sn}^{{{layout}}}$の分布', xlabel='パーセンタイル', ylabel=f'${sn}$')
        i_fig += 1
        for c, label in zip(cs, labels):
            ax.plot(np.linspace(0, 100, len(c)), [getattr(r, an) for r in c], marker=None, label=label)
        ax.legend()
        fig.savefig(f'evaluate-{sn}-{filename_suffix}.png')

    for i in range(1, 3):
        fig = plt.figure(
            figsize=(8., 8.),
            facecolor="w",
        )
        ax = fig.add_subplot(111, title=f'図{n_figs[i_fig]}({["a", "b"][i - 1]}) 4モーラ区間の$C_m^{{{layout}}}$が親の20-50モーラ区間の$C_m^{{{layout}}}$より\n{i * MAG_TRS_CLASS}以上大きい割合の分布', xlabel='パーセンタイル', ylabel='%')
        i_fig += 1
        for short, label in zip(fcms, labels):
            ax.plot(np.linspace(0, 100, len(short)), np.sort(short[:, i - 1]) * 100, marker=None, label=label)
        ax.legend()
        fig.savefig(f'evaluate_ratio-{i}-{filename_suffix}.png')


def draw_heat_map(mora_table: dict[str, list[tuple[str, str]]]):
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import japanize_matplotlib  # noqa
    from risc_kana_layout import INITIAL_MORAS, SHORTHANDS, SYMBOLS
    from risc_kana_layout.optimize import load_mora_freq
    moras = INITIAL_MORAS + SHORTHANDS + SYMBOLS
    mora_freq = load_mora_freq(moras, 'ave')
    key_freq: dict[str, float] = defaultdict(float)

    for m, kpl in mora_table.items():
        f = mora_freq[moras.index(m)]
        for kslr in kpl:
            for ks in kslr:
                for k in ks:
                    if k != '_':
                        key_freq[k] += f
    s = sum(key_freq.values())
    key_freq = {k: v * 100 / s for k, v in key_freq.items()}

    rows = ['zxcvbnm,', 'asdfghjkl', 'qwertyuiop']
    row_offset = [3, 1, 0]

    def k2pos(k: str):
        for i, row in enumerate(rows):
            if k in row:
                x = row.index(k)
                return x * 4 + row_offset[i], (2 - i) * 4
        raise Exception()

    matrix = np.zeros((3 * 4, 10 * 4))
    for k, f in key_freq.items():
        x, y = k2pos(k)
        matrix[y: y + 4, x:x + 4] = f

    fig = plt.figure(
        figsize=(8., 4.),
        facecolor="w",
    )
    ax = fig.add_subplot(1, 1, 1)
    c_matrix = ax.imshow(matrix)
    fig.colorbar(c_matrix, label="全打鍵中に占める割合（%）", ticks=mticker.MaxNLocator(nbins=3), orientation='horizontal')
    plt.xticks([])
    plt.yticks([])

    ax.set_xlabel("QWERTYUIOP/ASDFGHJKL/ZXCVBNM,")

    plt.savefig('heat_map.png')
