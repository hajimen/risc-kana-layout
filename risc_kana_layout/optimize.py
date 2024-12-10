from dataclasses import dataclass
from itertools import product, combinations, chain
import pickle
import os
import numpy as np
import mip
from risc_kana_layout.constant import LEFT_KEYS, LEFT_TKNS, RIGHT_KEYS, MARKOV_BEGIN, MARKOV_END, TRS_SCORE_TXT


#########################################
# Score and Markov Model
#########################################

def load_pair_score(
        left_keys: list[str] = LEFT_KEYS,
        right_keys: list[str] = RIGHT_KEYS,
        trs_score_txt: str = TRS_SCORE_TXT):
    pair_score = {
        frozenset(k): 5
        for k in chain(combinations(left_keys, 2), combinations(right_keys, 2))
    }

    for k in left_keys + right_keys:
        pair_score[frozenset([k, k])] = 2

    with open(trs_score_txt, 'r') as f:
        k0 = ''
        for li in f.readlines():
            raw_k0, k1, str_score = li.split('\t')
            if len(raw_k0) > 0:
                k0 = raw_k0
            else:
                score = int(str_score.strip())
                pair_score[frozenset([k0, k1])] = score
    return pair_score


PAIR_SCORE = load_pair_score()


def load_mora_freq(
        moras: list[str],
        markov_filename_suffix: str):
    mora_freq_dict: dict[str, float]
    with open(f'markov_{markov_filename_suffix}.pkl', 'rb') as f:
        _, mora_freq_dict = pickle.load(f)
    return np.array([mora_freq_dict[m] for m in moras])


def load_score_freq(
        moras: list[str],
        markov_filename_suffix: str,
        left_tkns: list[str] = LEFT_TKNS,  # tkns means twin, key, no-hit
        right_keys: list[str] = RIGHT_KEYS,
        pair_score: dict[frozenset[str], int] = PAIR_SCORE):

    def _calc_trs_score(tkns: list[str], is_left: bool):
        trs_score = np.zeros((len(tkns),) * 2, dtype=np.float64)
        for k0i, k1i in product(enumerate(tkns), repeat=2):
            k0 = k0i[1][-1]
            k1 = k1i[1][0]
            if k1 == '_':
                s = 0  # strategy 6
            elif k0 == '_':
                s = 2.7  # strategy 7.1
            else:
                s = pair_score[frozenset([k0, k1])]
                if is_left:
                    s += {1: 1.8, 2: 0.9, 3: 0, 4: 0, 5: 0}[s]  # strategy 7.2
            if 'fj' in k0 or 'fj' in k1:
                s -= 0.01  # strategy 5
            if 'xcvm' in k0 or 'xcvm' in k1:
                s += 0.01  # strategy 5
            if len(k1i[1]) == 2:
                s += 2  # strategy 8
            trs_score[k0i[0], k1i[0]] = s
        return trs_score

    left_trs_score = _calc_trs_score(left_tkns, True)
    right_trs_score = _calc_trs_score(right_keys, False)

    markov_model: dict[str, dict[str, float]]
    with open(f'markov_{markov_filename_suffix}.pkl', 'rb') as f:
        markov_model, raw_mora_freq = pickle.load(f)

    def _screen_mora_pairs(n_mora_pair: int, fixed_moras: list[str]):
        idx_fixed_moras = [moras.index(m) for m in fixed_moras]
        mora_freq = np.zeros(len(moras))
        mbs = moras + [MARKOV_BEGIN, MARKOV_END]
        pairs = []

        sum_f_be = 0.
        sum_mf = 0.
        for m0, d in markov_model.items():
            for m1, f in d.items():
                if m0 == MARKOV_BEGIN or m1 == MARKOV_END:
                    sum_f_be += f
                    sum_mf += f / 2
                else:
                    sum_mf += f
                if m1 in mbs and m0 in mbs:
                    pairs.append((m0, m1, f))

        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        mora_pairs: list[tuple[int, int, float]] = []

        for m0, m1, f in pairs:
            if m0 == MARKOV_BEGIN:
                mora_freq[moras.index(m1)] += f / 2
                continue
            if m1 == MARKOV_END:
                mora_freq[moras.index(m0)] += f / 2
                continue
            mora_freq[moras.index(m0)] += f / 2
            mora_freq[moras.index(m1)] += f / 2
            mora_pairs.append((moras.index(m0), moras.index(m1), f / (1. - sum_f_be)))
        
            if len(mora_pairs) == n_mora_pair:
                break

        mora_freq = mora_freq / sum_mf
        mora_pairs = [p for p in mora_pairs if not (p[0] in idx_fixed_moras and p[1] in idx_fixed_moras)]
        for i in idx_fixed_moras:
            mora_freq[i] = 0.

        return mora_pairs, mora_freq

    return left_trs_score, right_trs_score, raw_mora_freq, _screen_mora_pairs


#########################################
# Optimization Model
#########################################


@dataclass
class OptModel:
    model: mip.Model
    assign: mip.LinExprTensor
    left_tkns: list[str]
    right_keys: list[str]
    mora_freq: np.ndarray


def build_opt_model(
        moras: list[str],
        markov_filename_suffix: str,
        n_mora_pair: int,
        initial_solution_filename: str | os.PathLike | None = None,
        fixed_moras: list[str] = [],
        left_tkns: list[str] = LEFT_TKNS,
        right_keys: list[str] = RIGHT_KEYS,
        pair_score: dict[frozenset[str], int] = PAIR_SCORE,
        fix_all=False):
    left_trs_score, right_trs_score, raw_mora_freq, screen_mora_pairs = load_score_freq(
        moras,
        markov_filename_suffix,
        left_tkns,
        right_keys,
        pair_score)

    mora_pairs, mod_mora_freq = screen_mora_pairs(n_mora_pair, fixed_moras)

    model = mip.Model()

    assign = model.add_var_tensor(
        (len(moras), len(left_tkns), len(right_keys)),
        name='assign',
        var_type=mip.BINARY)

    for ic in product(range(len(left_tkns)), range(len(right_keys))):
        model.add_constr(mip.xsum(assign[:, *ic].flatten()) <= 1)  # type: ignore

    if ',' in right_keys and '、' in moras:
        irs_non_comma = [i for i, k in enumerate(right_keys) if k != ',']
        ir_comma = right_keys.index(',')
        for im in range(len(moras)):
            if moras[im] != '、':
                model.add_constr(mip.xsum(assign[im, :, irs_non_comma].flatten()) == 1.)
                for a in assign[im, :, ir_comma]:
                    model.add_constr(a == 0.)
        im_comma = moras.index('、')
        model.add_constr(assign[im_comma, left_tkns.index('_'), ir_comma] == 1.)
        for a in assign[im_comma, [i for i, k in enumerate(left_tkns) if k != '_'], ir_comma]:
            model.add_constr(a == 0.)
    else:
        for im in range(len(moras)):
            model.add_constr(mip.xsum(assign[im].flatten()) == 1.)

    if initial_solution_filename is not None:
        with open(initial_solution_filename, 'r', encoding='utf-8') as f:
            start = np.zeros((assign.shape))
            for li in f.readlines():
                ks, m = li.strip().split('\t')
                left, right = ks.split(' ')
                i_left = left_tkns.index(left)
                i_right = right_keys.index(right)
                im = moras.index(m)
                start[im, i_left, i_right] = 1.
                if (m in fixed_moras or fix_all) and m != '、':
                    model.add_constr(assign[im, i_left, i_right] == 1.)
        start_as_list = []
        for z1 in zip(start, assign):
            for z2 in zip(*z1):
                for ini_e, assign_e in zip(*z2):
                    start_as_list.append((assign_e, ini_e))
        model.start = start_as_list

    m_trs_left = model.add_var_tensor(
        (len(mora_pairs), len(left_tkns), len(left_tkns)),
        name='m_trs_left')

    m_trs_right = model.add_var_tensor(
        (len(mora_pairs), len(right_keys), len(right_keys)),
        name='m_trs_right')

    m_trs_cost = model.add_var_tensor(
        (len(mora_pairs), ),
        name='m_trs_cost')

    for i_mp in range(len(mora_pairs)):
        im0, im1, _ = mora_pairs[i_mp]

        for m0l, m1l, a0l, a1l in zip(
                m_trs_left[i_mp].sum(axis=1),
                m_trs_left[i_mp].sum(axis=0),
                assign[im0, :, :].sum(axis=1),
                assign[im1, :, :].sum(axis=1)):
            model.add_constr(m0l == a0l)
            model.add_constr(m1l == a1l)

        for m0r, m1r, a0r, a1r in zip(
                m_trs_right[i_mp].sum(axis=1),
                m_trs_right[i_mp].sum(axis=0),
                assign[im0, :, :].sum(axis=0),
                assign[im1, :, :].sum(axis=0)):
            model.add_constr(m0r == a0r)
            model.add_constr(m1r == a1r)

        model.add_constr(m_trs_cost[i_mp] >= mip.xsum(m_trs_left[i_mp].flatten() * left_trs_score.flatten()))  # type: ignore
        model.add_constr(m_trs_cost[i_mp] >= mip.xsum(m_trs_right[i_mp].flatten() * right_trs_score.flatten()))  # type: ignore

    objs = []

    objs.append(- mip.xsum(
        (assign[:, -1, :].sum(axis=1).T @ mod_mora_freq).flatten()  # no hit makes cost lower
    ))

    mora_pair_freq = np.array([f for _, _, f in mora_pairs])
    objs.append(mip.xsum(c * f for c, f in zip(m_trs_cost, mora_pair_freq) if f > 0.))

    if len(objs) > 1:
        model.objective = mip.minimize(mip.xsum(objs))
    else:
        model.objective = mip.minimize(objs[0])

    return OptModel(model, assign, left_tkns, right_keys, raw_mora_freq)


_to_float = np.frompyfunc(lambda x: float(x.x), 1, 1)


def do_optimize(
        opt_model: OptModel,
        moras: list[str],
        max_seconds_same_incumbent=mip.INF,
        max_seconds=mip.INF,
        result_filename: str | os.PathLike = ''):
    model = opt_model.model
    cc = os.cpu_count()
    model.threads = 1 if cc is None else max(cc - 2, 1)  # -2 looks good empirically
    opt = model.optimize(
        max_seconds_same_incumbent=max_seconds_same_incumbent,  # type: ignore
        max_seconds=max_seconds)  # type: ignore
    if opt.name == 'INFEASIBLE':
        raise Exception('The model is infeasible')
    result = _to_float(opt_model.assign).astype(np.float64)
    with open(result_filename, 'w', encoding='utf-8') as f:
        d = []
        for ic in product(range(result.shape[1]), range(result.shape[2])):
            for im, m in enumerate(moras):
                if result[im][ic] > 0.1:
                    d.append((opt_model.mora_freq[m], f'{opt_model.left_tkns[ic[0]]} {opt_model.right_keys[ic[1]]}\t{m}\n'))
                    break
        for _, s in sorted(d, key=lambda x: x[0], reverse=True):
            f.write(s)
    print(f'objective: {model.objective_value}')

    return float(0. if model.objective_value is None else model.objective_value), opt.name, result
