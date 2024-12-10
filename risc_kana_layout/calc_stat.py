from collections import defaultdict
from collections.abc import Iterable
import pickle
import markovify
from risc_kana_layout import separate_by_moras, INITIAL_MORAS, SHORTHANDS, SYMBOLS
from risc_kana_layout.constant import MARKOV_BEGIN


def to_raw_markov(kana_reader: Iterable[str], moras: list[str]):
    markov_model_raw: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    mora_count: dict[str, int] = {}
    begin_count = 0

    def parsed_sentences(gen):
        nonlocal mora_count
        for parsed, _mora_c in gen:
            mora_count = _mora_c
            yield parsed

    for ps in parsed_sentences(separate_by_moras(kana_reader, moras)):
        t_s1 = markovify.Text(None, state_size=1, parsed_sentences=[ps])
        begin_count += 1
        for (m0,), d in t_s1.chain.model.items():
            for m1, c in d.items():  # type: ignore
                markov_model_raw[m0][m1] += c

    return markov_model_raw, mora_count, begin_count


def calc_save_markov(kana_reader: Iterable[str], suffix: str):
    markov_model_raw, mora_count, begin_count = to_raw_markov(kana_reader, INITIAL_MORAS + SHORTHANDS + SYMBOLS)

    mora_sum = sum(mora_count.values())
    mora_freq_dict = {k: v / mora_sum for k, v in mora_count.items()}
    mb_freq_dict = {k: v / (mora_sum + begin_count) for k, v in mora_count.items()}
    mb_freq_dict[MARKOV_BEGIN] = begin_count / (mora_sum + begin_count)

    markov_model = {}
    for m0, d in markov_model_raw.items():
        markov_model[m0] = {}
        s1 = sum(d.values())
        m0f = mb_freq_dict[m0]
        for m1, c in d.items():
            markov_model[m0][m1] = c * m0f / s1

    with open(f'markov_{suffix}.pkl', 'wb') as f:
        pickle.dump((markov_model, mora_freq_dict), f)


def ave_markov(suffix0: str, suffix1: str, suffix_ave: str):
    mm0: dict[str, dict[str, float]]
    mm1: dict[str, dict[str, float]]
    mm_ave: dict[str, dict[str, float]] = defaultdict(dict)
    mora_freq0: dict[str, float]
    mora_freq1: dict[str, float]
    mora_freq_ave: dict[str, float] = {}

    with open(f'markov_{suffix0}.pkl', 'rb') as f:
        mm0, mora_freq0 = pickle.load(f)
    with open(f'markov_{suffix1}.pkl', 'rb') as f:
        mm1, mora_freq1 = pickle.load(f)

    for m0 in set(mm0.keys()) | set(mm1.keys()):
        mm01 = mm0[m0] if m0 in mm0 else {}
        mm11 = mm1[m0] if m0 in mm1 else {}
        m1s = set(mm01.keys()) | set(mm11.keys())
        for m1 in m1s:
            f = 0.
            for mmn1 in [mm01, mm11]:
                if m1 in mmn1:
                    f += mmn1[m1]
            mm_ave[m0][m1] = f / 2
    mm_ave = {k: v for k, v in mm_ave.items()}

    for m in set(mora_freq0.keys()) | set(mora_freq1.keys()):
        f = 0.
        c = 0
        for mf in [mora_freq0, mora_freq1]:
            if m in mf:
                f += mf[m]
                c += 1
        mora_freq_ave[m] = f / c

    with open(f'markov_{suffix_ave}.pkl', 'wb') as f:
        pickle.dump((mm_ave, mora_freq_ave), f)
