from collections.abc import Iterable
from collections import defaultdict
from risc_kana_layout import separate_by_moras, INITIAL_MORAS, SHORTHANDS


LIST_N_GRAM = [3, 4]
N_CANDIDATE = 100


def find(kana_reader: Iterable[str], filename_suffix: str, use_shorthand=False):
    gram_freq = defaultdict(lambda: defaultdict(int))
    for parsed, _ in separate_by_moras(kana_reader, INITIAL_MORAS + (SHORTHANDS if use_shorthand else [])):
        for n_gram in LIST_N_GRAM:
            for i in range(len(parsed) - n_gram + 1):
                gram_freq[n_gram][''.join(parsed[i: i + n_gram])] += 1
    sum_freq = sum([sum(gram_freq[n_gram].values()) for n_gram in LIST_N_GRAM])
    for n_gram in LIST_N_GRAM:
        with open(f'shorthand_candidates_{filename_suffix}_n{n_gram}.txt', 'w', encoding='utf-8') as f:
            freq = gram_freq[n_gram]
            c = 0
            for k, v in sorted(freq.items(), key=lambda x: x[1], reverse=True):
                f.write(f'{k}\t{v / sum_freq:.3f}\n')
                c += 1
                if c == N_CANDIDATE:
                    break
