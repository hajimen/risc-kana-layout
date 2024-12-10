from pathlib import Path

#########################################
# Path
#########################################

KANA_TXT = 'kana.txt'

CC100 = 'cc100'
CC100_SRC = 'cc100-ja\\*.parquet'
CC100_DIR = Path('cc100-sample')

ARM = 'arm'
ARM_SRC = 'amazon_reviews_multi_ja/train.jsonl'
ARM_DIR = Path('arm-sample')

MEIDAI = 'meidai'
MEIDAI_SRC = 'https://mmsrv.ninjal.ac.jp/nucc/nucc.zip'
MEIDAI_DIR = Path('meidai-dialogue')

WIKIPEDIA = 'wikipedia'
WIKIPEDIA_SRC = 'jawiki/data/'
WIKIPEDIA_DIR = Path('jawiki-sample')

AVE = 'ave'
OPT_LOG = 'opt_log.txt'

TRS_SCORE_TXT = 'key_transition_score.txt'

DATA_DIR = Path(__file__).parent / 'data'
INITIAL_MORA_TXT = 'initial_mora.txt'
SHORTHAND_TXT = 'shorthand.txt'
SYMBOL_TXT = 'symbol.txt'

#########################################
# Markov Model
#########################################

MARKOV_BEGIN = '___BEGIN__'
MARKOV_END = '___END__'

#########################################
# Key and Score
#########################################

LEFT_KEYS = list('wersdfgxcv')
TWINS = [
    'sd',
    'df',
    'ef',
]
TWINS.extend([k[::-1] for k in TWINS])
RIGHT_KEYS = list('uiophjklnm,')

LEFT_TKNS = LEFT_KEYS + TWINS + ['_']
