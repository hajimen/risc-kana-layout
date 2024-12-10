# RISCかな配列

RISCかな配列とは、本稿の筆者が設計・提案する、新しい日本語かな配列である。
これは、Google日本語入力のローマ字テーブルのカスタマイズ機能と、
一般的なQWERTY配列キーボードを前提にしている。

RISCかな配列を用いる人への案内は、[RISCかな配列 Wikiホーム](https://github.com/hajimen/risc-kana-layout/wiki)にある。

OS等の機能によらず、また独自のかな漢字変換やドライバ等のソフトウェアを要さず、
ローマ字テーブルのカスタマイズ機能だけで実装できることが最大の特徴である。

ローマ字入力と同様に、ラテン文字キーだけを用い、基本的に2打鍵と1モーラが対応する。
ローマ字入力との最大の違いは、配列に規則性がないことである。
規則性をなくすことで得た配列の自由度を活かして、打鍵しやすくなるよう最適化してある。
ローマ字入力より打ちやすいが、覚えるのは難しい。

設計の詳細は[RISCかな配列の設計](https://github.com/hajimen/risc-kana-layout/blob/main/doc/RISC-kana-layout.md)に記した。

## RISCかな配列の設計に用いたツール

RISCかな配列は、日本語の統計的な性質にもとづいて、組合せ最適化により決定されている。
本リポジトリのソフトウェア（RISCかな配列の設計に用いたツール、以下**RKT**と呼ぶ）は、
配列を決定する過程で実際に用いられたものである。

RISCかな配列が行った組合せ最適化は巨大な非凸問題なので、同じ環境を整えても、完全に同じ結果が得られることはない。
しかしRKTを検討することによって、定式化における判断や誤りを検証することができる。

### 準備

実行環境は2024年現在のPython 3.11とWindows 11を前提とする。
2024年現在、[Python MIP](https://github.com/coin-or/python-mip)がApple Silicon MacやPython 3.12に対応していないことに注意。
Pythonにはvenv等の仮想環境を用いること。作業用ディレクトリには60GB以上の空きが必要。
以下のコマンドはすべて作業用ディレクトリをカレントディレクトリとして実行する。

```powershell
python -m pip install git+https://github.com/hajimen/risc-kana-layout.git
git lfs install
git clone https://huggingface.co/datasets/hpprc/jawiki
git clone https://github.com/knok/make-meidai-dialogue.git make_meidai_dialogue
git clone https://huggingface.co/datasets/SetFit/amazon_reviews_multi_ja
git clone https://huggingface.co/datasets/range3/cc100-ja
python -c 'from risc_kana_layout.prepare import prepare_all; prepare_all()'
```

とすることで、RKTと日本語コーパスと設定ファイルが準備される。設定ファイルは作業用ディレクトリに作られる。

設定ファイルには：

- キー遷移スコア表の `key_transition_score.txt`
- モーラを列挙した `initial_mora.txt`
- ショートハンドを列挙した `shorthand.txt`
- 記号を列挙した `symbol.txt`

がある。RISCかな配列の結果を再現しようとする場合（完全に同じ結果は得られないが）、これらをそのまま用いる。

```powershell
python -c 'from risc_kana_layout import calc_save_markov_all; calc_save_markov_all()'
```

とすることで、RISCかな配列の最適化に用いた統計情報（以下**AVE**と呼ぶ）が準備される。
この統計情報は`initial_mora.txt`と`shorthand.txt`と`symbol.txt`に依存する。

### 最適化の実行

```powershell
python -c 'from risc_kana_layout import optimize_all; optimize_all()'
```

とすることで、最適化された配列が`opt_third_complete.txt`に出力される。実行には約20時間かかる。

この際、Python MIPのバグにより、メモリ不足でプロセスが終了することがある。
その場合は`opt_log.txt`のログと`opt_first_10()`の実装を照らし合わせて、
`opt_first_10()`に適切な引数を与えることで途中から再開できる。

### 最適化済みデータ

RISCかな配列の設計の結論となった`opt_third_complete.txt`は本リポジトリの
`risc_kana_layout/data/opt_third_complete.txt`にある。

### ツール

```powershell
python -c 'from risc_kana_layout import calc_mora_freq; calc_mora_freq()'
```

AVEにおける出力単位の使用頻度が`mora_freq_ave.md`に出力される。
また`calc_mora_freq("cc100")`のように引数を与えると、各コーパスにおける使用頻度が得られる。
コーパスとそれに対応する引数：

- 日本語版Wikipedia 2024年1月1日 `wikipedia`
- CC100データセット中の日本語 `cc100`
- amazon_reviews_multi中の日本語 `arm`
- 名大会話コーパス `meidai`

```powershell
python -c 'from risc_kana_layout import calc_relaxed_solution; calc_relaxed_solution()'
```

$Minimize(Q)$ とその線形緩和解 $LMQ$ を計算して標準出力に出す。この計算は`opt_third_complete.txt`に依存する。
実行には約40時間かかる。

```powershell
python -c 'from risc_kana_layout import eval_format_all; eval_format_all()'
```

最適化の結果を評価するのに用いた図表、配列のMarkdownテーブル `risc-md.txt`、
Google日本語入力のローマ字テーブル用ファイル `risc-romantable.txt`が出力される。
当然`opt_third_complete.txt`に依存する。

```powershell
python -c 'from risc_kana_layout import example_opt; example_opt()'
```

上位10出力単位・連接頻度100個（総和は約12.6%）の場合での厳密解と線形緩和解を求める。
実行には約30分かかる。

## RISCかな配列練習器

RISCかな配列を覚えるのは難しいので、練習用アプリを作成した。
本章の準備として、作業用ディレクトリで

```powershell
git clone https://github.com/hajimen/risc-kana-layout.git .
```

とする。

### 例文データの作成

本節では、練習用アプリで用いる例文データの作成について説明する。

```powershell
python -c 'from risc_kana_layout.trainer import find_example; find_example()'
```

出力単位の使用頻度によって難易度の階級分け（Basic / Advanced / Practical / Maniac）を行うため、
各階級の範囲内に収まる例文候補をコーパス（日本語版Wikipedia）中に探し、`bin_*.txt`に出力する。
実行には長時間かかる。

```powershell
python -c 'from risc_kana_layout.trainer import fill; fill(["bin_40"], "Basic", 0, 40, 4)'
```

出力単位の使用回数が一定数以上になる最小の例文集合を得る。上の例では、`bin_40.txt`が例文候補、
`Basic.txt`が例文集合の出力先、`0`が階級の下界、`40`が上界、`4`が出力単位の使用回数の最小個数である。

例文集合にはほぼ必ず、日本語として誤りがある（特に読み）か、難読な固有名詞が含まれている。
このため、`bin_*.txt`を修正して`fill()`を再度実行して誤り等を探す、という作業を繰り返す必要がある。
しかし手戻りの関係上、`Basic.txt`のほうを修正して済ませたい場合が出てくる。

```powershell
python -c 'from risc_kana_layout.trainer import find_deficient; find_deficient("Basic", 0, 40, 4)'
```

`Basic.txt`の出力単位の使用回数を調べて、不足があれば表示される。

```powershell
python -c 'from risc_kana_layout.trainer import generate_image; generate_image()'
```

Basic / Advanced / Practical / Maniacに含まれる各例文に対して、
OpenAIのDALL-E 2で挿絵画像を生成し、`trainer/public/assets/illust/*.jpg`に保存する。
例文に対応する`trainer/public/assets/illust/*.jpg`がすでに存在する場合には生成しない。
OpenAIのAPIキーが必要。

```powershell
python -c 'from risc_kana_layout.trainer import dump_stage_data; dump_stage_data()'
```

Basic / Advanced / Practical / Maniacの例文集合から、練習用アプリが読み込むJSONファイルを作成する。
`trainer/public/assets/stage-data.json`が生成される。

### その他のデータの作成

```powershell
python -c 'from risc_kana_layout.trainer import dump_kana_table_as; dump_kana_table_as()'
```

`opt_third_complete.txt`のかな配列から、練習用アプリが読み込むJSONファイルを生成する。

```powershell
python -c 'from risc_kana_layout.trainer import generate_table; generate_table()'
```

`opt_third_complete.txt`のかな配列からPDFの表を作る。Basic / Advanced / Practical / Maniacの階級分けに対応した表と、
階級分けせずに全部を含む表が生成される。

### 練習用アプリの実行とデプロイ

Node.js 20が必要。本節ではリポジトリの`trainer/`をカレントディレクトリとする。
`node_modules`の準備のため最初に`npm install`としておく。

```powershell
npm run dev
```

練習用アプリがローカルで開ける。

```powershell
npm run build
npm run deploy
```

練習用アプリがGitHubの`gh-pages`ブランチにプッシュされる。
`gh-pages`ブランチをGitHub Pagesに設定すると、練習用アプリがインターネットで開ける。

### ライセンス

MIT License
