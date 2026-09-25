# voicevox-jev-proxy

VOICEVOX の読み・アクセント・イントネーションを、TypeSafe の Jev に判断させて直すツール。1 文ずつ WAV に書き出すコマンドと、VOICEVOX 互換の API サーバーがある。

## しくみ

1. 文を Sudachi で単語に分け、読みの候補を集める。読みは UniDic (unidic-lite) と Sudachi 辞書から、アクセント型は UniDic から、英語の語義は JMdict から取る。
2. 1 回目のリクエストで Jev に聞く。
   - 漢字にひらがなが続くところで、語がどこで切れるか
   - VOICEVOX のアクセント句が文節をまたぐところで、どこで区切るのが正しいか
   - 単語の読み (ひらがなの語は、どの語か)
   - www、笑 が笑いを表しているか (W杯 の W などと見分ける)。笑いなら読まない。草 は聞かずに、書いてあるとおり読む
3. 答えに合わせて文を書き換え、VOICEVOX に読ませ直し、選ばれた読みの辞書のアクセント型を当てる。

既定ではここで終わる (読みだけ直すモード)。Jev へのリクエストは 1 文につき 1 回で、聞くことが無い文では送らない。`--intonation` を付けると、続けてイントネーションも直す。

4. 2 回目のリクエストで、アクセント句が前の句に付くかどうか、文末の型 (言い切り・疑問・念押しなど)、句どうしのかかり方を聞いて、クエリに反映する。かかり方の答えは、修飾する句のあとで VOICEVOX が上げた句を下げるのに使う。VOICEVOX がアクセント核を 1 拍遅らせるところは、Jev に聞かずに直す。

イントネーションの補正は開発途中で、精度がまだ微妙なので、オプションにしてある。`--intonation` を付けると Jev へのリクエストが 1 文につき 2 回走るので注意。`--intonation` に `--requests 1` を足すと、2 回のリクエストを 1 回にまとめる。

確信度がしきい値 (既定 0.6、句の高さは 0.8) に届かない答えは使わない。

## 必要なもの

- Python 3.12 以上と [uv](https://docs.astral.sh/uv/)
- VOICEVOX エンジン (既定の接続先は `http://127.0.0.1:50021`)
- TypeSafe の API キー
- Sudachi のシステム辞書。無くても動くが、読みの質問、アクセント句の区切りの質問、句をつなぐ処理をしなくなる。
- JMdict の `JMdict_e.gz` (任意)。無いと、語義を付けずに読みを聞く。

辞書はこのリポジトリに入っていない。各自でダウンロードする。

## セットアップ

```sh
uv sync
```

### 辞書

動作を確かめたのは、Sudachi 辞書 `20260723` の full 版と、2026-09-22 版の `JMdict_e.gz`。

```sh
mkdir -p .cache/sudachi .cache/jmdict
curl -L -o .cache/sudachi/sudachi-dictionary-20260723-full.zip \
  https://d2ej7fkh96fzlu.cloudfront.net/sudachidict/sudachi-dictionary-20260723-full.zip
unzip -d .cache/sudachi .cache/sudachi/sudachi-dictionary-20260723-full.zip
curl -L -o .cache/jmdict/JMdict_e.gz http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz
```

`JMdict_e.gz` は展開せずに使う。

### .env

リポジトリの直下に `.env` を置く。コマンドもリポジトリの直下で実行する。

```sh
TYPESAFE_API_KEY=...
SUDACHI_DICT_PATH=.cache/sudachi/sudachi-dictionary-20260723/system_full.dic
JMDICT_PATH=.cache/jmdict/JMdict_e.gz
```

`VOICEVOX_URL`、`TYPESAFE_URL`、`TYPESAFE_MODEL` (既定は `jev-latest`) も `.env` で変えられる。辞書のパスは `--sudachi-dict`、`--jmdict` でも渡せる。

## 使い方

### 1 文を直して WAV に書き出す

```sh
uv run voicevox-jev-correct "雨が降ってきたから、傘を持っていくのだ。"
```

`.build/out.before.wav` (VOICEVOX のまま) と `.build/out.wav` (直したもの) ができる。何を変えたかは標準出力に出る。

- `--speaker`: VOICEVOX の話者 ID。既定は 3 (ずんだもん ノーマル)。
- `--out`: 出力先。既定は `.build/out.wav`。
- `--dry-run`: Jev に送るリクエストを表示するだけで、送らない。
- `--intonation`: イントネーションも直す。Jev へのリクエストが 1 文につき 2 回になる。

### VOICEVOX 互換サーバー

```sh
uv run voicevox-jev-proxy
```

`http://127.0.0.1:50121` で待ち受ける。VOICEVOX を使うアプリの接続先をこのアドレスに変えると、`POST /audio_query` の結果が直したものになる。ほかのリクエストはそのまま VOICEVOX に転送する。

- `--request-cap`: このプロセスが Jev に送るリクエストの上限。既定は 20。超えると `/audio_query` は 503 を返す。
- `--request-interval`: リクエストの最小間隔 (秒)。既定は 0。
- `--allow-origin`: Web ページから `/audio_query` を呼ぶときに許可するオリジン。複数回指定できる。Origin ヘッダーの付いたリクエストは、ここに無いオリジンなら 403 を返す。
- `--dry-run`: Jev に送らず、リクエストをログに出して、VOICEVOX のクエリをそのまま返す。
- `--intonation`: イントネーションも直す。Jev へのリクエストが 1 文につき 2 回になり、`--request-cap` に届くのも早くなる。

## テスト

```sh
uv run pytest
```

Sudachi 辞書を使う 4 つのテストは、`SUDACHI_DICT_PATH` を渡したときだけ動く。

```sh
SUDACHI_DICT_PATH=.cache/sudachi/sudachi-dictionary-20260723/system_full.dic uv run pytest
```

## ライセンス

コードは MIT License (`LICENSE`)。辞書は含まない。使う辞書にはそれぞれのライセンスがある。

- Sudachi 辞書: Apache License 2.0 (zip に入っている `LEGAL` と `LICENSE-2.0.txt`)
- JMdict: EDRDG のライセンス (Creative Commons Attribution-ShareAlike 4.0)
- UniDic: unidic-lite に同梱の BSD ライセンス (The UniDic Consortium)

VOICEVOX で作った音声は、VOICEVOX と各キャラクターの利用規約に従う。

## 謝辞

This package uses the JMdict dictionary file. This file is the property of the Electronic Dictionary Research and Development Group, and is used in conformance with the Group's licence.

- JMdict: <https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project>
- EDRDG licence: <https://www.edrdg.org/edrdg/licence.html>
- SudachiDict: <https://github.com/WorksApplications/SudachiDict>
- UniDic: <https://unidic.ninjal.ac.jp/>
- unidic-lite: <https://github.com/polm/unidic-lite>
