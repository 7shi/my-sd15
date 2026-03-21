# my-sd15: Stable Diffusion 1.5 Scratch Implementation with PyTorch

このプロジェクトは、画像生成 AI のアーキテクチャを深く理解するために、[Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) の推論パイプラインをスクラッチ実装したものです。PyTorch の基本演算（行列積、unfold 等）を用いて、Conv2D、GroupNorm、U-Net、VAE などの仕組みを再現しています。アニメ風モデル [Anything V5](https://huggingface.co/genai-archive/anything-v5) にも対応しています。

姉妹プロジェクト [my-gpt2](https://github.com/7shi/my-gpt2) で GPT-2 の推論エンジンをスクラッチ実装した経験を土台に、同じ「動くコードから学ぶ」アプローチで画像生成の世界に踏み込みます。

## 主な特徴

- **自前演算**: `conv2d`（im2col + 行列積）、`group_norm`、`layer_norm`、`softmax` 等を `ops.py` にスクラッチ実装。
- **自作 CLIP トークナイザー**: GPT-2 スタイルのバイトレベル BPE に、CLIP 固有の小文字化・`</w>` サフィックス・固定 77 トークンを実装。
- **クリーンルーム開発**: diffusers / transformers のソースコードを参照せず、仕様書（SPEC.md）とテストデータのみから実装。
- **公式重みのロード**: Hugging Face の `safetensors` 形式の学習済み重みを読み込み、テキストから画像を生成。
- **テスト駆動開発 (TDD)**: 全 60 テストで各コンポーネントの出力を diffusers の結果と数値比較。
- **詳細な解説ドキュメント**: 12 章構成で、GPT-2 との対比を軸に SD 1.5 の仕組みを段階的に解説。

## 初期実装の生成

以下の情報から、Claude Opus 4.6 (1M Context) によって 1 時間程度で生成できます。

- [PLAN.md](PLAN.md): プロジェクトの全体計画と実装の段階的なロードマップ。（ドキュメント生成の手前までで止める）
- [SPEC.md](SPEC.md): SD 1.5 の推論パイプラインの仕様書。各コンポーネントの入出力の形状や処理内容を詳細に記述。
- [gen_testdata.py](gen_testdata.py): 各コンポーネントの出力を `testdata/` に保存するスクリプト。あらかじめ既存の実装で生成しておきます。
- [tests/](tests/): 各コンポーネントの出力をテストデータと数値比較するユニットテスト。事前に `testdata/` を生成しておく必要があります。

既にアルゴリズムは確立されており、LLM はそれについての知識を持っているため、データ入出力などの仕様を渡せば Vibe Coding の要領で生成できます。

## 技術解説

docs ディレクトリに、推論パイプラインの処理順序に沿った解説と実験スクリプトがあります。

- [docs/README.md](docs/README.md)

### なぜ PyTorch か

my-gpt2 では NumPy のみで実装しましたが、同じアルゴリズムでも CPU 実行時に PyTorch は NumPy の約 10 倍高速です。画像生成はテキスト生成より計算量が桁違いに大きいため、PyTorch を採用しました。ただし原理的な面を説明する目的から、`F.conv2d` 等の高レベル関数は使わず、NumPy と同水準の書き方（行列積、unfold、基本的なテンソル操作）に留めています。

### 依存関係の補足

コード自体は numpy を使用していませんが、numpy を未インストールの状態で PyTorch をインポートすると、PyTorch の C++ 初期化コード（`tensor_numpy.cpp`）が NumPy との連携を試みて以下の警告を出力します：

```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

この警告は Python レベルで回避できないため、numpy を通常の依存関係として追加しています。

## SD 1.5 の位置づけ

SD 1.5 は 2022 年に公開された約 10 億パラメータの拡散モデルで、テキストから画像を生成します。GPT-2 が「次のトークンを予測する」自己回帰モデルだったのに対し、SD 1.5 は「ノイズを少しずつ除去する」拡散モデルです。

内部では Attention や残差接続など GPT-2 と共通する概念が多く使われていますが、画像を扱うために畳み込み（Conv2D）、グループ正規化（GroupNorm）、Cross-Attention、VAE といった新しい概念が加わります。SD 1.5 の基本構造は SDXL や SD 3 / Flux にも受け継がれており、ここで得た理解は最新の画像生成モデルにそのまま応用できます。

### 画像サイズについて

本プロジェクトは CPU 実行を前提としています。SD 1.5 は 512×512 で訓練されているため、256×256 では意味のある画像が生成できません。一方、Anything V5 は 256×256 でも比較的良好な結果が得られます。CPU (WSL2) での実行時間は 256×256 で約 30 秒、512×512 で約 3 分です。サンプル画像は [samples/README.md](samples/README.md) を参照してください。

### デノイジング過程の可視化

`gen_steps.py` はデノイジングの各ステップで中間画像を `steps/` に保存します。各ステップで latents の先頭 3 チャンネルをそのまま画像化したもの（`xx-1.png`）と、VAE でデコードしたもの（`xx-2.jpg`）の 2 種類を出力します。生成結果は [steps/README.md](steps/README.md) で確認できます。

## ディレクトリ構成

```text
my-sd15/
├── my_sd15/
│   ├── __init__.py    # CLI エントリポイント
│   ├── ops.py         # 基本演算（conv2d, group_norm, softmax 等）
│   ├── tokenizer.py   # CLIP BPE トークナイザ
│   ├── clip.py        # CLIP Text Encoder（12層 Transformer）
│   ├── scheduler.py   # DDIM スケジューラ
│   ├── unet.py        # U-Net（ResBlock, CrossAttention, SpatialTransformer）
│   ├── vae.py         # VAE Decoder
│   ├── loader.py      # 重みロード
│   └── model.py       # パイプライン統合（CFG）
├── Makefile           # セットアップと実行の自動化
│   ├── weights/       # make download で生成
│   └── samples/       # サンプル画像と実行ログ
├── gen_steps.py       # デノイジング過程の中間画像を steps/ に保存するスクリプト
│   └── steps/         # デノイジング過程の可視化
├── gen_testdata.py    # 各コンポーネントの出力を testdata/ に保存するスクリプト
│   └── testdata/      # gen_testdata.py で生成したテストデータ
├── tests/             # ユニットテスト (60 テスト)
├── docs/              # 技術解説ドキュメント (.md) と実験スクリプト (.py)
├── PLAN.md            # プロジェクトの全体計画と実装の段階的なロードマップ
├── SPEC.md            # 推論パイプラインの仕様書（各コンポーネントの入出力の形状）
├── single2dir.py      # 単一ファイル形式のモデルを Diffusers 形式に変換するスクリプト
└── pyproject.toml     # プロジェクト設定 (hatchling)
```

## セットアップと使用方法

### 1. 環境構築

リポジトリをクローンして、依存関係をインストールします。

```bash
git clone https://github.com/7shi/my-sd15.git
cd my-sd15
uv sync
```

### 2. 重みファイルのダウンロード

SD 1.5 と Anything V5 の重みをダウンロードします。

```bash
make download          # 両モデルをまとめてダウンロード
make download-sd15     # stable-diffusion-v1-5 のみ
make download-any5     # genai-archive/anything-v5 のみ
```

Anything V5 のような単一ファイル形式のモデルは、`single2dir.py` で Diffusers 形式（コンポーネント別ディレクトリ）に自動分割されます。

### 3. 画像生成の実行

プロンプトを指定して画像を生成します。Sixel 対応ターミナルでは画像がインラインで表示されます。

```bash
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 -o output.png
```

Anything V5 を使ってアニメ風の画像を生成できます。

```bash
uv run my-sd15 -m genai-archive/anything-v5 -p "a cat sitting on a windowsill" --seed 123 -o output.png
```

#### 主なオプション:
- `-p`, `--prompt`: 生成するテキスト条件（必須）。
- `-m`, `--model`: モデル ID（デフォルト: `stable-diffusion-v1-5/stable-diffusion-v1-5`、例: `genai-archive/anything-v5`）。
- `--seed`: 乱数シード。指定すると生成結果を再現できます。
- `--steps`: デノイジングステップ数（デフォルト: 10）。
- `--cfg`: CFG スケール（デフォルト: 7.5）。値を大きくするとプロンプトに忠実になります。
- `-W`, `--width`: 画像の幅（デフォルト: 256）。8 の倍数に切り上げられます。
- `-H`, `--height`: 画像の高さ（デフォルト: 256）。8 の倍数に切り上げられます。
- `-n`, `--negative`: Negative Prompt。生成したくない要素を指定します。
- `-o`, `--output`: 出力ファイルパス（デフォルト: `output.png`）。

## テスト

`gen_testdata.py` で各コンポーネントの入出力を safetensors として `testdata/` に保存します。pytest はこのテストデータを使って、スクラッチ実装の出力を検証します。

```bash
uv run gen_testdata.py  # テストデータ生成
uv run pytest           # テスト実行
```

---
画像生成 AI の内部で何が起きているのかを、コードを通じて学ぶためのリポジトリです。
