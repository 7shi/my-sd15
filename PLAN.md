# Stable Diffusion 1.5: クリーンルーム実装プラン

## 背景

my-gpt2 プロジェクトでは、GPT-2 の推論エンジンをスクラッチ実装し、「動くトイモデルに沿って関連知識を膨らませる」アプローチで解説ドキュメントを整備した。同じ発想で Stable Diffusion 1.5（SD 1.5）の推論パイプラインをスクラッチ実装し、画像生成 AI の仕組みを理解することを目指す。

## クリーンルーム開発の方針

本実装は **diffusers のソースコードを参照しない**クリーンルーム開発として行う。

### 入力として使うもの
- **SPEC.md** — アーキテクチャ仕様書（レイヤ構成、重みキー名、テンソル形状、計算式）
- **testdata/** — 検証用テストデータ（safetensors 形式、各コンポーネントの入出力）
- **tests/** — テストスイート（60 テスト、pytest で実行）
- **weights/** — Hugging Face の公式重みファイル（safetensors）

### 参照しないもの
- diffusers のソースコード
- transformers のモデル実装
- 既存の my_sd15/ や my_sd15_pytorch/ の実装コード

## SD 1.5 の全体構造

```
テキスト
  ↓
CLIP Text Encoder（Transformer）
  ↓
条件ベクトル (77, 768)
  ↓                        ランダムノイズ (4, 32, 32)
  ↓                              ↓
  └──→ U-Net（ノイズ除去）←── DDIM Scheduler
         ↑        ↓                ↓
         └── 繰り返し(10step) ──→ 潜在表現 (4, 32, 32)
                                    ↓
                              VAE Decoder
                                    ↓
                              画像 (3, 256, 256)
```

| コンポーネント | パラメータ数 | 主な構成要素 |
|---|---|---|
| CLIP Text Encoder | ~123M | Transformer（GPT-2 と類似） |
| U-Net | ~860M | ResBlock, Self/Cross-Attention, Down/Up サンプリング |
| VAE Decoder | ~50M | ResBlock, Conv2D, GroupNorm, アップサンプリング |
| DDIM Scheduler | パラメータなし | ノイズ除去の反復計算式 |

合計約10億パラメータ。画像サイズ 256×256、DDIM 10step、CPU で実行。

## GPT-2 との対比

| | GPT-2 | SD 1.5 |
|---|---|---|
| 生成方式 | 自己回帰（1トークンずつ左→右） | 拡散（ノイズ→画像を反復除去） |
| 中核モデル | Transformer Decoder | U-Net + Cross-Attention |
| 入力 | テキスト | テキスト + ランダムノイズ |
| 出力 | テキスト | 画像 |
| 共通する概念 | Attention, LayerNorm, 残差接続, Linear, GELU, Softmax |
| 新たに必要な概念 | — | Conv2D, GroupNorm, Cross-Attention, VAE, 拡散過程 |

## 実装方針

- **PyTorch（自前演算）** — ops.py で conv2d, group_norm 等を自前実装（im2col + 行列積）
- **推論のみ**（学習は対象外）
- **公式重みのロード**: Hugging Face の `stable-diffusion-v1-5` (safetensors)
- **テスト駆動**: SPEC.md の仕様と testdata/ のテストデータで検証
- **解説ドキュメント**: 実装完了後に docs/ を整備

### なぜ PyTorch か

NumPy のみの実装も試したが、同一アルゴリズムで 10 倍以上遅い（benchmark.md 参照）。PyTorch の自前実装は diffusers とほぼ同じ速度で、アルゴリズムの理解にも支障がない。

## 重みファイルの配置

Hugging Face の [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) をダウンロードし、`weights/` に配置する。

```
weights/stable-diffusion-v1-5/stable-diffusion-v1-5/
├── text_encoder/model.safetensors    # CLIP Text Encoder (~492 MB)
├── unet/diffusion_pytorch_model.safetensors  # U-Net (~3.4 GB)
├── vae/diffusion_pytorch_model.safetensors   # VAE (~335 MB)
├── tokenizer/vocab.json, merges.txt  # CLIP Tokenizer
└── scheduler/scheduler_config.json   # DDIM Scheduler 設定
```

## ファイル構成

```
SPEC.md              # アーキテクチャ仕様書
testdata/            # 検証用テストデータ (safetensors)
tests/              # テストスイート (pytest)

sd15/                # 新規実装（クリーンルーム）
    __init__.py      # CLI エントリポイント
    ops.py           # conv2d, group_norm, silu, gelu, softmax 等
    vae.py           # VAE Decoder
    clip.py          # CLIP Text Encoder
    unet.py          # U-Net
    scheduler.py     # DDIM Scheduler
    tokenizer.py     # CLIP BPE Tokenizer
    loader.py        # 全モデルの重みロード
    pipeline.py      # テキスト→画像パイプライン（CFG 含む）
```

## 実装順序

SPEC.md §10 の推奨順序に従い、ボトムアップで進める。各段階で `pytest tests/` の対応テストが通ることを確認する。

### Stage 1: 基本演算 (ops.py) — 重み不要、25 テスト

SPEC.md §9 の演算を実装する。

- `conv2d(x, weight, bias, stride, padding)` — im2col + 行列積
- `group_norm(x, weight, bias, num_groups, eps)` — グループ正規化
- `layer_norm(x, weight, bias, eps)` — レイヤ正規化
- `linear(x, weight, bias)` — 全結合層
- `silu(x)`, `quick_gelu(x)`, `gelu(x)` — 活性化関数
- `softmax(x, axis)` — ソフトマックス
- `upsample_nearest_2d(x, scale)` — 最近傍アップサンプリング
- `embedding(indices, weight)` — 埋め込みテーブル参照

検証: `pytest tests/test_ops.py` — 手計算可能な小さい入力で検証。

### Stage 2: Tokenizer (tokenizer.py) — 重み不要、9 テスト

SPEC.md §3 の仕様で CLIP BPE トークナイザを実装する。

- GPT-2 スタイルのバイトレベル BPE
- 小文字化、`<|startoftext|>` / `<|endoftext|>` 特殊トークン
- 77 トークンにパディング/トランケーション

検証: `pytest tests/test_tokenizer.py` — プロパティテスト + testdata 比較。

### Stage 3: Scheduler (scheduler.py) — 重み不要、10 テスト

SPEC.md §7 の数式で DDIM スケジューラを実装する。

- scaled_linear ベータスケジュール
- タイムステップ選択
- DDIM ステップ（eta=0、決定的）

検証: `pytest tests/test_scheduler.py` — 数学的性質 + testdata 比較。

**ここまでで 44 テストが通る。重みファイル不要。**

### Stage 4: CLIP Text Encoder (clip.py + loader.py) — 重み必要、4 テスト

SPEC.md §4 の構造で CLIP Transformer を実装する。

- 12 層の Transformer（causal mask 付き）
- quick_gelu 活性化
- 重みローダー: `text_encoder/model.safetensors` のキーマッピング

検証: `pytest tests/test_clip.py` — testdata の埋め込みベクトルと比較。

### Stage 5: VAE Decoder (vae.py + loader.py) — 重み必要、2 テスト

SPEC.md §5 の構造で VAE Decoder を実装する。

- VaeResBlock（タイムステップ埋め込みなし）
- VaeAttention（単一ヘッド自己注意）
- Mid block + Up blocks (4個)
- 重みローダー: `vae/diffusion_pytorch_model.safetensors`

検証: `pytest tests/test_vae.py` — testdata の入出力と比較。

**ここで最初の目に見える成果: 潜在表現→画像の変換が動く。**

### Stage 6: U-Net (unet.py + loader.py) — 重み必要、2 テスト

SPEC.md §6 の構造で U-Net を実装する。最大かつ最も複雑なコンポーネント。

1. タイムステップ埋め込み: 正弦波 → linear → silu → linear
2. UNetResBlock: VaeResBlock + time_emb_proj
3. CrossAttention: num_heads=8, to_q/k/v にバイアスなし
4. GEGLU + FeedForward
5. BasicTransformerBlock: self-attn → cross-attn → feedforward
6. SpatialTransformer: GroupNorm → 1x1 conv → transformer → 1x1 conv + 残差
7. Down/Up/Mid ブロック: スキップ接続の管理
8. 重みローダー: `unet/diffusion_pytorch_model.safetensors`

検証: `pytest tests/test_unet.py` — testdata の入出力と比較。

### Stage 7: パイプライン (pipeline.py + __init__.py) — 重み必要、5 テスト

SPEC.md §8 のフローでパイプラインを統合する。

1. プロンプト → トークナイズ → CLIP エンコード → 条件ベクトル
2. 空文字列 → CLIP エンコード → 無条件ベクトル
3. ランダム潜在ノイズ生成
4. 10 ステップのデノイジングループ: U-Net ×2 → CFG → DDIM ステップ
5. VAE デコード → 画像 → PNG 保存
6. CLI: `sd15 --prompt "..." --seed 42 --steps 10 --cfg 7.5 -o output.png`

検証: `pytest tests/test_pipeline.py` — 各ステップの中間 latents を testdata と比較。

## テスト方針

| 段階 | テスト数 | 重み | 検証方法 |
|---|---|---|---|
| Stage 1: ops | 25 | 不要 | 手計算の期待値 |
| Stage 2: tokenizer | 9 | 不要 | プロパティ + testdata |
| Stage 3: scheduler | 10 | 不要 | 数学的性質 + testdata |
| Stage 4: CLIP | 4 | 必要 | testdata 比較 |
| Stage 5: VAE | 2 | 必要 | testdata 比較 |
| Stage 6: U-Net | 2 | 必要 | testdata 比較 |
| Stage 7: pipeline | 5 | 必要 | testdata ステップ比較 |
| **合計** | **60** | | |

### 許容誤差

| コンポーネント | atol |
|---|---|
| ops（手計算） | exact or 1e-5 |
| scheduler | 1e-5 |
| tokenizer | exact（整数） |
| CLIP | 1e-4 |
| VAE | 1e-3 |
| U-Net | 1e-3 |
| pipeline（per-step latents） | 1e-3 |
| pipeline（最終 uint8 画像） | exact |

## 章構成（案）

実装完了後にドキュメントを整備する。

| 章 | テーマ | 対応する新概念 |
|---|---|---|
| 01 | まず動かしてみよう — 画像生成体験 | — |
| 02 | 推論パイプラインの全体像 | 拡散モデルの概念 |
| 03 | CLIP Text Encoder — テキストを条件ベクトルに | Tokenizer, Transformer（GPT-2 の復習） |
| 04 | Conv2D — 画像処理の基本演算 | 畳み込み, im2col |
| 05 | GroupNorm — チャネル方向の正規化 | GroupNorm vs LayerNorm |
| 06 | ResBlock — 残差ブロックの画像版 | タイムステップ埋め込み |
| 07 | U-Net の全体構造 — ダウン・ミッド・アップ | エンコーダ・デコーダ構造, スキップ接続 |
| 08 | Cross-Attention — テキスト条件の注入 | Q=画像, KV=テキスト |
| 09 | DDIM Scheduler — ノイズ除去の数理 | 拡散過程, ノイズスケジュール |
| 10 | VAE Decoder — 潜在空間から画像へ | 変分オートエンコーダ |
| 11 | パイプライン統合 — テキストから画像へ | Classifier-Free Guidance |
| 12 | アーキテクチャ — SD 1.5 の設計思想 | GPT-2 との比較, 発展の方向 |

---

## 文体サンプル：04 Conv2D

※ `ページ：` 行は `pages.py` により自動生成される。各 .md ファイルにはプレースホルダーとして `ページ：` とだけ書いておけばよい。

ページ：

---

# Conv2D: 画像処理の基本演算

GPT-2 では全結合層（Linear）がベクトルの変換を担いました。画像を扱う SD 1.5 では、その役割を**畳み込み（Conv2D）** が担います。U-Net と VAE のほぼすべての層で使われる最も基本的な演算です。

1. テキスト
   - [CLIP Text Encoder](03_clip.md)
2. 条件ベクトル
3. ランダムノイズ
   - U-Net × 10 step
     - **Conv2D** ← この章
     - [GroupNorm](05_groupnorm.md)
     - [ResBlock](06_resblock.md)
     - [Cross-Attention](08_cross_attention.md)
4. 潜在表現
   - [VAE Decoder](10_vae.md)
5. 画像

## 1. なぜ全結合層ではなく畳み込みなのか

GPT-2 の全結合層は、各トークンの 768 次元ベクトルを別の 768 次元ベクトルに変換します（👉[02](02_overview.md)）。入力の全次元が出力の全次元に接続されるため「全結合」と呼ばれます。

画像にも同じ方式を適用できるでしょうか。512×512 の RGB 画像を 1 次元に展開すると 786,432 次元になります。全結合層の重み行列は 786,432×786,432 で、float32 なら約 2.3 TB です。これは明らかに非現実的です。

畳み込みはこの問題を、**局所性**と**重み共有**の 2 つのアイデアで解決します。

- **局所性**: 各出力ピクセルは、入力の小さな領域（例: 3×3）だけを見る
- **重み共有**: 同じ 3×3 のフィルタを画像全体でスライドさせて使い回す

3×3 フィルタで 256 チャネルから 256 チャネルへ変換する場合、重みは 256×256×3×3 = 589,824 パラメータです。全結合の 2.3 TB と比べて 6 桁小さくなります。

## 2. 畳み込みの計算

入力画像の各位置で、フィルタ（カーネル）を重ねて要素ごとの積を取り、合計します。

3×3 カーネルによる 1 チャネルの畳み込みを例に示します。

```
入力 (4×4):          カーネル (3×3):      出力 (2×2):
┌─────────────┐      ┌─────────┐         ┌───────┐
│ 1  2  3  0  │      │ 1  0  1 │         │ 12  7 │
│ 0  1  2  1  │  *   │ 0  1  0 │    =    │  9 11 │
│ 1  0  1  2  │      │ 1  0  1 │         └───────┘
│ 2  1  0  1  │      └─────────┘
└─────────────┘
```

左上の出力値 12 は、入力の左上 3×3 領域とカーネルの要素積の合計です。

```
1×1 + 2×0 + 3×1 +
0×0 + 1×1 + 2×0 +
1×1 + 0×0 + 1×1 = 12
```

カーネルを 1 ピクセルずつスライドさせて、すべての位置で同じ計算を繰り返します。

### GPT-2 の全結合層との対応

全結合層が `y = x @ W + b` であったように、畳み込みも本質的には線形変換にバイアスを加えたものです。違いは「全次元を接続する」か「局所領域だけを接続する」かです。

## 3. 複数チャネルの畳み込み

実際の SD 1.5 では入力・出力ともに複数チャネルを持ちます。入力が $C_{in}$ チャネル、出力が $C_{out}$ チャネルの場合、カーネルの形状は ($C_{out}$, $C_{in}$, $k_H$, $k_W$) です。

```python
# weight: (C_out, C_in, kH, kW) — 例: (256, 256, 3, 3)
# bias:   (C_out,)
# input:  (C_in, H, W)
# output: (C_out, H', W')
```

出力の各チャネルは、入力の**全チャネル**に対してそれぞれ異なるカーネルを適用し、その結果を合計したものです。GPT-2 の全結合層で、出力の各次元が入力の全次元の重み付き和であったのと同じ構造です（👉[02](02_overview.md)）。

## 4. PyTorch での実装

素朴な 4 重ループ実装は極端に遅いため、**im2col** という手法を使います。畳み込みの各パッチを列として並べた行列を作り、カーネルとの行列積に変換します。行列積に帰着させるのは、GPT-2 の Attention で Q·K^T を `@` 演算子で一括計算したのと同じ発想です（👉[03](03_clip.md) で再登場）。

```python
def conv2d(x, weight, bias, stride=1, padding=0):
    C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    # パディング
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    # im2col: unfold で各パッチを取り出す
    col = x.unfold(1, kH, stride).unfold(2, kW, stride)
    col = col.permute(0, 3, 4, 1, 2)          # (C_in, kH, kW, H_out, W_out)
    col = col.reshape(C_in * kH * kW, H_out * W_out)
    # カーネルを行に展開して行列積
    kernel = weight.reshape(C_out, -1)          # (C_out, C_in*kH*kW)
    out = kernel @ col                          # (C_out, H_out*W_out)
    out = out.reshape(C_out, H_out, W_out) + bias.reshape(-1, 1, 1)
    return out
```

畳み込みが行列積に帰着するため、最適化された BLAS ルーチンが使え、素朴なループの数百倍の速度が得られます。

## 実験：畳み込みの動作確認

3×3 カーネルで小さな入力画像を畳み込み、手計算の結果と一致することを確認します。また、PyTorch の `F.conv2d` と数値比較し、im2col 実装の正しさを検証します。実行結果は本文中で引用しています。

**実行方法**: ([04_conv2d.py](04_conv2d.py))

```bash
uv run docs/04_conv2d.py
```

---

ページ：
