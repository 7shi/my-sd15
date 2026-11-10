ページ：[01](01_quickstart.md) | [02](02_overview.md) | **03** | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)

---

# CLIP Text Encoder: テキストを条件ベクトルに

SD 1.5 のパイプラインは、まずテキストを条件ベクトルに変換するところから始まります。この役割を担うのが **CLIP Text Encoder** です。内部は GPT-2 とほぼ同じ Transformer ですが、トークナイザーにいくつかの違いがあります。

1. テキスト
   - **CLIP Text Encoder** ← この章
2. 条件ベクトル
3. ランダムノイズ
   - U-Net × 10 step
     - [Conv2D](04_conv2d.md)
     - [GroupNorm](05_groupnorm.md)
     - [ResBlock](06_resblock.md)
     - [Cross-Attention](08_cross_attention.md)
4. 潜在表現
   - [VAE Decoder](10_vae.md)
5. 画像

## 1. CLIP とは

CLIP (Contrastive Language-Image Pre-training) は、OpenAI が 2021 年に公開した、画像とテキストを結び付けるモデルです。画像を読む Image Encoder と、テキストを読む Text Encoder の 2 つからなり、大量の「画像とその説明文」のペアを使って、**対応する画像と説明文が近いベクトルになる**ように学習されています。

SD 1.5 はこのうち Text Encoder だけを借りてきて使います。画像と突き合わせながら鍛えられた Text Encoder は、「cat」という語がどんな見た目に対応するかを反映したベクトルを出力します。いわば「絵のことをよく知っている文章の読み手」であり、画像生成の指示役に適しています。

SD 1.5 の学習でも CLIP Text Encoder の重みは凍結されており、U-Net のほうが CLIP の出力を読み取れるように学習されました。

## 2. CLIP トークナイザー

CLIP トークナイザーは GPT-2 と同じバイトレベル BPE を基盤としています。BPE の仕組み自体は「[GPT-2 推論エンジン入門 03](https://github.com/7shi/my-gpt2/blob/main/docs/03_tokenizer.md)」を参照してください。ここでは GPT-2 との違いに焦点を当てます。

### 小文字化

GPT-2 のトークナイザーは大文字・小文字を区別しますが、CLIP は入力テキストを `text.lower()` で小文字に変換してからトークン化します。画像とテキストの対応付けでは大文字・小文字の区別が重要でないためです。

### `</w>` サフィックス

GPT-2 では単語の先頭にスペースを含めて `Ġhello` のようにエンコードしました（`Ġ` はスペースのバイト表現）。CLIP では逆に、単語の**末尾**に `</w>` (word end) サフィックスを付けます。

```
GPT-2:  "hello world" → ["hello", "Ġworld"]      （先頭にスペース）
CLIP:   "hello world" → ["hello</w>", "world</w>"]（末尾に </w>）
```

### 特殊トークン

| | GPT-2 | CLIP |
|---|---|---|
| 開始トークン | なし | `<\|startoftext\|>` (ID: 49406) |
| 終了トークン | `<\|endoftext\|>` (ID: 50256) | `<\|endoftext\|>` (ID: 49407) |

CLIP は系列の先頭に `<|startoftext|>` を挿入し、末尾に `<|endoftext|>` を挿入します。

### 固定長 77 トークン

GPT-2 は可変長の系列を扱いましたが、CLIP は常に **77 トークン**（`<|startoftext|>` + 最大 75 テキストトークン + `<|endoftext|>`）に固定します。短い場合は `<|endoftext|>` でパディング、長い場合はトランケーションします。

```python
def encode(self, text):
    text = text.lower().strip()
    tokens = [self._bos_token_id]
    # ... BPE トークン化 ...
    tokens.append(self._eos_token_id)
    if len(tokens) > 77:
        tokens = tokens[:76] + [self._eos_token_id]
    tokens = tokens + [self._eos_token_id] * (77 - len(tokens))
    return tokens
```

長さを固定するのは、後段の都合です。条件ベクトルの形状が常に (77, 768) になるため、U-Net は長さの違いを気にせずに済みます。

### トークン化の例

"a cat sitting on a windowsill" をトークン化した結果です。

```
プロンプト: 'a cat sitting on a windowsill'
トークン数: 77 (固定)
先頭: 49406 (<|startoftext|>)
末尾: 49407 (<|endoftext|>)
テキスト部分: [320, 2368, 4919, 525, 320, 3110, 6300, 660] (8 トークン)
```

テキスト部分は 8 トークンです（"windowsill" が 3 トークンに分割されています）。2 回現れる "a" がどちらも 320 になっていることも確認できます。全体は次のような並びになります。

```
[49406] [320, 2368, 4919, 525, 320, 3110, 6300, 660] [49407] [49407, 49407, ...]
 開始     テキスト 8 トークン                          終了     パディング 67 個
```

また、`'Hello World'` と `'hello world'` は同一のトークン列になり、小文字化が機能していることが分かります。

## 3. CLIP Transformer

CLIP Text Encoder の Transformer は GPT-2 と非常によく似た構造です。

| | GPT-2 | CLIP Text Encoder |
|---|---|---|
| レイヤ数 | 12 | 12 |
| 隠れ次元 | 768 | 768 |
| ヘッド数 | 12 | 12 |
| ヘッド次元 | 64 | 64 |
| 活性化関数 | GELU | Quick GELU |
| 出力 | 最後のトークンの 768 次元 | 全系列 (77, 768) |

Attention の仕組み（Q, K, V の計算、マルチヘッド分割、スケーリング）は GPT-2 と同一です。詳細は「[GPT-2 推論エンジン入門 07](https://github.com/7shi/my-gpt2/blob/main/docs/07_attention.md)」を参照してください。

### Quick GELU

GPT-2 が使う GELU は `0.5 * x * (1 + erf(x / sqrt(2)))` ですが、CLIP は **Quick GELU** という近似を使います。

```python
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)
```

GELU の式に含まれる `erf`（誤差関数）を、形のよく似た `sigmoid(1.702 * x)` で置き換えたもので、計算が高速です。CLIP はこの関数で学習されているため、推論でも同じ関数を使う必要があります。

### 全系列を出力

GPT-2 は次のトークンを予測するために最後のトークンの出力だけを使いましたが、CLIP Text Encoder は全 77 トークンの出力 `(77, 768)` をそのまま返します。

```
出力形状: (77, 768)
平均: -0.1073
標準偏差: 1.0256
```

プロンプト全体を 1 本のベクトルに要約するのではなく、トークンごとのベクトルを残しておくのがポイントです。この (77, 768) の行列が U-Net の Cross-Attention（👉[08](08_cross_attention.md)）にテキスト条件として渡され、画像の各位置が「cat」や「windowsill」といった個々のトークンを別々に参照できるようになります。

## 4. なぜ因果マスクがあるのか

CLIP Text Encoder は因果マスク（上三角が $-\infty$ のマスク）を使います。GPT-2 が因果マスクを使うのは「未来のトークンを見てはいけない」という自己回帰の制約からですが、次のトークンを予測しない CLIP がなぜ使うのでしょうか。

```python
mask = torch.full((seq_len, seq_len), float("-inf"))
mask = torch.triu(mask, diagonal=1)
```

理由は CLIP の学習方法にあります。CLIP は画像とテキストのペアを使って対照学習 (contrastive learning) で訓練されました。テキスト側のエンコーダは GPT-2 と同じアーキテクチャを使い、文末の `<|endoftext|>` の位置の出力を、文全体を表すベクトルとして画像埋め込みと比較します。因果マスクがあっても、文末の位置からはそれより前のトークンがすべて見えるので、文全体の要約として使えるわけです。この学習時に因果マスクがあったため、推論時にも同じマスクを使う必要があります。

SD 1.5 では文末の位置だけでなく全系列を使いますが、因果マスクは学習済み重みとの整合性のために必要です。

因果マスクの帰結として、各位置のベクトルは**自分より前のトークンの情報しか含んでいません**。"a cat sitting on a windowsill" の "cat" の位置は "sitting" 以降を知らず、後ろの位置ほど文全体を踏まえたベクトルになります。パディング部分の `<|endoftext|>` もプロンプト全体が見える位置にあるため、単なる空白ではなく、文全体の情報を持っています。

## 実験：CLIP の動作確認

テキストをトークン化し、CLIP Text Encoder で条件ベクトルに変換します。トークン化と出力形状の結果は本文中で引用しています。

もう 1 つ、プロンプトを空文字列にした場合の**無条件ベクトル**も計算します。

```
空文字列のトークン先頭 5: [49406, 49407, 49407, 49407, 49407]
出力形状: (77, 768)
条件付き vs 無条件の平均絶対差: 0.9450
```

空文字列でも 77 トークン（開始トークン 1 個と `<|endoftext|>` 76 個）になり、同じ形状のベクトルが得られます。条件付きベクトルとの平均絶対差 0.9450 は、出力の標準偏差が約 1.03 であることを考えると大きな値で、テキスト条件の有無で出力が大きく異なることを示しています。この違いが U-Net のノイズ予測の違いを生み、CFG（👉[11](11_pipeline.md)）で増幅されます。

**実行方法**: ([03_clip.py](03_clip.py))

```bash
uv run docs/03_clip.py
```

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | **03** | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)
