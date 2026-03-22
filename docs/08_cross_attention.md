ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | **08** | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)

---

# Cross-Attention: テキスト条件の注入

SD 1.5 がテキストに応じた画像を生成できるのは、U-Net 内部で **Cross-Attention** によりテキスト条件を画像特徴に注入しているからです。

1. テキスト
   - [CLIP Text Encoder](03_clip.md)
2. 条件ベクトル
3. ランダムノイズ
   - U-Net × 10 step
     - [Conv2D](04_conv2d.md)
     - [GroupNorm](05_groupnorm.md)
     - [ResBlock](06_resblock.md)
     - **Cross-Attention** ← この章
4. 潜在表現
   - [VAE Decoder](10_vae.md)
5. 画像

## 1. Self-Attention の復習

GPT-2 の Self-Attention（👉[GPT-2 07](https://github.com/7shi/my-gpt2/tree/main/docs/07_attention.md)）では、Q, K, V すべてが**同じ入力**から計算されます。

```python
# Self-Attention: Q, K, V はすべて x から
q = x @ W_q    # 自分自身に「何を求めているか」
k = x @ W_k    # 自分自身に「何を持っているか」
v = x @ W_v    # 自分自身の「値」
attn = softmax(q @ k.T / sqrt(d))
out = attn @ v
```

各トークンが他のすべてのトークンを参照し、文脈を理解します。

## 2. Self-Attention vs Cross-Attention

Cross-Attention では、Q は**画像特徴**から、K と V は**テキスト埋め込み**から計算されます。

```python
# Cross-Attention: Q は画像、K/V はテキスト
q = image_features @ W_q   # 画像が「何を求めているか」
k = text_embedding @ W_k   # テキストが「何を持っているか」
v = text_embedding @ W_v   # テキストの「値」
attn = softmax(q @ k.T / sqrt(d))
out = attn @ v
```

画像の各位置が「テキストのどの部分に注目すべきか」を学習し、テキスト条件に合った特徴を画像に注入します。例えば「cat」に対応するテキスト埋め込みが、画像中の猫が描かれるべき位置で高い Attention スコアを持ちます。

| | Self-Attention | Cross-Attention |
|---|---|---|
| Q の出所 | 入力系列 | 画像特徴 |
| K, V の出所 | 入力系列（= Q と同じ） | テキスト埋め込み（外部） |
| 系列長 | Q = K = V が同じ長さ | Q と K/V が異なる長さ |
| 用途 | 系列内の文脈理解 | 外部条件の注入 |

## 3. CrossAttention クラス

```python
class CrossAttention:
    def __init__(self, state, prefix, dim, num_heads=8):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def __call__(self, x, context=None):
        if context is None:
            context = x  # context なし → Self-Attention
        q = x @ W_q.T           # (seq, dim)
        k = context @ W_k.T     # (ctx_seq, dim)
        v = context @ W_v.T     # (ctx_seq, dim)
        # マルチヘッド分割、Attention 計算、結合
        ...
        out = linear(out, W_out, b_out)
        return out
```

**重要なポイント:**

- `context=None` のとき `context = x` とし、Self-Attention として動作
- GPT-2 は 12 ヘッド × 64 次元、SD 1.5 の U-Net は **8 ヘッド** × (dim/8) 次元
- GPT-2 の Attention は Q, K, V にバイアスがあったが、SD 1.5 の CrossAttention は**バイアスなし**（`to_q`, `to_k`, `to_v` は重み行列のみ）

## 4. BasicTransformerBlock

`BasicTransformerBlock` は 3 つのサブ層で構成されます。

```python
class BasicTransformerBlock:
    def __call__(self, x, context):
        x = x + self.attn1(layer_norm(x, ...))              # Self-Attention
        x = x + self.attn2(layer_norm(x, ...), context)     # Cross-Attention
        x = x + self.ff(layer_norm(x, ...))                 # GEGLU FFN
        return x
```

GPT-2 の Transformer Block が 2 サブ層 (Attention + MLP) だったのに対し、SD 1.5 は **3 サブ層**です。

| サブ層 | GPT-2 | SD 1.5 BasicTransformerBlock |
|---|---|---|
| 1 | Self-Attention | Self-Attention (`attn1`, context=None)  |
| 2 | MLP (GELU)  | Cross-Attention（`attn2`, context=テキスト） |
| 3 | — | GEGLU FFN |

各サブ層は LayerNorm → 変換 → 残差接続の構造で、GPT-2 と同じ Pre-Norm パターンです。

## 5. GEGLU

GPT-2 の MLP は `Linear → GELU → Linear` でしたが、SD 1.5 は **GEGLU** (Gated GELU) を使います。

```python
class GEGLU:
    def __call__(self, x):
        h = linear(x, ...)        # dim → dim*8 (2倍分を一度に計算)
        h, gate = h.chunk(2, dim=-1)  # 半分に分割
        h = h * gelu(gate)        # ゲートで制御
        h = linear(h, ...)        # dim*4 → dim
        return h
```

通常の MLP が `x → 拡大 → GELU → 縮小` であるのに対し、GEGLU は拡大した出力を 2 つに分け、一方を GELU に通して「ゲート」として使います。もう一方をこのゲートで乗算することで、通す情報を動的に制御します。

## 6. SpatialTransformer

画像特徴マップ `(C, H, W)` は 2D ですが、Attention は 1D の系列を扱います。`SpatialTransformer` はこの変換を担います。

```python
class SpatialTransformer:
    def __call__(self, x, context):
        C, H, W = x.shape
        residual = x
        x = group_norm(x, ...)                  # GroupNorm で正規化
        x = conv2d(x, ..., proj_in)             # 1×1 conv（チャネル方向の変換）
        x = x.reshape(C, H * W).T               # (C, H, W) → (H×W, C)
        x = self.block(x, context)              # BasicTransformerBlock
        x = x.T.reshape(C, H, W)               # (H×W, C) → (C, H, W)
        x = conv2d(x, ..., proj_out)            # 1×1 conv
        return residual + x                      # 残差接続
```

核心は `reshape(C, H*W).T` の 1 行です。`(C, H, W)` の画像を `(H×W, C)` の系列に変換することで、各ピクセルを 1 つの「トークン」として Attention に渡します。32×32 の特徴マップなら 1024 トークンの系列になります。

この発想は、GPT-2 が「テキストを系列として扱う」のと同じです。SD 1.5 は「画像のピクセルを系列として扱う」ことで、Transformer の仕組みを画像に適用しています。

## 実験：Cross-Attention の動作確認

Self-Attention と Cross-Attention の違い、SpatialTransformer の reshape を確認します。実行結果は以下のとおりです。

```
=== 1. Self-Attention vs Cross-Attention ===
Self-Attention:
  入力 x:     (1024, 320)
  出力:       (1024, 320)
Cross-Attention:
  入力 x:     (1024, 320) (Q の出所)
  context:    (77, 768) (K/V の出所)
  出力:       (1024, 320)

=== 2. BasicTransformerBlock ===
入力:   (1024, 320)
出力:   (1024, 320)
残差の効果: 入出力の差の平均 = 0.2815

=== 3. SpatialTransformer (2D → 系列 → 2D) ===
入力 (画像): (320, 32, 32)
  → reshape: (320, 1024) → (1024, 320) [系列として処理]
  → reshape: (320, 32, 32) [画像に戻す]
出力 (画像): (320, 32, 32)

=== 4. パラメータ比較 ===
Self-Attention (attn1):
  to_q.weight: (320, 320)
  to_q.bias:   なし（GPT-2 との違い）
Cross-Attention (attn2):
  to_q.weight: (320, 320) (入力: 320 dim)
  to_k.weight: (320, 768) (入力: 768 dim = CLIP 出力)
```

Cross-Attention の `to_k.weight` が `(320, 768)` であることに注目してください。K と V の入力次元が 768（CLIP の出力次元）であるのに対し、Q の入力次元は 320（画像特徴の次元）です。これが「画像が問いかけ、テキストが答える」という Cross-Attention の構造を反映しています。

また、Self-Attention の `to_q` にバイアスがないことは GPT-2 との違いです。GPT-2 の Attention は Q, K, V すべてにバイアスがありました。

**実行方法**: ([08_cross_attention.py](08_cross_attention.py))

```bash
uv run docs/08_cross_attention.py
```

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | **08** | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)
