ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | **08** | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)

---

# Cross-Attention: テキスト条件の注入

SD 1.5 がテキストに応じた画像を生成できるのは、U-Net 内部で **Cross-Attention** によりテキスト条件を画像特徴に注入しているからです。プロンプトが画像に影響を与える入口は、ここだけです。

仕組みは GPT-2 の Attention とほとんど同じで、違いは「誰が誰を参照するか」だけです。GPT-2 ではトークンが他のトークンを参照しましたが、Cross-Attention では**画像の各位置がテキストのトークンを参照します**。

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

GPT-2 の Self-Attention（👉[GPT-2 07](https://github.com/7shi/my-gpt2/blob/main/docs/07_attention.md)）では、Q, K, V すべてが**同じ入力**から計算されます。

```python
# Self-Attention: Q, K, V はすべて x から
q = x @ W_q    # 自分自身に「何を求めているか」
k = x @ W_k    # 自分自身に「何を持っているか」
v = x @ W_v    # 自分自身の「値」
attn = softmax(q @ k.T / sqrt(d))
out = attn @ v
```

Q と K の内積で「どこにどれだけ注目するか」を決め、その注目度で V を混ぜ合わせて取り出す、という処理でした。探す側（Q）と探される側（K, V）が同じ系列なので **Self**-Attention と呼びます。

## 2. Self-Attention vs Cross-Attention

Cross-Attention では、探す側と探される側が別の系列になります。Q は**画像特徴**から、K と V は**テキスト埋め込み**（CLIP の出力、👉[03](03_clip.md)）から計算されます。

```python
# Cross-Attention: Q は画像、K/V はテキスト
q = image_features @ W_q   # 画像が「何を求めているか」
k = text_embedding @ W_k   # テキストが「何を持っているか」
v = text_embedding @ W_v   # テキストの「値」
attn = softmax(q @ k.T / sqrt(d))
out = attn @ v
```

ここでの画像特徴は、特徴マップの各ピクセルを 1 つのトークンとみなして並べた系列です（変換方法は 6 節）。32×32 の特徴マップなら 1024 トークンになります。

```
Q: 画像の 1024 位置   「この位置には何を描けばよいか？」
K: テキスト 77 トークン 「私は cat です」「私は windowsill です」
V: テキスト 77 トークン  それぞれのトークンが渡す情報

注目度 attn: (1024, 77)  画像の各位置が、77 トークンにどう注目するか
出力 out:    (1024, 320) 画像の各位置が受け取った情報
```

画像の各位置は、自分の Q とテキストの各トークンの K を照合して注目度を決め、その注目度に応じてテキストの V を受け取ります。行ごとに Softmax を取るので、注目度は位置ごとに合計 1 です。例えば猫が描かれるべき位置は「cat」のトークンに強く注目して猫らしい特徴を受け取り、窓辺になる位置は「windowsill」に注目する、といった形で、テキストの内容が画像の適切な場所に配分されます。

出力の形状は Q の側（画像）と同じです。テキストの長さが 77 でも、出力は画像の 1024 位置分になります。Cross-Attention は「画像の形を保ったまま、テキストの情報を流し込む」操作と言えます。

| | Self-Attention | Cross-Attention |
|---|---|---|
| Q の出所 | 入力系列 | 画像特徴 |
| K, V の出所 | 入力系列（= Q と同じ） | テキスト埋め込み（外部） |
| 系列長 | Q = K = V が同じ長さ | Q と K/V が異なる長さ |
| 用途 | 系列内の文脈理解 | 外部条件の注入 |

なお、GPT-2 の Self-Attention にあった因果マスクは、どちらの Attention にもありません。画像の位置には「前後」がなく、すべての位置が互いを、またテキストの全トークンを参照できます。

### Encoder-Decoder との関係

Cross-Attention は、もともと Transformer の原論文で、Decoder が Encoder の出力を参照するための仕組みとして導入されました（👉[GPT-2 07](https://github.com/7shi/my-gpt2/blob/main/docs/07_attention.md)）。GPT-2 は Decoder だけのモデルなので、Cross-Attention はありませんでした。では、Cross-Attention を使う SD 1.5 には Encoder と Decoder があるのでしょうか。

役割の分担としては、そう読めます。「入力を読んでベクトル列にする側」が CLIP Text Encoder、「それを参照しながら出力を作る側」が U-Net です。

| | 原論文の Transformer（翻訳） | SD 1.5 |
|---|---|---|
| 参照される側（K, V を出す） | Encoder: 原文を読む | CLIP Text Encoder: プロンプトを読む |
| 参照する側（Q を出す） | Decoder: 訳文を生成する | U-Net: ノイズを予測する |
| 参照される側の実行回数 | 1 回 | 1 回（条件あり・なしで各 1 回） |
| 参照する側の実行回数 | 生成するトークンの数だけ | デノイジングのステップ数だけ |

参照される側は最初に 1 回だけ実行され、参照する側がその出力を繰り返し読みに行く、という使い方も共通しています。

ただし、原論文の Encoder-Decoder と同じものではありません。違いは次のとおりです。

- **U-Net は Transformer ではありません。** 畳み込みが主体で、その中に SpatialTransformer が挟まっている構造です（👉[07](07_unet.md)）。
- **U-Net は自己回帰をしません。** 原論文の Decoder は因果マスクを使ってトークンを 1 つずつ生成しますが、U-Net はマスクなしで全位置を同時に更新します。
- **CLIP Text Encoder には因果マスクがあります。** 原論文の Encoder は前後の両方向を参照しますが、CLIP Text Encoder の中身は GPT-2 と同じ Decoder 型の構造です（👉[03](03_clip.md)）。名前は Encoder でも、構造は逆になっています。
- **2 つは別々に学習されました。** 原論文では Encoder と Decoder をまとめて学習しますが、SD 1.5 では学習済みの CLIP を凍結して使い、U-Net だけが拡散の学習を受けています。

なお、SD 1.5 には「Encoder / Decoder」と呼ばれるものがほかにもあり、それぞれ意味が異なります。

| 組 | 意味 | つなぐもの |
|---|---|---|
| CLIP Text Encoder と U-Net | テキストを読む側と、それを参照して画像を作る側 | Cross-Attention |
| U-Net の前半と後半 | 空間を縮小する側と、拡大する側（👉[07](07_unet.md)） | スキップ接続 |
| VAE Encoder と VAE Decoder | 画像を潜在表現に圧縮する側と、復元する側（👉[10](10_vae.md)） | 潜在表現 |

Cross-Attention と対応するのは 1 つ目の組だけです。

## 3. CrossAttention クラス

Self-Attention と Cross-Attention は、K と V の出所が違うだけです。そのため 1 つのクラスで両方を実装できます。

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

重みの形状を見ると、Self-Attention と Cross-Attention の違いがはっきり現れます。

```
Self-Attention (attn1):
  to_q.weight: (320, 320)
  to_q.bias:   なし（GPT-2 との違い）
Cross-Attention (attn2):
  to_q.weight: (320, 320) (入力: 320 dim)
  to_k.weight: (320, 768) (入力: 768 dim = CLIP 出力)
```

Cross-Attention の `to_k.weight` は `(320, 768)` です。Q の入力は 320 次元（画像特徴）、K と V の入力は 768 次元（CLIP の出力）と次元が異なりますが、`to_q` と `to_k` がどちらも 320 次元に変換するので、内積で照合できるようになります。**画像とテキストという別々の世界のベクトルを、比較できる共通の空間に持ち込む**のが、これらの重み行列の役割です。

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
| 2 | — | Cross-Attention（`attn2`, context=テキスト） |
| 3 | MLP (GELU) | GEGLU FFN |

各サブ層は LayerNorm → 変換 → 残差接続の構造で、GPT-2 と同じ Pre-Norm パターンです。系列として処理している間は、GroupNorm ではなく LayerNorm を使います。

3 つのサブ層の分担は次のように整理できます。

1. **Self-Attention**: 画像の各位置が、画像の他の位置を参照する（離れた場所同士の整合性を取る）
2. **Cross-Attention**: 画像の各位置が、テキストを参照する（何を描くかを受け取る）
3. **FFN**: 位置ごとに、集めた情報を変換する（GPT-2 の MLP と同じ役割）

畳み込みは近傍しか見られませんが、Self-Attention は 1 回で画像全体を参照できます。「左の耳と右の耳を揃える」といった離れた位置同士の調整に向いています。

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

```
通常の MLP:  h = gelu(h)           値そのものに活性化関数を適用
GEGLU:       h = h * gelu(gate)    別に計算したゲートを掛ける
```

通常の MLP では、ある成分を通すかどうかがその成分自身の値で決まります。GEGLU では「通す中身」(`h`) と「どれだけ通すか」(`gate`) を別々の重みで計算するため、中身とは独立に通し具合を決められます。蛇口にたとえると、流れる水と、蛇口の開き具合を別々に制御するイメージです。

## 6. SpatialTransformer

画像特徴マップ `(C, H, W)` は 2D ですが、Attention は 1D の系列を扱います。`SpatialTransformer` は、2D から系列への変換と、その逆変換を担う外枠です。U-Net の各ブロックからは、この `SpatialTransformer` が呼び出されます（👉[07](07_unet.md)）。

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

核心は `reshape(C, H*W).T` の 1 行です。`(C, H, W)` の画像を `(H×W, C)` の系列に変換することで、各ピクセルを 1 つの「トークン」として Attention に渡します。

```
特徴マップ (320, 32, 32)          系列 (1024, 320)

  320 チャネル                     トークン 0:    320 次元（左上のピクセル）
  ┌────────┐                      トークン 1:    320 次元（その右隣）
  │ 32×32  │        →               ...
  │ ピクセル │                      トークン 1023: 320 次元（右下のピクセル）
  └────────┘
```

1 つのピクセルが持つ 320 チャネル分の値が、そのまま 320 次元のトークンベクトルになります。GPT-2 でトークンが 768 次元のベクトルだったのと同じ形です。32×32 の特徴マップなら 1024 トークンの系列になります。

この発想は、GPT-2 が「テキストを系列として扱う」のと同じです。SD 1.5 は「画像のピクセルを系列として扱う」ことで、Transformer の仕組みを画像に適用しています。

なお、Self-Attention の計算量は系列長の 2 乗に比例します。32×32 では 1024 × 1024 の注目度を計算することになり、512×512 の生成（特徴マップは 64×64 で 4096 トークン）ではその 16 倍です。ここでも、潜在空間で処理して空間サイズを抑えていることが効いています。

## 実験：Cross-Attention の動作確認

Self-Attention と Cross-Attention の違い、SpatialTransformer の reshape を確認します。重みの形状は本文中で引用しています。

**実行方法**: ([08_cross_attention.py](08_cross_attention.py))

```bash
uv run docs/08_cross_attention.py
```

```
=== 1. Self-Attention vs Cross-Attention ===
Self-Attention:
  入力 x:     (1024, 320)
  出力:       (1024, 320)
Cross-Attention:
  入力 x:     (1024, 320) (Q の出所)
  context:    (77, 768) (K/V の出所)
  出力:       (1024, 320)

=== 3. SpatialTransformer (2D → 系列 → 2D) ===
入力 (画像): (320, 32, 32)
  → reshape: (320, 1024) → (1024, 320) [系列として処理]
  → reshape: (320, 32, 32) [画像に戻す]
出力 (画像): (320, 32, 32)
```

Cross-Attention では、(77, 768) のテキストを参照しても、出力は入力 `x` と同じ (1024, 320) です。「画像が問いかけ、テキストが答える」という構造のとおり、形状を決めるのは問いかける側です。SpatialTransformer 全体でも入出力の形状は変わらないため、U-Net のどの段階にも差し込めます。

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | **08** | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)
