ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | **06** | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)

---

# ResBlock: 残差ブロックの画像版

GPT-2 の Transformer Block では、Attention と MLP の出力を入力に加算する**残差接続**が使われていました（👉[GPT-2 09](https://github.com/7shi/my-gpt2/blob/main/docs/09_residual.md)）。SD 1.5 でも同じ原理が、畳み込みベースの **ResBlock** として使われています。

1. テキスト
   - [CLIP Text Encoder](03_clip.md)
2. 条件ベクトル
3. ランダムノイズ
   - U-Net × 10 step
     - [Conv2D](04_conv2d.md)
     - [GroupNorm](05_groupnorm.md)
     - **ResBlock** ← この章
     - [Cross-Attention](08_cross_attention.md)
4. 潜在表現
   - [VAE Decoder](10_vae.md)
5. 画像

## 1. 残差接続の復習

残差接続は `output = x + f(x)` という単純な構造です。層の出力をそのまま使うのではなく、「元の入力に変更分を加える」という考え方でした。`f(x)` が 0 を出力するだけで「何も変更しない」ブロックになるため、各ブロックは必要な修正だけを学習すれば済みます。また、加算の経路を通って勾配が直接伝播するため、深いネットワークでも学習が安定します。

GPT-2 では次の形でした。

```python
x = x + attention(layer_norm(x))  # 残差接続 1
x = x + mlp(layer_norm(x))        # 残差接続 2
```

SD 1.5 の ResBlock も同じ原理ですが、Linear の代わりに Conv2D（👉[04](04_conv2d.md)）を、LayerNorm の代わりに GroupNorm（👉[05](05_groupnorm.md)）を使います。これまでの 2 章で見た部品を組み立てる章です。

## 2. VaeResBlock

VAE Decoder（👉[10](10_vae.md)）で使われる ResBlock です。最もシンプルな形なので、こちらから見ていきます。「正規化 → 活性化 → 畳み込み」を 2 回繰り返し、その結果を入力に加算します。

```python
class VaeResBlock:
    def __call__(self, x):
        h = silu(group_norm(x, ...))      # GroupNorm → SiLU
        h = conv2d(h, ..., padding=1)      # Conv2D (3×3)
        h = silu(group_norm(h, ...))       # GroupNorm → SiLU
        h = conv2d(h, ..., padding=1)      # Conv2D (3×3)
        if self.c_in != self.c_out:
            x = conv2d(x, ...)             # ショートカット (1×1 conv)
        return x + h                       # 残差接続
```

GPT-2 の Transformer Block と対比すると：

| | GPT-2 Transformer Block | VaeResBlock |
|---|---|---|
| 正規化 | LayerNorm | GroupNorm |
| 変換 | Linear | Conv2D (3×3) |
| 活性化 | GELU | SiLU |
| ショートカット | 常に恒等写像 | チャネル数が変わる場合は 1×1 conv |

正規化を変換の前に置く点は、GPT-2 の Pre-LayerNorm と同じ配置です。

入力と出力のチャネル数が異なる場合（例: 512 → 256）、そのままでは `x + h` の形状が合わず加算できません。そこで `x` 側も 1×1 の畳み込みでチャネル数を合わせます。1×1 の畳み込みは周囲のピクセルを見ないため、「各ピクセルのチャネル方向のベクトルに全結合層を適用する」のと同じ操作です。

## 3. SiLU 活性化関数

GPT-2 が GELU を使ったのに対し、SD 1.5 の U-Net と VAE は **SiLU**（Sigmoid Linear Unit、別名 Swish）を使います。

```python
def silu(x):
    return x * torch.sigmoid(x)
```

GELU と SiLU はどちらも滑らかな活性化関数で、形状もよく似ています。

```
GELU:  0.5 * x * (1 + erf(x / sqrt(2)))
SiLU:  x * sigmoid(x)
```

どちらも「`x` に 0〜1 の係数を掛ける」という形をしており、係数が `x` 自身から決まります。`x` が大きいと係数がほぼ 1 になって `x` をそのまま通し、`x` が負に大きいと係数がほぼ 0 になって遮断します。ReLU と違って原点付近が滑らかで、微小な負の値を通す点が共通しています。

CLIP の Quick GELU（👉[03](03_clip.md)）は `x * sigmoid(1.702 * x)` でした。SiLU はその係数 1.702 を 1 にしたもの、と見ることもできます。

## 4. UNetResBlock

U-Net で使われる ResBlock は、VaeResBlock に**タイムステップ埋め込み**の注入が加わります。

```python
class UNetResBlock:
    def __call__(self, x, temb):
        h = silu(group_norm(x, ...))      # GroupNorm → SiLU
        h = conv2d(h, ..., padding=1)      # Conv2D (3×3)
        # タイムステップ埋め込みを加算
        t = linear(silu(temb), ...)
        h = h + t.reshape(-1, 1, 1)        # ブロードキャスト加算
        h = silu(group_norm(h, ...))       # GroupNorm → SiLU
        h = conv2d(h, ..., padding=1)      # Conv2D (3×3)
        if self.c_in != self.c_out:
            x = conv2d(x, ...)
        return x + h
```

VaeResBlock との唯一の違いは、2 つの Conv2D の間にタイムステップ埋め込み `temb` を加算する 2 行です。

`temb` は 1280 次元のベクトルです。これを Linear でチャネル数と同じ次元に変換し、`reshape(-1, 1, 1)` で `(C, 1, 1)` の形にしてから加算します。つまり、**チャネルごとに 1 つの値を、画像の全ピクセルに一律に足します**。「今はこの種類の特徴を強め、あの種類は弱める」という指示を、画像全体に対して出しているイメージです。

## 5. タイムステップ埋め込み

タイムステップは「いま潜在表現にどれくらいノイズが残っているか」を表す 0〜999 の目盛りで、大きいほどノイズが多いことを意味します（👉[09](09_ddim.md)）。

U-Net は同じ重みのまま、デノイジングの全ステップで繰り返し使われます。しかし、やるべき仕事は段階によって違います。ノイズだらけのタイムステップ 900 では大まかな構図を決め、ほぼ完成しているタイムステップ 100 では細部を整える必要があります。そのため、U-Net に「いまどの段階か」を知らせなければなりません。この情報をベクトルに変換するのがタイムステップ埋め込みです。

```python
def _timestep_embedding(self, timestep):
    half = 160
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = float(timestep) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)])  # (320,)
    emb = linear(emb, ...)   # 320 → 1280
    emb = silu(emb)
    emb = linear(emb, ...)   # 1280 → 1280
    return emb
```

番号をベクトルに変換するという点は、GPT-2 の位置埋め込み（👉[GPT-2 05](https://github.com/7shi/my-gpt2/blob/main/docs/05_embedding.md)）と同じ発想です。GPT-2 では位置 (0, 1, 2, ...) をベクトルに変換しましたが、ここではタイムステップ (900, 800, 700, ...) をベクトルに変換しています。

ただし作り方が違います。GPT-2 の位置埋め込みは**学習されたテーブル**から行を取り出すものでしたが、ここで使う**正弦波エンコーディング**は Transformer 原論文（Attention Is All You Need）の方式で、タイムステップの値から計算で求めます。160 種類の周波数それぞれについて cos と sin を計算し、320 次元のベクトルにします。

直観的には、時計の針にたとえられます。秒針・分針・時針は回る速さが違い、秒針は細かい時刻の違いを、時針は大まかな時間帯を表します。3 本を組み合わせることで時刻が一意に決まるように、速く振動する成分から遅く振動する成分までを組み合わせることで、タイムステップの値を細かい違いも大まかな違いも読み取れる形で表現できます。

正弦波の後に 2 層の MLP（Linear → SiLU → Linear）を通して 1280 次元に拡張します。この MLP は学習済みで、機械的に計算した正弦波のベクトルを、U-Net の各 ResBlock が利用しやすい表現に変換します。タイムステップ埋め込みは U-Net の 1 回の実行につき 1 度だけ計算され、すべての ResBlock に同じ `temb` が渡されます。

実際に t=900 と t=100 の埋め込みを比較してみます。

```
t=900 の埋め込み: shape=(1280,), mean=-0.0064
t=100 の埋め込み: shape=(1280,), mean=0.0035
t=900 と t=100 のコサイン類似度: -0.6039
```

コサイン類似度が **-0.6039** と負の値で、2 つのベクトルはむしろ逆に近い方向を向いています。デノイジングの初期（ノイズが多い）と後期（ノイズが少ない）とで、U-Net にまったく異なる指示が渡されていることが分かります。

## 実験：ResBlock の動作確認

VaeResBlock と UNetResBlock の動作を確認します。タイムステップ埋め込みの結果は本文中で引用しています。

**実行方法**: ([06_resblock.py](06_resblock.py))

```bash
uv run docs/06_resblock.py
```

```
=== 1. VaeResBlock (512 → 512) ===
入力:  (512, 4, 4)
出力:  (512, 4, 4)
残差の平均絶対値: 0.4068

=== 2. VaeResBlock (512 → 256) ===
入力:  (512, 8, 8)
出力:  (256, 8, 8) (1×1 conv でチャネル数を変換)

=== 4. UNetResBlock (320 → 320) ===
入力:  (320, 32, 32)
temb:  (1280,)
出力:  (320, 32, 32)
```

VaeResBlock (512→512) では入出力が同じ形状です。「残差の平均絶対値」は出力と入力の差、つまりブロックが加えた修正量 `h` の大きさです。入力は標準偏差 1 の乱数なので、0.4 程度という値は、入力を作り変えているのではなく「入力に対する修正」を加えていることを示しています（入力が乱数のため、この値は実行ごとに多少変わります）。

VaeResBlock (512→256) ではチャネル数が変わるため、1×1 conv によるショートカットが使われます。UNetResBlock では、1280 次元の `temb` を受け取りつつ、入出力の形状は変わりません。

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | **06** | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)
