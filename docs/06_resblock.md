ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | **06** | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)

---

# ResBlock: 残差ブロックの画像版

GPT-2 の Transformer Block では、Attention と MLP の出力を入力に加算する**残差接続**が使われていました（👉[GPT-2 09](https://github.com/7shi/my-gpt2/tree/main/docs/09_residual.md)）。SD 1.5 でも同じ原理が、畳み込みベースの **ResBlock** として使われています。

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

残差接続は `output = x + f(x)` という単純な構造です。`f(x)` が恒等写像を学習しやすく（何も変更しないことが簡単）、勾配が直接伝播するため深いネットワークの学習が安定します。

GPT-2 では：
```python
x = x + attention(layer_norm(x))  # 残差接続 1
x = x + mlp(layer_norm(x))        # 残差接続 2
```

SD 1.5 の ResBlock も同じ原理ですが、Linear の代わりに Conv2D を、LayerNorm の代わりに GroupNorm を使います。

## 2. VaeResBlock

VAE Decoder で使われる ResBlock です。最もシンプルな形です。

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

入力と出力のチャネル数が異なる場合（例: 512 → 256）、残差接続のために `x` 側も 1×1 の畳み込みでチャネル数を合わせます。

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

どちらも `x` が大きいとほぼ `x`（線形）、`x` が小さいとほぼ 0 に近づきます。ReLU と違って原点付近が滑らかで、微小な負の値を通す点が共通しています。

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

## 5. タイムステップ埋め込み

U-Net は「今ステップ 900 のノイズ除去をしている」「今ステップ 100 のノイズ除去をしている」という情報を知る必要があります。この情報をベクトルに変換するのがタイムステップ埋め込みです。

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

正弦波エンコーディングの部分は、GPT-2 の位置埋め込み（👉[GPT-2 05](https://github.com/7shi/my-gpt2/tree/main/docs/05_embedding.md)）と同じ発想です。GPT-2 では位置 (0, 1, 2, ...) をベクトルに変換しましたが、ここではタイムステップ (999, 900, 800, ...) をベクトルに変換しています。異なる周波数の正弦波・余弦波を組み合わせることで、タイムステップの値を豊かに表現します。

正弦波の後に 2 層の MLP（Linear → SiLU → Linear）を通して 1280 次元に拡張します。この MLP により、モデルはタイムステップ情報を非線形に変換して利用できます。

## 実験：ResBlock の動作確認

VaeResBlock と UNetResBlock の動作を確認します。VaeResBlock (512→512) では入出力が同じ形状で、残差の平均絶対値 0.3590 は「入力に対する小さな修正」であることを示しています。VaeResBlock (512→256) ではチャネル数が変わるため、1×1 conv によるショートカットが使われます。

タイムステップ埋め込みでは、t=900 と t=100 のコサイン類似度が **-0.6039** と負の値です。これはデノイジングの初期（ノイズが多い）と後期（ノイズが少ない）で、U-Net が大きく異なる挙動をすることを意味します。

**実行方法**: ([06_resblock.py](06_resblock.py))

```bash
uv run docs/06_resblock.py
```

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | **06** | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_lora.md) | [13](13_architecture.md)
