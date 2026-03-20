ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | **05** | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_architecture.md)

---

# GroupNorm: チャネル方向の正規化

GPT-2 では LayerNorm がベクトルの各次元を正規化しました（👉[GPT-2 06](https://github.com/7shi/my-gpt2/tree/main/docs/06_layer_norm.md)）。SD 1.5 の画像処理部（U-Net、VAE）では、代わりに **GroupNorm** を使います。

1. テキスト
   - [CLIP Text Encoder](03_clip.md)
2. 条件ベクトル
3. ランダムノイズ
   - U-Net × 10 step
     - [Conv2D](04_conv2d.md)
     - **GroupNorm** ← この章
     - [ResBlock](06_resblock.md)
     - [Cross-Attention](08_cross_attention.md)
4. 潜在表現
   - [VAE Decoder](10_vae.md)
5. 画像

## 1. なぜ LayerNorm ではなく GroupNorm なのか

LayerNorm は入力の**最後の次元**（GPT-2 では 768 次元）の平均と分散で正規化します。テキストのような 1 次元の系列データには適していますが、画像データには問題があります。

画像の特徴マップは `(C, H, W)` の 3 次元です。例えば `(320, 32, 32)` なら 320 チャネル、32×32 ピクセルです。LayerNorm を最後の次元に適用すると、幅方向の 32 ピクセルだけを正規化することになり、チャネルや高さの情報を活用できません。全次元に適用すると、異なるチャネル（異なる特徴）を混ぜて正規化してしまいます。

## 2. GroupNorm の仕組み

GroupNorm は、チャネルを**グループに分けて**、各グループ内のチャネル×空間全体で正規化します。

SD 1.5 では `num_groups=32` を使います。320 チャネルなら 32 グループ×10 チャネルになります。

```
入力: (320, 32, 32)
  ↓ 32 グループに分割
グループ 0: チャネル 0〜9   の (10, 32, 32) → 平均・分散を計算 → 正規化
グループ 1: チャネル 10〜19 の (10, 32, 32) → 平均・分散を計算 → 正規化
  ...
グループ 31: チャネル 310〜319 の (10, 32, 32) → 平均・分散を計算 → 正規化
  ↓ チャネルごとの weight, bias でスケール・シフト
出力: (320, 32, 32)
```

各グループ内では、関連するチャネルの空間全体を使って統計量を計算するため、空間サイズやバッチサイズに依存しない安定した正規化が得られます。

## 3. 正規化手法の比較

| 手法 | 正規化の単位 | 使用場所 |
|---|---|---|
| LayerNorm | 最後の次元（768 次元）全体 | GPT-2 の各層、CLIP |
| BatchNorm | バッチ×空間（チャネルごと） | 画像分類（ResNet 等） |
| GroupNorm | グループ×空間 | SD 1.5 の U-Net、VAE |

BatchNorm はバッチサイズに依存するため、推論時（バッチサイズ 1）では学習時の移動平均を使う必要があります。GroupNorm はバッチサイズに依存しないため、推論でも学習と同じ計算ができます。

## 4. 実装

```python
def group_norm(x, weight, bias, num_groups, eps=1e-5):
    shape = x.shape
    C = shape[0]
    x = x.reshape(num_groups, -1)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x.reshape(shape)
    extra_dims = len(shape) - 1
    view_shape = (C,) + (1,) * extra_dims
    x = x * weight.reshape(view_shape) + bias.reshape(view_shape)
    return x
```

LayerNorm との核心的な違いは `reshape(num_groups, -1)` の 1 行だけです。チャネルをグループに分割し、各グループ内で平均と分散を計算しています。最後に `weight` と `bias`（チャネルごとのパラメータ）でスケール・シフトする点は LayerNorm と同じです。

## 実験：GroupNorm の動作確認

小さな入力で GroupNorm を計算し、PyTorch の `F.group_norm` と比較します。実行結果は以下のとおりです。

```
=== 1. 基本動作 ===
入力:     (4, 3, 3)
グループ数: 2 (各グループ 2 チャネル)
最大誤差:  1.19e-07

=== 2. 正規化の効果 ===
グループ 0 正規化前: mean=-0.0392, std=1.1984
グループ 0 正規化後: mean=0.0000, std=1.0000

=== 3. SD 1.5 の典型サイズ ===
入力:     (320, 32, 32)
グループ数: 32 (各グループ 10 チャネル)
最大誤差:  1.91e-06
```

正規化後のグループ 0 は平均 0.0000、標準偏差 1.0000 になっており、正規化が正しく機能しています。SD 1.5 の典型的なサイズ (320, 32, 32) でも PyTorch との誤差は $10^{-6}$ 程度です。

**実行方法**: ([05_groupnorm.py](05_groupnorm.py))

```bash
uv run docs/05_groupnorm.py
```

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | **05** | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_architecture.md)
