ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | **12**

---

# アーキテクチャ: SD 1.5 の設計思想

最終章では、SD 1.5 の全体像を振り返り、GPT-2 との構造比較を通じてアーキテクチャの設計思想を整理します。

## 1. パイプラインの要約

SD 1.5 の推論パイプラインは、わずか 20 行足らずのコードに集約されます。

```python
# 1. テキスト → 条件ベクトル
cond_emb = text_encoder(tokenizer.encode(prompt))        # (77, 768)
uncond_emb = text_encoder(tokenizer.encode(""))           # (77, 768)

# 2. ランダムノイズ
latents = torch.randn(4, height // 8, width // 8)        # (4, 32, 32)

# 3. デノイジングループ
for t in scheduler.timesteps:
    noise_cond = unet(latents, t, cond_emb)               # (4, 32, 32)
    noise_uncond = unet(latents, t, uncond_emb)            # (4, 32, 32)
    noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
    latents = scheduler.step(noise_pred, t, latents)       # (4, 32, 32)

# 4. 画像化
image = vae(latents / 0.18215)                            # (3, 256, 256)
```

4 つのコンポーネントが明確に分離され、それぞれが独立した役割を担っています。

## 2. GPT-2 との構造比較

### 共通点

| 概念 | GPT-2 | SD 1.5 |
|---|---|---|
| Attention | Self-Attention（12 ヘッド×64 次元） | Self/Cross-Attention（8 or 12 ヘッド） |
| 正規化 | LayerNorm | LayerNorm（CLIP）+ GroupNorm（U-Net/VAE） |
| 活性化関数 | GELU | SiLU（U-Net/VAE）、Quick GELU（CLIP）、GELU（GEGLU） |
| 残差接続 | `x = x + f(x)` | `x = x + f(x)`（ResBlock、SpatialTransformer） |
| 埋め込み | 位置埋め込み（学習済み） | 位置埋め込み + タイムステップ埋め込み（正弦波） |
| BPE | バイトレベル BPE | バイトレベル BPE（小文字化 + `</w>`） |

### 差異

| 概念 | GPT-2 | SD 1.5 |
|---|---|---|
| 生成方式 | 自己回帰（1 トークンずつ） | 拡散（ノイズ→画像を反復） |
| 主な線形変換 | Linear（全結合） | Conv2D（畳み込み）+ Linear |
| 条件付け | なし | Cross-Attention（テキスト→画像） |
| MLP | Linear → GELU → Linear | GEGLU（ゲート付き） |
| ネットワーク構造 | 単一方向（12 層の積み重ね） | U-Net（エンコーダ・デコーダ + スキップ接続） |
| 潜在空間 | なし（トークン空間で直接） | VAE（48 倍圧縮） |

## 3. 2 つの残差ストリーム

SD 1.5 には 2 種類の残差接続があります。

### ResBlock 内の残差接続

```python
return x + h   # h = GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv
```

GPT-2 の Transformer Block 内の残差接続と同じ原理です。ブロック内の変換を「修正量」として加算します。

### U-Net のスキップ接続

```python
x = torch.cat([x, skip], dim=0)  # チャネル方向に結合
```

こちらは加算ではなく**結合**です。エンコーダの高解像度な空間情報をデコーダに直接伝達します。ResBlock の残差接続が「局所的な情報の保存」であるのに対し、スキップ接続は「エンコーダ・デコーダ間の長距離な情報伝達」です。

## 4. Transformer の汎用性

SD 1.5 は 3 つの異なる文脈で Transformer を使っています。

| コンポーネント | 構造 | 入力 |
|---|---|---|
| CLIP Text Encoder | GPT-2 類似（因果マスク付き） | テキストトークン系列 (77, 768) |
| SpatialTransformer | Self-Attn + Cross-Attn + GEGLU | 画像特徴 (H×W, C) |
| VaeAttention | 単一ヘッド Self-Attn | 画像特徴 (H×W, C) |

核心は同じ `softmax(Q·K^T / √d)·V` の計算です。テキストに適用すれば言語モデル、画像に適用すれば画像処理、テキストと画像を結べば条件付き生成になります。Transformer の「系列に対する重み付き集約」という抽象的な操作が、さまざまなモダリティに適用できることを SD 1.5 は示しています。

## 5. SD 1.5 からの発展

SD 1.5 以降、画像生成モデルは急速に進化しています。

### SDXL (2023)

SD 1.5 の拡張版。U-Net のチャネル数を増加、テキストエンコーダを 2 つ（CLIP + OpenCLIP）に増やし、高解像度（1024×1024）に対応。基本アーキテクチャは SD 1.5 と同じです。

### SD 3 / Flux (2024)

U-Net を廃止し、**DiT（Diffusion Transformer）** を採用。画像パッチをトークンとして扱い、テキストトークンと結合して単一の Transformer で処理します。SpatialTransformer の「画像を系列に変換する」アイデアを全面的に採用した形です。

### 一貫性モデル (Consistency Models)

拡散モデルの「多ステップ」という制約を取り除き、1〜2 ステップで画像を生成する手法。DDIM の「決定的な軌跡」の考え方を発展させています。

これらの発展はいずれも、SD 1.5 で学んだ概念（Attention、残差接続、潜在空間、CFG）の延長線上にあります。SD 1.5 の理解は、最新の画像生成モデルを学ぶ確かな土台になるはずです。

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | **12**
