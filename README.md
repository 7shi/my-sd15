# my-sd15: Stable Diffusion 1.5 クリーンルーム再実装

Stable Diffusion 1.5 の推論パイプラインを PyTorch の基本演算からスクラッチ実装するプロジェクト。

## 現状

全 7 Stage の実装が完了し、60 テストすべて通過。

| Stage | ファイル | 内容 | テスト数 |
|---|---|---|---|
| 1 | `my_sd15/ops.py` | 基本演算（conv2d im2col, group_norm, layer_norm, linear, silu, gelu, softmax, upsample, embedding） | 27 |
| 2 | `my_sd15/tokenizer.py` | CLIP BPE トークナイザ | 9 |
| 3 | `my_sd15/scheduler.py` | DDIM スケジューラ | 11 |
| 4 | `my_sd15/clip.py` | CLIP Text Encoder（12層 Transformer） | 4 |
| 5 | `my_sd15/vae.py` | VAE Decoder（ResBlock, Attention, UpBlock） | 2 |
| 6 | `my_sd15/unet.py` | U-Net（ResBlock, CrossAttention, GEGLU, SpatialTransformer, Down/Mid/Up blocks） | 2 |
| 7 | `my_sd15/pipeline.py`, `my_sd15/__init__.py` | テキスト→画像パイプライン（CFG）、CLI | 5 |

## 使い方

```bash
# テスト実行
uv run pytest

# 画像生成
uv run my-sd15 -m genai-archive/anything-v5 --prompt "a cat sitting on a windowsill" --seed 123 --steps 10 --cfg 7.5 -o output.png
```

## 必要なファイル

- `weights/stable-diffusion-v1-5/stable-diffusion-v1-5/` — Hugging Face の公式重み
- `testdata/` — 検証用テストデータ（safetensors）

## 依存関係の補足

コード自体は numpy を使用していないが、numpy を未インストールの状態で PyTorch をインポートすると、PyTorch の C++ 初期化コード（`tensor_numpy.cpp`）が NumPy との連携を試みて以下の警告を出力する：

```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

この警告は Python レベルで回避できないため、numpy を通常の依存関係として追加している。
