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
| 7 | `my_sd15/model.py`, `my_sd15/__init__.py` | テキスト→画像パイプライン（CFG）、CLI、Sixel 表示 | 5 |

## 重みのダウンロード

`make download` で Hugging Face から SD 1.5 と Anything V5 の重みをダウンロードする。

```bash
make download
```

Anything V5 のような単一ファイル形式のモデルは、`single2dir.py` で Diffusers 形式（コンポーネント別ディレクトリ）に自動分割される。

## 使い方

画像生成（Sixel 対応ターミナルで画像表示）

```bash
uv run my-sd15 -m genai-archive/anything-v5 -p "a cat sitting on a windowsill" --seed 123 --steps 10 --cfg 7.5 -o output.png
```

## デノイジング過程の可視化

`gen_steps.py` はデノイジングループの各ステップで中間画像を `steps/` に保存する。各ステップで latents の先頭 3 チャンネルをそのまま画像化したもの（`xx-1.png`）と、VAE でデコードしたもの（`xx-2.jpg`）の 2 種類を出力する。Sixel 対応ターミナルではリアルタイムに表示される。

```bash
uv run gen_steps.py
```

## テスト

`gen_testdata.py` で diffusers パイプラインから各コンポーネントの入出力を safetensors として `testdata/` に保存する。pytest はこのテストデータを使って、スクラッチ実装の出力を検証する。

```bash
uv run gen_testdata.py  # テストデータ生成
uv run pytest           # テスト実行
```

## 依存関係の補足

コード自体は numpy を使用していないが、numpy を未インストールの状態で PyTorch をインポートすると、PyTorch の C++ 初期化コード（`tensor_numpy.cpp`）が NumPy との連携を試みて以下の警告を出力する：

```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

この警告は Python レベルで回避できないため、numpy を通常の依存関係として追加している。
