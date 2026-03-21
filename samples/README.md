# Sample Images

`make run` により、画像をまとめて生成します。👉[Makefile](../Makefile)

共通パラメーター: `-p "a cat sitting on a windowsill" --seed 123 --steps 10 --cfg 7.5`

CPU (WSL2) での実行時間を併記しています。

## Stable Diffusion 1.5

モデル: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

256x256 は SD 1.5 の訓練解像度（512x512）と異なるため、まともな画像が生成できません。512x512 ではプロンプト通りの画像が得られます。

| 256x256 (0m28s) | 512x512 (3m11s) |
|:---:|:---:|
| ![sd15-256x256](sd15-256x256.jpg) | ![sd15-512x512](sd15-512x512.jpg) |

## miniSD

モデル: https://huggingface.co/webui/miniSD （[justinpinkney/miniSD](https://huggingface.co/justinpinkney/miniSD) の safetensors 変換版）

miniSD は SD 1.4 を 256x256 で訓練した軽量モデルです。256x256 でも比較的良好な画像が生成されますが、512x512 ではパターンの繰り返しが見られます。

| 256x256 (0m28s) | 512x512 (3m38s) |
|:---:|:---:|
| ![minisd-256x256](minisd-256x256.jpg) | ![minisd-512x512](minisd-512x512.jpg) |

## Anything V5

モデル: https://huggingface.co/genai-archive/anything-v5

Anything V5 はアニメに特化したモデルです。256x256 でも形状は認識できますが、細部がやや崩れています。512x512 ではより精細な画像になります。

| 256x256 (0m28s) | 512x512 (3m05s) |
|:---:|:---:|
| ![any5-256x256](any5-256x256.jpg) | ![any5-512x512](any5-512x512.jpg) |
