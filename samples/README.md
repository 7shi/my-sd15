# Sample Images

`make samples` により、画像をまとめて生成します。👉[Makefile](../Makefile)

AMD Ryzen 5 5600X (WSL2) での実行時間を記載しています。実行時間はステップ数にほぼ比例します。本来 30 程度は必要ですが、時間節約のため 10 に減らしているため、画像の細部が甘くなっています。

- 共通パラメーター: `-p "a cat sitting on a windowsill" --seed 123`

## Stable Diffusion 1.5

- モデル: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- 個別パラメーター: `--steps 10 --cfg 7.5`
  ステップ数 10 は、画像が形になる最低限のステップです。👉[steps](../steps/README.md)

256x256 は SD 1.5 の訓練解像度（512x512）と異なるため、まともな画像が生成できません。512x512 ではプロンプト通りの画像が得られます。

| 256x256 (26s) | 512x512 (2m49s) |
|:---:|:---:|
| ![sd15-256x256](sd15-256x256.jpg) | ![sd15-512x512](sd15-512x512.jpg) |

### TAESD 適用

- VAE: https://huggingface.co/madebyollin/taesd
- 個別パラメーター: `--steps 10 --cfg 7.5 --vae madebyollin/taesd`

TAESD (Tiny AutoEncoder) は VAE デコーダーの軽量版（パラメータ数約 1/20）で、Conv + ReLU のみのシンプルな構造により高速にデコードできます。ただし画質が低下するため、最終出力には向きません。

まず TAESD で試行錯誤して、気に入った seed やプロンプトが見つかったら、通常の VAE（`--vae` 指定なし）で再生成するような使い方を推奨します。

| 256x256 (22s) | 512x512 (2m27s) |
|:---:|:---:|
| ![sd15-taesd-256x256](sd15-taesd-256x256.jpg) | ![sd15-taesd-512x512](sd15-taesd-512x512.jpg) |

### LCM LoRA 適用

- LoRA: https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
- 個別パラメーター: `--steps 3 --cfg 1.0 --lcm --lora latent-consistency/lcm-lora-sdv1-5`

LCM LoRA を適用すると、2～4 ステップで画像を生成できます。また `--cfg 1.0` では U-Net の呼び出しが 1 回だけになるため、ステップ数の削減と合わせて大幅な高速化が得られます。ただし Negative Prompt は効きません。また、seed が同じでもスケジューラーが異なるため異なる画像になります。

| 256x256 (12s) | 512x512 (48s) |
|:---:|:---:|
| ![sd15-lcm-256x256](sd15-lcm-256x256.jpg) | ![sd15-lcm-512x512](sd15-lcm-512x512.jpg) |

### LCM LoRA + TAESD 適用

- 個別パラメーター: `--steps 3 --cfg 1.0 --lcm --lora latent-consistency/lcm-lora-sdv1-5 --vae madebyollin/taesd`

LCM LoRA と TAESD の組み合わせると所要時間が大幅に短縮できます。ただし画質は低下します。

| 256x256 (9s) | 512x512 (31s) |
|:---:|:---:|
| ![sd15-lcm-taesd-256x256](sd15-lcm-taesd-256x256.jpg) | ![sd15-lcm-taesd-512x512](sd15-lcm-taesd-512x512.jpg) |

## miniSD

- モデル: https://huggingface.co/webui/miniSD （[justinpinkney/miniSD](https://huggingface.co/justinpinkney/miniSD) の safetensors 変換版）
- 個別パラメーター: `--steps 10 --cfg 7.5`

miniSD は SD 1.4 を 256x256 でチューニングしたモデルです。256x256 でも比較的良好な画像が生成されます。想定解像度より大きい 512x512 では、パターンの繰り返しが見られます。

| 256x256 (27s) | 512x512 (2m55s) |
|:---:|:---:|
| ![minisd-256x256](minisd-256x256.jpg) | ![minisd-512x512](minisd-512x512.jpg) |

### TAESD 適用

| 256x256 (22s) | 512x512 (2m30s) |
|:---:|:---:|
| ![minisd-taesd-256x256](minisd-taesd-256x256.jpg) | ![minisd-taesd-512x512](minisd-taesd-512x512.jpg) |

### LCM LoRA 適用

| 256x256 (12s) | 512x512 (48s) |
|:---:|:---:|
| ![minisd-lcm-256x256](minisd-lcm-256x256.jpg) | ![minisd-lcm-512x512](minisd-lcm-512x512.jpg) |

### LCM LoRA + TAESD 適用

| 256x256 (9s) | 512x512 (31s) |
|:---:|:---:|
| ![minisd-lcm-taesd-256x256](minisd-lcm-taesd-256x256.jpg) | ![minisd-lcm-taesd-512x512](minisd-lcm-taesd-512x512.jpg) |

## Anything V5

- モデル: https://huggingface.co/genai-archive/anything-v5
- 個別パラメーター: `--steps 10 --cfg 7.5`

Anything V5 はアニメ絵に特化したモデルです。256x256 でも形状は認識できますが、細部がやや崩れています。512x512 ではより精細な画像になります。

| 256x256 (27s) | 512x512 (2m56s) |
|:---:|:---:|
| ![any5-256x256](any5-256x256.jpg) | ![any5-512x512](any5-512x512.jpg) |

### TAESD 適用

画像にメリハリが出て、アニメ風の絵には合っているように見えます。

| 256x256 (23s) | 512x512 (2m30s) |
|:---:|:---:|
| ![any5-taesd-256x256](any5-taesd-256x256.jpg) | ![any5-taesd-512x512](any5-taesd-512x512.jpg) |

### LCM LoRA 適用

| 256x256 (12s) | 512x512 (48s) |
|:---:|:---:|
| ![any5-lcm-256x256](any5-lcm-256x256.jpg) | ![any5-lcm-512x512](any5-lcm-512x512.jpg) |

### LCM LoRA + TAESD 適用

| 256x256 (9s) | 512x512 (30s) |
|:---:|:---:|
| ![any5-lcm-taesd-256x256](any5-lcm-taesd-256x256.jpg) | ![any5-lcm-taesd-512x512](any5-lcm-taesd-512x512.jpg) |
