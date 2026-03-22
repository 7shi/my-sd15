"""12: LoRA の構造確認と LCM LoRA による画像生成。"""

import struct
import json
from collections import Counter

import torch

# === 1. LoRA ファイルの構造 ===
print("=== 1. LoRA ファイルの構造 ===")

lora_path = "weights/latent-consistency/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors"
with open(lora_path, "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_size))

keys = sorted(k for k in header if k != "__metadata__")
bases = set()
for k in keys:
    base = k.replace(".lora_down.weight", "").replace(".lora_up.weight", "").replace(".alpha", "")
    bases.add(base)

print("各ターゲットのキー:")
print("  *.alpha              (スカラー: スケーリング係数)")
print("  *.lora_down.weight   (A 行列: 入力 → r)")
print("  *.lora_up.weight     (B 行列: r → 出力)")
print()

# ランクと alpha の確認
sample_base = sorted(bases)[0]
down_shape = header[sample_base + ".lora_down.weight"]["shape"]
rank = down_shape[0]

# alpha の値を読む
with open(lora_path, "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    header_bytes = f.read(header_size)
    header = json.loads(header_bytes)
    alpha_key = sample_base + ".alpha"
    start, end = header[alpha_key]["data_offsets"]
    f.seek(8 + header_size + start)
    alpha_val = struct.unpack("<e", f.read(end - start))[0]

print(f"LoRA ターゲット数: {len(bases)}")
print(f"ランク r: {rank}, alpha: {alpha_val}, scale: {alpha_val / rank}")
print()

# === 2. 適用対象の内訳 ===
print("=== 2. 適用対象の内訳 ===")
categories = Counter()
for b in bases:
    name = b.removeprefix("lora_unet_")
    if "attn2_to_" in name:
        categories["Cross-Attention (to_q/k/v/out)"] += 1
    elif "attn1_to_" in name:
        categories["Self-Attention (to_q/k/v/out)"] += 1
    elif "ff_net" in name:
        categories["FFN (GEGLU)"] += 1
    elif "proj_in" in name or "proj_out" in name:
        categories["proj_in / proj_out"] += 1
    else:
        categories["ResBlock / Conv"] += 1

for cat in ["ResBlock / Conv", "FFN (GEGLU)", "proj_in / proj_out",
            "Self-Attention (to_q/k/v/out)", "Cross-Attention (to_q/k/v/out)"]:
    print(f"  {cat + ':':38s}{categories[cat]:3d}")
print()

# === 3. shape の例 ===
print("=== 3. shape の例 ===")
# Linear 層の例
linear_base = [b for b in sorted(bases) if "attn1_to_q" in b][0]
ld = header[linear_base + ".lora_down.weight"]["shape"]
lu = header[linear_base + ".lora_up.weight"]["shape"]
print(f"Linear 層:")
print(f"  down: {tuple(ld)}, up: {tuple(lu)}")
print(f"  → delta: ({lu[0]}, {ld[1]})")

# Conv 層の例
conv_base = [b for b in sorted(bases) if "resnets" in b and "conv1" in b][0]
cd = header[conv_base + ".lora_down.weight"]["shape"]
cu = header[conv_base + ".lora_up.weight"]["shape"]
print(f"Conv 層:")
print(f"  down: {tuple(cd)}, up: {tuple(cu)}")
print(f"  → delta: ({cu[0]}, {cd[1]}, {cd[2]}, {cd[3]})")
print()

# === 4. LCM スケジューラー ===
print("=== 4. LCM スケジューラー ===")
from my_sd15.scheduler import LCMScheduler

lcm = LCMScheduler()
lcm.set_timesteps(2)
print(f"タイムステップ (2 steps): {lcm.timesteps.tolist()}")
print(f"prev_timestep: {lcm._prev_timestep}")
print()

# === 5. 画像生成 ===
print("=== 5. 画像生成 (2 steps) ===")
from my_sd15.loader import load_model
from my_sd15.model import save_image

model = load_model(lora_path=lora_path, scheduler=LCMScheduler())
image = model.generate(
    prompt="a cat sitting on a windowsill",
    seed=42, steps=2, cfg_scale=1.0,
    height=512, width=512, show_progress=True,
)
save_image("images/lcm_lora.png", image, show=True, mkdir=True)
print("生成完了: images/lcm_lora.png")
