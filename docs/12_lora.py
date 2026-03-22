"""12: LoRA の構造確認。"""

from collections import Counter
from safetensors import safe_open

# === 1. LoRA ファイルの構造 ===
print("=== 1. LoRA ファイルの構造 ===")

lora_path = "weights/latent-consistency/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors"
with safe_open(lora_path, framework="pt") as f:
    all_keys = sorted(f.keys())
    bases = set()
    for k in all_keys:
        base = k.replace(".lora_down.weight", "").replace(".lora_up.weight", "").replace(".alpha", "")
        bases.add(base)

    sample_base = sorted(bases)[0]
    rank = f.get_tensor(sample_base + ".lora_down.weight").shape[0]
    alpha_val = f.get_tensor(sample_base + ".alpha").item()

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
with safe_open(lora_path, framework="pt") as f:
    # Linear 層の例
    linear_base = [b for b in sorted(bases) if "attn1_to_q" in b][0]
    ld = f.get_tensor(linear_base + ".lora_down.weight").shape
    lu = f.get_tensor(linear_base + ".lora_up.weight").shape
    print(f"Linear 層:")
    print(f"  down: {tuple(ld)}, up: {tuple(lu)}")
    print(f"  → delta: ({lu[0]}, {ld[1]})")

    # Conv 層の例
    conv_base = [b for b in sorted(bases) if "resnets" in b and "conv1" in b][0]
    cd = f.get_tensor(conv_base + ".lora_down.weight").shape
    cu = f.get_tensor(conv_base + ".lora_up.weight").shape
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
