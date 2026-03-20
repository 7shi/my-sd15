"""11: パイプラインの各段階を追跡し、CFG の効果を確認する。"""

import torch
from my_sd15.loader import load_model

model = load_model()

# 1. テキストエンコーディング
print("=== 1. テキストエンコーディング ===")
prompt = "a cat sitting on a windowsill"
negative = ""

cond_tokens = model.tokenizer.encode(prompt)
uncond_tokens = model.tokenizer.encode(negative)

with torch.no_grad():
    cond_emb = model.text_encoder(cond_tokens)
    uncond_emb = model.text_encoder(uncond_tokens)

print(f"条件付きベクトル:  {tuple(cond_emb.shape)}")
print(f"無条件ベクトル:    {tuple(uncond_emb.shape)}")
diff = (cond_emb - uncond_emb).abs().mean().item()
print(f"条件付き vs 無条件の平均絶対差: {diff:.4f}")
print()

# 2. デノイジングの追跡
print("=== 2. デノイジングの追跡 ===")
model.scheduler.set_timesteps(10)
latents = torch.randn(4, 32, 32, generator=torch.manual_seed(123))
print(f"初期 latents: mean={latents.mean():.4f}, std={latents.std():.4f}")

with torch.no_grad():
    for i, t in enumerate(model.scheduler.timesteps):
        t_int = int(t)
        noise_cond = model.unet(latents, t_int, cond_emb)
        noise_uncond = model.unet(latents, t_int, uncond_emb)

        # CFG の効果を可視化
        guidance = noise_cond - noise_uncond
        noise_pred = noise_uncond + 7.5 * guidance

        latents = model.scheduler.step(noise_pred, t_int, latents)

        if i in [0, 4, 9]:  # 最初、中間、最後のステップ
            print(f"  Step {i} (t={t_int}):")
            print(f"    guidance の大きさ: {guidance.abs().mean():.4f}")
            print(f"    latents: mean={latents.mean():.4f}, std={latents.std():.4f}")

print()

# 3. CFG スケールの効果
print("=== 3. CFG スケールの効果 ===")
latents_orig = torch.randn(4, 32, 32, generator=torch.manual_seed(123))

for cfg_scale in [1.0, 7.5, 15.0]:
    model.scheduler.set_timesteps(10)
    latents = latents_orig.clone()
    with torch.no_grad():
        for t in model.scheduler.timesteps:
            t_int = int(t)
            nc = model.unet(latents, t_int, cond_emb)
            nu = model.unet(latents, t_int, uncond_emb)
            noise_pred = nu + cfg_scale * (nc - nu)
            latents = model.scheduler.step(noise_pred, t_int, latents)
    print(f"  cfg={cfg_scale:5.1f}: latents mean={latents.mean():.4f}, std={latents.std():.4f}")
