"""02: 推論パイプラインの各段階のテンソル形状を追跡する。"""

import torch
from my_sd15.loader import load_model

model = load_model()

# 1. テキスト → トークン ID
prompt = "a cat sitting on a windowsill"
tokens = model.tokenizer.encode(prompt)
print(f"1. トークン化: {len(tokens)} トークン")
print(f"   先頭 10 トークン: {tokens[:10]}")

# 2. トークン ID → 条件ベクトル
with torch.no_grad():
    cond_emb = model.text_encoder(tokens)
print(f"2. CLIP Text Encoder 出力: {tuple(cond_emb.shape)}")

# 3. ランダムノイズ生成
latents = torch.randn(4, 32, 32, generator=torch.manual_seed(123))
print(f"3. 初期ノイズ: {tuple(latents.shape)}")

# 4. デノイジングループ（1 ステップだけ追跡）
model.scheduler.set_timesteps(10)
uncond_emb = model.text_encoder(model.tokenizer.encode(""))
print(f"4. デノイジング（10 ステップ）:")
print(f"   タイムステップ: {model.scheduler.timesteps.tolist()}")
with torch.no_grad():
    t = int(model.scheduler.timesteps[0])
    noise_cond = model.unet(latents, t, cond_emb)
    print(f"   U-Net 出力（ノイズ予測）: {tuple(noise_cond.shape)}")
    noise_uncond = model.unet(latents, t, uncond_emb)
    noise_pred = noise_uncond + 7.5 * (noise_cond - noise_uncond)
    print(f"   CFG 後のノイズ予測: {tuple(noise_pred.shape)}")
    latents_next = model.scheduler.step(noise_pred, t, latents)
    print(f"   Scheduler step 後: {tuple(latents_next.shape)}")

# 5. VAE デコード（最終ステップの latents を使用）
with torch.no_grad():
    decoded = model.vae(latents_next / 0.18215)
print(f"5. VAE Decoder 出力: {tuple(decoded.shape)}")
print(f"   → 画像サイズ: {decoded.shape[1]}x{decoded.shape[2]} (RGB)")
