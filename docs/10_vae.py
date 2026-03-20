"""10: VAE Decoder の動作確認。"""

import torch
from my_sd15.loader import load_model
from my_sd15.model import decode_to_image

model = load_model()

# 1. 入出力の形状
print("=== 1. VAE Decoder の入出力 ===")
latents = torch.randn(4, 32, 32)
with torch.no_grad():
    decoded = model.vae(latents / 0.18215)
print(f"入力（潜在表現）: {tuple(latents.shape)}")
print(f"スケーリング後:   {tuple((latents / 0.18215).shape)}")
print(f"出力（画像）:     {tuple(decoded.shape)}")
print(f"空間の拡大: {latents.shape[1]}×{latents.shape[2]} → {decoded.shape[1]}×{decoded.shape[2]} (×{decoded.shape[1] // latents.shape[1]})")
print(f"チャネル:   {latents.shape[0]} → {decoded.shape[0]}")
print()

# 2. 圧縮率
print("=== 2. 圧縮率 ===")
pixel_size = 3 * 256 * 256
latent_size = 4 * 32 * 32
print(f"ピクセル空間: {pixel_size:,} 要素")
print(f"潜在空間:     {latent_size:,} 要素")
print(f"圧縮率:       {pixel_size / latent_size:.0f}×")
print()

# 3. パラメータ数
print("=== 3. パラメータ数 ===")
s = model.vae.state
total = sum(v.numel() for v in s.values())
print(f"VAE Decoder 総パラメータ: {total:,} ({total / 1e6:.0f}M)")
print()

# 4. スケーリング係数の効果
print("=== 4. スケーリング係数の効果 ===")
with torch.no_grad():
    decoded_scaled = model.vae(latents / 0.18215)
    decoded_unscaled = model.vae(latents)
print(f"スケーリングあり: mean={decoded_scaled.mean():.4f}, std={decoded_scaled.std():.4f}")
print(f"スケーリングなし: mean={decoded_unscaled.mean():.4f}, std={decoded_unscaled.std():.4f}")
print(f"スケーリングにより入力の振幅が {1/0.18215:.1f} 倍に拡大")
print()

# 5. 画像への変換
print("=== 5. 画像への変換 ===")
image = decode_to_image(decoded_scaled)
print(f"PIL Image: size={image.size}, mode={image.mode}")
print(f"値の範囲: decoded [{decoded_scaled.min():.2f}, {decoded_scaled.max():.2f}] → pixel [0, 255]")
