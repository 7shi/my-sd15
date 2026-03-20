"""06: ResBlock の動作確認。"""

import math

import torch
from my_sd15.loader import load_model

model = load_model()

# 1. VaeResBlock（同チャネル数）
print("=== 1. VaeResBlock (512 → 512) ===")
from my_sd15.vae import VaeResBlock
vae_block = VaeResBlock(model.vae.state, "decoder.mid_block.resnets.0.", 512, 512)
x_vae = torch.randn(512, 4, 4)
with torch.no_grad():
    y_vae = vae_block(x_vae)
print(f"入力:  {tuple(x_vae.shape)}")
print(f"出力:  {tuple(y_vae.shape)}")
residual = (y_vae - x_vae).abs().mean().item()
print(f"残差の平均絶対値: {residual:.4f}")
print()

# 2. VaeResBlock（異チャネル数）
print("=== 2. VaeResBlock (512 → 256) ===")
vae_block2 = VaeResBlock(model.vae.state, "decoder.up_blocks.2.resnets.0.", 512, 256)
x_vae2 = torch.randn(512, 8, 8)
with torch.no_grad():
    y_vae2 = vae_block2(x_vae2)
print(f"入力:  {tuple(x_vae2.shape)}")
print(f"出力:  {tuple(y_vae2.shape)} (1×1 conv でチャネル数を変換)")
print()

# 3. タイムステップ埋め込み
print("=== 3. タイムステップ埋め込み ===")
from my_sd15.unet import UNet
unet = UNet(model.unet.state)
with torch.no_grad():
    temb_900 = unet._timestep_embedding(900)
    temb_100 = unet._timestep_embedding(100)
print(f"t=900 の埋め込み: shape={tuple(temb_900.shape)}, mean={temb_900.mean():.4f}")
print(f"t=100 の埋め込み: shape={tuple(temb_100.shape)}, mean={temb_100.mean():.4f}")
cos_sim = torch.cosine_similarity(temb_900.unsqueeze(0), temb_100.unsqueeze(0)).item()
print(f"t=900 と t=100 のコサイン類似度: {cos_sim:.4f}")
print()

# 4. UNetResBlock
print("=== 4. UNetResBlock (320 → 320) ===")
from my_sd15.unet import UNetResBlock
unet_block = UNetResBlock(model.unet.state, "down_blocks.0.resnets.0.", 320, 320)
x_unet = torch.randn(320, 32, 32)
with torch.no_grad():
    y_unet = unet_block(x_unet, temb_900)
print(f"入力:  {tuple(x_unet.shape)}")
print(f"temb:  {tuple(temb_900.shape)}")
print(f"出力:  {tuple(y_unet.shape)}")
