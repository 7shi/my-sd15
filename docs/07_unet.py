"""07: U-Net の各段階のテンソル形状を追跡する。"""

import torch
from my_sd15.loader import load_model
from my_sd15.ops import conv2d, group_norm, silu
from my_sd15.unet import UNetResBlock, SpatialTransformer

model = load_model()
s = model.unet.state

# テスト用の入力
x = torch.randn(4, 32, 32)
context = torch.randn(77, 768)  # CLIP 出力（ダミー）
with torch.no_grad():
    temb = model.unet._timestep_embedding(900)

print("=== U-Net のテンソル形状追跡 ===")
print(f"入力:      {tuple(x.shape)}")
print(f"context:   {tuple(context.shape)}")
print(f"temb:      {tuple(temb.shape)}")
print()

with torch.no_grad():
    # conv_in
    x = conv2d(x, s["conv_in.weight"], s["conv_in.bias"], padding=1)
    print(f"conv_in:   {tuple(x.shape)}")

    skips = [x]

    # Down block 0
    for j in range(2):
        x = UNetResBlock(s, f"down_blocks.0.resnets.{j}.", 320, 320)(x, temb)
        x = SpatialTransformer(s, f"down_blocks.0.attentions.{j}.", 320)(x, context)
        skips.append(x)
    x = conv2d(x, s["down_blocks.0.downsamplers.0.conv.weight"],
               s["down_blocks.0.downsamplers.0.conv.bias"], stride=2, padding=1)
    skips.append(x)
    print(f"Down 0:    {tuple(x.shape)} (skips: {len(skips)})")

    # Down block 1
    c_ins = [320, 640]
    for j in range(2):
        x = UNetResBlock(s, f"down_blocks.1.resnets.{j}.", c_ins[j], 640)(x, temb)
        x = SpatialTransformer(s, f"down_blocks.1.attentions.{j}.", 640)(x, context)
        skips.append(x)
    x = conv2d(x, s["down_blocks.1.downsamplers.0.conv.weight"],
               s["down_blocks.1.downsamplers.0.conv.bias"], stride=2, padding=1)
    skips.append(x)
    print(f"Down 1:    {tuple(x.shape)} (skips: {len(skips)})")

    # Down block 2
    c_ins = [640, 1280]
    for j in range(2):
        x = UNetResBlock(s, f"down_blocks.2.resnets.{j}.", c_ins[j], 1280)(x, temb)
        x = SpatialTransformer(s, f"down_blocks.2.attentions.{j}.", 1280)(x, context)
        skips.append(x)
    x = conv2d(x, s["down_blocks.2.downsamplers.0.conv.weight"],
               s["down_blocks.2.downsamplers.0.conv.bias"], stride=2, padding=1)
    skips.append(x)
    print(f"Down 2:    {tuple(x.shape)} (skips: {len(skips)})")

    # Down block 3
    for j in range(2):
        x = UNetResBlock(s, f"down_blocks.3.resnets.{j}.", 1280, 1280)(x, temb)
        skips.append(x)
    print(f"Down 3:    {tuple(x.shape)} (skips: {len(skips)})")

    print()
    print(f"skips の形状:")
    for i, sk in enumerate(skips):
        print(f"  [{i}]: {tuple(sk.shape)}")
    print()

    # Mid block
    from my_sd15.unet import UNetResBlock, SpatialTransformer
    x = UNetResBlock(s, "mid_block.resnets.0.", 1280, 1280)(x, temb)
    x = SpatialTransformer(s, "mid_block.attentions.0.", 1280)(x, context)
    x = UNetResBlock(s, "mid_block.resnets.1.", 1280, 1280)(x, temb)
    print(f"Mid:       {tuple(x.shape)}")

    # 以降の Up blocks の各段階の入力チャネル数を表示
    print()
    print(f"Up blocks のスキップ結合:")
    # Up block 0 の最初の結合
    sk = skips[-1]
    print(f"  Up 0: x {tuple(x.shape)} + skip {tuple(sk.shape)} → cat {x.shape[0] + sk.shape[0]} ch")

print()
print("=== パラメータ数 ===")
total = sum(v.numel() for v in s.values())
print(f"U-Net 総パラメータ: {total:,} ({total / 1e6:.0f}M)")
