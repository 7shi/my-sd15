"""U-Net for SD 1.5."""

import math

import torch

from my_sd15.ops import (
    conv2d,
    gelu,
    group_norm,
    layer_norm,
    linear,
    silu,
    softmax,
    upsample_nearest_2d,
)


class UNetResBlock:
    def __init__(self, state, prefix, c_in, c_out):
        self.state = state
        self.prefix = prefix
        self.c_in = c_in
        self.c_out = c_out

    def __call__(self, x, temb):
        s = self.state
        p = self.prefix
        h = silu(group_norm(x, s[p + "norm1.weight"], s[p + "norm1.bias"], 32))
        h = conv2d(h, s[p + "conv1.weight"], s[p + "conv1.bias"], padding=1)
        # Timestep embedding
        t = linear(silu(temb), s[p + "time_emb_proj.weight"], s[p + "time_emb_proj.bias"])
        h = h + t.reshape(-1, 1, 1)
        h = silu(group_norm(h, s[p + "norm2.weight"], s[p + "norm2.bias"], 32))
        h = conv2d(h, s[p + "conv2.weight"], s[p + "conv2.bias"], padding=1)
        if self.c_in != self.c_out:
            x = conv2d(x, s[p + "conv_shortcut.weight"], s[p + "conv_shortcut.bias"])
        return x + h


class CrossAttention:
    def __init__(self, state, prefix, dim, num_heads=8):
        self.state = state
        self.prefix = prefix
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def __call__(self, x, context=None):
        if context is None:
            context = x
        s = self.state
        p = self.prefix
        q = x @ s[p + "to_q.weight"].T
        k = context @ s[p + "to_k.weight"].T
        v = context @ s[p + "to_v.weight"].T

        seq = q.shape[0]
        ctx_seq = k.shape[0]
        nh = self.num_heads
        hd = self.head_dim

        q = q.reshape(seq, nh, hd).permute(1, 0, 2)
        k = k.reshape(ctx_seq, nh, hd).permute(1, 0, 2)
        v = v.reshape(ctx_seq, nh, hd).permute(1, 0, 2)

        attn = softmax(q @ k.transpose(-2, -1) / (hd ** 0.5))
        out = (attn @ v).permute(1, 0, 2).reshape(seq, self.dim)
        out = linear(out, s[p + "to_out.0.weight"], s[p + "to_out.0.bias"])
        return out


class GEGLU:
    def __init__(self, state, prefix):
        self.state = state
        self.prefix = prefix

    def __call__(self, x):
        s = self.state
        p = self.prefix
        h = linear(x, s[p + "net.0.proj.weight"], s[p + "net.0.proj.bias"])
        h, gate = h.chunk(2, dim=-1)
        h = h * gelu(gate)
        h = linear(h, s[p + "net.2.weight"], s[p + "net.2.bias"])
        return h


class BasicTransformerBlock:
    def __init__(self, state, prefix, dim):
        self.state = state
        self.prefix = prefix
        self.attn1 = CrossAttention(state, prefix + "attn1.", dim)
        self.attn2 = CrossAttention(state, prefix + "attn2.", dim)
        self.ff = GEGLU(state, prefix + "ff.")

    def __call__(self, x, context):
        s = self.state
        p = self.prefix
        x = x + self.attn1(layer_norm(x, s[p + "norm1.weight"], s[p + "norm1.bias"]))
        x = x + self.attn2(layer_norm(x, s[p + "norm2.weight"], s[p + "norm2.bias"]), context)
        x = x + self.ff(layer_norm(x, s[p + "norm3.weight"], s[p + "norm3.bias"]))
        return x


class SpatialTransformer:
    def __init__(self, state, prefix, dim):
        self.state = state
        self.prefix = prefix
        self.block = BasicTransformerBlock(state, prefix + "transformer_blocks.0.", dim)

    def __call__(self, x, context):
        s = self.state
        p = self.prefix
        C, H, W = x.shape
        residual = x
        x = group_norm(x, s[p + "norm.weight"], s[p + "norm.bias"], 32)
        x = conv2d(x, s[p + "proj_in.weight"], s[p + "proj_in.bias"])
        x = x.reshape(C, H * W).T  # (H*W, C)
        x = self.block(x, context)
        x = x.T.reshape(C, H, W)
        x = conv2d(x, s[p + "proj_out.weight"], s[p + "proj_out.bias"])
        return residual + x


class UNet:
    def __init__(self, state):
        self.state = state

    def _timestep_embedding(self, timestep):
        half = 160
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args = float(timestep) * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)])
        s = self.state
        emb = linear(emb, s["time_embedding.linear_1.weight"], s["time_embedding.linear_1.bias"])
        emb = silu(emb)
        emb = linear(emb, s["time_embedding.linear_2.weight"], s["time_embedding.linear_2.bias"])
        return emb

    def __call__(self, x, timestep, context):
        s = self.state
        temb = self._timestep_embedding(timestep)

        # conv_in
        x = conv2d(x, s["conv_in.weight"], s["conv_in.bias"], padding=1)

        skips = [x]

        # Down blocks
        # Block 0: CrossAttnDown, 320->320
        for j in range(2):
            p = f"down_blocks.0.resnets.{j}."
            x = UNetResBlock(s, p, 320, 320)(x, temb)
            x = SpatialTransformer(s, f"down_blocks.0.attentions.{j}.", 320)(x, context)
            skips.append(x)
        x = conv2d(x, s["down_blocks.0.downsamplers.0.conv.weight"],
                   s["down_blocks.0.downsamplers.0.conv.bias"], stride=2, padding=1)
        skips.append(x)

        # Block 1: CrossAttnDown, 320->640
        c_ins = [320, 640]
        for j in range(2):
            p = f"down_blocks.1.resnets.{j}."
            x = UNetResBlock(s, p, c_ins[j], 640)(x, temb)
            x = SpatialTransformer(s, f"down_blocks.1.attentions.{j}.", 640)(x, context)
            skips.append(x)
        x = conv2d(x, s["down_blocks.1.downsamplers.0.conv.weight"],
                   s["down_blocks.1.downsamplers.0.conv.bias"], stride=2, padding=1)
        skips.append(x)

        # Block 2: CrossAttnDown, 640->1280
        c_ins = [640, 1280]
        for j in range(2):
            p = f"down_blocks.2.resnets.{j}."
            x = UNetResBlock(s, p, c_ins[j], 1280)(x, temb)
            x = SpatialTransformer(s, f"down_blocks.2.attentions.{j}.", 1280)(x, context)
            skips.append(x)
        x = conv2d(x, s["down_blocks.2.downsamplers.0.conv.weight"],
                   s["down_blocks.2.downsamplers.0.conv.bias"], stride=2, padding=1)
        skips.append(x)

        # Block 3: Down, 1280->1280
        for j in range(2):
            p = f"down_blocks.3.resnets.{j}."
            x = UNetResBlock(s, p, 1280, 1280)(x, temb)
            skips.append(x)

        # Mid block
        x = UNetResBlock(s, "mid_block.resnets.0.", 1280, 1280)(x, temb)
        x = SpatialTransformer(s, "mid_block.attentions.0.", 1280)(x, context)
        x = UNetResBlock(s, "mid_block.resnets.1.", 1280, 1280)(x, temb)

        # Up blocks
        # Block 0: UpBlock, 1280
        up0_ins = [2560, 2560, 2560]
        for j in range(3):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=0)
            p = f"up_blocks.0.resnets.{j}."
            x = UNetResBlock(s, p, up0_ins[j], 1280)(x, temb)
        x = upsample_nearest_2d(x, 2)
        x = conv2d(x, s["up_blocks.0.upsamplers.0.conv.weight"],
                   s["up_blocks.0.upsamplers.0.conv.bias"], padding=1)

        # Block 1: CrossAttnUp, 1280
        up1_ins = [2560, 2560, 1920]
        for j in range(3):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=0)
            p = f"up_blocks.1.resnets.{j}."
            x = UNetResBlock(s, p, up1_ins[j], 1280)(x, temb)
            x = SpatialTransformer(s, f"up_blocks.1.attentions.{j}.", 1280)(x, context)
        x = upsample_nearest_2d(x, 2)
        x = conv2d(x, s["up_blocks.1.upsamplers.0.conv.weight"],
                   s["up_blocks.1.upsamplers.0.conv.bias"], padding=1)

        # Block 2: CrossAttnUp, 640
        up2_ins = [1920, 1280, 960]
        for j in range(3):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=0)
            p = f"up_blocks.2.resnets.{j}."
            x = UNetResBlock(s, p, up2_ins[j], 640)(x, temb)
            x = SpatialTransformer(s, f"up_blocks.2.attentions.{j}.", 640)(x, context)
        x = upsample_nearest_2d(x, 2)
        x = conv2d(x, s["up_blocks.2.upsamplers.0.conv.weight"],
                   s["up_blocks.2.upsamplers.0.conv.bias"], padding=1)

        # Block 3: CrossAttnUp, 320
        up3_ins = [960, 640, 640]
        for j in range(3):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=0)
            p = f"up_blocks.3.resnets.{j}."
            x = UNetResBlock(s, p, up3_ins[j], 320)(x, temb)
            x = SpatialTransformer(s, f"up_blocks.3.attentions.{j}.", 320)(x, context)

        # Output
        x = silu(group_norm(x, s["conv_norm_out.weight"], s["conv_norm_out.bias"], 32))
        x = conv2d(x, s["conv_out.weight"], s["conv_out.bias"], padding=1)
        return x
