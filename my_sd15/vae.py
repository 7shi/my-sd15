"""VAE Decoder for SD 1.5."""

import torch

from my_sd15.ops import conv2d, group_norm, linear, silu, softmax, upsample_nearest_2d


class VaeResBlock:
    def __init__(self, state, prefix, c_in, c_out):
        self.state = state
        self.prefix = prefix
        self.c_in = c_in
        self.c_out = c_out

    def __call__(self, x):
        s = self.state
        p = self.prefix
        h = silu(group_norm(x, s[p + "norm1.weight"], s[p + "norm1.bias"], 32))
        h = conv2d(h, s[p + "conv1.weight"], s[p + "conv1.bias"], padding=1)
        h = silu(group_norm(h, s[p + "norm2.weight"], s[p + "norm2.bias"], 32))
        h = conv2d(h, s[p + "conv2.weight"], s[p + "conv2.bias"], padding=1)
        if self.c_in != self.c_out:
            x = conv2d(x, s[p + "conv_shortcut.weight"], s[p + "conv_shortcut.bias"])
        return x + h


class VaeAttention:
    def __init__(self, state, prefix):
        self.state = state
        self.prefix = prefix

    def __call__(self, x):
        s = self.state
        p = self.prefix
        C, H, W = x.shape
        h = group_norm(x, s[p + "group_norm.weight"], s[p + "group_norm.bias"], 32)
        h = h.reshape(C, H * W).T  # (H*W, C)
        q = linear(h, s[p + "query.weight"], s[p + "query.bias"])
        k = linear(h, s[p + "key.weight"], s[p + "key.bias"])
        v = linear(h, s[p + "value.weight"], s[p + "value.bias"])
        attn = softmax(q @ k.T / (C ** 0.5))
        h = attn @ v
        h = linear(h, s[p + "proj_attn.weight"], s[p + "proj_attn.bias"])
        h = h.T.reshape(C, H, W)
        return x + h


class VaeDecoder:
    def __init__(self, state):
        self.state = state

    def __call__(self, x):
        s = self.state
        # post_quant_conv
        x = conv2d(x, s["post_quant_conv.weight"], s["post_quant_conv.bias"])
        # conv_in
        x = conv2d(x, s["decoder.conv_in.weight"], s["decoder.conv_in.bias"], padding=1)

        # Mid block
        p = "decoder.mid_block."
        x = VaeResBlock(s, p + "resnets.0.", 512, 512)(x)
        x = VaeAttention(s, p + "attentions.0.")(x)
        x = VaeResBlock(s, p + "resnets.1.", 512, 512)(x)

        # Up blocks
        block_configs = [
            # (c_in for resnet.0, c_out, has_upsample)
            [(512, 512), (512, 512), (512, 512), True],
            [(512, 512), (512, 512), (512, 512), True],
            [(512, 256), (256, 256), (256, 256), True],
            [(256, 128), (128, 128), (128, 128), False],
        ]
        for i, (r0, r1, r2, has_up) in enumerate(block_configs):
            bp = f"decoder.up_blocks.{i}."
            for j, (ci, co) in enumerate([r0, r1, r2]):
                x = VaeResBlock(s, bp + f"resnets.{j}.", ci, co)(x)
            if has_up:
                x = upsample_nearest_2d(x, 2)
                x = conv2d(x, s[bp + "upsamplers.0.conv.weight"], s[bp + "upsamplers.0.conv.bias"], padding=1)

        # Output
        x = silu(group_norm(x, s["decoder.conv_norm_out.weight"], s["decoder.conv_norm_out.bias"], 32))
        x = conv2d(x, s["decoder.conv_out.weight"], s["decoder.conv_out.bias"], padding=1)
        return x
