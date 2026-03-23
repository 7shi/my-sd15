"""VAE Decoder for SD 1.5."""

import torch

from my_sd15.ops import conv2d, group_norm, linear, silu, softmax, upsample_nearest_2d


def relu(x):
    """ReLU activation."""
    return torch.relu(x)


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
        q = linear(h, s[p + "to_q.weight"], s[p + "to_q.bias"])
        k = linear(h, s[p + "to_k.weight"], s[p + "to_k.bias"])
        v = linear(h, s[p + "to_v.weight"], s[p + "to_v.bias"])
        attn = softmax(q @ k.T / (C ** 0.5))
        h = attn @ v
        h = linear(h, s[p + "to_out.0.weight"], s[p + "to_out.0.bias"])
        h = h.T.reshape(C, H, W)
        return x + h


class VaeDecoder:
    scale_factor = 0.18215

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


class TaesdResBlock:
    """TAESD residual block: Conv-ReLU-Conv-ReLU-Conv + skip, then ReLU."""

    def __init__(self, state, prefix, act_fn=relu):
        self.state = state
        self.prefix = prefix
        self.act_fn = act_fn

    def __call__(self, x):
        s = self.state
        p = self.prefix
        h = self.act_fn(conv2d(x, s[p + "conv.0.weight"], s[p + "conv.0.bias"], padding=1))
        h = self.act_fn(conv2d(h, s[p + "conv.2.weight"], s[p + "conv.2.bias"], padding=1))
        h = conv2d(h, s[p + "conv.4.weight"], s[p + "conv.4.bias"], padding=1)
        return self.act_fn(h + x)


class TaesdDecoder:
    """Tiny AutoEncoder (TAESD) decoder.

    Output is converted to [-1, 1] range to match VaeDecoder convention.
    """

    def __init__(self, state, config):
        self.state = state
        self.scale_factor = config.get("scaling_factor", 1.0)
        act_fn_name = config.get("act_fn", "relu")
        self.act_fn = {"relu": relu, "silu": silu}[act_fn_name]

    def __call__(self, x):
        s = self.state
        p = "decoder.layers."
        act = self.act_fn

        # Clamp input
        x = x.clamp(-3, 3)

        # Layer 0: Input Conv (4 → 64)
        x = conv2d(x, s[p + "0.weight"], s[p + "0.bias"], padding=1)

        # Layer 1: ReLU (no parameters)
        x = act(x)

        # Layers 2-4: ResBlock × 3
        for i in (2, 3, 4):
            x = TaesdResBlock(s, p + f"{i}.", act)(x)

        # Layers 5-9: Upsample + Conv + ResBlock × 3
        x = upsample_nearest_2d(x, 2)
        x = conv2d(x, s[p + "6.weight"], padding=1)
        for i in (7, 8, 9):
            x = TaesdResBlock(s, p + f"{i}.", act)(x)

        # Layers 10-14: Upsample + Conv + ResBlock × 3
        x = upsample_nearest_2d(x, 2)
        x = conv2d(x, s[p + "11.weight"], padding=1)
        for i in (12, 13, 14):
            x = TaesdResBlock(s, p + f"{i}.", act)(x)

        # Layers 15-17: Upsample + Conv + ResBlock × 1
        x = upsample_nearest_2d(x, 2)
        x = conv2d(x, s[p + "16.weight"], padding=1)
        x = TaesdResBlock(s, p + "17.", act)(x)

        # Layer 18: Output Conv (64 → 3)
        x = conv2d(x, s[p + "18.weight"], s[p + "18.bias"], padding=1)

        # Convert from [0, 1] to [-1, 1] to match VaeDecoder convention
        return x * 2 - 1
