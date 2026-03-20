"""Basic operations for SD 1.5 inference."""

import torch
import torch.nn.functional as F


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """Conv2d via im2col + matmul. x: (C_in, H, W), weight: (C_out, C_in, kH, kW)."""
    C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    # im2col
    col = x.unfold(1, kH, stride).unfold(2, kW, stride)  # (C_in, H_out, W_out, kH, kW)
    col = col.permute(0, 3, 4, 1, 2)  # (C_in, kH, kW, H_out, W_out)
    col = col.reshape(C_in * kH * kW, H_out * W_out)
    kernel = weight.reshape(C_out, -1)  # (C_out, C_in*kH*kW)
    out = kernel @ col  # (C_out, H_out*W_out)
    out = out.reshape(C_out, H_out, W_out)
    if bias is not None:
        out = out + bias.reshape(-1, 1, 1)
    return out


def group_norm(x, weight, bias, num_groups, eps=1e-5):
    """Group normalization. x: (C, *spatial)."""
    shape = x.shape
    C = shape[0]
    x = x.reshape(num_groups, -1)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x.reshape(shape)
    # Reshape weight/bias for broadcasting over spatial dims
    extra_dims = len(shape) - 1
    view_shape = (C,) + (1,) * extra_dims
    x = x * weight.reshape(view_shape) + bias.reshape(view_shape)
    return x


def layer_norm(x, weight, bias, eps=1e-5):
    """Layer normalization over the last dimension."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return x * weight + bias


def linear(x, weight, bias=None):
    """Linear: x @ W.T + b."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def silu(x):
    """SiLU activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def quick_gelu(x):
    """Quick GELU: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)


def gelu(x):
    """GELU: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    m = x.max(dim=axis, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=axis, keepdim=True)


def upsample_nearest_2d(x, scale=2):
    """Nearest-neighbor 2x upsampling. x: (C, H, W)."""
    C, H, W = x.shape
    x = x.unsqueeze(2).unsqueeze(4)  # (C, H, 1, W, 1)
    x = x.expand(C, H, scale, W, scale)  # (C, H, scale, W, scale)
    return x.reshape(C, H * scale, W * scale)


def embedding(indices, weight):
    """Table lookup. indices: (N,), weight: (V, D) -> (N, D)."""
    return weight[indices]
