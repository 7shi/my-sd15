"""Verify basic operations with hand-computed and property-based tests.

These tests do NOT require model weights. They verify that each operation
is implemented correctly using small, deterministic inputs.
"""

import numpy as np
import pytest

from my_sd15.ops import (
    conv2d,
    embedding,
    gelu,
    group_norm,
    layer_norm,
    linear,
    quick_gelu,
    silu,
    softmax,
    upsample_nearest_2d,
)
import torch


class TestConv2dHandComputed:
    """Test conv2d with hand-computable examples."""

    def test_1x1_is_linear(self):
        """1x1 conv is equivalent to a per-pixel linear transformation."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                          [[5.0, 6.0], [7.0, 8.0]]])  # (2, 2, 2)
        w = torch.tensor([[[[1.0]], [[2.0]]]])  # (1, 2, 1, 1): out=1, in=2
        b = torch.tensor([0.5])
        out = conv2d(x, w, b)
        # out[0, i, j] = 1*x[0,i,j] + 2*x[1,i,j] + 0.5
        expected = torch.tensor([[[11.5, 14.5], [17.5, 20.5]]])
        torch.testing.assert_close(out, expected)

    def test_3x3_no_padding(self):
        """3x3 conv on 3x3 input with no padding gives 1x1 output."""
        x = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
        w = torch.ones(1, 1, 3, 3)
        out = conv2d(x, w)
        expected = torch.tensor([[[36.0]]])  # sum of 0..8
        torch.testing.assert_close(out, expected)

    def test_3x3_with_padding(self):
        """3x3 conv with padding=1 preserves spatial size."""
        x = torch.ones(1, 5, 5)
        w = torch.ones(1, 1, 3, 3)
        out = conv2d(x, w, padding=1)
        assert out.shape == (1, 5, 5)
        # Interior pixel sees all 9 ones
        assert out[0, 2, 2].item() == 9.0
        # Corner pixel sees 4 ones (rest is padding zeros)
        assert out[0, 0, 0].item() == 4.0
        # Edge pixel sees 6 ones
        assert out[0, 0, 2].item() == 6.0

    def test_stride2_halves_spatial(self):
        """Stride 2 halves the spatial dimensions."""
        x = torch.ones(1, 8, 8)
        w = torch.ones(1, 1, 3, 3)
        out = conv2d(x, w, stride=2, padding=1)
        assert out.shape == (1, 4, 4)

    def test_no_bias(self):
        """Conv without bias should not add anything."""
        x = torch.zeros(1, 3, 3)
        w = torch.ones(1, 1, 3, 3)
        out = conv2d(x, w)
        assert out[0, 0, 0].item() == 0.0

    def test_multi_channel_output(self):
        """Multiple output channels produce independent results."""
        x = torch.ones(1, 2, 2)
        w = torch.zeros(3, 1, 1, 1)
        w[0, 0, 0, 0] = 1.0
        w[1, 0, 0, 0] = 2.0
        w[2, 0, 0, 0] = 3.0
        out = conv2d(x, w)
        assert out.shape == (3, 2, 2)
        assert out[0, 0, 0].item() == 1.0
        assert out[1, 0, 0].item() == 2.0
        assert out[2, 0, 0].item() == 3.0


class TestGroupNorm:
    def test_zero_mean_unit_var(self):
        """After group norm (without affine), each group has zero mean and unit var."""
        torch.manual_seed(0)
        x = torch.randn(32, 4, 4)
        w = torch.ones(32)
        b = torch.zeros(32)
        out = group_norm(x, w, b, num_groups=8)
        # Check each group has mean≈0 and population var≈1
        grouped = out.reshape(8, -1)
        np.testing.assert_allclose(grouped.mean(dim=1).numpy(), 0.0, atol=1e-5)
        # Use unbiased=False (population variance) to match group_norm's normalization
        np.testing.assert_allclose(
            grouped.var(dim=1, unbiased=False).numpy(), 1.0, atol=1e-4
        )

    def test_affine_transform(self):
        """Weight and bias are applied after normalization."""
        x = torch.tensor([[[1.0, 3.0]],
                          [[5.0, 7.0]]])  # (2, 1, 2)
        w = torch.tensor([2.0, 3.0])
        b = torch.tensor([10.0, 20.0])
        out = group_norm(x, w, b, num_groups=2)
        # Each channel is its own group with 2 elements
        # ch0: [1, 3] → mean=2, var=1 → normalized=[-1, 1] → 2*[-1,1]+10 = [8, 12]
        assert abs(out[0, 0, 0].item() - 8.0) < 1e-5
        assert abs(out[0, 0, 1].item() - 12.0) < 1e-5


class TestLayerNorm:
    def test_zero_mean_unit_var(self):
        """After layer norm (without affine), last dim has zero mean and population var≈1."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        w = torch.ones(4)
        b = torch.zeros(4)
        out = layer_norm(x, w, b)
        np.testing.assert_allclose(out.mean(dim=-1).numpy(), 0.0, atol=1e-6)
        np.testing.assert_allclose(
            out.var(dim=-1, unbiased=False).numpy(), 1.0, atol=1e-5
        )

    def test_known_values(self):
        """Test with known input/output."""
        x = torch.tensor([[0.0, 4.0]])  # mean=2, var=4, std=2
        w = torch.ones(2)
        b = torch.zeros(2)
        out = layer_norm(x, w, b)
        # (0-2)/2 = -1, (4-2)/2 = 1
        np.testing.assert_allclose(out.numpy(), [[-1.0, 1.0]], atol=1e-5)


class TestLinear:
    def test_identity(self):
        """Identity weight matrix should pass through input."""
        x = torch.tensor([1.0, 2.0, 3.0])
        w = torch.eye(3)
        out = linear(x, w)
        torch.testing.assert_close(out, x)

    def test_with_bias(self):
        """Linear with bias."""
        x = torch.tensor([1.0, 2.0])
        w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = torch.tensor([10.0, 20.0, 30.0])
        out = linear(x, w, b)
        expected = torch.tensor([11.0, 22.0, 33.0])
        torch.testing.assert_close(out, expected)

    def test_batch(self):
        """Linear should work on batched input."""
        x = torch.ones(5, 3)
        w = torch.ones(2, 3)
        out = linear(x, w)
        assert out.shape == (5, 2)
        assert out[0, 0].item() == 3.0


class TestActivations:
    def test_silu_zero(self):
        """silu(0) = 0 * sigmoid(0) = 0."""
        assert silu(torch.tensor(0.0)).item() == 0.0

    def test_silu_positive(self):
        """silu is approximately identity for large positive x."""
        x = torch.tensor(10.0)
        assert abs(silu(x).item() - 10.0) < 0.01

    def test_quick_gelu_zero(self):
        """quick_gelu(0) = 0."""
        assert quick_gelu(torch.tensor(0.0)).item() == 0.0

    def test_quick_gelu_positive(self):
        """quick_gelu is approximately identity for large positive x."""
        x = torch.tensor(10.0)
        assert abs(quick_gelu(x).item() - 10.0) < 0.01

    def test_gelu_zero(self):
        """gelu(0) = 0."""
        assert abs(gelu(torch.tensor(0.0)).item()) < 1e-7

    def test_gelu_symmetry(self):
        """gelu is NOT symmetric: gelu(-x) != -gelu(x) in general,
        but gelu(-x) + gelu(x) approaches x for large x."""
        x = torch.tensor(3.0)
        # gelu(3) ≈ 3, gelu(-3) ≈ 0
        assert gelu(x).item() > 2.9
        assert abs(gelu(-x).item()) < 0.01

    def test_silu_shape(self):
        """silu preserves shape."""
        x = torch.randn(3, 4, 5)
        assert silu(x).shape == (3, 4, 5)


class TestSoftmax:
    def test_sums_to_one(self):
        """Softmax output sums to 1 along the specified axis."""
        x = torch.randn(3, 5)
        out = softmax(x)
        np.testing.assert_allclose(out.sum(dim=-1).numpy(), [1.0, 1.0, 1.0], atol=1e-6)

    def test_all_non_negative(self):
        """Softmax output is always non-negative."""
        x = torch.tensor([-100.0, -50.0, 0.0, 50.0, 100.0])
        out = softmax(x)
        assert (out >= 0).all()

    def test_uniform(self):
        """Equal inputs produce uniform distribution."""
        x = torch.tensor([1.0, 1.0, 1.0, 1.0])
        out = softmax(x)
        np.testing.assert_allclose(out.numpy(), [0.25, 0.25, 0.25, 0.25], atol=1e-6)

    def test_numerical_stability(self):
        """Should not overflow/underflow with large values."""
        x = torch.tensor([1000.0, 1001.0, 1002.0])
        out = softmax(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        np.testing.assert_allclose(out.sum().numpy(), 1.0, atol=1e-6)


class TestUpsampleNearest2d:
    def test_doubles_size(self):
        """Upsampling by 2 doubles H and W."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        out = upsample_nearest_2d(x, scale=2)
        assert out.shape == (1, 4, 4)

    def test_values_repeated(self):
        """Each value is repeated in a 2x2 block."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        out = upsample_nearest_2d(x, scale=2)
        expected = torch.tensor([[[1, 1, 2, 2],
                                  [1, 1, 2, 2],
                                  [3, 3, 4, 4],
                                  [3, 3, 4, 4]]], dtype=torch.float32)
        torch.testing.assert_close(out, expected)


class TestEmbedding:
    def test_lookup(self):
        """Embedding is just a table lookup."""
        table = torch.tensor([[10.0, 20.0],
                              [30.0, 40.0],
                              [50.0, 60.0]])
        indices = torch.tensor([2, 0, 1])
        out = embedding(indices, table)
        expected = torch.tensor([[50.0, 60.0],
                                 [10.0, 20.0],
                                 [30.0, 40.0]])
        torch.testing.assert_close(out, expected)
