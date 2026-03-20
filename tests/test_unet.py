"""Verify U-Net against test data."""

import os

import pytest
import torch

from my_sd15.loader import DEFAULT_WEIGHTS_DIR, load_unet, _resolve_path

ATOL = 1e-3


def weights_available():
    return os.path.exists(
        _resolve_path(os.path.join(DEFAULT_WEIGHTS_DIR, "unet", "diffusion_pytorch_model.safetensors"))
    )


@pytest.mark.skipif(not weights_available(), reason="weights not found")
class TestUNet:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = load_unet(DEFAULT_WEIGHTS_DIR)

    def test_output_shape(self, unet_data):
        """U-Net output has same shape as input."""
        x = unet_data["input"]
        ctx = unet_data["context"]
        t = int(unet_data["t"][0])
        with torch.no_grad():
            out = self.model(x, t, ctx)
        assert out.shape == (4, 32, 32)

    def test_forward_matches(self, unet_data):
        """U-Net forward pass matches saved data."""
        x = unet_data["input"]
        ctx = unet_data["context"]
        t = int(unet_data["t"][0])
        with torch.no_grad():
            out = self.model(x, t, ctx)
        torch.testing.assert_close(
            out,
            unet_data["output"],
            atol=ATOL,
            rtol=0,
        )
