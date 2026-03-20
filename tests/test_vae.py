"""Verify VAE decoder against test data."""

import os

import numpy as np
import pytest
import torch

from my_sd15.loader import DEFAULT_WEIGHTS_DIR, load_vae_decoder

ATOL = 1e-3


def weights_available():
    return os.path.exists(
        os.path.join(DEFAULT_WEIGHTS_DIR, "vae", "diffusion_pytorch_model.safetensors")
    )


@pytest.mark.skipif(not weights_available(), reason="weights not found")
class TestVaeDecoder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.decoder = load_vae_decoder(DEFAULT_WEIGHTS_DIR)

    def test_output_shape(self, vae_data):
        """VAE decoder outputs (3, H*8, W*8) for input (4, H, W)."""
        x = torch.from_numpy(vae_data["input"])
        with torch.no_grad():
            out = self.decoder(x)
        assert out.shape == (3, 256, 256)

    def test_decode_matches(self, vae_data):
        """Decoded image matches saved data."""
        x = torch.from_numpy(vae_data["input"])
        with torch.no_grad():
            out = self.decoder(x)
        np.testing.assert_allclose(
            out.numpy(),
            vae_data["output"],
            atol=ATOL,
        )
