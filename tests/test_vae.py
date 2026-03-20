"""Verify VAE decoder against test data."""

import pytest
import torch

from tests.conftest import single_file_available

ATOL = 1e-3


@pytest.mark.skipif(not single_file_available(), reason="weights not found")
class TestVaeDecoder:
    @pytest.fixture(autouse=True)
    def setup(self, models):
        self.decoder = models[2]

    def test_output_shape(self, vae_data):
        """VAE decoder outputs (3, H*8, W*8) for input (4, H, W)."""
        x = vae_data["input"]
        with torch.no_grad():
            out = self.decoder(x)
        assert out.shape == (3, 256, 256)

    def test_decode_matches(self, vae_data):
        """Decoded image matches saved data."""
        x = vae_data["input"]
        with torch.no_grad():
            out = self.decoder(x)
        torch.testing.assert_close(
            out,
            vae_data["output"],
            atol=ATOL,
            rtol=0,
        )
