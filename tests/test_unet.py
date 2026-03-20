"""Verify U-Net against test data."""

import pytest
import torch

from tests.conftest import single_file_available

ATOL = 1e-3


@pytest.mark.skipif(not single_file_available(), reason="weights not found")
class TestUNet:
    @pytest.fixture(autouse=True)
    def setup(self, models):
        self.model = models[1]

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
