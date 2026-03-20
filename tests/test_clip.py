"""Verify CLIP text encoder against test data."""

import os

import pytest
import torch

from my_sd15.loader import DEFAULT_WEIGHTS_DIR, load_clip_text_model

ATOL = 1e-4


def weights_available():
    return os.path.exists(
        os.path.join(DEFAULT_WEIGHTS_DIR, "text_encoder", "model.safetensors")
    )


@pytest.mark.skipif(not weights_available(), reason="weights not found")
class TestCLIPTextModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = load_clip_text_model(DEFAULT_WEIGHTS_DIR)

    def test_output_shape(self, clip_data):
        ids = clip_data["cond_ids"].tolist()
        out = self.model(ids)
        assert out.shape == (77, 768)

    def test_cond_embedding(self, clip_data):
        """CLIP output for the test prompt matches saved data."""
        ids = clip_data["cond_ids"].tolist()
        out = self.model(ids)
        torch.testing.assert_close(
            out.detach(),
            clip_data["cond_emb"],
            atol=ATOL,
            rtol=0,
        )

    def test_uncond_embedding(self, clip_data):
        """CLIP output for empty string matches saved data."""
        ids = clip_data["uncond_ids"].tolist()
        out = self.model(ids)
        torch.testing.assert_close(
            out.detach(),
            clip_data["uncond_emb"],
            atol=ATOL,
            rtol=0,
        )

    def test_different_prompts_differ(self, clip_data):
        """Different prompts produce different embeddings."""
        cond = clip_data["cond_emb"]
        uncond = clip_data["uncond_emb"]
        assert not torch.allclose(cond, uncond, atol=0.1)
