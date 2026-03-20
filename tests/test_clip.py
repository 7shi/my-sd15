"""Verify CLIP text encoder against test data."""

import pytest
import torch

from tests.conftest import single_file_available

ATOL = 1e-4


@pytest.mark.skipif(not single_file_available(), reason="weights not found")
class TestCLIPTextModel:
    @pytest.fixture(autouse=True)
    def setup(self, models):
        self.model = models[0]

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
