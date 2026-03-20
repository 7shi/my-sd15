"""Verify full pipeline step-by-step against test data."""

import os

import pytest
import torch

from tests.conftest import single_file_available
from my_sd15.loader import DEFAULT_WEIGHTS_DIR
from my_sd15.tokenizer import CLIPTokenizer
from my_sd15.scheduler import DDIMScheduler

ATOL_LATENT = 1e-3
ATOL_IMAGE = 1e-2


@pytest.mark.skipif(not single_file_available(), reason="weights not found")
class TestPipelineStepByStep:
    @pytest.fixture(autouse=True)
    def setup(self, models):
        self.clip, self.unet, self.vae = models
        self.tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(DEFAULT_WEIGHTS_DIR, "tokenizer")
        )
        self.scheduler = DDIMScheduler()
        self.scheduler.set_timesteps(10)

    def test_initial_latents(self, pipeline_data):
        """Initial noise generated with torch.manual_seed(42) matches."""
        generator = torch.manual_seed(42)
        latents = torch.randn(4, 32, 32, generator=generator)
        torch.testing.assert_close(
            latents,
            pipeline_data["latents_init"],
            atol=1e-6,
            rtol=0,
        )

    def test_each_step(self, pipeline_data, metadata):
        """Each denoising step matches the saved intermediate latents."""
        cond_emb = self.clip(self.tokenizer.encode(metadata["prompt"]))
        uncond_emb = self.clip(self.tokenizer.encode(""))

        latents = pipeline_data["latents_init"].clone()
        cfg = metadata["cfg_scale"]

        with torch.no_grad():
            for i, t in enumerate(self.scheduler.timesteps):
                t_int = int(t)
                noise_cond = self.unet(latents, t_int, cond_emb)
                noise_uncond = self.unet(latents, t_int, uncond_emb)
                noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)
                latents = self.scheduler.step(noise_pred, t_int, latents)

                expected = pipeline_data[f"latents_step{i:02d}"]
                diff = (latents - expected).abs().max().item()
                assert diff < ATOL_LATENT, (
                    f"Step {i} (t={t_int}): max diff {diff:.6f} > {ATOL_LATENT}"
                )

    def test_final_latents(self, pipeline_data, metadata):
        """Final latents after all steps match."""
        cond_emb = self.clip(self.tokenizer.encode(metadata["prompt"]))
        uncond_emb = self.clip(self.tokenizer.encode(""))

        latents = pipeline_data["latents_init"].clone()
        cfg = metadata["cfg_scale"]

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t_int = int(t)
                noise_cond = self.unet(latents, t_int, cond_emb)
                noise_uncond = self.unet(latents, t_int, uncond_emb)
                noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)
                latents = self.scheduler.step(noise_pred, t_int, latents)

        torch.testing.assert_close(
            latents,
            pipeline_data["latents_final"],
            atol=ATOL_LATENT,
            rtol=0,
        )

    def test_decoded_image(self, pipeline_data):
        """VAE decode of final latents matches."""
        latents = pipeline_data["latents_final"].clone()
        with torch.no_grad():
            decoded = self.vae(latents / 0.18215)
        torch.testing.assert_close(
            decoded,
            pipeline_data["decoded"],
            atol=ATOL_IMAGE,
            rtol=0,
        )

    def test_final_image(self, pipeline_data):
        """Final uint8 image matches."""
        decoded = pipeline_data["decoded"].clone()
        image = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        image = (image * 255).byte().permute(1, 2, 0)
        torch.testing.assert_close(image, pipeline_data["image"])
