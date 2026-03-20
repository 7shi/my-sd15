"""Verify DDIM scheduler against test data and mathematical properties."""

import pytest
import torch

from my_sd15.scheduler import DDIMScheduler


class TestBetaSchedule:
    def test_alphas_cumprod_shape(self):
        sched = DDIMScheduler()
        assert sched.alphas_cumprod.shape == (1000,)

    def test_alphas_cumprod_monotonic(self):
        """alpha_cumprod must be monotonically decreasing."""
        sched = DDIMScheduler()
        assert (torch.diff(sched.alphas_cumprod) < 0).all()

    def test_alphas_cumprod_range(self):
        """alpha_cumprod[0] ≈ 1, alpha_cumprod[-1] ≈ 0."""
        sched = DDIMScheduler()
        acp = sched.alphas_cumprod
        assert acp[0].item() > 0.99
        assert acp[-1].item() < 0.01

    def test_alphas_cumprod_matches(self, scheduler_data):
        """Match saved alphas_cumprod."""
        sched = DDIMScheduler()
        torch.testing.assert_close(
            sched.alphas_cumprod,
            scheduler_data["alphas_cumprod"],
            atol=1e-6,
            rtol=0,
        )


class TestTimesteps:
    def test_count(self):
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        assert len(sched.timesteps) == 10

    def test_descending(self):
        """Timesteps are in descending order."""
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        assert (torch.diff(sched.timesteps) < 0).all()

    def test_known_values(self):
        """10 steps from 1000 training steps → [900, 800, ..., 0]."""
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        expected = torch.tensor([900, 800, 700, 600, 500, 400, 300, 200, 100, 0])
        torch.testing.assert_close(sched.timesteps, expected)

    def test_matches_saved(self, scheduler_data):
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        torch.testing.assert_close(sched.timesteps, scheduler_data["timesteps"])


class TestDDIMStep:
    def test_deterministic(self):
        """Same inputs produce same outputs (eta=0)."""
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        sample = torch.randn(4, 4, 4)
        noise = torch.randn(4, 4, 4)
        t = int(sched.timesteps[0])
        out1 = sched.step(noise, t, sample)
        out2 = sched.step(noise, t, sample)
        torch.testing.assert_close(out1, out2)

    def test_output_shape(self):
        """Output has same shape as input."""
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        sample = torch.randn(4, 8, 8)
        noise = torch.randn(4, 8, 8)
        out = sched.step(noise, int(sched.timesteps[0]), sample)
        assert out.shape == sample.shape

    def test_matches_saved(self, scheduler_data):
        """Single step matches saved test data."""
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        sample = scheduler_data["step_sample"]
        noise = scheduler_data["step_noise"]
        t = int(scheduler_data["step_t"][0])
        out = sched.step(noise, t, sample)
        torch.testing.assert_close(
            out,
            scheduler_data["step_out"],
            atol=1e-5,
            rtol=0,
        )
