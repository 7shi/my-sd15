"""Verify DDIM scheduler against test data and mathematical properties."""

import numpy as np
import pytest

from my_sd15.scheduler import DDIMScheduler


class TestBetaSchedule:
    def test_alphas_cumprod_shape(self):
        sched = DDIMScheduler()
        assert sched.alphas_cumprod.shape == (1000,)

    def test_alphas_cumprod_monotonic(self):
        """alpha_cumprod must be monotonically decreasing."""
        sched = DDIMScheduler()
        acp = sched.alphas_cumprod.numpy()
        assert np.all(np.diff(acp) < 0)

    def test_alphas_cumprod_range(self):
        """alpha_cumprod[0] ≈ 1, alpha_cumprod[-1] ≈ 0."""
        sched = DDIMScheduler()
        acp = sched.alphas_cumprod.numpy()
        assert acp[0] > 0.99
        assert acp[-1] < 0.01

    def test_alphas_cumprod_matches(self, scheduler_data):
        """Match saved alphas_cumprod."""
        sched = DDIMScheduler()
        np.testing.assert_allclose(
            sched.alphas_cumprod.numpy(),
            scheduler_data["alphas_cumprod"],
            atol=1e-6,
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
        ts = sched.timesteps.numpy()
        assert np.all(np.diff(ts) < 0)

    def test_known_values(self):
        """10 steps from 1000 training steps → [900, 800, ..., 0]."""
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        expected = [900, 800, 700, 600, 500, 400, 300, 200, 100, 0]
        np.testing.assert_array_equal(sched.timesteps.numpy(), expected)

    def test_matches_saved(self, scheduler_data):
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        np.testing.assert_array_equal(
            sched.timesteps.numpy(),
            scheduler_data["timesteps"],
        )


class TestDDIMStep:
    def test_deterministic(self):
        """Same inputs produce same outputs (eta=0)."""
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        import torch
        sample = torch.randn(4, 4, 4)
        noise = torch.randn(4, 4, 4)
        t = int(sched.timesteps[0])
        out1 = sched.step(noise, t, sample)
        out2 = sched.step(noise, t, sample)
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())

    def test_output_shape(self):
        """Output has same shape as input."""
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        import torch
        sample = torch.randn(4, 8, 8)
        noise = torch.randn(4, 8, 8)
        out = sched.step(noise, int(sched.timesteps[0]), sample)
        assert out.shape == sample.shape

    def test_matches_saved(self, scheduler_data):
        """Single step matches saved test data."""
        import torch
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        sample = torch.from_numpy(scheduler_data["step_sample"])
        noise = torch.from_numpy(scheduler_data["step_noise"])
        t = int(scheduler_data["step_t"][0])
        out = sched.step(noise, t, sample)
        np.testing.assert_allclose(
            out.numpy(),
            scheduler_data["step_out"],
            atol=1e-5,
        )
