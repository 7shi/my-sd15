"""DDIM Scheduler for SD 1.5."""

import torch


class DDIMScheduler:
    def __init__(self):
        # scaled_linear beta schedule
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.timesteps = None
        self._step_ratio = None

    def set_timesteps(self, num_steps):
        step_ratio = 1000 / num_steps
        self._step_ratio = step_ratio
        ts = torch.round(torch.arange(0, num_steps).float() * step_ratio).long()
        self.timesteps = ts.flip(0)

    def step(self, noise_pred, t, sample):
        """DDIM step (eta=0, deterministic)."""
        alpha_t = self.alphas_cumprod[t]
        t_prev = t - int(self._step_ratio)
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        pred_x0 = (sample - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        prev_sample = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1.0 - alpha_t_prev) * noise_pred
        return prev_sample
