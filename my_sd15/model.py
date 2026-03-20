"""SD 1.5 model container with text-to-image generation."""

from dataclasses import dataclass

import torch
from PIL import Image

from my_sd15.scheduler import DDIMScheduler
from my_sd15.tokenizer import CLIPTokenizer


@dataclass
class SD15Model:
    tokenizer: CLIPTokenizer
    text_encoder: object
    unet: object
    vae: object

    def generate(self, prompt, seed=42, steps=10, cfg_scale=7.5, height=256, width=256, show_progress=False):
        scheduler = DDIMScheduler()
        scheduler.set_timesteps(steps)

        cond_emb = self.text_encoder(self.tokenizer.encode(prompt))
        uncond_emb = self.text_encoder(self.tokenizer.encode(""))

        generator = torch.manual_seed(seed)
        latents = torch.randn(4, height // 8, width // 8, generator=generator)

        from tqdm import tqdm

        with torch.no_grad():
            for t in tqdm(scheduler.timesteps, disable=not show_progress):
                t_int = int(t)
                noise_cond = self.unet(latents, t_int, cond_emb)
                noise_uncond = self.unet(latents, t_int, uncond_emb)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
                latents = scheduler.step(noise_pred, t_int, latents)

            decoded = self.vae(latents / 0.18215)

        image = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        image = (image * 255).byte().permute(1, 2, 0).contiguous()
        return Image.frombytes("RGB", (image.shape[1], image.shape[0]), bytes(image.untyped_storage()))
