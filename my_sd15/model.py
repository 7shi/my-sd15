"""SD 1.5 model container with text-to-image generation."""

import sys
import os
from dataclasses import dataclass

import torch
from PIL import Image
from sixel.converter import SixelConverter

from my_sd15.tokenizer import CLIPTokenizer


def decode_to_image(decoded):
    image = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
    image = (image * 255).byte().permute(1, 2, 0).contiguous()
    return Image.frombytes("RGB", (image.shape[1], image.shape[0]), bytes(image.untyped_storage()))


def save_image(path, image, show=False, mkdir=False):
    if mkdir:
        dir = os.path.dirname(path)
        if dir:
            os.makedirs(dir, exist_ok=True)
    image.save(path)
    if show:
        SixelConverter(path).write(sys.stdout)
        print()


@dataclass
class SD15Model:
    tokenizer: CLIPTokenizer
    text_encoder: object
    unet: object
    vae: object
    scheduler: object = None

    def generate(self, prompt, negative_prompt="", seed=None, steps=10, cfg_scale=7.5, height=256, width=256, show_progress=False):
        self.scheduler.set_timesteps(steps)

        cond_emb = self.text_encoder(self.tokenizer.encode(prompt))
        uncond_emb = self.text_encoder(self.tokenizer.encode(negative_prompt))

        generator = torch.manual_seed(seed) if seed is not None else None
        latents = torch.randn(4, height // 8, width // 8, generator=generator)

        from tqdm import tqdm

        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps, disable=not show_progress):
                t_int = int(t)
                if cfg_scale == 0.0:
                    noise_pred = self.unet(latents, t_int, uncond_emb)
                elif cfg_scale == 1.0:
                    noise_pred = self.unet(latents, t_int, cond_emb)
                else:
                    noise_cond = self.unet(latents, t_int, cond_emb)
                    noise_uncond = self.unet(latents, t_int, uncond_emb)
                    noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
                latents = self.scheduler.step(noise_pred, t_int, latents, generator=generator)

        return decode_to_image(self.vae(latents / 0.18215))
