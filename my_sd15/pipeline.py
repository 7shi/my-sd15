"""Text-to-image pipeline for SD 1.5."""

import os

import torch
from PIL import Image

from my_sd15.loader import DEFAULT_WEIGHTS_DIR, load_clip_text_model, load_unet, load_vae_decoder
from my_sd15.scheduler import DDIMScheduler
from my_sd15.tokenizer import CLIPTokenizer


def generate(prompt, seed=42, steps=10, cfg_scale=7.5, height=256, width=256, weights_dir=None):
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR

    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(weights_dir, "tokenizer"))
    clip = load_clip_text_model(weights_dir)
    unet = load_unet(weights_dir)
    vae = load_vae_decoder(weights_dir)
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(steps)

    cond_emb = clip(tokenizer.encode(prompt))
    uncond_emb = clip(tokenizer.encode(""))

    generator = torch.manual_seed(seed)
    latents = torch.randn(4, height // 8, width // 8, generator=generator)

    with torch.no_grad():
        for t in scheduler.timesteps:
            t_int = int(t)
            noise_cond = unet(latents, t_int, cond_emb)
            noise_uncond = unet(latents, t_int, uncond_emb)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, t_int, latents)

        decoded = vae(latents / 0.18215)

    image = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
    image = (image * 255).byte().permute(1, 2, 0).numpy()
    return image
