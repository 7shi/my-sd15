"""Visualize intermediate latents at each denoising step."""

import torch
from PIL import Image

from my_sd15.loader import load_model
from my_sd15.model import decode_to_image, save_image


def latents_to_image(latents):
    """Convert 4-channel latents to 2x2 grayscale: each pixel's 4 components become a 2x2 block."""
    c, h, w = latents.shape  # (4, H, W)
    lo, hi = latents.min(), latents.max()
    norm = ((latents - lo) / (hi - lo) * 255).byte() if hi > lo else torch.zeros(c, h, w, dtype=torch.uint8)
    # norm: (4, H, W) -> reshape to (2, 2, H, W) -> permute to (H, 2, W, 2) -> reshape to (2H, 2W)
    grid = norm.reshape(2, 2, h, w).permute(2, 0, 3, 1).reshape(h * 2, w * 2)
    return Image.frombytes("L", (grid.shape[1], grid.shape[0]), bytes(grid.contiguous().untyped_storage()))


def save_show_image(path, image):
    save_image(path, image, show=True, mkdir=True)


def main():
    model_id = "genai-archive/anything-v5"
    prompt = "a cat sitting on a windowsill"
    negative_prompt = ""
    seed = 123
    steps = 10
    cfg_scale = 7.5
    size = 256

    model = load_model(model_id=model_id)
    model.scheduler.set_timesteps(steps)

    cond_emb = model.text_encoder(model.tokenizer.encode(prompt))
    uncond_emb = model.text_encoder(model.tokenizer.encode(negative_prompt))

    generator = torch.manual_seed(seed)
    latents = torch.randn(4, size // 8, size // 8, generator=generator)

    with torch.no_grad():
        for i, t in enumerate(model.scheduler.timesteps):
            print(f"step {i}/{steps} (t={int(t)})")
            save_show_image(f"steps/{i:02d}-1.png", latents_to_image(latents))
            save_show_image(f"steps/{i:02d}-2.jpg", decode_to_image(model.vae(latents / 0.18215)))

            t_int = int(t)
            noise_cond = model.unet(latents, t_int, cond_emb)
            noise_uncond = model.unet(latents, t_int, uncond_emb)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            latents = model.scheduler.step(noise_pred, t_int, latents)

    # Save final result
    print(f"step {steps}/{steps}")
    save_show_image(f"steps/{steps:02d}-1.png", latents_to_image(latents))
    save_show_image(f"steps/{steps:02d}-2.jpg", decode_to_image(model.vae(latents / 0.18215)))
    print(f"Saved steps/00-{{1,2}}.png ~ steps/{steps:02d}-{{1,2}}.png")

    # Generate finished images with fewer steps for comparison
    for n in range(1, steps):
        print(f"generating {n}-step result...")
        model.scheduler.set_timesteps(n)
        generator = torch.manual_seed(seed)
        lat = torch.randn(4, size // 8, size // 8, generator=generator)
        with torch.no_grad():
            for t in model.scheduler.timesteps:
                t_int = int(t)
                noise_cond = model.unet(lat, t_int, cond_emb)
                noise_uncond = model.unet(lat, t_int, uncond_emb)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
                lat = model.scheduler.step(noise_pred, t_int, lat)
        save_show_image(f"steps/{n:02d}-3.jpg", decode_to_image(model.vae(lat / 0.18215)))
    print(f"Saved steps/01-3.jpg ~ steps/{steps - 1:02d}-3.jpg")


if __name__ == "__main__":
    main()
