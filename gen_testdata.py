"""Generate test data for clean-room implementation verification.

Each component's inputs and outputs are saved as safetensors so that
a new implementation can verify correctness step by step.
"""

import os
import json

import numpy as np
import torch
from safetensors.numpy import save_file

TESTDATA_DIR = "testdata"
PROMPT = "a cat sitting on a windowsill"
SEED = 42
STEPS = 10
CFG = 7.5
HEIGHT = 256
WIDTH = 256


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().numpy()
    elif isinstance(data, list):
        return np.array(data)
    return data


def save_safetensors(name, tensors):
    """Save a dict of named arrays as a safetensors file."""
    path = os.path.join(TESTDATA_DIR, f"{name}.safetensors")
    arrays = {k: to_numpy(v) for k, v in tensors.items()}
    save_file(arrays, path)
    print(f"  {path}")
    for k, v in arrays.items():
        print(f"    {k}: {v.shape} {v.dtype}")


def main():
    os.makedirs(TESTDATA_DIR, exist_ok=True)

    from my_sd15.loader import (
        DEFAULT_WEIGHTS_DIR,
        load_clip_text_model,
        load_unet,
        load_vae_decoder,
    )
    from my_sd15.tokenizer import CLIPTokenizer
    from my_sd15.scheduler import DDIMScheduler

    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(DEFAULT_WEIGHTS_DIR, "tokenizer")
    )
    clip = load_clip_text_model(DEFAULT_WEIGHTS_DIR)
    unet = load_unet(DEFAULT_WEIGHTS_DIR)
    vae = load_vae_decoder(DEFAULT_WEIGHTS_DIR)
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(STEPS)

    # =========================================================
    # 1. Tokenizer
    # =========================================================
    print("=== Tokenizer ===")
    cond_ids = tokenizer.encode(PROMPT)
    uncond_ids = tokenizer.encode("")
    save_safetensors("tokenizer", {
        "cond_ids": np.array(cond_ids, dtype=np.int64),
        "uncond_ids": np.array(uncond_ids, dtype=np.int64),
    })

    # =========================================================
    # 2. CLIP Text Encoder
    # =========================================================
    print("=== CLIP ===")
    with torch.no_grad():
        cond_emb = clip(cond_ids)
        uncond_emb = clip(uncond_ids)
    save_safetensors("clip", {
        "cond_ids": np.array(cond_ids, dtype=np.int64),
        "cond_emb": cond_emb,
        "uncond_ids": np.array(uncond_ids, dtype=np.int64),
        "uncond_emb": uncond_emb,
    })

    # =========================================================
    # 3. Scheduler
    # =========================================================
    print("=== Scheduler ===")
    np.random.seed(0)
    sched_sample = torch.from_numpy(
        np.random.randn(4, 4, 4).astype(np.float32)
    )
    sched_noise = torch.from_numpy(
        np.random.randn(4, 4, 4).astype(np.float32)
    )
    sched_t = int(scheduler.timesteps[0])
    sched_out = scheduler.step(sched_noise, sched_t, sched_sample)
    save_safetensors("scheduler", {
        "alphas_cumprod": scheduler.alphas_cumprod,
        "timesteps": scheduler.timesteps,
        "step_sample": sched_sample,
        "step_noise": sched_noise,
        "step_t": np.array([sched_t], dtype=np.int64),
        "step_out": sched_out,
    })

    # =========================================================
    # 4. VAE Decoder
    # =========================================================
    print("=== VAE ===")
    np.random.seed(1)
    vae_input = torch.from_numpy(
        np.random.randn(4, 32, 32).astype(np.float32) * 0.5
    )
    with torch.no_grad():
        vae_output = vae(vae_input)
    save_safetensors("vae", {
        "input": vae_input,
        "output": vae_output,
    })

    # =========================================================
    # 5. U-Net (single forward pass)
    # =========================================================
    print("=== U-Net ===")
    np.random.seed(2)
    unet_x = torch.from_numpy(
        np.random.randn(4, 32, 32).astype(np.float32) * 0.1
    )
    unet_ctx = torch.from_numpy(
        np.random.randn(77, 768).astype(np.float32) * 0.1
    )
    unet_t = 500
    with torch.no_grad():
        unet_out = unet(unet_x, unet_t, unet_ctx)
    save_safetensors("unet", {
        "input": unet_x,
        "context": unet_ctx,
        "t": np.array([unet_t], dtype=np.int64),
        "output": unet_out,
    })

    # =========================================================
    # 6. Full pipeline (step-by-step intermediates)
    # =========================================================
    print("=== Pipeline ===")
    latent_h, latent_w = HEIGHT // 8, WIDTH // 8

    generator = torch.manual_seed(SEED)
    latents = torch.randn(4, latent_h, latent_w, generator=generator)

    pipe_tensors = {
        "latents_init": latents,
    }

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            t_int = int(t)
            noise_cond = unet(latents, t_int, cond_emb)
            noise_uncond = unet(latents, t_int, uncond_emb)
            noise_pred = noise_uncond + CFG * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, t_int, latents)
            pipe_tensors[f"latents_step{i:02d}"] = latents
            print(f"    step {i}: t={t_int}")

        decoded = vae(latents / 0.18215)

    image = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
    image = (image * 255).byte().permute(1, 2, 0)

    pipe_tensors["latents_final"] = latents
    pipe_tensors["decoded"] = decoded
    pipe_tensors["image"] = image

    save_safetensors("pipeline", pipe_tensors)

    # =========================================================
    # Save metadata
    # =========================================================
    meta = {
        "prompt": PROMPT,
        "seed": SEED,
        "steps": STEPS,
        "cfg_scale": CFG,
        "height": HEIGHT,
        "width": WIDTH,
        "files": {
            "tokenizer.safetensors": {
                "cond_ids": "int64 (77,) — tokenized prompt",
                "uncond_ids": "int64 (77,) — tokenized empty string",
            },
            "clip.safetensors": {
                "cond_ids": "int64 (77,) — input token IDs",
                "cond_emb": "float32 (77, 768) — CLIP output for prompt",
                "uncond_ids": "int64 (77,) — input token IDs",
                "uncond_emb": "float32 (77, 768) — CLIP output for empty string",
            },
            "scheduler.safetensors": {
                "alphas_cumprod": "float32 (1000,) — cumulative product of alphas",
                "timesteps": "int64 (10,) — selected timesteps for inference",
                "step_sample": "float32 (4,4,4) — input sample for step test",
                "step_noise": "float32 (4,4,4) — predicted noise for step test",
                "step_t": "int64 (1,) — timestep for step test",
                "step_out": "float32 (4,4,4) — output of one DDIM step",
            },
            "vae.safetensors": {
                "input": "float32 (4,32,32) — latent input",
                "output": "float32 (3,256,256) — decoded image",
            },
            "unet.safetensors": {
                "input": "float32 (4,32,32) — noisy latent",
                "context": "float32 (77,768) — encoder hidden states",
                "t": "int64 (1,) — timestep",
                "output": "float32 (4,32,32) — predicted noise",
            },
            "pipeline.safetensors": {
                "latents_init": "float32 (4,32,32) — initial noise",
                "latents_step00..09": "float32 (4,32,32) — latents after each step",
                "latents_final": "float32 (4,32,32) — same as step09",
                "decoded": "float32 (3,256,256) — VAE decoded image [-1,1]",
                "image": "uint8 (256,256,3) — final image [0,255]",
            },
        },
    }
    meta_path = os.path.join(TESTDATA_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  {meta_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
