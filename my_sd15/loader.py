"""Weight loader for SD 1.5 components."""

import os

from safetensors import safe_open

DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights",
    "stable-diffusion-v1-5",
    "stable-diffusion-v1-5",
)


def load_safetensors(path):
    """Load all tensors from a safetensors file as a dict of torch tensors."""
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_clip_text_model(weights_dir):
    """Load CLIP text encoder from weights."""
    from my_sd15.clip import CLIPTextModel

    path = os.path.join(weights_dir, "text_encoder", "model.safetensors")
    state = load_safetensors(path)
    model = CLIPTextModel(state)
    return model


def load_vae_decoder(weights_dir):
    """Load VAE decoder from weights."""
    from my_sd15.vae import VaeDecoder

    path = os.path.join(weights_dir, "vae", "diffusion_pytorch_model.safetensors")
    state = load_safetensors(path)
    model = VaeDecoder(state)
    return model


def load_unet(weights_dir):
    """Load U-Net from weights."""
    from my_sd15.unet import UNet

    path = os.path.join(weights_dir, "unet", "diffusion_pytorch_model.safetensors")
    state = load_safetensors(path)
    model = UNet(state)
    return model
