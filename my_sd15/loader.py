"""Weight loader for SD 1.5 components."""

import os

from safetensors import safe_open

DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights",
    "stable-diffusion-v1-5",
    "stable-diffusion-v1-5",
)


def _resolve_path(path):
    """Return path, falling back to .fp16/.fp8.safetensors if .safetensors is missing."""
    if not os.path.exists(path) and path.endswith(".safetensors"):
        for alt_suffix in (".fp16.safetensors", ".fp8.safetensors"):
            alt = path.removesuffix(".safetensors") + alt_suffix
            if os.path.exists(alt):
                return alt
    return path


def load_safetensors(path):
    """Load all tensors from a safetensors file as a dict of torch tensors."""
    tensors = {}
    with safe_open(_resolve_path(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).float()
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


def _resolve_weights_dir(model_id=None):
    """Resolve model_id to a weights directory path."""
    if model_id is None:
        return DEFAULT_WEIGHTS_DIR
    weights_base = os.path.normpath(os.path.join(os.path.dirname(DEFAULT_WEIGHTS_DIR), ".."))
    return os.path.join(weights_base, model_id)


def load_model(model_id=None):
    """Load all SD 1.5 components and return an SD15Model instance."""
    from my_sd15.model import SD15Model
    from my_sd15.scheduler import DDIMScheduler
    from my_sd15.tokenizer import CLIPTokenizer

    weights_dir = _resolve_weights_dir(model_id)

    tokenizer_dir = os.path.join(weights_dir, "tokenizer")
    if not os.path.isdir(tokenizer_dir):
        tokenizer_dir = os.path.join(DEFAULT_WEIGHTS_DIR, "tokenizer")

    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
    text_encoder = load_clip_text_model(weights_dir)
    unet = load_unet(weights_dir)
    vae = load_vae_decoder(weights_dir)
    scheduler = DDIMScheduler()

    return SD15Model(tokenizer=tokenizer, text_encoder=text_encoder, unet=unet, vae=vae, scheduler=scheduler)
