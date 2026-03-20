"""Weight loader for SD 1.5 components."""

import os

from safetensors import safe_open

DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights",
    "stable-diffusion-v1-5",
    "stable-diffusion-v1-5",
)

SINGLE_FILE_PATH = os.path.join(
    DEFAULT_WEIGHTS_DIR,
    "v1-5-pruned-emaonly.safetensors",
)


def load_safetensors(path):
    """Load all tensors from a safetensors file as a dict of torch tensors."""
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


_UNET_BLOCK_MAP = {
    # conv_in
    "input_blocks.0.0.": "conv_in.",
    # down_blocks.0
    "input_blocks.1.0.": "down_blocks.0.resnets.0.",
    "input_blocks.1.1.": "down_blocks.0.attentions.0.",
    "input_blocks.2.0.": "down_blocks.0.resnets.1.",
    "input_blocks.2.1.": "down_blocks.0.attentions.1.",
    "input_blocks.3.0.": "down_blocks.0.downsamplers.0.",
    # down_blocks.1
    "input_blocks.4.0.": "down_blocks.1.resnets.0.",
    "input_blocks.4.1.": "down_blocks.1.attentions.0.",
    "input_blocks.5.0.": "down_blocks.1.resnets.1.",
    "input_blocks.5.1.": "down_blocks.1.attentions.1.",
    "input_blocks.6.0.": "down_blocks.1.downsamplers.0.",
    # down_blocks.2
    "input_blocks.7.0.": "down_blocks.2.resnets.0.",
    "input_blocks.7.1.": "down_blocks.2.attentions.0.",
    "input_blocks.8.0.": "down_blocks.2.resnets.1.",
    "input_blocks.8.1.": "down_blocks.2.attentions.1.",
    "input_blocks.9.0.": "down_blocks.2.downsamplers.0.",
    # down_blocks.3
    "input_blocks.10.0.": "down_blocks.3.resnets.0.",
    "input_blocks.11.0.": "down_blocks.3.resnets.1.",
    # mid_block
    "middle_block.0.": "mid_block.resnets.0.",
    "middle_block.1.": "mid_block.attentions.0.",
    "middle_block.2.": "mid_block.resnets.1.",
    # up_blocks.0
    "output_blocks.0.0.": "up_blocks.0.resnets.0.",
    "output_blocks.1.0.": "up_blocks.0.resnets.1.",
    "output_blocks.2.0.": "up_blocks.0.resnets.2.",
    "output_blocks.2.1.": "up_blocks.0.upsamplers.0.",
    # up_blocks.1
    "output_blocks.3.0.": "up_blocks.1.resnets.0.",
    "output_blocks.3.1.": "up_blocks.1.attentions.0.",
    "output_blocks.4.0.": "up_blocks.1.resnets.1.",
    "output_blocks.4.1.": "up_blocks.1.attentions.1.",
    "output_blocks.5.0.": "up_blocks.1.resnets.2.",
    "output_blocks.5.1.": "up_blocks.1.attentions.2.",
    "output_blocks.5.2.": "up_blocks.1.upsamplers.0.",
    # up_blocks.2
    "output_blocks.6.0.": "up_blocks.2.resnets.0.",
    "output_blocks.6.1.": "up_blocks.2.attentions.0.",
    "output_blocks.7.0.": "up_blocks.2.resnets.1.",
    "output_blocks.7.1.": "up_blocks.2.attentions.1.",
    "output_blocks.8.0.": "up_blocks.2.resnets.2.",
    "output_blocks.8.1.": "up_blocks.2.attentions.2.",
    "output_blocks.8.2.": "up_blocks.2.upsamplers.0.",
    # up_blocks.3
    "output_blocks.9.0.": "up_blocks.3.resnets.0.",
    "output_blocks.9.1.": "up_blocks.3.attentions.0.",
    "output_blocks.10.0.": "up_blocks.3.resnets.1.",
    "output_blocks.10.1.": "up_blocks.3.attentions.1.",
    "output_blocks.11.0.": "up_blocks.3.resnets.2.",
    "output_blocks.11.1.": "up_blocks.3.attentions.2.",
    # time embedding
    "time_embed.0.": "time_embedding.linear_1.",
    "time_embed.2.": "time_embedding.linear_2.",
    # output
    "out.0.": "conv_norm_out.",
    "out.2.": "conv_out.",
}

_UNET_RESBLOCK_MAP = {
    "in_layers.0.": "norm1.",
    "in_layers.2.": "conv1.",
    "emb_layers.1.": "time_emb_proj.",
    "out_layers.0.": "norm2.",
    "out_layers.3.": "conv2.",
    "skip_connection.": "conv_shortcut.",
}


def _remap_unet_state(state):
    """Remap single-file UNet keys (LDM format) to split-file (Diffusers) format."""
    remapped = {}
    for key, value in state.items():
        k = key
        for src, dst in _UNET_BLOCK_MAP.items():
            if k.startswith(src):
                rest = k[len(src):]
                # ResBlock internal key remapping
                for rsrc, rdst in _UNET_RESBLOCK_MAP.items():
                    if rest.startswith(rsrc):
                        rest = rdst + rest[len(rsrc):]
                        break
                # Downsampler: op.* → conv.*
                if rest.startswith("op."):
                    rest = "conv." + rest[3:]
                k = dst + rest
                break
        remapped[k] = value
    return remapped


def _remap_vae_state(state):
    """Remap single-file VAE keys (after prefix strip) to split-file format."""
    attn_map = {
        "norm.": "group_norm.",
        "q.": "query.",
        "k.": "key.",
        "v.": "value.",
        "proj_out.": "proj_attn.",
    }
    # Attention weights are stored as [C, C, 1, 1] conv2d in the single file;
    # squeeze to [C, C] to match the split-file (linear) format.
    attn_weight_suffixes = {"query.weight", "key.weight", "value.weight", "proj_attn.weight"}

    remapped = {}
    for key, value in state.items():
        k = key
        # Mid block ResNets
        k = k.replace("decoder.mid.block_1.", "decoder.mid_block.resnets.0.")
        k = k.replace("decoder.mid.block_2.", "decoder.mid_block.resnets.1.")
        # Mid block Attention
        for src, dst in attn_map.items():
            k = k.replace(f"decoder.mid.attn_1.{src}", f"decoder.mid_block.attentions.0.{dst}")
        # Up blocks: up.{i} → up_blocks.{3-i}, order reversed
        for i in range(4):
            old = f"decoder.up.{i}."
            if k.startswith(old):
                rest = k[len(old):]
                new_prefix = f"decoder.up_blocks.{3 - i}."
                rest = rest.replace("block.", "resnets.", 1)
                rest = rest.replace(".nin_shortcut.", ".conv_shortcut.")
                rest = rest.replace("upsample.conv.", "upsamplers.0.conv.")
                k = new_prefix + rest
                break
        # Output norm
        k = k.replace("decoder.norm_out.", "decoder.conv_norm_out.")
        if any(k.endswith(s) for s in attn_weight_suffixes):
            value = value.squeeze()
        remapped[k] = value
    return remapped


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


def load_from_single_file(path=None):
    """Load all SD 1.5 components from a single safetensors file."""
    from my_sd15.clip import CLIPTextModel
    from my_sd15.unet import UNet
    from my_sd15.vae import VaeDecoder

    if path is None:
        path = SINGLE_FILE_PATH

    raw = load_safetensors(path)

    # Text encoder: strip "cond_stage_model.transformer." prefix
    clip_prefix = "cond_stage_model.transformer."
    clip_state = {k[len(clip_prefix):]: v for k, v in raw.items() if k.startswith(clip_prefix)}

    # U-Net: strip "model.diffusion_model." prefix, then remap LDM→Diffusers keys
    unet_prefix = "model.diffusion_model."
    unet_raw = {k[len(unet_prefix):]: v for k, v in raw.items() if k.startswith(unet_prefix)}
    unet_state = _remap_unet_state(unet_raw)

    # VAE: strip "first_stage_model." prefix, then remap keys
    vae_prefix = "first_stage_model."
    vae_raw = {k[len(vae_prefix):]: v for k, v in raw.items() if k.startswith(vae_prefix)}
    vae_state = _remap_vae_state(vae_raw)

    return CLIPTextModel(clip_state), UNet(unet_state), VaeDecoder(vae_state)
