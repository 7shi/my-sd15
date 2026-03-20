"""Convert SD 1.5 single-file safetensors to split-file (Diffusers) format."""

import argparse
import os

from safetensors import safe_open
from safetensors.torch import save_file


def load_safetensors(path):
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
    remapped = {}
    for key, value in state.items():
        k = key
        for src, dst in _UNET_BLOCK_MAP.items():
            if k.startswith(src):
                rest = k[len(src):]
                for rsrc, rdst in _UNET_RESBLOCK_MAP.items():
                    if rest.startswith(rsrc):
                        rest = rdst + rest[len(rsrc):]
                        break
                if rest.startswith("op."):
                    rest = "conv." + rest[3:]
                k = dst + rest
                break
        remapped[k] = value
    return remapped


def _remap_vae_state(state):
    attn_map = {
        "norm.": "group_norm.",
        "q.": "query.",
        "k.": "key.",
        "v.": "value.",
        "proj_out.": "proj_attn.",
    }
    attn_weight_suffixes = {"query.weight", "key.weight", "value.weight", "proj_attn.weight"}

    remapped = {}
    for key, value in state.items():
        k = key
        k = k.replace("decoder.mid.block_1.", "decoder.mid_block.resnets.0.")
        k = k.replace("decoder.mid.block_2.", "decoder.mid_block.resnets.1.")
        for src, dst in attn_map.items():
            k = k.replace(f"decoder.mid.attn_1.{src}", f"decoder.mid_block.attentions.0.{dst}")
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
        k = k.replace("decoder.norm_out.", "decoder.conv_norm_out.")
        if any(k.endswith(s) for s in attn_weight_suffixes):
            value = value.squeeze()
        remapped[k] = value
    return remapped


def convert(src_path, out_dir, fp16=False):
    print(f"Loading {src_path} ...")
    raw = load_safetensors(src_path)

    dtype_label = "fp16" if fp16 else "fp32"
    suffix = ".fp16.safetensors" if fp16 else ".safetensors"

    def maybe_half(d):
        if fp16:
            return {k: v.half() for k, v in d.items()}
        return d

    # CLIP
    clip_prefix = "cond_stage_model.transformer."
    clip_state = {k[len(clip_prefix):]: v for k, v in raw.items() if k.startswith(clip_prefix)}
    clip_out = os.path.join(out_dir, "text_encoder", f"model{suffix}")
    os.makedirs(os.path.dirname(clip_out), exist_ok=True)
    print(f"Saving text_encoder ({len(clip_state)} keys, {dtype_label}):\n  {clip_out}")
    save_file(maybe_half(clip_state), clip_out)

    # UNet
    unet_prefix = "model.diffusion_model."
    unet_raw = {k[len(unet_prefix):]: v for k, v in raw.items() if k.startswith(unet_prefix)}
    unet_state = _remap_unet_state(unet_raw)
    unet_out = os.path.join(out_dir, "unet", f"diffusion_pytorch_model{suffix}")
    os.makedirs(os.path.dirname(unet_out), exist_ok=True)
    print(f"Saving unet ({len(unet_state)} keys, {dtype_label}):\n  {unet_out}")
    save_file(maybe_half(unet_state), unet_out)

    # VAE
    vae_prefix = "first_stage_model."
    vae_raw = {k[len(vae_prefix):]: v for k, v in raw.items() if k.startswith(vae_prefix)}
    vae_state = _remap_vae_state(vae_raw)
    vae_out = os.path.join(out_dir, "vae", f"diffusion_pytorch_model{suffix}")
    os.makedirs(os.path.dirname(vae_out), exist_ok=True)
    print(f"Saving vae ({len(vae_state)} keys, {dtype_label}):\n  {vae_out}")
    save_file(maybe_half(vae_state), vae_out)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Convert SD 1.5 single-file to split-file (Diffusers) format.")
    parser.add_argument("src", help="Path to source safetensors file")
    parser.add_argument("--out-dir", help="Output directory (default: same directory as src)")
    parser.add_argument("--fp16", action="store_true", help="Save weights in float16")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.dirname(args.src) or "."
    convert(args.src, out_dir, fp16=args.fp16)


if __name__ == "__main__":
    main()
