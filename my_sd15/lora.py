"""LoRA (Low-Rank Adaptation) loader for SD 1.5."""

from safetensors import safe_open


def _build_key_map(state):
    """Build a mapping from LoRA underscore keys to model state dict keys.

    LoRA key:  lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
    Model key: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight
    """
    key_map = {}
    for model_key in state:
        if not model_key.endswith(".weight"):
            continue
        # Remove .weight suffix, replace dots with underscores
        base = model_key.removesuffix(".weight").replace(".", "_")
        key_map[base] = model_key
    return key_map


def apply_lora(state, lora_path, scale=1.0):
    """Apply LoRA weights to model state dict (in-place).

    For each LoRA target with matrices A (down) and B (up):
        W' = W + scale * (alpha / rank) * B @ A

    Args:
        state: Model state dict (modified in-place).
        lora_path: Path to LoRA safetensors file.
        scale: User-specified scaling factor (default 1.0).
    """
    key_map = _build_key_map(state)

    with safe_open(lora_path, framework="pt") as f:
        lora_keys = f.keys()

        # Collect LoRA base names (unique targets)
        bases = set()
        for k in lora_keys:
            if k.endswith(".lora_down.weight"):
                bases.add(k.removesuffix(".lora_down.weight"))

        for base in sorted(bases):
            down = f.get_tensor(base + ".lora_down.weight").float()
            up = f.get_tensor(base + ".lora_up.weight").float()
            alpha = f.get_tensor(base + ".alpha").float().item()
            rank = down.shape[0]

            # Map LoRA key to model key
            lora_name = base.removeprefix("lora_unet_")
            if lora_name not in key_map:
                raise KeyError(f"LoRA key '{lora_name}' not found in model state dict")
            model_key = key_map[lora_name]

            # Compute delta: scale * (alpha / rank) * up @ down
            if down.dim() == 2:
                # Linear layer: down (rank, in), up (out, rank)
                delta = up @ down
            else:
                # Conv layer: down (rank, C_in, kH, kW), up (C_out, rank, 1, 1)
                delta = (up.squeeze(-1).squeeze(-1) @ down.reshape(rank, -1))
                delta = delta.reshape(state[model_key].shape)

            state[model_key] = state[model_key] + scale * (alpha / rank) * delta
