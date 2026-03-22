# Stable Diffusion 1.5 Inference Specification

This document specifies the architecture and algorithms needed to implement
Stable Diffusion 1.5 text-to-image inference from scratch. All information
is derived from the published paper, model card, and weight file inspection.

## 1. Pipeline Overview

```
prompt (string)
    │
    ▼
CLIP Tokenizer ──► token_ids (77,) int
    │
    ▼
CLIP Text Encoder ──► cond_emb (77, 768) float32
    │
    │   "" (empty string)
    │       │
    │       ▼
    │   CLIP Tokenizer + Text Encoder ──► uncond_emb (77, 768)
    │
    ▼
┌─── Denoising Loop (N steps) ───────────────────────┐
│                                                     │
│  latents (4, H/8, W/8)                             │
│      │                                              │
│      ├──► U-Net(latents, t, cond_emb) ──► noise_c  │
│      ├──► U-Net(latents, t, uncond_emb) ──► noise_u│
│      │                                              │
│      │  CFG: noise = noise_u + scale*(noise_c-noise_u)
│      │                                              │
│      ▼                                              │
│  DDIM step ──► updated latents                      │
│                                                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
latents / 0.18215
    │
    ▼
VAE Decoder ──► image (3, H, W) float32 in [-1, 1]
    │
    ▼
(image + 1) / 2 ──► [0, 1] ──► uint8 ──► PNG
```

## 2. Weight Files

Source: `stable-diffusion-v1-5/stable-diffusion-v1-5` on Hugging Face.
Format: safetensors. All weights are loaded as float32.

### Single-file format (primary)

```
tokenizer/vocab.json                        # BPE vocabulary: token_string -> token_id
tokenizer/merges.txt                        # BPE merge rules
v1-5-pruned-emaonly.safetensors             # All model weights in one file (~4.0 GB)
```

The single file contains 1145 keys using the original LDM naming convention.
Use `load_from_single_file()` in `loader.py` to load all three models at once.

**Note:** The tokenizer (`vocab.json`, `merges.txt`) is **not** included in the
single file and must be loaded separately from the `tokenizer/` directory.

#### Key inventory

Of the 1145 keys, 1022 are used by the inference code:

| Component | Prefix | Total keys | Used |
|---|---|---|---|
| U-Net | `model.diffusion_model.` | 686 | 686 |
| CLIP text encoder | `cond_stage_model.transformer.` | 197 | 196 |
| VAE | `first_stage_model.` | 248 | 140 |

The remaining 123 unused keys:

| Category | Keys | Description |
|---|---|---|
| VAE encoder | 108 | `first_stage_model.encoder.*` + `first_stage_model.quant_conv.*` — encoding path, not needed for inference |
| Scheduler precomputed | 14 | `alphas_cumprod`, `betas`, `sqrt_*`, `posterior_*`, etc. — recomputed from scratch by `DDIMScheduler` |
| CLIP position_ids | 1 | `cond_stage_model.transformer.text_model.embeddings.position_ids` — integer index tensor, not used |

#### Key name differences from split-file format

The single file uses LDM-style names that differ from the Diffusers-style names
used in the split files. `loader.py` remaps them automatically.

**CLIP** — strip prefix only, key names are identical:
```
cond_stage_model.transformer.text_model.* → text_model.*
```

**U-Net** — strip prefix and remap LDM block names to Diffusers names:
```
time_embed.{0,2}.*         → time_embedding.linear_{1,2}.*
input_blocks.0.0.*         → conv_in.*
input_blocks.{1-11}.*      → down_blocks.{0-3}.*
middle_block.{0,1,2}.*     → mid_block.{resnets.0, attentions.0, resnets.1}.*
output_blocks.{0-11}.*     → up_blocks.{0-3}.*
out.{0,2}.*                → {conv_norm_out, conv_out}.*
```

ResBlock internal key remapping:
```
in_layers.0.*   → norm1.*
in_layers.2.*   → conv1.*
emb_layers.1.*  → time_emb_proj.*
out_layers.0.*  → norm2.*
out_layers.3.*  → conv2.*
skip_connection.*→ conv_shortcut.*
op.*            → conv.*          (downsampler)
```

**VAE** — strip prefix, remap block names, and squeeze attention weights:
```
first_stage_model.decoder.mid.block_{1,2}.* → decoder.mid_block.resnets.{0,1}.*
first_stage_model.decoder.mid.attn_1.norm.* → decoder.mid_block.attentions.0.group_norm.*
first_stage_model.decoder.mid.attn_1.{q,k,v}.* → decoder.mid_block.attentions.0.{to_q,to_k,to_v}.*
first_stage_model.decoder.mid.attn_1.proj_out.* → decoder.mid_block.attentions.0.to_out.0.*
first_stage_model.decoder.up.{i}.block.{j}.* → decoder.up_blocks.{3-i}.resnets.{j}.*  (order reversed)
first_stage_model.decoder.up.{i}.upsample.conv.* → decoder.up_blocks.{3-i}.upsamplers.0.conv.*
first_stage_model.decoder.norm_out.*        → decoder.conv_norm_out.*
nin_shortcut.*  → conv_shortcut.*
```

Attention weights (`to_q`, `to_k`, `to_v`, `to_out.0`) are stored as
`(C, C, 1, 1)` conv2d tensors in the single file; they are squeezed to
`(C, C)` to match the linear format expected by the implementation.

**Note:** Older Diffusers versions used `query`/`key`/`value`/`proj_attn` for these keys.
The current format uses `to_q`/`to_k`/`to_v`/`to_out.0` (matching UNet attention naming).

### Split-file format (legacy)

```
tokenizer/vocab.json          # BPE vocabulary: token_string -> token_id
tokenizer/merges.txt          # BPE merge rules
text_encoder/model.safetensors  # CLIP text encoder (~492 MB)
unet/diffusion_pytorch_model.safetensors  # U-Net (~3.4 GB)
vae/diffusion_pytorch_model.safetensors   # VAE (~335 MB)
```

Uses Diffusers-style key names (same as used throughout this spec).
The split files were converted from the single file by Hugging Face and may
have minor floating-point differences due to rounding during conversion.

## 3. CLIP Tokenizer

BPE tokenizer (GPT-2 style byte-level BPE with modifications for CLIP).

### Vocabulary
- vocab.json: 49408 entries mapping token strings to integer IDs
- Tokens use `</w>` suffix to mark word boundaries

### Special tokens
- `<|startoftext|>` (BOS): prepended to every sequence
- `<|endoftext|>` (EOS): appended after the text, also used as padding

### Encoding procedure
1. Lowercase the input text and strip whitespace
2. Split using regex: `'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`
3. For each match, encode bytes using the GPT-2 byte-to-unicode mapping
4. Apply BPE merges (from merges.txt, skipping the `#version:` header line)
5. BPE adds `</w>` to the last character of each word before merging
6. Look up each BPE token in vocab.json to get token IDs
7. Prepend BOS token ID, append EOS token ID
8. Truncate to 77 tokens (keeping BOS, replacing last with EOS if needed)
9. Pad with EOS token ID to exactly 77 tokens

### GPT-2 byte-to-unicode mapping
Maps each byte (0-255) to a printable unicode character. The printable ASCII
ranges (33-126, 161-172, 174-255) map to themselves as characters. The
remaining bytes (0-32, 127-160, 173) map to characters starting at U+0100.

## 4. CLIP Text Encoder

Architecture: 12-layer Transformer with causal (GPT-2 style) attention mask.

### Weight key prefix: `text_model.`

### Embeddings
- `embeddings.token_embedding.weight`: (49408, 768) — token embedding table
- `embeddings.position_embedding.weight`: (77, 768) — learned positional embeddings

### Transformer layers (x12)
Prefix: `encoder.layers.{i}.` for i in 0..11

Each layer (pre-norm style):
```
x = x + self_attn(layer_norm1(x))
x = x + mlp(layer_norm2(x))
```

**Layer norm** (2 per layer):
- `layer_norm1.weight`, `layer_norm1.bias`: (768,)
- `layer_norm2.weight`, `layer_norm2.bias`: (768,)

**Self-attention**:
- `self_attn.q_proj.weight`, `.bias`: (768, 768)
- `self_attn.k_proj.weight`, `.bias`: (768, 768)
- `self_attn.v_proj.weight`, `.bias`: (768, 768)
- `self_attn.out_proj.weight`, `.bias`: (768, 768)
- num_heads = 12, head_dim = 64
- Causal mask: upper-triangular matrix of -inf

**MLP**:
- `mlp.fc1.weight`, `.bias`: (3072, 768)
- `mlp.fc2.weight`, `.bias`: (768, 3072)
- Activation: quick_gelu = `x * sigmoid(1.702 * x)`

### Final layer norm
- `final_layer_norm.weight`, `.bias`: (768,)

### Forward pass
```
x = token_embedding[input_ids] + position_embedding[:seq_len]
causal_mask = upper_triangular(-inf, shape=(77, 77))
for layer in layers:
    x = layer(x, causal_mask)
x = final_layer_norm(x)
return x  # shape: (77, 768)
```

### Output
Returns the full sequence (77, 768), not just the EOS position.
This is passed to U-Net cross-attention as encoder_hidden_states.

## 5. VAE Decoder

Decodes latent representation (4, H/8, W/8) to image (3, H, W).
Range of output: approximately [-1, 1].

### Scaling factor
Before decoding, divide latents by 0.18215.

### Architecture overview
```
post_quant_conv (1x1)  ──► conv_in (3x3, pad=1)
    ──► mid_block
    ──► up_block_0 (with upsample)
    ──► up_block_1 (with upsample)
    ──► up_block_2 (with upsample)
    ──► up_block_3 (no upsample)
    ──► group_norm ──► silu ──► conv_out (3x3, pad=1)
```

### Channel configuration
- post_quant_conv: (4, 4, 1, 1) — 4 channels in, 4 channels out
- conv_in: (512, 4, 3, 3)
- up_block_0: 512 → 512, with upsample
- up_block_1: 512 → 512, with upsample
- up_block_2: 512 → 256, with upsample
- up_block_3: 256 → 128, no upsample
- conv_norm_out: 128 channels
- conv_out: (3, 128, 3, 3)

### Weight key prefix: `decoder.`

### VaeResBlock
Used in mid_block and up_blocks. No timestep embedding (unlike U-Net).

Prefix: `{parent}.resnets.{j}.`
- `norm1.weight`, `norm1.bias`: (C_in,) — group_norm, 32 groups
- `conv1.weight`, `conv1.bias`: (C_out, C_in, 3, 3) — padding=1
- `norm2.weight`, `norm2.bias`: (C_out,) — group_norm, 32 groups
- `conv2.weight`, `conv2.bias`: (C_out, C_out, 3, 3) — padding=1
- `conv_shortcut.weight`, `conv_shortcut.bias`: (C_out, C_in, 1, 1) — only when C_in != C_out

```
h = silu(group_norm(x))
h = conv1(h)
h = silu(group_norm(h))
h = conv2(h)
if C_in != C_out:
    x = conv_shortcut(x)
return x + h
```

Activation: silu = `x * sigmoid(x)`

### VaeAttention (mid_block only)
Single-head self-attention over spatial positions.

Prefix: `mid_block.attentions.0.`
- `group_norm.weight`, `group_norm.bias`: (512,)
- `to_q.weight`, `to_q.bias`: (512, 512)
- `to_k.weight`, `to_k.bias`: (512, 512)
- `to_v.weight`, `to_v.bias`: (512, 512)
- `to_out.0.weight`, `to_out.0.bias`: (512, 512)

```
h = group_norm(x)                  # (C, H, W)
h = h.reshape(C, H*W).T           # (H*W, C)
q, k, v = linear(h) for each      # (H*W, C)
attn = softmax(q @ k.T / sqrt(C)) # (H*W, H*W)
h = attn @ v                       # (H*W, C)
h = linear(h)                      # to_out.0
h = h.T.reshape(C, H, W)
return x + h
```

### Mid block
Prefix: `mid_block.`
- `resnets.0`: VaeResBlock (512 → 512)
- `attentions.0`: VaeAttention (512 channels)
- `resnets.1`: VaeResBlock (512 → 512)

```
x = resnet_0(x)
x = attention(x)
x = resnet_1(x)
```

### Up decoder block (x4)
Prefix: `up_blocks.{i}.`
- `resnets.{0,1,2}`: VaeResBlock (3 per block)
- `upsamplers.0.conv.weight`, `.bias`: (C, C, 3, 3) — only for blocks 0,1,2

```
for resnet in resnets:
    x = resnet(x)
if has_upsampler:
    x = nearest_upsample_2x(x)
    x = conv(x, padding=1)
```

#### Per-block resnet input/output channels:
| Block | resnet.0 | resnet.1 | resnet.2 | Upsample |
|---|---|---|---|---|
| up_blocks.0 | 512→512 | 512→512 | 512→512 | yes (512) |
| up_blocks.1 | 512→512 | 512→512 | 512→512 | yes (512) |
| up_blocks.2 | 512→256 | 256→256 | 256→256 | yes (256) |
| up_blocks.3 | 256→128 | 128→128 | 128→128 | no |

## 6. U-Net

The largest component. Predicts noise given noisy latents, timestep, and
text conditioning.

Input: (4, H/8, W/8), Output: (4, H/8, W/8)

### Architecture overview
```
timestep ──► sinusoidal_embedding(320) ──► linear(1280) ──► silu ──► linear(1280) ──► temb

x = conv_in(latents)                    # (4,H,W) → (320,H,W)

# Down path
down_0: CrossAttnDownBlock  320ch  ──► downsample
down_1: CrossAttnDownBlock  640ch  ──► downsample
down_2: CrossAttnDownBlock  1280ch ──► downsample
down_3: DownBlock           1280ch     (no downsample)

# Mid
mid: resnet ──► spatial_transformer ──► resnet  (1280ch)

# Up path (with skip connections from down path)
up_0: UpBlock          1280ch ──► upsample
up_1: CrossAttnUpBlock 1280ch ──► upsample
up_2: CrossAttnUpBlock 640ch  ──► upsample
up_3: CrossAttnUpBlock 320ch     (no upsample)

x = group_norm(x) ──► silu ──► conv_out  # (320,H,W) → (4,H,W)
```

### Timestep embedding

```
half = 160  # 320 // 2
freqs = exp(-log(10000) * arange(half) / half)
args = float(timestep) * freqs
emb = concat([cos(args), sin(args)])  # (320,)
```

Weight keys:
- `time_embedding.linear_1.weight`, `.bias`: (1280, 320)
- `time_embedding.linear_2.weight`, `.bias`: (1280, 1280)

```
emb = linear_1(emb)  # (320,) → (1280,)
emb = silu(emb)
emb = linear_2(emb)  # (1280,) → (1280,)
```

### UNetResBlock

Like VaeResBlock but with timestep embedding injection.

Prefix: `{parent}.resnets.{j}.`
- `norm1.weight`, `norm1.bias`: (C_in,)
- `conv1.weight`, `conv1.bias`: (C_out, C_in, 3, 3)
- `time_emb_proj.weight`, `time_emb_proj.bias`: (C_out, 1280)
- `norm2.weight`, `norm2.bias`: (C_out,)
- `conv2.weight`, `conv2.bias`: (C_out, C_out, 3, 3)
- `conv_shortcut.weight`, `conv_shortcut.bias`: (C_out, C_in, 1, 1) — when C_in != C_out

```
h = silu(group_norm(x))
h = conv1(h, padding=1)
t = linear(silu(temb))           # (1280,) → (C_out,)
h = h + t.reshape(-1, 1, 1)     # broadcast over spatial dims
h = silu(group_norm(h))
h = conv2(h, padding=1)
if C_in != C_out:
    x = conv_shortcut(x)
return x + h
```

All group_norm uses num_groups=32.

### CrossAttention

Used for both self-attention and cross-attention.
Note: to_q, to_k, to_v have **no bias**.

Prefix: `{parent}.attn{1,2}.`
- `to_q.weight`: (dim, dim) — for self-attn; (dim, dim) for cross-attn query
- `to_k.weight`: (dim, dim) for self-attn; (dim, 768) for cross-attn
- `to_v.weight`: (dim, dim) for self-attn; (dim, 768) for cross-attn
- `to_out.0.weight`, `to_out.0.bias`: (dim, dim)

num_heads = 8, head_dim = dim / 8.

```
# For self-attention: context = x
# For cross-attention: context = encoder_hidden_states
q = x @ to_q.T                    # (seq, dim)
k = context @ to_k.T              # (ctx_seq, dim)
v = context @ to_v.T              # (ctx_seq, dim)

# Reshape to multi-head
q = q.reshape(seq, num_heads, head_dim).permute(1,0,2)   # (8, seq, head_dim)
k = k.reshape(ctx_seq, num_heads, head_dim).permute(1,0,2)
v = v.reshape(ctx_seq, num_heads, head_dim).permute(1,0,2)

attn = softmax(q @ k.T / sqrt(head_dim))
out = (attn @ v).permute(1,0,2).reshape(seq, dim)
out = out @ to_out.weight.T + to_out.bias
```

> **Note:** Use division (`/ sqrt(head_dim)`) rather than multiplication by the
> precomputed inverse (`* (1 / sqrt(head_dim))`). In float32, `a / b` involves
> one rounding step, while `a * (1/b)` involves two (computing `1/b` then
> multiplying), so division is theoretically at least as accurate. The two
> methods can differ by ~5e-7 per operation. Because the U-Net contains many
> attention blocks and the denoising loop feeds each step's output into the next,
> this per-operation difference compounds across steps and can reach ~0.003 after
> 10 steps — enough to fail verification at atol=1e-3.

### GEGLU FeedForward

Prefix: `{parent}.ff.`
- `net.0.proj.weight`, `net.0.proj.bias`: (2 * inner_dim, dim) — inner_dim = dim * 4
- `net.2.weight`, `net.2.bias`: (dim, inner_dim)

```
h = linear(x, proj_weight, proj_bias)  # (seq, 2*inner_dim)
h, gate = split(h, 2, dim=-1)          # each (seq, inner_dim)
h = h * gelu(gate)                     # GEGLU activation
h = linear(h, net2_weight, net2_bias)  # (seq, dim)
```

gelu: exact erf-based `0.5 * x * (1 + erf(x / sqrt(2)))`

### BasicTransformerBlock

Prefix: `{parent}.transformer_blocks.0.`
- `norm1.weight`, `norm1.bias`: (dim,) — layer_norm
- `attn1`: CrossAttention (self-attention, context = x itself)
- `norm2.weight`, `norm2.bias`: (dim,) — layer_norm
- `attn2`: CrossAttention (cross-attention, context = encoder_hidden_states)
- `norm3.weight`, `norm3.bias`: (dim,) — layer_norm
- `ff`: GEGLU FeedForward

```
x = x + attn1(layer_norm(x))                          # self-attention
x = x + attn2(layer_norm(x), encoder_hidden_states)   # cross-attention
x = x + ff(layer_norm(x))                             # feedforward
```

### SpatialTransformer

Wraps BasicTransformerBlock with spatial reshape and 1x1 convolutions.

Prefix: `{parent}.attentions.{j}.`
- `norm.weight`, `norm.bias`: (C,) — group_norm, 32 groups
- `proj_in.weight`, `proj_in.bias`: (C, C, 1, 1) — 1x1 conv
- `transformer_blocks.0`: BasicTransformerBlock
- `proj_out.weight`, `proj_out.bias`: (C, C, 1, 1) — 1x1 conv

```
residual = x                            # (C, H, W)
x = group_norm(x)
x = conv1x1(x, proj_in)                # (C, H, W)
x = x.reshape(C, H*W).T               # (H*W, C)
x = transformer_block(x, context)      # (H*W, C)
x = x.T.reshape(C, H, W)
x = conv1x1(x, proj_out)              # (C, H, W)
return residual + x
```

### Down blocks

#### CrossAttnDownBlock (blocks 0, 1, 2)
Prefix: `down_blocks.{i}.`
- `resnets.{0,1}`: UNetResBlock (2 resnets per block)
- `attentions.{0,1}`: SpatialTransformer (1 per resnet)
- `downsamplers.0.conv.weight`, `.bias`: (C, C, 3, 3) — stride=2, padding=1

```
outputs = []
for resnet, attn in zip(resnets, attentions):
    x = resnet(x, temb)
    x = attn(x, context)
    outputs.append(x)        # skip connection
x = conv(x, stride=2, padding=1)  # downsample
outputs.append(x)                  # skip connection
```

#### DownBlock (block 3)
Prefix: `down_blocks.3.`
- `resnets.{0,1}`: UNetResBlock (2 resnets)
- No attentions, no downsampler

```
outputs = []
for resnet in resnets:
    x = resnet(x, temb)
    outputs.append(x)
```

#### Channel dimensions per down block:
| Block | Type | Input ch | Output ch | Downsample |
|---|---|---|---|---|
| 0 | CrossAttnDown | 320 | 320 | yes |
| 1 | CrossAttnDown | 320 | 640 | yes |
| 2 | CrossAttnDown | 640 | 1280 | yes |
| 3 | Down | 1280 | 1280 | no |

Note: block 1 resnet.0 has conv_shortcut (320→640), block 2 resnet.0 has
conv_shortcut (640→1280).

### Mid block

Prefix: `mid_block.`
- `resnets.0`: UNetResBlock (1280 → 1280)
- `attentions.0`: SpatialTransformer (1280 channels)
- `resnets.1`: UNetResBlock (1280 → 1280)

```
x = resnet_0(x, temb)
x = spatial_transformer(x, context)
x = resnet_1(x, temb)
```

### Up blocks

Up blocks consume skip connections from the down path. Each resnet
concatenates its input with a skip connection along the channel dimension.

Skip connections are accumulated during the down path (including the
conv_in output) as a list, and consumed in reverse order (pop from end).

Total skip connections: 1 (conv_in) + 3 + 3 + 3 + 2 = 12.
Each up block has 3 resnets, consuming 3 skip connections each (3 × 4 = 12).

#### UpBlock (block 0)
Prefix: `up_blocks.0.`
- `resnets.{0,1,2}`: UNetResBlock (3 resnets)
- `upsamplers.0.conv.weight`, `.bias`: (1280, 1280, 3, 3)
- No attentions

```
for resnet in resnets:
    skip = skip_connections.pop()  # from end
    x = concat([x, skip], dim=channel)
    x = resnet(x, temb)
x = nearest_upsample_2x(x)
x = conv(x, padding=1)
```

#### CrossAttnUpBlock (blocks 1, 2, 3)
Prefix: `up_blocks.{i}.`
- `resnets.{0,1,2}`: UNetResBlock (3 resnets)
- `attentions.{0,1,2}`: SpatialTransformer (1 per resnet)
- `upsamplers.0.conv.weight`, `.bias` — blocks 1 and 2 only

```
for resnet, attn in zip(resnets, attentions):
    skip = skip_connections.pop()
    x = concat([x, skip], dim=channel)
    x = resnet(x, temb)
    x = attn(x, context)
if has_upsampler:
    x = nearest_upsample_2x(x)
    x = conv(x, padding=1)
```

#### Up block resnet input channels (after skip concatenation):

| Block | Type | resnet.0 in | resnet.1 in | resnet.2 in | Out ch | Upsample |
|---|---|---|---|---|---|---|
| 0 | Up | 2560 | 2560 | 2560 | 1280 | yes |
| 1 | CrossAttnUp | 2560 | 2560 | 1920 | 1280 | yes |
| 2 | CrossAttnUp | 1920 | 1280 | 960 | 640 | yes |
| 3 | CrossAttnUp | 960 | 640 | 640 | 320 | no |

All resnets with input != output channels have conv_shortcut.

### Output
- `conv_norm_out.weight`, `conv_norm_out.bias`: (320,) — group_norm, 32 groups
- `conv_out.weight`, `conv_out.bias`: (4, 320, 3, 3) — padding=1

```
x = silu(group_norm(x))
x = conv_out(x, padding=1)   # (320,H,W) → (4,H,W)
```

## 7. DDIM Scheduler

No learned parameters. Pure mathematical computation.

### Beta schedule (scaled_linear)
```
betas = linspace(sqrt(0.00085), sqrt(0.012), 1000) ** 2
alphas = 1 - betas
alpha_cumprod = cumprod(alphas)   # (1000,)
```

### Timestep selection
```
step_ratio = 1000 / num_inference_steps
timesteps = round(arange(0, num_inference_steps) * step_ratio)
timesteps = reverse(timesteps)    # descending order
```

For 10 steps: [900, 800, 700, 600, 500, 400, 300, 200, 100, 0]

### DDIM step (eta=0, deterministic)
```
alpha_t      = alpha_cumprod[t]
alpha_t_prev = alpha_cumprod[t - step_ratio]  # or 1.0 if t_prev < 0

pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)

prev_sample = sqrt(alpha_t_prev) * pred_x0
            + sqrt(1 - alpha_t_prev) * noise_pred
```

## 7b. LCM Scheduler

Used with LCM LoRA for few-step inference (1-4 steps).
Same beta schedule and alpha_cumprod as DDIM. No learned parameters.

Reference: Luo et al., "Latent Consistency Models" (2023), Algorithm 2.

### Timestep selection
```
original_steps = 50
c = 1000 // original_steps                              # 20
lcm_timesteps = arange(1, original_steps + 1) * c - 1   # [19, 39, ..., 999]
skip = len(lcm_timesteps) // num_inference_steps
timesteps = reverse(lcm_timesteps)[::skip][:num_inference_steps]
```

For 2 steps: [999, 499]
For 4 steps: [999, 739, 479, 219]

Previous timestep is looked up from the schedule (not computed by subtraction):
```
prev_timestep = {999: 499, 499: -1}   # example for 2 steps
```

### LCM step (Algorithm 2: Multistep Latent Consistency Sampling)

Each step predicts z0 directly, then re-noises with **random noise** (not noise_pred).

```
alpha_t = alpha_cumprod[t]
pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)

if t_prev >= 0:
    # Intermediate step: re-noise with random noise
    alpha_t_prev = alpha_cumprod[t_prev]
    noise = randn_like(pred_x0, generator=generator)
    prev_sample = sqrt(alpha_t_prev) * pred_x0
                + sqrt(1 - alpha_t_prev) * noise
else:
    # Final step: return pred_x0 directly
    prev_sample = pred_x0
```

Key differences from DDIM:

| | DDIM | LCM |
|---|---|---|
| Re-noising | noise_pred (deterministic) | randn (random) |
| Final step | Same formula (alpha_t_prev=1 zeroes noise term) | Return pred_x0 directly |
| pred_x0 clamp | None | None |
| CFG scale | 7.5 (typical) | 1.0 (guidance baked into LoRA) |

**Important**: Do NOT clamp pred_x0 to [-1, 1]. Latent space values routinely
exceed this range. Clamping destroys detail (e.g., fine textures become blurred).

## 8. Full Pipeline

```python
# 1. Tokenize and encode
cond_emb  = clip_encode(tokenize(prompt))     # (77, 768)
uncond_emb = clip_encode(tokenize(""))         # (77, 768)

# 2. Initialize latents
latents = random_normal(4, H/8, W/8)          # float32

# 3. Set up scheduler
scheduler.set_timesteps(num_steps)

# 4. Denoising loop
for t in scheduler.timesteps:
    noise_cond   = unet(latents, t, cond_emb)
    noise_uncond = unet(latents, t, uncond_emb)
    noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
    latents = scheduler.step(noise_pred, t, latents)

# 5. Decode
image = vae_decode(latents / 0.18215)          # (3, H, W) in [-1, 1]
image = clip((image + 1) / 2, 0, 1)            # [0, 1]
image = to_uint8(image * 255)
```

### Default parameters
- Image size: 512x512 (latent 4x64x64), also works at 256x256 (4x32x32)
- num_inference_steps: 10-50 (10 is minimum for reasonable quality)
- cfg_scale: 7.5
- All computation in float32 on CPU

## 9. Operations Reference

All operations used in the model:

| Operation | Formula | Used in |
|---|---|---|
| conv2d | im2col + matmul, or equivalent | VAE, U-Net |
| group_norm | per-group mean/var normalize, 32 groups, eps=1e-5 | VAE, U-Net |
| layer_norm | per-feature mean/var normalize, eps=1e-5 | CLIP, U-Net transformers |
| linear | x @ W.T + b | All |
| silu | x * sigmoid(x) | VAE, U-Net |
| quick_gelu | x * sigmoid(1.702 * x) | CLIP |
| gelu | 0.5 * x * (1 + erf(x / sqrt(2))) | U-Net GEGLU |
| softmax | exp(x - max) / sum(exp(x - max)) | All attention |
| upsample_nearest_2d | repeat each pixel 2x in H and W | VAE, U-Net |
| embedding | table lookup by integer index | CLIP |

## 10. Verification

Test data is provided in `testdata/` as safetensors files. Metadata
(prompt, seed, parameters) is in `testdata/metadata.json`.

Verification tests are in `tests/`. Run with `pytest tests/`.

### Test data files

| File | Keys | Description |
|---|---|---|
| `tokenizer.safetensors` | `cond_ids`, `uncond_ids` | Token IDs for prompt and empty string |
| `clip.safetensors` | `cond_ids`, `cond_emb`, `uncond_ids`, `uncond_emb` | CLIP encoder inputs and outputs |
| `scheduler.safetensors` | `alphas_cumprod`, `timesteps`, `step_*` | Beta schedule, timesteps, and one DDIM step |
| `vae.safetensors` | `input`, `output` | VAE decoder input (4,32,32) and output (3,256,256) |
| `unet.safetensors` | `input`, `context`, `t`, `output` | Single U-Net forward pass |
| `pipeline.safetensors` | `latents_init`, `latents_step00`..`09`, `decoded`, `image` | Full pipeline intermediates |

### Test parameters
- Prompt: "a cat sitting on a windowsill"
- Seed: 42 (torch.manual_seed for initial latents)
- Steps: 10, CFG: 7.5, Size: 256x256

### Recommended implementation order with tests

Each stage has tests that pass independently, allowing incremental development.

**Stage 1: Operations (no weights needed, 25 tests)**

Implement `ops.py` with conv2d, group_norm, layer_norm, linear, silu,
quick_gelu, gelu, softmax, upsample_nearest_2d, embedding.

Tests in `tests/test_ops.py` use hand-computed values:
- conv2d: 1x1 (equivalent to linear), 3x3 with/without padding, stride, multi-channel
- group_norm: zero mean / unit variance after normalization, affine transform
- layer_norm: zero mean / unit variance, known input-output pairs
- linear: identity matrix, with bias, batched input
- activations: known values at 0, large positive, shape preservation
- softmax: sums to 1, non-negative, uniform for equal inputs, numerical stability
- upsample: size doubling, value repetition pattern
- embedding: table lookup

**Stage 2: Tokenizer (no weights needed, 9 tests)**

Implement CLIP BPE tokenizer.

Tests in `tests/test_tokenizer.py`:
- Property tests (7): output length is always 77, starts with BOS, has EOS,
  padding is EOS, lowercased, empty string handling, truncation
- Test data comparison (2): match saved token IDs for prompt and empty string

**Stage 3: Scheduler (no weights needed, 10 tests)**

Implement DDIM scheduler.

Tests in `tests/test_scheduler.py`:
- Beta schedule (4): shape, monotonically decreasing, range, match saved values
- Timesteps (4): count, descending order, known values [900..0], match saved
- DDIM step (3): deterministic, output shape, match saved single step

**Stage 4: CLIP Text Encoder (weights needed, 4 tests)**

Implement CLIP transformer and weight loader.

Tests in `tests/test_clip.py`:
- Output shape (77, 768)
- Match saved embeddings for prompt and empty string
- Different prompts produce different embeddings

**Stage 5: VAE Decoder (weights needed, 2 tests)**

Implement VAE decoder (ResBlock, Attention, UpBlock) and weight loader.

Tests in `tests/test_vae.py`:
- Output shape (3, 256, 256)
- Match saved decoded image

**Stage 6: U-Net (weights needed, 2 tests)**

Implement U-Net (ResBlock, CrossAttention, GEGLU, SpatialTransformer,
Down/Mid/Up blocks) and weight loader.

Tests in `tests/test_unet.py`:
- Output shape (4, 32, 32)
- Match saved U-Net output

**Stage 7: Pipeline (weights needed, 5 tests)**

Assemble the full pipeline.

Tests in `tests/test_pipeline.py`:
- Initial latents from torch.manual_seed(42)
- Each of the 10 denoising steps matches saved intermediates
- Final latents match
- VAE decoded image matches
- Final uint8 image matches exactly

### Tolerance guidelines

| Component | Recommended atol |
|---|---|
| Operations (hand-computed) | exact or 1e-5 |
| Scheduler | 1e-5 |
| Tokenizer | exact (integer) |
| CLIP encoder | 1e-4 |
| VAE decoder | 1e-3 |
| U-Net | 1e-3 |
| Pipeline per-step latents | 1e-3 |
| Pipeline decoded image | 1e-2 |
| Pipeline final uint8 image | exact |
