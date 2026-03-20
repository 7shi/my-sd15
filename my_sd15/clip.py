"""CLIP Text Encoder for SD 1.5."""

import torch

from my_sd15.ops import embedding, layer_norm, linear, quick_gelu, softmax


class CLIPTextModel:
    def __init__(self, state):
        self.state = state

    def __call__(self, input_ids):
        s = self.state
        prefix = "text_model."

        # Embeddings
        token_emb = s[prefix + "embeddings.token_embedding.weight"]
        pos_emb = s[prefix + "embeddings.position_embedding.weight"]

        ids = torch.tensor(input_ids, dtype=torch.long)
        x = embedding(ids, token_emb) + pos_emb[:len(input_ids)]

        # Causal mask
        seq_len = x.shape[0]
        mask = torch.full((seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)

        # 12 transformer layers
        for i in range(12):
            lp = prefix + f"encoder.layers.{i}."

            # Self-attention with pre-norm
            ln1_w = s[lp + "layer_norm1.weight"]
            ln1_b = s[lp + "layer_norm1.bias"]
            h = layer_norm(x, ln1_w, ln1_b)

            q = linear(h, s[lp + "self_attn.q_proj.weight"], s[lp + "self_attn.q_proj.bias"])
            k = linear(h, s[lp + "self_attn.k_proj.weight"], s[lp + "self_attn.k_proj.bias"])
            v = linear(h, s[lp + "self_attn.v_proj.weight"], s[lp + "self_attn.v_proj.bias"])

            # Multi-head attention: 12 heads, head_dim=64
            num_heads = 12
            head_dim = 64
            q = q.reshape(seq_len, num_heads, head_dim).permute(1, 0, 2)
            k = k.reshape(seq_len, num_heads, head_dim).permute(1, 0, 2)
            v = v.reshape(seq_len, num_heads, head_dim).permute(1, 0, 2)

            attn = q @ k.transpose(-2, -1) / (head_dim ** 0.5)
            attn = attn + mask
            attn = softmax(attn, axis=-1)
            h = (attn @ v).permute(1, 0, 2).reshape(seq_len, 768)

            h = linear(h, s[lp + "self_attn.out_proj.weight"], s[lp + "self_attn.out_proj.bias"])
            x = x + h

            # MLP with pre-norm
            ln2_w = s[lp + "layer_norm2.weight"]
            ln2_b = s[lp + "layer_norm2.bias"]
            h = layer_norm(x, ln2_w, ln2_b)
            h = linear(h, s[lp + "mlp.fc1.weight"], s[lp + "mlp.fc1.bias"])
            h = quick_gelu(h)
            h = linear(h, s[lp + "mlp.fc2.weight"], s[lp + "mlp.fc2.bias"])
            x = x + h

        # Final layer norm
        x = layer_norm(x, s[prefix + "final_layer_norm.weight"], s[prefix + "final_layer_norm.bias"])
        return x
