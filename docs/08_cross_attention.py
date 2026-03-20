"""08: Cross-Attention の動作確認。"""

import torch
from my_sd15.loader import load_model
from my_sd15.unet import CrossAttention, BasicTransformerBlock, SpatialTransformer

model = load_model()
s = model.unet.state

# 1. Self-Attention vs Cross-Attention
print("=== 1. Self-Attention vs Cross-Attention ===")
prefix = "down_blocks.0.attentions.0.transformer_blocks.0."
attn1 = CrossAttention(s, prefix + "attn1.", 320)  # Self-Attention
attn2 = CrossAttention(s, prefix + "attn2.", 320)  # Cross-Attention

x = torch.randn(1024, 320)    # 画像特徴 (H*W=32*32, dim=320)
context = torch.randn(77, 768)  # テキスト条件

with torch.no_grad():
    # Self-Attention: context=None → x を Q,K,V すべてに使う
    self_out = attn1(x, context=None)
    print(f"Self-Attention:")
    print(f"  入力 x:     {tuple(x.shape)}")
    print(f"  出力:       {tuple(self_out.shape)}")

    # Cross-Attention: Q=x, K/V=context
    cross_out = attn2(x, context=context)
    print(f"Cross-Attention:")
    print(f"  入力 x:     {tuple(x.shape)} (Q の出所)")
    print(f"  context:    {tuple(context.shape)} (K/V の出所)")
    print(f"  出力:       {tuple(cross_out.shape)}")
print()

# 2. BasicTransformerBlock (3 サブ層)
print("=== 2. BasicTransformerBlock ===")
block = BasicTransformerBlock(s, prefix, 320)
with torch.no_grad():
    block_out = block(x, context)
print(f"入力:   {tuple(x.shape)}")
print(f"出力:   {tuple(block_out.shape)}")
print(f"残差の効果: 入出力の差の平均 = {(block_out - x).abs().mean():.4f}")
print()

# 3. SpatialTransformer (2D ↔ 1D 変換)
print("=== 3. SpatialTransformer (2D → 系列 → 2D) ===")
st = SpatialTransformer(s, "down_blocks.0.attentions.0.", 320)
x_2d = torch.randn(320, 32, 32)  # 画像特徴マップ

with torch.no_grad():
    st_out = st(x_2d, context)
print(f"入力 (画像): {tuple(x_2d.shape)}")
print(f"  → reshape: ({320}, {32*32}) → ({32*32}, {320}) [系列として処理]")
print(f"  → reshape: ({320}, {32}, {32}) [画像に戻す]")
print(f"出力 (画像): {tuple(st_out.shape)}")
print()

# 4. Attention パラメータの確認
print("=== 4. パラメータ比較 ===")
q_weight = s[prefix + "attn1.to_q.weight"]
print(f"Self-Attention (attn1):")
print(f"  to_q.weight: {tuple(q_weight.shape)}")
has_q_bias = (prefix + "attn1.to_q.bias") in s
print(f"  to_q.bias:   {'あり' if has_q_bias else 'なし（GPT-2 との違い）'}")

q_weight2 = s[prefix + "attn2.to_q.weight"]
k_weight2 = s[prefix + "attn2.to_k.weight"]
print(f"Cross-Attention (attn2):")
print(f"  to_q.weight: {tuple(q_weight2.shape)} (入力: 320 dim)")
print(f"  to_k.weight: {tuple(k_weight2.shape)} (入力: 768 dim = CLIP 出力)")
