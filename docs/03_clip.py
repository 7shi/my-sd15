"""03: CLIP Text Encoder の動作確認。"""

import torch
from my_sd15.loader import load_model

model = load_model()

# 1. トークン化
print("=== 1. トークン化 ===")
prompt = "a cat sitting on a windowsill"
tokens = model.tokenizer.encode(prompt)
print(f"プロンプト: '{prompt}'")
print(f"トークン数: {len(tokens)} (固定)")
print(f"先頭: {tokens[0]} (<|startoftext|>)")
print(f"末尾: {tokens[-1]} (<|endoftext|>)")

# テキスト部分のトークン ID
text_tokens = []
for t in tokens[1:]:
    if t == model.tokenizer.eos_token_id:
        break
    text_tokens.append(t)
print(f"テキスト部分: {text_tokens} ({len(text_tokens)} トークン)")
print()

# 2. GPT-2 との比較: 小文字化
print("=== 2. 小文字化の確認 ===")
upper = model.tokenizer.encode("Hello World")
lower = model.tokenizer.encode("hello world")
print(f"'Hello World' と 'hello world' のトークン: {'同一' if upper == lower else '異なる'}")
print()

# 3. CLIP Text Encoder
print("=== 3. CLIP Text Encoder ===")
with torch.no_grad():
    cond_emb = model.text_encoder(tokens)
print(f"出力形状: {tuple(cond_emb.shape)}")
print(f"出力 dtype: {cond_emb.dtype}")
print(f"平均: {cond_emb.mean():.4f}")
print(f"標準偏差: {cond_emb.std():.4f}")
print(f"最小: {cond_emb.min():.4f}")
print(f"最大: {cond_emb.max():.4f}")
print()

# 4. 空文字列（無条件ベクトル）
print("=== 4. 無条件ベクトル ===")
empty_tokens = model.tokenizer.encode("")
with torch.no_grad():
    uncond_emb = model.text_encoder(empty_tokens)
print(f"空文字列のトークン先頭 5: {empty_tokens[:5]}")
print(f"出力形状: {tuple(uncond_emb.shape)}")

# 条件付きと無条件の差
diff = (cond_emb - uncond_emb).abs().mean().item()
print(f"条件付き vs 無条件の平均絶対差: {diff:.4f}")
