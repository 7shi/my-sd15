"""05: GroupNorm の動作確認 — 手計算と PyTorch 検証。"""

import torch
import torch.nn.functional as F
from my_sd15.ops import group_norm

# 1. 小さな入力で動作確認
print("=== 1. 基本動作 ===")
torch.manual_seed(42)
x = torch.randn(4, 3, 3)  # 4 チャネル、3×3
weight = torch.ones(4)
bias = torch.zeros(4)
num_groups = 2  # 4チャネル → 2グループ×2チャネル

result = group_norm(x, weight, bias, num_groups)
ref = F.group_norm(x.unsqueeze(0), num_groups, weight, bias).squeeze(0)
diff = (result - ref).abs().max().item()
print(f"入力:     {tuple(x.shape)}")
print(f"グループ数: {num_groups} (各グループ {4 // num_groups} チャネル)")
print(f"最大誤差:  {diff:.2e}")
print()

# 2. 正規化の効果を確認
print("=== 2. 正規化の効果 ===")
# グループ 0 (チャネル 0,1) の統計
g0 = x[:2].reshape(-1)
print(f"グループ 0 正規化前: mean={g0.mean():.4f}, std={g0.std(correction=0):.4f}")
g0_norm = result[:2].reshape(-1)
print(f"グループ 0 正規化後: mean={g0_norm.mean():.4f}, std={g0_norm.std(correction=0):.4f}")
print()

# 3. SD 1.5 の典型的なサイズ（num_groups=32）
print("=== 3. SD 1.5 の典型サイズ ===")
x_sd = torch.randn(320, 32, 32)
w_sd = torch.randn(320)
b_sd = torch.randn(320)

result_sd = group_norm(x_sd, w_sd, b_sd, 32)
ref_sd = F.group_norm(x_sd.unsqueeze(0), 32, w_sd, b_sd).squeeze(0)
diff_sd = (result_sd - ref_sd).abs().max().item()
print(f"入力:     {tuple(x_sd.shape)}")
print(f"グループ数: 32 (各グループ {320 // 32} チャネル)")
print(f"最大誤差:  {diff_sd:.2e}")
