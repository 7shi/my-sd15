"""04: Conv2D の動作確認 — 手計算との比較と PyTorch 検証。"""

import torch
import torch.nn.functional as F
from my_sd15.ops import conv2d

# 1. 手計算との比較（1 チャネル、パディングなし）
print("=== 1. 手計算との比較 ===")
x = torch.tensor([
    [1., 2., 3., 0.],
    [0., 1., 2., 1.],
    [1., 0., 1., 2.],
    [2., 1., 0., 1.],
]).unsqueeze(0)  # (1, 4, 4)

kernel = torch.tensor([
    [1., 0., 1.],
    [0., 1., 0.],
    [1., 0., 1.],
]).reshape(1, 1, 3, 3)  # (1, 1, 3, 3)

result = conv2d(x, kernel)
print(f"入力:    {tuple(x.shape)}")
print(f"カーネル: {tuple(kernel.shape)}")
print(f"出力:    {tuple(result.shape)}")
print(f"出力値:\n{result[0]}")
print(f"期待値: [[7, 6], [4, 5]]")
print()

# 2. PyTorch F.conv2d との比較
print("=== 2. PyTorch との比較 ===")
ref = F.conv2d(x.unsqueeze(0), kernel).squeeze(0)
diff = (result - ref).abs().max().item()
print(f"最大誤差: {diff}")
print()

# 3. 複数チャネル + パディング + バイアス
print("=== 3. 複数チャネル ===")
x_multi = torch.randn(3, 8, 8)       # 3チャネル、8×8
w_multi = torch.randn(16, 3, 3, 3)   # 16出力チャネル
b_multi = torch.randn(16)

result_multi = conv2d(x_multi, w_multi, b_multi, padding=1)
ref_multi = F.conv2d(x_multi.unsqueeze(0), w_multi, b_multi, padding=1).squeeze(0)
diff_multi = (result_multi - ref_multi).abs().max().item()
print(f"入力:    {tuple(x_multi.shape)}")
print(f"カーネル: {tuple(w_multi.shape)}")
print(f"出力:    {tuple(result_multi.shape)}")
print(f"最大誤差: {diff_multi:.2e}")
print()

# 4. ストライド=2（ダウンサンプリング）
print("=== 4. stride=2 ===")
x_down = torch.randn(4, 32, 32)
w_down = torch.randn(4, 4, 3, 3)
b_down = torch.randn(4)

result_down = conv2d(x_down, w_down, b_down, stride=2, padding=1)
ref_down = F.conv2d(x_down.unsqueeze(0), w_down, b_down, stride=2, padding=1).squeeze(0)
diff_down = (result_down - ref_down).abs().max().item()
print(f"入力:    {tuple(x_down.shape)}")
print(f"出力:    {tuple(result_down.shape)} (空間サイズが半分)")
print(f"最大誤差: {diff_down:.2e}")
