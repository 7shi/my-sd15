"""09: DDIM Scheduler の動作確認。"""

import torch
from my_sd15.scheduler import DDIMScheduler

scheduler = DDIMScheduler()

# 1. ノイズスケジュール
print("=== 1. ノイズスケジュール ===")
acp = scheduler.alphas_cumprod
print(f"alphas_cumprod の長さ: {len(acp)}")
print(f"  t=0:   ᾱ={acp[0]:.6f} (ほぼ元の画像)")
print(f"  t=500: ᾱ={acp[500]:.6f}")
print(f"  t=999: ᾱ={acp[999]:.6f} (ほぼ純粋なノイズ)")
print()

# 2. タイムステップ
print("=== 2. タイムステップの選択 ===")
for steps in [5, 10, 20]:
    scheduler.set_timesteps(steps)
    print(f"  {steps} ステップ: {scheduler.timesteps.tolist()}")
print()

# 3. 前方過程のシミュレーション
print("=== 3. 前方過程（x_0 → x_t）===")
x0 = torch.ones(4, 4, 4) * 0.5  # "きれいな画像"
epsilon = torch.randn_like(x0)   # ノイズ
for t in [0, 100, 500, 999]:
    alpha_t = acp[t]
    x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * epsilon
    signal = torch.sqrt(alpha_t).item()
    noise = torch.sqrt(1.0 - alpha_t).item()
    print(f"  t={t:4d}: signal={signal:.4f}, noise={noise:.4f}, "
          f"x_t mean={x_t.mean():.4f}, std={x_t.std():.4f}")
print()

# 4. DDIM ステップ
print("=== 4. DDIM ステップ ===")
scheduler.set_timesteps(10)
# シンプルなテスト: ノイズ予測が正確なら x_0 に近づく
sample = torch.sqrt(acp[900]) * x0 + torch.sqrt(1.0 - acp[900]) * epsilon
noise_pred = epsilon  # 完璧なノイズ予測
result = scheduler.step(noise_pred, 900, sample)
print(f"t=900 → t=800:")
print(f"  入力の x_0 からの距離: {(sample - x0).abs().mean():.4f}")
print(f"  出力の x_0 からの距離: {(result - x0).abs().mean():.4f}")

# 5. 決定性の確認
print()
print("=== 5. 決定性（同じ入力 → 同じ出力）===")
result2 = scheduler.step(noise_pred, 900, sample)
print(f"同じ入力での最大差: {(result - result2).abs().max().item():.2e}")
