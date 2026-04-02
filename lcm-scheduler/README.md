# LCMScheduler の実装と修正の経緯

画像生成のステップ数を短縮する LCM の調査のため、スケジューラーを独自実装しました。ちょっとしたコードの違いで絵柄が変わるのが興味深かったため、修正の経緯を残しておきます。

- 実装: [my_sd15/scheduler.py](../my_sd15/scheduler.py)

※ 引用したコードは、変更箇所が分かりやすいように整理しています。実際のコミット履歴とは異なりますが、動作自体は同じです。

## LCMScheduler の目的

通常の Stable Diffusion 1.5 では、DDIM スケジューラーを使って 20〜50 ステップかけてノイズを除去します。各ステップで U-Net を呼び出すため、ステップ数がそのまま推論時間に直結します。さらに CFG (Classifier-Free Guidance) のために条件あり・なしの 2 回の U-Net 呼び出しが必要なので、実質的な計算量はステップ数の 2 倍です。

LCM (Latent Consistency Model) は、この推論ステップ数を 2〜4 回にまで削減する手法です。専用の LoRA を適用した U-Net と組み合わせて使います。LCM LoRA は学習時に CFG の効果を LoRA の重みに焼き込んでいるため、推論時は `cfg_scale=1.0` として U-Net を 1 回だけ呼び出せば済みます（ネガティブプロンプトが効かなくなります）。ステップ数の削減と CFG 不要化の両方により、大幅な高速化が得られます。

## 仕様（SPEC.md §7b / 論文 Algorithm 2）

論文: [Luo et al. "Latent Consistency Models" (2023)](https://arxiv.org/abs/2310.04378)

LCMScheduler の処理は、論文の Algorithm 2 (Multistep Latent Consistency Sampling) に基づいています。

### タイムステップの選択

LCM のタイムステップは DDIM とは異なる方法で選択されます。

DDIM では 0〜999 の範囲から `num_steps` 個を等間隔に選びます。

1. `step_ratio = 1000 / num_steps` で間隔を求める
2. `[0, 1, ..., num_steps-1] * step_ratio` を整数に丸めて基準点を生成する
3. 基準点を逆順にしてタイムステップを得る

`num_steps=15` の場合、`step_ratio = 1000 / 15 ≈ 66.67` となり、丸めにより以下のタイムステップが得られます。

```python
timesteps = [933, 867, 800, 733, ..., 133, 67, 0]
```

LCM では学習時に使われた `original_steps`（デフォルト 50）に基づく格子点からさらに `num_steps` 個を選びます。これは LCM が consistency distillation という手法で学習されており、モデルが `original_steps` で定義された格子点上で蒸留されているためです。

具体的な手順は以下のとおりです。

1. `c = 1000 // original_steps` で格子の間隔を求める
2. `lcm_timesteps = [c-1, 2c-1, ..., original_steps*c-1]` で `original_steps` 個の格子点を生成する
3. 格子点を逆順にし、`skip = original_steps // num_steps` 間隔で `num_steps` 個を抽出する
4. `step_ratio = skip * c` をステップ間隔として保存する

`num_steps=4` の場合、格子点は `c=20` 間隔で 50 個（`[19, 39, 59, ..., 999]`）、そこから `skip=12` 間隔で抽出するので `step_ratio = 12 * 20 = 240` となります。

```python
timesteps = [999, 759, 519, 279]
```

DDIM と同様に `t_prev = t - step_ratio` で次のタイムステップを求めます。DDIM の `step_ratio` は `1000 / num_steps` ですが、LCM では `skip * c` であり、`original_steps` に依存する値になります。すべて整数除算で求まるため、DDIM のような丸め処理は不要です。

### step() の処理

各ステップの処理は以下のとおりです。式中の `alpha_t` と `alpha_t_prev` は `alphas_cumprod` テーブルからタイムステップに対応する値を取得したもので、各時点でのノイズレベルを表します。`alpha` が大きいほど信号成分が多く（ノイズが少ない）、`sqrt(alpha)` が信号の係数、`sqrt(1 - alpha)` がノイズの係数として使われます。このテーブルは beta schedule から計算される 1000 要素の配列で、DDIM と LCM で共通です。

まず、現在のノイズ付きサンプル `sample` と U-Net が予測したノイズ `noise_pred` から、元の画像 `pred_x0` を逆算します。

```
pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
```

この式自体は DDIM と同じです。ここから先の処理が LCM 固有の部分になります。

中間ステップ（最終ステップ以外）では、予測した `pred_x0` にランダムノイズを加えて、次のタイムステップのノイズレベルまで再ノイズ化します。

```
noise = randn_like(pred_x0)
prev_sample = sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise
```

最終ステップ（`t == timesteps[-1]`）では再ノイズ化は行わず、`pred_x0` をそのまま返します。

これらは DDIM との違いが、初期実装では正しく実装されていませんでした。

## DDIM

出発点となった DDIM の `step()` は次のようなものです。

```python
def step(self, noise_pred, t, sample):
    alpha_t = self.alphas_cumprod[t]
    t_prev = t - int(self._step_ratio)
    alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
    pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    prev_sample = sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise_pred
    return prev_sample
```

最終ステップでは `t_prev < 0` となり `alpha_t_prev = 1.0` が使われます。このとき `sqrt(1 - alpha_t_prev) = 0` なので `noise_pred` の項が消え、`prev_sample = pred_x0` となります。つまり DDIM では最終ステップで暗黙的に `noise_pred` が無視されます。これを条件分岐で明示すると次のようになります。

```python
def step(self, noise_pred, t, sample):
    alpha_t = self.alphas_cumprod[t]
    pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    if t == self.timesteps[-1]:  # 最終ステップ
        return pred_x0
    else:
        t_prev = t - int(self._step_ratio)
        alpha_t_prev = self.alphas_cumprod[t_prev]
        return sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise_pred
```

比較のため、LCM LoRA を適用して DDIM で 10 ステップ生成しました。

![ddim.jpg](ddim.jpg)

形状は認識できますが、画像全体にメッシュ状のノイズパターンが発生しています。

## LCM

LCMScheduler は LoRA サポートの一部として DDIM をベースに実装されました。

### 初期実装

初期 LCM 実装は、DDIM から 2 点を変更しました。

1. `_step_ratio` を LCM 用の値 `skip * c`（整数値）に変更
2. `pred_x0.clamp(-1.0, 1.0)` を追加

1 は、前述のとおり LCM のタイムステップ間隔が DDIM と異なるために必要な変更です。2 は latent の値域を制限する意図で Sonnet が追加したようです（問題を起こすため不要だったことが判明します）。

```python
def step(self, noise_pred, t, sample):
    alpha_t = self.alphas_cumprod[t]
    pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    pred_x0 = pred_x0.clamp(-1.0, 1.0)  # 追加
    if t == self.timesteps[-1]:
        return pred_x0
    else:
        t_prev = t - self._step_ratio   # int() 不要
        alpha_t_prev = self.alphas_cumprod[t_prev]
        return sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise_pred
```

しかし DDIM と LCM では再ノイズ化の意味が異なるため、この流用には問題がありました。

![lcm-1.jpg](lcm-1.jpg)

DDIM とはやや絵柄が変わりましたが、構図は似ています。画像全体にメッシュ状のノイズパターンが発生しているのも同様です。

### 修正 1：再ノイズ化とステップ分岐

メッシュ状ノイズの原因は、再ノイズ化に `noise_pred` を使っていたことです。`noise_pred` は U-Net の出力であり、画像の構造に対応した空間的パターンを含んでいます。このパターンを含んだノイズで再ノイズ化すると、次のステップの U-Net がまた似たパターンを予測し、それがさらに次のステップに持ち込まれるという自己強化フィードバックが発生します。結果としてメッシュ状のノイズが蓄積されます。修正では `noise_pred` を `torch.randn_like()` に置き換えました。ランダムノイズは全周波数が均等な白色雑音なので、このフィードバックループを断ち切ります。

DDIM でも同じ `sqrt(alpha) * pred_x0 + sqrt(1 - alpha) * noise_pred` の形で `noise_pred` を使いますが、こちらでは問題になりません。DDIM のこの式は決定論的な常微分方程式 (ODE) ソルバーの 1 ステップであり、`noise_pred` は U-Net が推定した「現在のサンプルに含まれているノイズ」です。これを使って「タイムステップ `t_prev` でサンプルがどう見えるか」を計算する決定論的な写像なので、自己強化フィードバックは生じません。一方 LCM の再ノイズ化は `pred_x0` に新たなノイズを加えて確率的に拡散し直す操作であり、ランダムノイズでなければなりません。

最終ステップの処理については、`t == self.timesteps[-1]` の分岐で再ノイズ化を行わず `pred_x0` をそのまま返すようにしました。

修正後の `step()` は次のようになりました。

```python
def step(self, noise_pred, t, sample, generator=None):
    alpha_t = self.alphas_cumprod[t]
    pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    pred_x0 = pred_x0.clamp(-1.0, 1.0)
    if t == self.timesteps[-1]:
        return pred_x0
    else:
        t_prev = t - self._step_ratio
        alpha_t_prev = self.alphas_cumprod[t_prev]
        noise = torch.randn_like(pred_x0, generator=generator)  # 追加
        return sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise
```

![lcm-2.jpg](lcm-2.jpg)

メッシュ状のノイズは解消されましたが、窓の外の草木がすりガラスのようにぼやけています。

### 修正 2：clamp の削除

残る問題は `pred_x0.clamp(-1.0, 1.0)` でした。画像のピクセル値は `[-1, 1]` の範囲に正規化されますが、VAE エンコード後の latent space の値はこの範囲に収まらないことが多いです。clamp によって範囲外の値が切り捨てられ、細部の情報が失われます（窓の外の草がすりガラスのように不鮮明になる等）。DDIM の `step()` にも同様のクランプはなく、不要と判断して削除しました。

修正後の `step()` が現在の最終実装です。

```python
def step(self, noise_pred, t, sample, generator=None):
    alpha_t = self.alphas_cumprod[t]
    pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    if t == self.timesteps[-1]:
        return pred_x0
    else:
        t_prev = t - self._step_ratio
        alpha_t_prev = self.alphas_cumprod[t_prev]
        noise = torch.randn_like(pred_x0, generator=generator)
        return sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise
```

![lcm-3.jpg](lcm-3.jpg)

なお、当初は論文に基づいて boundary condition scaling（c_skip, c_out）を追加してみたのですが、大きな timestep ではほぼ恒等変換となり効果がありませんでした。

## まとめ：DDIM との違い

DDIM と LCM の `step()` は `pred_x0` を逆算する部分まで同一で、骨格はほぼ同じです。本質的な違いはノイズ項にあります。DDIM では `noise_pred` を使った決定論的な ODE ステップで、各ステップは軌道に沿った小さな移動です。一方 LCM では再ノイズ化にランダムノイズを使い、各ステップで `pred_x0` を改めて予測し直します。consistency distillation により少ないステップでも質の高い予測が得られるよう学習されています。

LCM の再拡散は、予測した `pred_x0` をより低いノイズレベルまで再拡散し、そこから再度予測をやり直すことで精度を上げる仕組みです。これは I2I（Image-to-Image）で入力画像にノイズを加えてからデノイズするのと同じ発想で、完全なランダムノイズからよりも、ある程度構造を持った出発点からの方が精度の高い結果が得られます。この「予測→再拡散→より良い予測」のサイクルにより、2〜4 ステップで十分な品質が得られます。

タイムステップの選択方法（`set_timesteps()`）も異なりますが、これは学習方法に由来する違いであり、`step()` のロジック上の核心は再拡散の違いです。
