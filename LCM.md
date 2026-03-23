# LCMScheduler の実装と修正の経緯

この文書では、`my_sd15/scheduler.py` に実装された `LCMScheduler` の目的、仕様、そして初期実装に含まれていた 3 つの問題がどのように発見・修正されたかを記録します。

## LCMScheduler の目的

通常の Stable Diffusion 1.5 では、DDIM スケジューラーを使って 20〜50 ステップかけてノイズを除去します。各ステップで U-Net を呼び出すため、ステップ数がそのまま推論時間に直結します。さらに CFG (Classifier-Free Guidance) のために条件あり・なしの 2 回の U-Net 呼び出しが必要なので、実質的な計算量はステップ数の 2 倍です。

LCM (Latent Consistency Model) は、この推論ステップ数を 2〜4 回にまで削減する手法です。専用の LoRA を適用した U-Net と組み合わせて使います。LCM LoRA は学習時に CFG の効果を LoRA の重みに焼き込んでいるため、推論時は `cfg_scale=1.0` として U-Net を 1 回だけ呼び出せば済みます。ステップ数の削減と CFG 不要化の両方により、大幅な高速化が得られます。

## 仕様（SPEC.md §7b / 論文 Algorithm 2）

論文: [Luo et al. "Latent Consistency Models" (2023)](https://arxiv.org/abs/2310.04378)

LCMScheduler の `step()` 関数は、論文の Algorithm 2 (Multistep Latent Consistency Sampling) に基づいています。各ステップの処理は以下のとおりです。

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

最終ステップ（`t_prev < 0`）では再ノイズ化は行わず、`pred_x0` をそのまま返します。

ここで重要な点が 3 つあります。再ノイズ化には `noise_pred` ではなくランダムノイズを使うこと、最終ステップでは `pred_x0` を直接返すこと、そして `pred_x0` に対して値域のクランプを行わないことです。これらはすべて DDIM との違いであり、初期実装ではいずれも正しく実装されていませんでした。

## 初期実装（ae87882）

LCMScheduler は LoRA サポートの一部として `ae87882` で初めて実装されました。初期の `step()` 関数は次のようなものでした。

```python
def step(self, noise_pred, t, sample):
    alpha_t = self.alphas_cumprod[t]
    t_prev = self._prev_timestep[t]
    alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

    pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    pred_x0 = pred_x0.clamp(-1.0, 1.0)
    prev_sample = sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise_pred
    return prev_sample
```

この実装は DDIM の `step()` をほぼそのまま流用したもので、3 つの問題を含んでいました。

第一に、再ノイズ化に U-Net の出力 `noise_pred` をそのまま使っています。第二に、`pred_x0` に `clamp(-1, 1)` を適用しています。第三に、最終ステップの分岐がなく、`alpha_t_prev = 1.0` を代入することで数式的にノイズ項がゼロになることに依存しています。

## 修正 1：再ノイズ化とステップ分岐（2840190）

最初の修正は 2 つの問題を同時に解決しました。

再ノイズ化について、`noise_pred` を `torch.randn_like()` に置き換えました。`noise_pred` は U-Net の出力であり、画像の構造に対応した空間的パターンを含んでいます。このパターンを含んだノイズで再ノイズ化すると、次のステップの U-Net がまた似たパターンを予測し、それがさらに次のステップに持ち込まれるという自己強化フィードバックが発生します。結果として、メッシュ状の artifact が画像全体に蓄積されます。ランダムノイズは全周波数が均等な白色雑音なので、このフィードバックループを断ち切ります。

最終ステップの処理については、`t_prev >= 0` の分岐を追加し、最終ステップでは `pred_x0` をそのまま返すようにしました。初期実装では `alpha_t_prev = 1.0` とすることで `sqrt(1 - 1.0) * noise = 0` となりノイズ項が消えるという暗黙の処理に頼っていましたが、再ノイズ化をランダムノイズに変更したことで、明示的な分岐が必須になりました。最終ステップでランダムノイズを加えてしまうと、出力画像にノイズが残ります。

修正後の `step()` は次のようになりました。

```python
def step(self, noise_pred, t, sample, generator=None):
    alpha_t = self.alphas_cumprod[t]
    t_prev = self._prev_timestep[t]

    pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    pred_x0 = pred_x0.clamp(-1.0, 1.0)

    if t_prev >= 0:
        alpha_t_prev = self.alphas_cumprod[t_prev]
        noise = torch.randn_like(pred_x0, generator=generator)
        return sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise
    else:
        return pred_x0
```

## 修正 2：clamp の削除（54cd1d0）

残る問題は `pred_x0.clamp(-1.0, 1.0)` でした。画像のピクセル値は `[-1, 1]` の範囲に正規化されますが、VAE エンコード後の latent space の値はこの範囲を日常的に超えます。`pred_x0` は latent space 上の値なので、`[-1, 1]` でクランプすると、範囲外の値が切り捨てられて細部の情報が失われます（窓の外の草がすりガラスのように不鮮明になる等）。DDIM の `step()` にも同様のクランプはなく、不要と判断して削除しました。

修正後の `step()` が現在の最終実装です。

```python
def step(self, noise_pred, t, sample, generator=None):
    alpha_t = self.alphas_cumprod[t]
    t_prev = self._prev_timestep[t]

    pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)

    if t_prev >= 0:
        alpha_t_prev = self.alphas_cumprod[t_prev]
        noise = torch.randn_like(pred_x0, generator=generator)
        return sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise
    else:
        return pred_x0
```

なお、当初は論文に基づいて boundary condition scaling（c_skip, c_out）を追加してみたのですが、大きな timestep では恒等変換となり効果がありませんでした。

## 補足：DDIM では問題にならなかった理由

初期実装の 3 つの問題はいずれも、DDIM のコードを LCM に流用したことに起因しています。DDIM では `noise_pred` による再ノイズ化が仕様どおりの正しい動作であり、最終ステップの分岐も不要（`alpha_t_prev = 1.0` で自然にノイズ項が消える）、そして `pred_x0` のクランプも行っていません。つまり DDIM 自体は正しく実装されていましたが、LCM は同じ数式の形をしていながら再ノイズ化の意味が異なるため、そのまま流用すると問題が生じたということです。
