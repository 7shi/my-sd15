ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | **12** | [13](13_architecture.md)

---

# LoRA: 低ランク適応による重みの修正

これまでの章では SD 1.5 の推論パイプラインをスクラッチ実装し、テキストから画像を生成できるようになりました。この章では、学習済みモデルの振る舞いを少量のパラメータで変更する **LoRA (Low-Rank Adaptation)** を扱います。

## 1. LoRA の動機

SD 1.5 の U-Net には約 8.6 億個のパラメータがあります。特定のスタイルやキャラクターを学習させるためにこれらすべてを再学習 (fine-tuning) すると、計算コストが高く、保存も大変です。

LoRA は「重み行列全体を動かす代わりに、少数の方向だけを動かす」というアイデアで、元の重みを凍結したまま、わずかな追加パラメータで振る舞いを変えます。ソフトウェア開発に例えると、ソースコード全体を書き換える代わりに**パッチ（差分）**だけを配布するようなものです。LoRA はさらに、その差分自体を低ランク分解で圧縮しています。

## 2. 低ランク分解

元の線形変換 $y = Wx$ に対し、LoRA は差分 $\Delta W$ を**低ランク行列の積**として表現します。

$$y = Wx + \Delta Wx = Wx + BAx$$

ここで $W$ は $(m \times n)$ の元の重み行列、$A$ は $(r \times n)$、$B$ は $(m \times r)$ です。$r$ はランクと呼ばれ、通常 4〜128 の小さな値です。

### ランク r=1 の場合

$A$ と $B$ はそれぞれベクトルになり、$BA$ は外積（直積）です。

$$\Delta W = \mathbf{b} \mathbf{a}^T \quad (m \times n \text{ の行列})$$

2 本のベクトル（$m + n$ 個のパラメータ）だけで $m \times n$ の行列を表現しますが、変化の「方向」は 1 つだけに限られます。

### ランク r>1 の場合

$r$ 個のベクトル対による外積の重ね合わせになります。

$$\Delta W = \sum_{i=1}^{r} \mathbf{b}_i \mathbf{a}_i^T$$

パラメータ数は $r \times (m + n)$ で、$r$ が小さければ元の $m \times n$ より大幅に少なくなります。

### 具体例

[Cross-Attention](08_cross_attention.md) の `to_k.weight` $(320 \times 768)$ に $r=4$ の LoRA を適用する場合:

| | パラメータ数 | 比率 |
|---|---|---|
| 元の重み行列 | $320 \times 768 = 245{,}760$ | 100% |
| LoRA ($A$ + $B$) | $4 \times (320 + 768) = 4{,}352$ | 1.8% |

わずか 1.8% のパラメータで重み行列の修正を表現できます。

## 3. スケーリング

実際の LoRA には**スケーリング係数**があります。

$$W' = W + \frac{\alpha}{r} \cdot BA$$

$\alpha$ は LoRA の学習時に設定される定数で、推論時にさらにユーザー指定の `scale` を掛けることもあります。$\alpha / r$ で正規化することで、ランク $r$ を変えても出力のスケールが大きく変わらないようにしています。

## 4. 重みのマージ

LoRA は推論時に元の重みと事前にマージできます。

$$W' = W + \frac{\alpha}{r} \cdot BA$$

マージ後は追加のコストなく通常どおり推論できます。これが LoRA の実用上の大きな利点です。本実装でもこの方式を採用しています。

```python
# Linear 層の場合: up (m, r) @ down (r, n) -> (m, n)
delta = up @ down
state[key] += scale * (alpha / r) * delta
```

## 5. SD 1.5 での適用対象

LoRA は原理上どの線形層にも適用できますが、SD 1.5 では主に **Text Encoder (CLIP)** と **U-Net** が対象になります。

### Text Encoder (CLIP) LoRA

テキストエンコーダーに LoRA を適用すると、テキストの解釈自体を変えることができます。たとえばキャラクター LoRA では、特定のトークンに新しい概念を紐づけるために、U-Net と Text Encoder の両方に LoRA を適用することがあります。LoRA ファイル内では U-Net 向けのキーに `lora_unet_`、Text Encoder 向けのキーに `lora_te_` というプレフィックスが付きます。

### U-Net LoRA

U-Net は画像生成の中核であり、LoRA の最も一般的な適用先です。スタイル/キャラクター LoRA では [Cross-Attention](08_cross_attention.md) の線形層（`to_q`, `to_k`, `to_v`, `to_out`）だけを変更するのが一般的です。Cross-Attention はテキストと画像の対応関係を決める層なので、ここを変えることで「特定の単語に特定の画像特徴を対応させる」ことができます。

### LCM LoRA: U-Net 全層への適用

本章で扱う LCM (Latent Consistency Model) LoRA は U-Net LoRA の一例です。多ステップの拡散モデルから少ステップで同等の出力を得られるモデルを**蒸留 (distillation)** で学習し、その差分を LoRA として抽出したものです。通常 50 ステップ必要なデノイジングを **2〜4 ステップ**に短縮します。これはテキストと画像の対応関係だけでなく、デノイジングプロセス自体を変える必要があるため、U-Net の**ほぼ全層**に LoRA を適用します。

```
適用対象の内訳（LCM LoRA の場合、U-Net の全 278 箇所）:

  ResBlock / Conv                 86
  FFN (GEGLU)                     32
  proj_in / proj_out              32
  Self-Attention  to_q/k/v/out    64
  Cross-Attention to_q/k/v/out    64
```

ランクは全て $r=64$、$\alpha=8$（スケール $\alpha/r = 0.125$）です。

### Conv 層への LoRA

[04 章](04_conv2d.md)で Conv2D を im2col + 行列積として実装したことを思い出してください。Conv2D も結局は行列積なので、同じ手法が使えます。

LoRA の重みの shape は通常の行列とは異なります。Linear 層では 2 次元ですが、Conv 層では 4 次元になります。

```
Linear LoRA の例（in=320, out=320, r=64）:
  down: (64, 320)      ← (r, n)
  up:   (320, 64)      ← (m, r)

Conv LoRA の例（3×3 カーネル、C_in=320, C_out=320, r=64）:
  down: (64, 320, 3, 3)    ← 入力チャネルとカーネルを保持
  up:   (320, 64, 1, 1)    ← 出力チャネル方向のみ
```

`up` が $1 \times 1$ カーネルなのは、出力チャネル方向の混合だけを LoRA が担い、空間方向のパターン（$3 \times 3$）は `down` 側に任せているためです。実装では reshape して行列積を取り、元の shape に戻します。

```python
# Conv 層の場合: down (r, C_in, kH, kW), up (C_out, r, 1, 1)
delta = (up.squeeze(-1).squeeze(-1) @ down.reshape(r, -1))  # (C_out, C_in*kH*kW)
delta = delta.reshape(weight.shape)                          # (C_out, C_in, kH, kW)
state[key] += scale * (alpha / r) * delta
```

## 6. キー名のマッピング

LoRA ファイル内のキー名はモデルの重み名と形式が異なります。

```
LoRA:  lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
モデル: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight
```

`.` が `_` に置換され、`lora_unet_` プレフィックスが付加されています。逆変換は、モデルの全キーからアンダースコア版を生成して逆引き辞書を作ることで実現します。

```python
def _build_key_map(state):
    key_map = {}
    for model_key in state:
        if not model_key.endswith(".weight"):
            continue
        base = model_key.removesuffix(".weight").replace(".", "_")
        key_map[base] = model_key
    return key_map
```

## 7. LCM スケジューラー

LCM LoRA を使うには、DDIM とは異なるスケジューラーが必要です。

### タイムステップの選択

DDIM は 1000 ステップを等間隔に間引きます（10 ステップなら `[900, 800, ..., 0]`）。LCM は学習時の `original_steps`（通常 50）に基づく格子点からタイムステップを選びます。これは LCM が consistency distillation で `original_steps` の格子点上で蒸留されているためです。

```python
c = 1000 // original_steps  # 20
# 格子点: [19, 39, 59, ..., 999]
lcm_timesteps = torch.arange(1, original_steps + 1) * c - 1
# 逆順から等間隔に num_steps 個を抽出
skip = len(lcm_timesteps) // num_steps
timesteps = lcm_timesteps.flip(0)[::skip][:num_steps]
# 2 ステップなら [999, 499]、4 ステップなら [999, 759, 519, 279]
```

### 前のタイムステップの決定

DDIM も LCM も、前のタイムステップは `t_prev = t - step_ratio` で求めます。`step_ratio` の計算方法が異なるだけです。

```python
# DDIM: step_ratio = 1000 / num_steps（例: 100.0）
# LCM:  step_ratio = skip * c（例: 500）
t_prev = t - self._step_ratio
```

LCM の `step_ratio` は `skip = original_steps // num_steps` と `c = 1000 // original_steps` の積で、すべて整数除算のため丸め処理は不要です。

### ステップ関数（Algorithm 2）

LCM のステップ関数は DDIM と似ていますが、**ノイズ項の意味が異なります**。DDIM の式は決定論的な常微分方程式 (ODE) ソルバーの一ステップであり、`noise_pred` を使って「タイムステップ `t_prev` でサンプルがどう見えるか」を計算する決定論的な写像です。一方 LCM は `pred_x0` に**ランダムノイズ**を加えて確率的に再拡散する操作です（論文 Algorithm 2: Multistep Latent Consistency Sampling）。

```python
def step(self, noise_pred, t, sample, generator=None):
    alpha_t = self.alphas_cumprod[t]
    pred_x0 = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    if t == self.timesteps[-1]:
        # 最終ステップ: pred_x0 をそのまま返す
        return pred_x0
    else:
        # 中間ステップ: ランダムノイズで再ノイズ化
        t_prev = t - self._step_ratio
        alpha_t_prev = self.alphas_cumprod[t_prev]
        noise = torch.randn_like(pred_x0, generator=generator)
        return sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise
```

DDIM との違いをまとめます。

| | DDIM | LCM |
|---|---|---|
| 再ノイズ化 | `noise_pred`（予測ノイズ、決定的） | `randn`（ランダムノイズ） |
| 最終ステップ | 他と同じ式（`alpha_t_prev=1` で自然に消える） | `pred_x0` を直接返す |
| `pred_x0` のクランプ | なし | なし |

再ノイズ化に `noise_pred` を使うと、U-Net の出力に含まれる構造的なパターンが次のステップに持ち込まれ、メッシュ状のノイズが蓄積されます。ランダムノイズは全周波数が均等な白色雑音なので、この自己強化フィードバックを断ち切ります。

なお、LCM 論文では boundary condition scaling（$c_{skip}$, $c_{out}$）も定義されていますが、LCM LoRA が使う大きな timestep では恒等変換となるため、本実装では省略しています（詳細は [lcm-scheduler/README.md](../lcm-scheduler/README.md) 参照）。

### CFG の扱い

通常の推論では CFG (Classifier-Free Guidance) のために U-Net を**条件あり・なしの 2 回**呼び出します（[11 章](11_pipeline.md)参照）。LCM LoRA は学習時に CFG の効果を LoRA 内に取り込んでいるため、推論時は `cfg_scale=1.0` とします。[11 章](11_pipeline.md)で説明したとおり、`cfg_scale=1.0` では CFG の式が $\epsilon_{cond}$ そのものになり、無条件側の U-Net 呼び出しが不要になります。Negative Prompt も効きません。ステップあたりの U-Net 呼び出しが 2 回から 1 回に半減し、ステップ数の削減と合わせて大幅な高速化が得られます。

## 実験：LCM LoRA の適用

LCM LoRA ファイルの構造や適用対象の内訳など、本文中の数値を確認するためのスクリプトです。

**実行方法**: ([12_lora.py](12_lora.py))

```bash
uv run docs/12_lora.py
```

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | **12** | [13](13_architecture.md)
