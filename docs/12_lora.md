ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | **12** | [13](13_architecture.md)

---

# LoRA: 低ランク適応による重みの修正

これまでの章では SD 1.5 の推論パイプラインをスクラッチ実装し、テキストから画像を生成できるようになりました。この章では、学習済みモデルの振る舞いを少量のパラメータで変更する **LoRA (Low-Rank Adaptation)** を扱います。

> この章は推論パイプラインの一区間ではなく、パイプライン全体に対する応用を扱う番外編です（03〜11 章にあるパイプライン位置リストはありません）。

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

小さな例で確かめます。$\mathbf{b} = (1, 2)$、$\mathbf{a} = (1, 0, 3)$ とすると、5 個の数値から 2×3 = 6 要素の行列ができます。

```
            a = [1  0  3]
b = [1]         [1  0  3]     ← a の 1 倍
    [2]         [2  0  6]     ← a の 2 倍
```

どの行も $\mathbf{a}$ の定数倍になっています。行列としては 6 要素ありますが、中身は「$\mathbf{a}$ という 1 種類のパターン」と「それを各行に何倍で配るか」だけで決まっており、自由に 6 個の値を選べるわけではありません。これが「ランク 1」の意味です。

この行列を入力 $x$ に掛けると、$\Delta W x = \mathbf{b} (\mathbf{a} \cdot x)$ となります。入力から $\mathbf{a}$ 方向の成分だけを内積で読み取り、その大きさに応じて出力に $\mathbf{b}$ 方向の変化を加える、という働きです。2 本のベクトル（$m + n$ 個のパラメータ）だけで $m \times n$ の行列を表現できる代わりに、「読み取る方向」と「書き込む方向」は 1 組だけに限られます。

### ランク r>1 の場合

$r$ 個のベクトル対による外積の重ね合わせになります。

$$\Delta W = \sum_{i=1}^{r} \mathbf{b}_i \mathbf{a}_i^T$$

「読み取る方向と書き込む方向の組」を $r$ 組持てるようになります。パラメータ数は $r \times (m + n)$ で、$r$ が小さければ元の $m \times n$ より大幅に少なくなります。$r$ を大きくするほど表現できる修正は豊かになりますが、その分ファイルサイズも増えます。

少数の方向だけで足りるのは、追加学習で必要になる修正が元の重みに比べてずっと単純だからです。モデルはすでに画像の描き方を一通り身に付けており、特定の画風やキャラクターを覚えさせる程度の修正なら、少数の方向を動かすだけで表現できることが経験的に知られています。

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

$\alpha$ は LoRA の学習時に設定される定数で、推論時にさらにユーザー指定の `scale` を掛けることもあります。$\alpha / r$ で正規化することで、ランク $r$ を変えても出力のスケールが大きく変わらないようにしています。`scale` は LoRA の効き具合を調整するツマミで、0 なら元のモデルのまま、1 なら LoRA をそのまま適用します。

## 4. 重みのマージ

上の式の $W'$ は、$W$ と同じ形状の 1 つの行列です。そこで、推論を始める前に $W'$ を計算して元の重みを置き換えてしまうことができます。これを**マージ**と呼びます。

マージ後のモデルは、重みの値が変わっただけで構造は元のままです。推論コードは一切変更する必要がなく、追加の計算コストもかかりません。これが LoRA の実用上の大きな利点です。本実装でもこの方式を採用しており、これまでの章で実装した U-Net や CLIP がそのまま使えます。

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

本章で扱う LCM (Latent Consistency Model) LoRA は U-Net LoRA の一例です。多ステップの拡散モデルから少ステップで同等の出力を得られるモデルを**蒸留 (distillation)** で学習し、その差分を LoRA として抽出したものです。DDIM では通常 10〜30 ステップかかるデノイジングを **2〜4 ステップ**に短縮します。

通常の U-Net は「少しずつノイズを除去する」前提で学習されているため、ノイズだらけの段階での完成画像の見積もりは粗いものでした（👉[09](09_ddim.md)）。LCM は、元のモデルが何ステップもかけてたどり着く結果を、**どの段階からでも一気に言い当てる**ように学習し直したものです。

これはテキストと画像の対応関係だけでなく、デノイジングプロセス自体を変える必要があるため、U-Net の**ほぼ全層**に LoRA を適用します。

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

👉[04](04_conv2d.md) で Conv2D を im2col + 行列積として実装したことを思い出してください。カーネル `(C_out, C_in, kH, kW)` を `(C_out, C_in×kH×kW)` の行列に展開して計算しました。Conv2D も結局は行列積なので、この展開した行列に対して同じ低ランク分解が使えます。

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

全体像を先に示します。LCM LoRA を適用した U-Net は完成画像を一気に見積もれるので、1 ステップ目の見積もりがすでにかなり良いものになります。ただし 1 回では細部が甘いため、次のように「見積もり → 少しノイズを戻す → 見積もり直す」を繰り返して仕上げます。

```
純粋なノイズ
  ↓ U-Net: 完成画像を見積もる
見積もり 1（大まかには合っているが細部が甘い）
  ↓ 新しいランダムノイズを混ぜて、途中のタイムステップの状態に戻す
ノイズ混じりの見積もり 1
  ↓ U-Net: 完成画像を見積もり直す
見積もり 2（細部が改善）= 2 ステップの場合はこれが出力
```

描いた絵をいったん霞ませて、その上から描き直すようなものです。DDIM が階段を一段ずつ降りていくのに対し、LCM は毎回ゴールまで跳んでから、少し戻ってやり直します。以下、DDIM との違いを順に見ていきます。

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
| 同じ入力に対する結果 | 常に同じ（決定的） | 乱数に依存（確率的） |

再ノイズ化に `noise_pred` を使うと、U-Net の出力に含まれる構造的なパターンが次のステップに持ち込まれ、メッシュ状のノイズが蓄積されます。ランダムノイズは全周波数が均等な白色雑音なので、この自己強化フィードバックを断ち切ります。

なお、LCM 論文では boundary condition scaling（$c_{skip}$, $c_{out}$）も定義されていますが、LCM LoRA が使う大きな timestep では恒等変換となるため、本実装では省略しています（詳細は [lcm-scheduler/README.md](../lcm-scheduler/README.md) 参照）。

### CFG の扱い

通常の推論では CFG (Classifier-Free Guidance) のために U-Net を**条件あり・なしの 2 回**呼び出します（👉[11](11_pipeline.md)）。LCM LoRA は学習時に CFG の効果を LoRA 内に取り込んでいるため、推論時は `cfg_scale=1.0` とします。👉[11](11_pipeline.md) で説明したとおり、`cfg_scale=1.0` では CFG の式が $\epsilon_{cond}$ そのものになり、無条件側の U-Net 呼び出しが不要になります。Negative Prompt も効きません。

ステップあたりの U-Net 呼び出しが 2 回から 1 回に半減し、ステップ数の削減と合わせて大幅な高速化が得られます。

| | U-Net 呼び出し回数 |
|---|---|
| DDIM 10 ステップ + CFG | 10 × 2 = 20 回 |
| LCM 2 ステップ（CFG なし） | 2 × 1 = 2 回 |

## 実験：LCM LoRA の適用

LCM LoRA ファイルの構造や適用対象の内訳など、本文中の数値を確認するためのスクリプトです。実行結果は本文中で引用しています。

**実行方法**: ([12_lora.py](12_lora.py))

```bash
uv run docs/12_lora.py
```

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | **12** | [13](13_architecture.md)
