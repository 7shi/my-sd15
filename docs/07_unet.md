ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | **07** | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_architecture.md)

---

# U-Net の全体構造

U-Net は SD 1.5 の中核をなすモデルで、パラメータの約 85% (~860M) を占めます。テキスト条件とタイムステップを受け取り、潜在表現に含まれるノイズを予測します。

1. テキスト
   - [CLIP Text Encoder](03_clip.md)
2. 条件ベクトル
3. ランダムノイズ
   - **U-Net** ← この章 × 10 step
     - [Conv2D](04_conv2d.md)
     - [GroupNorm](05_groupnorm.md)
     - [ResBlock](06_resblock.md)
     - [Cross-Attention](08_cross_attention.md)
4. 潜在表現
   - [VAE Decoder](10_vae.md)
5. 画像

## 1. U-Net とは

U-Net はもともと医療画像のセグメンテーション用に提案されたアーキテクチャで、エンコーダ（ダウンサンプリング）とデコーダ（アップサンプリング）を**スキップ接続**で結んだ U 字型の構造です。

```
入力 (4, 32, 32)
  ↓ conv_in
(320, 32, 32)
  ↓ Down block 0
(320, 32, 32) ──────────────────────┐ skip
  ↓ downsample (stride=2 conv)     │
(320, 16, 16)                       │
  ↓ Down block 1                    │
(640, 16, 16) ──────────────────┐   │ skip
  ↓ downsample                  │   │
(640, 8, 8)                     │   │
  ↓ Down block 2                │   │
(1280, 8, 8) ──────────────┐   │   │ skip
  ↓ downsample              │   │   │
(1280, 4, 4)                │   │   │
  ↓ Down block 3            │   │   │
(1280, 4, 4) ──────────┐   │   │   │ skip
                        │   │   │   │
  ↓ Mid block           │   │   │   │
(1280, 4, 4)            │   │   │   │
                        │   │   │   │
  ↓ Up block 0 ←───────┘   │   │   │ cat
(1280, 4, 4)                │   │   │
  ↓ upsample               │   │   │
  ↓ Up block 1 ←───────────┘   │   │ cat
(1280, 8, 8)                    │   │
  ↓ upsample                   │   │
  ↓ Up block 2 ←───────────────┘   │ cat
(640, 16, 16)                       │
  ↓ upsample                       │
  ↓ Up block 3 ←───────────────────┘ cat
(320, 32, 32)
  ↓ conv_out
出力 (4, 32, 32)
```

## 2. チャネル数の変化

U-Net の各段階でチャネル数と空間サイズが変化します。

| 段階 | チャネル | 空間サイズ | ブロック構成 |
|---|---|---|---|
| conv_in | 4 → 320 | 32×32 | 3×3 conv |
| Down 0 | 320 | 32×32 | ResBlock×2 + Attention×2 |
| Down 1 | 320 → 640 | 16×16 | ResBlock×2 + Attention×2 |
| Down 2 | 640 → 1280 | 8×8 | ResBlock×2 + Attention×2 |
| Down 3 | 1280 | 4×4 | ResBlock×2（Attention なし） |
| Mid | 1280 | 4×4 | ResBlock + Attention + ResBlock |
| Up 0 | 1280 | 4×4 | ResBlock×3 |
| Up 1 | 1280 | 8×8 | ResBlock×3 + Attention×3 |
| Up 2 | → 640 | 16×16 | ResBlock×3 + Attention×3 |
| Up 3 | → 320 | 32×32 | ResBlock×3 + Attention×3 |
| conv_out | 320 → 4 | 32×32 | GroupNorm + SiLU + 3×3 conv |

入力と出力はどちらも `(4, 32, 32)` です。入力は「ノイズを含む潜在表現」、出力は「予測されたノイズ」で、同じ形状のテンソルです。

## 3. スキップ接続

U-Net の最大の特徴はスキップ接続です。ダウンサンプリング側の各段階の出力を保存 (`skips`) し、アップサンプリング側で `torch.cat` により**チャネル方向に結合**します。

```python
skips = [x]  # conv_in の出力

# Down blocks で skips に追加
for ...:
    x = resblock(x, temb)
    x = attention(x, context)
    skips.append(x)

# Up blocks で skips から取り出して結合
for ...:
    skip = skips.pop()          # 対応するダウン側の出力
    x = torch.cat([x, skip], dim=0)  # チャネル方向に結合
    x = resblock(x, temb)      # 結合後のチャネル数から処理
```

GPT-2 の残差接続（👉[GPT-2 09](https://github.com/7shi/my-gpt2/tree/main/docs/09_residual.md)）が `x + h` で加算だったのに対し、U-Net のスキップ接続は `torch.cat([x, skip])` で結合です。これにより、エンコーダの詳細な空間情報をデコーダに直接伝えることができます。

結合によりチャネル数が倍増するため（例: 1280 + 1280 = 2560）、Up ブロックの最初の ResBlock は結合後のチャネル数を入力として受け取ります。

## 4. ダウンサンプリングとアップサンプリング

### ダウンサンプリング

`stride=2` の 3×3 畳み込みで空間サイズを半分にします（👉[04](04_conv2d.md)）。

```python
x = conv2d(x, weight, bias, stride=2, padding=1)
# (C, 32, 32) → (C, 16, 16)
```

### アップサンプリング

最近傍補間で空間サイズを倍にした後、3×3 畳み込みで平滑化します。

```python
x = upsample_nearest_2d(x, 2)  # (C, 8, 8) → (C, 16, 16)
x = conv2d(x, weight, bias, padding=1)
```

`upsample_nearest_2d` は各ピクセルを 2×2 に複製するだけの単純な操作です。

## 5. 各ブロックの構成要素

各 Down/Up ブロックは、以下のコンポーネントの組み合わせです。

- **UNetResBlock** — 特徴量の変換 + タイムステップ埋め込み注入（👉[06](06_resblock.md)）
- **SpatialTransformer** — Self-Attention + Cross-Attention + GEGLU FFN（👉[08](08_cross_attention.md)）

Down block 3 と Up block 0 だけは SpatialTransformer を持たず、ResBlock のみで構成されています。最も解像度が低い（4×4）段階では、空間的な Attention の効果が限定的なためです。

## 実験：U-Net の構造を追跡

U-Net の各段階でテンソル形状がどう変化するかを追跡します。実行結果は以下のとおりです。

```
入力:      (4, 32, 32)
context:   (77, 768)
temb:      (1280,)

conv_in:   (320, 32, 32)
Down 0:    (320, 16, 16) (skips: 4)
Down 1:    (640, 8, 8) (skips: 7)
Down 2:    (1280, 4, 4) (skips: 10)
Down 3:    (1280, 4, 4) (skips: 12)

skips の形状:
  [0]: (320, 32, 32)
  [1]: (320, 32, 32)
  [2]: (320, 32, 32)
  [3]: (320, 16, 16)
  [4]: (640, 16, 16)
  [5]: (640, 16, 16)
  [6]: (640, 8, 8)
  [7]: (1280, 8, 8)
  [8]: (1280, 8, 8)
  [9]: (1280, 4, 4)
  [10]: (1280, 4, 4)
  [11]: (1280, 4, 4)

Mid:       (1280, 4, 4)

Up blocks のスキップ結合:
  Up 0: x (1280, 4, 4) + skip (1280, 4, 4) → cat 2560 ch

U-Net 総パラメータ: 859,520,964 (860M)
```

空間サイズが 32→16→8→4 と半減していく様子、チャネル数が 320→640→1280 と増加していく様子が分かります。skips には 12 個のテンソルが保存され、Up blocks で逆順に取り出されてチャネル方向に結合されます。結合により Up block 0 の最初の ResBlock は 2560 チャネル (1280+1280) を入力として受け取ります。

**実行方法**: ([07_unet.py](07_unet.py))

```bash
uv run docs/07_unet.py
```

---

ページ：[01](01_quickstart.md) | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | **07** | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_architecture.md)
