ページ：**01** | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_architecture.md)

---

# まず動かしてみよう

技術的な詳細に入る前に、SD 1.5 が実際にどう動くのかを体験しましょう。この章ではセットアップから画像生成まで、手を動かしながら SD 1.5 の振る舞いを観察します。

## セットアップ

依存関係をインストールします。

```bash
uv sync
```

SD 1.5 の重みファイルをダウンロードします。

```bash
make download
```

これにより、`weights/` ディレクトリに Stable Diffusion 1.5 と Anything V5 の 2 つのモデルがダウンロードされます。

## 画像生成

プロンプト（指示文）を与えると、SD 1.5 がそれに沿った画像を生成します。

```bash
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 -o output.png
```

GPT-2 が「次に来る単語を予測する」ことでテキストを生成したのに対し、SD 1.5 は「ノイズを少しずつ除去する」ことで画像を生成します。ランダムノイズから出発して、10 ステップのデノイジングを経て最終的な画像が得られます。

## パラメータの効果

いくつかのオプションで生成の振る舞いを調整できます。

### ステップ数 (`--steps`)

デノイジングの反復回数を指定します。多いほど品質が上がりますが、生成に時間がかかります。

```bash
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 --steps 5 -o steps5.png
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 --steps 10 -o steps10.png
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 --steps 20 -o steps20.png
```

ステップ数が少ないとノイズが残り、多すぎると過度にシャープになる傾向があります。10 ステップは速度と品質のバランスが良い設定です。

### CFG スケール (`--cfg`)

Classifier-Free Guidance（CFG）スケールは、テキスト条件にどの程度忠実に従うかを制御します。

```bash
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 --cfg 1.0 -o cfg1.png
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 --cfg 7.5 -o cfg7.png
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 --cfg 15.0 -o cfg15.png
```

CFG 1.0 ではプロンプトの影響が弱くぼんやりした画像、7.5 はバランスの良いデフォルト、15.0 ではプロンプトに強く従いますがコントラストが高くなりすぎることがあります。GPT-2 の temperature と似た役割ですが、方向が逆です（高いほどプロンプトに忠実）。

### シード値 (`--seed`)

乱数のシード値を指定します。同じシードなら同じ画像が再現されます。

```bash
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 42 -o seed42.png
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 -o seed123.png
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 456 -o seed456.png
```

GPT-2 でもシードを固定するとサンプリング結果が再現されましたが、SD 1.5 ではシードが初期ノイズを決定するため、画像全体の構図が大きく変わります。

### Negative Prompt (`-n`)

生成したくない要素を指定します。

```bash
uv run my-sd15 -p "a cat sitting on a windowsill" --seed 123 -n "blurry, low quality" -o neg.png
```

Negative Prompt は CFG の仕組みを利用して、指定した要素を画像から「遠ざける」方向に働きます。詳しくは [11 章](11_pipeline.md)で解説します。

## 別モデルでの生成

`-m` オプションでモデルを切り替えられます。Anything V5 はアニメ風の画像を生成するモデルです。

```bash
uv run my-sd15 -m genai-archive/anything-v5 -p "a cat sitting on a windowsill" --seed 123 -o any5.png
```

SD 1.5 と同じアーキテクチャの重みを差し替えるだけで、画風が大きく変わります。これはモデルの構造ではなく、学習データの違いによるものです。

## まとめ

この章では SD 1.5 をブラックボックスとして使い、プロンプト・ステップ数・CFG スケール・シード・Negative Prompt といったパラメータが生成結果にどう影響するかを体験しました。次章からは、このパイプラインの内部でどのような計算が行われているかを順に解きほぐしていきます。

---

ページ：**01** | [02](02_overview.md) | [03](03_clip.md) | [04](04_conv2d.md) | [05](05_groupnorm.md) | [06](06_resblock.md) | [07](07_unet.md) | [08](08_cross_attention.md) | [09](09_ddim.md) | [10](10_vae.md) | [11](11_pipeline.md) | [12](12_architecture.md)
