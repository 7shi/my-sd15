def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="SD 1.5 text-to-image")
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="Model ID under weights/ (e.g. genai-archive/anything-v5)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("-W", "--width", type=int, default=256)
    parser.add_argument("-H", "--height", type=int, default=256)
    parser.add_argument("-n", "--negative", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("-o", "--output", type=str, default="output.png")
    args = parser.parse_args()

    from datetime import datetime
    start = datetime.now()

    print("Loading libraries...")
    import torch
    from my_sd15.loader import load_model
    from my_sd15.model import save_show_image

    print("Loading model...")
    model = load_model(model_id=args.model)

    seed = args.seed
    if args.seed is None:
        seed = torch.randint(0, 2**30, (1,)).item()
        print(f"Seed set to {seed}")

    def align8(x):
        return (x + 7) // 8 * 8

    w = align8(args.width)
    h = align8(args.height)
    if w != args.width or h != args.height:
        print(f"Size adjusted to {w}x{h}")

    image = model.generate(
        prompt=args.prompt,
        negative_prompt=args.negative,
        seed=seed,
        steps=args.steps,
        cfg_scale=args.cfg,
        height=h,
        width=w,
        show_progress=True,
    )
    print("VAE decoding...")
    save_show_image(args.output, image)
    print(f"Saved to {args.output}")

    end = datetime.now()
    print(f"Elapsed time: {end - start}")
