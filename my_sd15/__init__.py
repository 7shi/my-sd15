import argparse


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SD 1.5 text-to-image")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="Model ID under weights/ (e.g. genai-archive/anything-v5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("-o", "--output", type=str, default="output.png")
    args = parser.parse_args()

    from my_sd15.loader import load_model

    model = load_model(model_id=args.model)

    image = model.generate(
        prompt=args.prompt,
        seed=args.seed,
        steps=args.steps,
        cfg_scale=args.cfg,
        height=args.size,
        width=args.size,
        show_progress=True,
    )
    image.save(args.output)
    print(f"Saved to {args.output}")
