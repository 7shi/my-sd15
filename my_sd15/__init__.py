import argparse
import os


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

    from my_sd15.loader import DEFAULT_WEIGHTS_DIR
    from my_sd15.pipeline import generate

    if args.model is not None:
        weights_base = os.path.normpath(os.path.join(os.path.dirname(DEFAULT_WEIGHTS_DIR), ".."))
        weights_dir = os.path.join(weights_base, args.model)
    else:
        weights_dir = None

    image = generate(
        prompt=args.prompt,
        seed=args.seed,
        steps=args.steps,
        cfg_scale=args.cfg,
        height=args.size,
        width=args.size,
        weights_dir=weights_dir,
        show_progress=True,
    )
    image.save(args.output)
    print(f"Saved to {args.output}")
