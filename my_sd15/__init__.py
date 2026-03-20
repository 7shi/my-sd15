import argparse

from PIL import Image


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SD 1.5 text-to-image")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("-o", "--output", type=str, default="output.png")
    args = parser.parse_args()

    from my_sd15.pipeline import generate

    image = generate(
        prompt=args.prompt,
        seed=args.seed,
        steps=args.steps,
        cfg_scale=args.cfg,
        height=args.size,
        width=args.size,
    )
    Image.fromarray(image).save(args.output)
    print(f"Saved to {args.output}")
