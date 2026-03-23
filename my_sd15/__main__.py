def main():
    """CLI entry point."""
    import argparse
    from my_sd15 import __version__
    parser = argparse.ArgumentParser(description="SD 1.5 text-to-image",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-p", "--prompt", type=str, required=True,
                        help="Text prompt for image generation")
    parser.add_argument("-n", "--negative", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("-m", "--model", type=str, default="webui/miniSD",
                        help="Model ID under weights/ (e.g. genai-archive/anything-v5)")
    parser.add_argument("-s", "--seed", type=int, action="append", dest="seeds",
                        help="Random seed (can be specified multiple times)")
    parser.add_argument("-S", "--steps", type=int, default=10,
                        help="Number of denoising steps")
    parser.add_argument("-C", "--cfg", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("-W", "--width", type=int, default=256,
                        help="Image width in pixels")
    parser.add_argument("-H", "--height", type=int, default=256,
                        help="Image height in pixels")
    parser.add_argument("-c", "--count", type=int, default=1,
                        help="Number of images to generate")
    parser.add_argument("-o", "--output", type=str, default="output/%s.png",
                        help="Output file path (use %s to include seed)")
    parser.add_argument("--vae", type=str, default=None,
                        help="VAE model ID (e.g. madebyollin/taesd)")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA safetensors file")
    parser.add_argument("--lora-scale", type=float, default=1.0,
                        help="LoRA scaling factor")
    parser.add_argument("--lcm", action="store_true",
                        help="Use LCM scheduler (for LCM LoRA)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save image without displaying")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress display")
    args = parser.parse_args()
    seeds = args.seeds or []
    if len(seeds) > args.count:
        parser.error(f"Too many seeds ({len(seeds)}) for count ({args.count})")

    import os
    from datetime import datetime
    start = datetime.now()

    print("Loading libraries...")
    import torch
    from my_sd15.loader import load_model
    from my_sd15.model import save_image

    scheduler = None
    if args.lcm:
        from my_sd15.scheduler import LCMScheduler
        scheduler = LCMScheduler()

    print("Loading model...")
    model = load_model(model_id=args.model, lora_path=args.lora,
                       lora_scale=args.lora_scale, scheduler=scheduler,
                       vae=args.vae)

    if len(seeds) < args.count:
        seeds += torch.randint(0, 2**30, (args.count - len(seeds),)).tolist()

    def align8(x):
        return (x + 7) // 8 * 8

    w = align8(args.width)
    h = align8(args.height)
    if w != args.width or h != args.height:
        print(f"Size adjusted to {w}x{h}")

    for i, seed in enumerate(seeds, start=1):
        loop_start = datetime.now()
        print(f"Generating image ({i}/{len(seeds)}, seed={seed})...")
        image = model.generate(
            prompt=args.prompt,
            negative_prompt=args.negative,
            seed=seed,
            steps=args.steps,
            cfg_scale=args.cfg,
            height=h,
            width=w,
            show_progress=not args.no_progress,
        )
        if "%s" in args.output:
            output = args.output.replace("%s", f"{seed:010d}")
        elif args.count > 1:
            name, ext = os.path.splitext(args.output)
            output = f"{name}-{seed:010d}{ext}"
        else:
            output = args.output
        save_image(output, image, show=not args.no_show, mkdir=True)
        loop_end = datetime.now()
        print(f"Saved to {output} ({loop_end - loop_start})")

    end = datetime.now()
    print(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()
