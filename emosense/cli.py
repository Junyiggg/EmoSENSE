from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import Paths, RuntimeConfig, resolve_device
from .mapping import SentimentSemanticMapper, train_mapping_space
from .generation import OmniGenImageGenerator
from .prompting import QwenPromptGenerator, TemplatePromptGenerator
from .rl import EmoSENSEPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EmoSENSE emotional image generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-mapping", help="Train the VAD prototype/projection mapping space")
    train.add_argument("--annotation-dir", required=True, help="Path to filtered EmoSet annotation directory")
    train.add_argument("--output-dir", default="projection")
    train.add_argument("--bge-model", default="BGE")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--device", default=None)
    train.add_argument("--embedding-batch-size", type=int, default=128)
    train.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests")
    train.add_argument("--validation-ratio", type=float, default=0.1)
    train.add_argument("--random-seed", type=int, default=42)

    generate = subparsers.add_parser("generate", help="Generate an emotional image")
    generate.add_argument("--object", required=True, dest="object_text")
    generate.add_argument("--emotion", required=True)
    generate.add_argument("--output-dir", default=None)
    generate.add_argument("--high-episodes", type=int, default=1)
    generate.add_argument("--low-steps", type=int, default=5)
    generate.add_argument("--inference-steps", type=int, default=None)
    generate.add_argument("--qwen-model", default=None, help="Qwen model id or local path")
    generate.add_argument("--omnigen-model", default=None, help="OmniGen model id or local path")
    generate.add_argument("--device", default=None)
    generate.add_argument("--dry-run", action="store_true", help="Build prompts and update fuzzy states without loading OmniGen/CLIP")
    generate.add_argument("--quiet", action="store_true", help="Suppress stage progress logs")

    inspect = subparsers.add_parser("inspect", help="Inspect sentiment-semantic correlation and top-K references")
    inspect.add_argument("--object", required=True, dest="object_text")
    inspect.add_argument("--emotion", required=True)
    inspect.add_argument("--top-k", type=int, default=5)
    inspect.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train-mapping":
        train_mapping_space(
            annotation_dir=args.annotation_dir,
            output_dir=args.output_dir,
            bge_model_path=args.bge_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            embedding_batch_size=args.embedding_batch_size,
            limit=args.limit,
            validation_ratio=args.validation_ratio,
            random_seed=args.random_seed,
        )
        return

    device = resolve_device(args.device)
    mapper = SentimentSemanticMapper(Paths(), device=device)

    if args.command == "inspect":
        result = mapper.evaluate(args.object_text, args.emotion, top_k=args.top_k)
        print(
            json.dumps(
                {
                    "emotion": result.emotion,
                    "object": result.object_text,
                    "correlation": result.correlation,
                    "gamma": result.gamma,
                    "object_projection": result.object_projection,
                    "closest_objects": [asdict(obj) for obj in result.closest_objects],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "generate":
        output_dir = args.output_dir
        if output_dir is None:
            safe_object = "".join(ch if ch.isalnum() else "_" for ch in args.object_text.lower()).strip("_")
            output_dir = str(Path("runs") / f"{args.emotion.lower()}_{safe_object}")
        prompt_generator = (
            TemplatePromptGenerator()
            if args.dry_run
            else QwenPromptGenerator(model_name=args.qwen_model, device=str(device))
        )
        image_generator = None
        cfg = RuntimeConfig()
        if args.inference_steps is not None:
            cfg = RuntimeConfig(num_inference_steps=args.inference_steps)
        if args.omnigen_model is not None:
            image_generator = OmniGenImageGenerator(
                model_name_or_path=args.omnigen_model,
                height=cfg.image_height,
                width=cfg.image_width,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                seed=cfg.seed,
            )
        pipeline = EmoSENSEPipeline(
            device=device,
            mapper=mapper,
            prompt_generator=prompt_generator,
            image_generator=image_generator,
            config=cfg,
        )
        result = pipeline.generate(
            args.object_text,
            args.emotion,
            output_dir=output_dir,
            high_episodes=args.high_episodes,
            low_steps=args.low_steps,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
        print(json.dumps({k: v for k, v in asdict(result).items() if k != "records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None
