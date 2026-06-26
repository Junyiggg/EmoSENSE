from __future__ import annotations

from pathlib import Path

import torch

from .config import RuntimeConfig


class OmniGenImageGenerator:
    """Frozen OmniGen image generator."""

    def __init__(
        self,
        model_name_or_path: str = "Shitao/OmniGen-v1",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
    ):
        cfg = RuntimeConfig()
        self.model_name_or_path = model_name_or_path
        self.height = height or cfg.image_height
        self.width = width or cfg.image_width
        self.num_inference_steps = num_inference_steps or cfg.num_inference_steps
        self.guidance_scale = guidance_scale or cfg.guidance_scale
        self.seed = seed if seed is not None else cfg.seed
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            from OmniGen import OmniGenPipeline

            self._pipe = OmniGenPipeline.from_pretrained(self.model_name_or_path)
            if hasattr(self._pipe, "model"):
                self._pipe.model.eval()
                for param in self._pipe.model.parameters():
                    param.requires_grad_(False)
        return self._pipe

    def generate(self, prompt: str, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cuda_available = torch.cuda.is_available()
        with torch.no_grad():
            images = self.pipe(
                prompt=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
                use_kv_cache=cuda_available,
                offload_kv_cache=cuda_available,
                dtype=torch.bfloat16 if cuda_available else torch.float32,
            )
        images[0].save(output_path)
        return output_path
