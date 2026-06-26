from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


# Paper order from Table II. Do not sort directory names when assigning VAD
# vertices; alphabetical order maps several emotions to the wrong cube corners.
EMOTION_ORDER = (
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
)

VAD_COORDINATES = {
    "amusement": (1.0, 0.0, 1.0),
    "awe": (1.0, 1.0, 0.0),
    "contentment": (1.0, 0.0, 0.0),
    "excitement": (1.0, 1.0, 1.0),
    "anger": (0.0, 1.0, 1.0),
    "disgust": (0.0, 0.0, 1.0),
    "fear": (0.0, 1.0, 0.0),
    "sadness": (0.0, 0.0, 0.0),
}

POSITIVE_EMOTIONS = {"amusement", "awe", "contentment", "excitement"}
NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}

DEFAULT_BRIGHTNESS_THRESHOLDS = (1.0 / 3.0, 2.0 / 3.0)
DEFAULT_COLORFULNESS_THRESHOLDS = (
    1.0 / 6.0,
    2.0 / 6.0,
    3.0 / 6.0,
    4.0 / 6.0,
    5.0 / 6.0,
)
DEFAULT_FUZZY_STATE = DEFAULT_BRIGHTNESS_THRESHOLDS + DEFAULT_COLORFULNESS_THRESHOLDS


@dataclass(frozen=True)
class Paths:
    bge_model: Path = Path("BGE")
    projection_dir: Path = Path("projection")
    projection_weights: Path = Path("projection/cube_projection_weights.pth")
    emotion_prototypes: Path = Path("projection/emotion_prototypes.pt")
    prototype_objects: Path = Path("projection/prototypes.pkl")
    mapping_metadata: Path = Path("projection/mapping_metadata.json")
    emotion_classifier: Path = Path("docs/Clip_emotion_classifier.pth")
    prompt_system: Path = Path("prompt.txt")


@dataclass(frozen=True)
class RuntimeConfig:
    top_k: int = 5
    lambda_discount: float = 0.95
    alpha_clip: float = 1.0
    beta_emotion: float = 1.0
    low_level_sigma: float = 0.08
    low_level_action_scale: float = 0.08
    high_level_reward_threshold: float = 10.0
    image_height: int = 512
    image_width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 2.5
    seed: int = 200
    qwen_model: str = "Qwen/Qwen2.5-3B-Instruct"
    qwen_max_new_tokens: int = 50


def validate_emotion(emotion: str) -> str:
    value = emotion.strip().lower()
    if value not in EMOTION_ORDER:
        raise ValueError(f"Unsupported emotion '{emotion}'. Expected one of: {', '.join(EMOTION_ORDER)}")
    return value


def resolve_device(device: str | torch.device | None = None, *, require_cuda: bool = False) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        message = (
            "CUDA was requested but PyTorch cannot see a CUDA device. "
            "Check that an NVIDIA GPU, driver, and CUDA-enabled PyTorch runtime are available."
        )
        if require_cuda:
            raise RuntimeError(message)
        raise RuntimeError(message)
    return resolved
