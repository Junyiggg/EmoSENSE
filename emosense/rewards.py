from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import EMOTION_ORDER, Paths, resolve_device, validate_emotion
from .fuzzy import stability_reward


class EmoClassifier(nn.Module):
    def __init__(self, input_dim: int = 768, num_classes: int = 8):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class RewardEvaluator:
    """CLIPScore + EmoAccuracy reward components with lazy model loading."""

    def __init__(
        self,
        classifier_weight: str | Path | None = None,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        device: str | torch.device | None = None,
        local_files_only: bool = True,
    ):
        paths = Paths()
        self.classifier_weight = Path(classifier_weight or paths.emotion_classifier)
        self.clip_model_name = clip_model_name
        self.device = resolve_device(device)
        self.local_files_only = local_files_only
        self._clip_model = None
        self._processor = None
        self._classifier = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(
                self.clip_model_name,
                local_files_only=self.local_files_only,
            )
        return self._processor

    @property
    def clip_model(self):
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(
                self.clip_model_name,
                local_files_only=self.local_files_only,
            ).to(self.device)
            self._clip_model.eval()
        return self._clip_model

    @property
    def classifier(self):
        if self._classifier is None:
            if not self.classifier_weight.exists():
                raise FileNotFoundError(f"Emotion classifier weight not found: {self.classifier_weight}")
            classifier = EmoClassifier().to(self.device)
            classifier.load_state_dict(torch.load(self.classifier_weight, map_location=self.device))
            classifier.eval()
            self._classifier = classifier
        return self._classifier

    def clip_score(self, image_path: str | Path, prompt: str) -> float:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float(torch.matmul(image_embeds, text_embeds.T).item())

    def emotion_confidence(self, image_path: str | Path, target_emotion: str) -> float:
        target_emotion = validate_emotion(target_emotion)
        image = Image.open(image_path).convert("RGB")
        data = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_values = data["pixel_values"].to(self.device)
        with torch.no_grad():
            clip_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            logits = self.classifier(clip_features)
            probabilities = F.softmax(logits, dim=1)
        return float(probabilities[0][EMOTION_ORDER.index(target_emotion)].item())


def low_level_reward(
    clip_reward: float,
    emotion_reward: float,
    stable_reward: float,
    gamma: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    return alpha * clip_reward + beta * emotion_reward + gamma * stable_reward


__all__ = ["RewardEvaluator", "EmoClassifier", "stability_reward", "low_level_reward"]
