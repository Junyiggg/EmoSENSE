from __future__ import annotations

import json
import pickle
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import EMOTION_ORDER, VAD_COORDINATES, Paths, resolve_device, validate_emotion
from .fuzzy import FuzzyAttributeSet


class CubeProjection(nn.Module):
    """MLP projection network h_phi from BGE embeddings to the VAD cube."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class AnnotationSample:
    emotion: str
    object_text: str
    brightness: float
    colorfulness: float


@dataclass(frozen=True)
class MappingResult:
    emotion: str
    object_text: str
    correlation: float
    gamma: float
    object_projection: tuple[float, float, float]
    closest_objects: tuple[FuzzyAttributeSet, ...]
    cosine_similarities: tuple[float, ...]
    object_embedding: np.ndarray


def normalize_object(value: object) -> str:
    if isinstance(value, list):
        text = " ".join(str(v) for v in value)
    else:
        text = str(value)
    return " ".join(text.lower().strip().split())


def load_annotation_samples(annotation_dir: str | Path) -> list[AnnotationSample]:
    root = Path(annotation_dir)
    samples: list[AnnotationSample] = []
    for emotion in EMOTION_ORDER:
        emotion_dir = root / emotion
        if not emotion_dir.is_dir():
            continue
        for json_path in emotion_dir.glob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "object" not in data or "brightness" not in data or "colorfulness" not in data:
                continue
            obj = normalize_object(data["object"])
            if not obj:
                continue
            samples.append(
                AnnotationSample(
                    emotion=emotion,
                    object_text=obj,
                    brightness=float(data["brightness"]),
                    colorfulness=float(data["colorfulness"]),
                )
            )
    return samples


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def _encode_batch(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    normalize: bool = False,
) -> torch.Tensor:
    inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]
        if normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.detach().cpu()


class AnnotationEmbeddingDataset(Dataset):
    def __init__(self, samples: Sequence[AnnotationSample], embeddings: torch.Tensor):
        self.samples = list(samples)
        self.embeddings = embeddings.float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        label = EMOTION_ORDER.index(sample.emotion)
        return self.embeddings[idx], label


def _compute_centroids(samples: Sequence[AnnotationSample], embeddings: torch.Tensor) -> torch.Tensor:
    by_emotion: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_emotion[sample.emotion].append(idx)

    missing = [emotion for emotion in EMOTION_ORDER if emotion not in by_emotion]
    if missing:
        raise ValueError(f"Missing emotion categories in annotation data: {missing}")

    centroids = []
    for emotion in EMOTION_ORDER:
        indices = by_emotion[emotion]
        centroids.append(embeddings[indices].mean(dim=0))
    return torch.stack(centroids, dim=0)


def _group_samples_by_emotion(samples: Sequence[AnnotationSample]) -> dict[str, list[AnnotationSample]]:
    grouped: dict[str, list[AnnotationSample]] = {emotion: [] for emotion in EMOTION_ORDER}
    for sample in samples:
        grouped[sample.emotion].append(sample)
    return grouped


def stratified_sample(
    samples: Sequence[AnnotationSample],
    limit: int | None,
    random_seed: int = 42,
) -> list[AnnotationSample]:
    if limit is None or limit >= len(samples):
        return list(samples)
    if limit <= 0:
        raise ValueError("limit must be positive")

    grouped = _group_samples_by_emotion(samples)
    rng = random.Random(random_seed)
    non_empty = {emotion: values[:] for emotion, values in grouped.items() if values}
    if limit < len(non_empty):
        raise ValueError(f"limit={limit} is too small; at least {len(non_empty)} samples are needed for all emotions")

    quotas = {emotion: 1 for emotion in non_empty}
    remaining = limit - sum(quotas.values())
    if remaining <= 0:
        raw_quotas = {emotion: 0.0 for emotion in non_empty}
    else:
        alloc_total = sum(max(0, len(values) - 1) for values in non_empty.values())
        raw_quotas = {
            emotion: remaining * max(0, len(values) - 1) / max(1, alloc_total)
            for emotion, values in non_empty.items()
        }
        for emotion, raw in raw_quotas.items():
            quotas[emotion] += min(len(non_empty[emotion]) - quotas[emotion], floor(raw))
        remaining = limit - sum(quotas.values())
    if remaining <= 0:
        raw_quotas = {
            emotion: quotas[emotion]
            for emotion in non_empty
        }
    for emotion, _ in sorted(raw_quotas.items(), key=lambda item: item[1] - floor(item[1]), reverse=True):
        if remaining <= 0:
            break
        if quotas[emotion] < len(non_empty[emotion]):
            quotas[emotion] += 1
            remaining -= 1

    selected: list[AnnotationSample] = []
    for emotion in EMOTION_ORDER:
        values = non_empty.get(emotion, [])
        rng.shuffle(values)
        selected.extend(values[: quotas.get(emotion, 0)])
    return selected


def stratified_train_validation_split(
    samples: Sequence[AnnotationSample],
    validation_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[AnnotationSample], list[AnnotationSample]]:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    if validation_ratio == 0.0:
        return list(samples), []

    grouped = _group_samples_by_emotion(samples)
    rng = random.Random(random_seed)
    train_samples: list[AnnotationSample] = []
    val_samples: list[AnnotationSample] = []

    for emotion in EMOTION_ORDER:
        values = grouped[emotion][:]
        rng.shuffle(values)
        if len(values) <= 1:
            train_samples.extend(values)
            continue
        n_val = max(1, int(round(len(values) * validation_ratio)))
        n_val = min(n_val, len(values) - 1)
        val_samples.extend(values[:n_val])
        train_samples.extend(values[n_val:])
    return train_samples, val_samples


def save_prototype_objects(
    samples: Sequence[AnnotationSample],
    embeddings: torch.Tensor,
    output_path: str | Path,
) -> None:
    metadata: dict[str, list[tuple[np.ndarray, float, float, str]]] = {emotion: [] for emotion in EMOTION_ORDER}
    for sample, embedding in zip(samples, embeddings):
        metadata[sample.emotion].append(
            (
                embedding.numpy(),
                sample.brightness,
                sample.colorfulness,
                sample.object_text,
            )
        )
    with Path(output_path).open("wb") as f:
        pickle.dump(metadata, f)


def train_mapping_space(
    annotation_dir: str | Path,
    output_dir: str | Path = "projection",
    bge_model_path: str | Path = "BGE",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str | torch.device | None = None,
    embedding_batch_size: int = 128,
    limit: int | None = None,
    validation_ratio: float = 0.1,
    random_seed: int = 42,
) -> None:
    """Train h_phi with the paper's Lproto + Lcube + Lobj objective.

    BGE remains frozen. Emotion prototypes are fixed centroids of BGE object
    embeddings, matching Eq. 1. Lproto is reported as the centroid clustering
    term; Lcube and Lobj update the projection network.
    """

    device = resolve_device(device)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = load_annotation_samples(annotation_dir)
    if not samples:
        raise ValueError(f"No valid annotation samples found under {annotation_dir}")
    samples = stratified_sample(samples, limit=limit, random_seed=random_seed)
    train_samples, val_samples = stratified_train_validation_split(
        samples,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )
    if not train_samples:
        raise ValueError("No training samples remain after stratified split")

    tokenizer = AutoTokenizer.from_pretrained(str(bge_model_path), local_files_only=True)
    bge = AutoModel.from_pretrained(str(bge_model_path), local_files_only=True).to(device).eval()
    for param in bge.parameters():
        param.requires_grad_(False)

    all_embeddings = []
    for start in tqdm(range(0, len(train_samples), embedding_batch_size), desc="Encoding train objects"):
        batch = train_samples[start : start + embedding_batch_size]
        all_embeddings.append(_encode_batch([s.object_text for s in batch], tokenizer, bge, device))
    embeddings = torch.cat(all_embeddings, dim=0)
    val_embeddings = None
    if val_samples:
        encoded_val = []
        for start in tqdm(range(0, len(val_samples), embedding_batch_size), desc="Encoding validation objects"):
            batch = val_samples[start : start + embedding_batch_size]
            encoded_val.append(_encode_batch([s.object_text for s in batch], tokenizer, bge, device))
        val_embeddings = torch.cat(encoded_val, dim=0)

    centroids = _compute_centroids(train_samples, embeddings)
    vad_vertices = torch.tensor([VAD_COORDINATES[e] for e in EMOTION_ORDER], dtype=torch.float32, device=device)

    dataset = AnnotationEmbeddingDataset(train_samples, embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    projection = CubeProjection(input_dim=embeddings.shape[1]).to(device)
    optimizer = optim.Adam(projection.parameters(), lr=lr)

    centroids_device = centroids.to(device)
    for epoch in range(epochs):
        projection.train()
        total_loss = total_proto = total_cube = total_obj = 0.0
        for batch_embeddings, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch_embeddings = batch_embeddings.to(device)
            labels = labels.to(device)

            selected_centroids = centroids_device[labels]
            proto_loss = torch.mean((batch_embeddings - selected_centroids).pow(2))

            projected_centroids = projection(centroids_device)
            cube_loss = torch.mean((projected_centroids - vad_vertices).pow(2))

            projected_objects = projection(batch_embeddings)
            projected_selected_centroids = projected_centroids[labels]
            obj_loss = torch.mean((projected_objects - projected_selected_centroids).pow(2))

            loss = proto_loss + cube_loss + obj_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_proto += proto_loss.item()
            total_cube += cube_loss.item()
            total_obj += obj_loss.item()

        n = max(1, len(loader))
        print(
            f"Epoch {epoch + 1}: "
            f"Loss={total_loss / n:.4f} "
            f"Proto={total_proto / n:.4f} "
            f"Cube={total_cube / n:.4f} "
            f"Obj={total_obj / n:.4f}"
        )
        if val_embeddings is not None:
            projection.eval()
            val_labels = torch.tensor([EMOTION_ORDER.index(sample.emotion) for sample in val_samples], device=device)
            val_batch = val_embeddings.to(device)
            with torch.no_grad():
                projected_centroids = projection(centroids_device)
                selected_centroids = centroids_device[val_labels]
                val_proto = torch.mean((val_batch - selected_centroids).pow(2))
                val_cube = torch.mean((projected_centroids - vad_vertices).pow(2))
                val_obj = torch.mean((projection(val_batch) - projected_centroids[val_labels]).pow(2))
            print(
                f"Validation: "
                f"Loss={(val_proto + val_cube + val_obj).item():.4f} "
                f"Proto={val_proto.item():.4f} "
                f"Cube={val_cube.item():.4f} "
                f"Obj={val_obj.item():.4f}"
            )

    torch.save(projection.state_dict(), output_root / "cube_projection_weights.pth")
    torch.save({emotion: centroids[idx].cpu() for idx, emotion in enumerate(EMOTION_ORDER)}, output_root / "emotion_prototypes.pt")
    save_prototype_objects(train_samples, embeddings, output_root / "prototypes.pkl")
    metadata = {
        "emotion_order": list(EMOTION_ORDER),
        "vad_coordinates": VAD_COORDINATES,
        "bge_model_path": str(bge_model_path),
        "sample_count": len(samples),
        "train_count": len(train_samples),
        "validation_count": len(val_samples),
        "validation_ratio": validation_ratio,
        "random_seed": random_seed,
        "embedding_dim": int(embeddings.shape[1]),
        "frozen_bge": True,
    }
    (output_root / "mapping_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class SentimentSemanticMapper:
    """Prototype network plus projection network used by the high-level agent."""

    def __init__(
        self,
        paths: Paths = Paths(),
        device: str | torch.device | None = None,
    ):
        self.paths = paths
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(paths.bge_model), local_files_only=True)
        self.bge = AutoModel.from_pretrained(str(paths.bge_model), local_files_only=True).to(self.device).eval()
        for param in self.bge.parameters():
            param.requires_grad_(False)

        self.emotion_prototypes = torch.load(paths.emotion_prototypes, map_location="cpu")
        first_proto = next(iter(self.emotion_prototypes.values()))
        self.projection = CubeProjection(input_dim=int(first_proto.numel())).to(self.device)
        self.projection.load_state_dict(torch.load(paths.projection_weights, map_location=self.device))
        self.projection.eval()

        with Path(paths.prototype_objects).open("rb") as f:
            self.prototype_objects = pickle.load(f)

        self.metadata = {}
        if Path(paths.mapping_metadata).exists():
            self.metadata = json.loads(Path(paths.mapping_metadata).read_text(encoding="utf-8"))
            if self.metadata.get("emotion_order") != list(EMOTION_ORDER):
                warnings.warn(
                    "Mapping metadata emotion_order does not match the paper order. "
                    "Retrain with `python -m emosense.cli train-mapping` for paper-aligned correlations.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "projection/mapping_metadata.json is missing. Existing mapping artifacts may be from the legacy "
                "alphabetical emotion-order training script. Retrain mapping artifacts for full paper alignment.",
                RuntimeWarning,
                stacklevel=2,
            )

    def encode(self, text: str) -> np.ndarray:
        return _encode_batch([text], self.tokenizer, self.bge, self.device).squeeze(0).numpy()

    def project(self, embedding: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(embedding, np.ndarray):
            tensor = torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            tensor = embedding.float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.projection(tensor).squeeze(0).cpu().numpy()

    def closest_objects(self, object_embedding: np.ndarray, emotion: str, top_k: int = 5) -> tuple[tuple[FuzzyAttributeSet, ...], tuple[float, ...]]:
        emotion = validate_emotion(emotion)
        candidates = []
        for emb, brightness, colorfulness, obj_text in self.prototype_objects.get(emotion, []):
            sim = _cosine_similarity(object_embedding, np.asarray(emb))
            candidates.append((sim, FuzzyAttributeSet(obj_text, float(brightness), float(colorfulness), sim)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        top = candidates[:top_k]
        return tuple(item[1] for item in top), tuple(float(item[0]) for item in top)

    def evaluate(self, object_text: str, emotion: str, top_k: int = 5) -> MappingResult:
        emotion = validate_emotion(emotion)
        normalized_object = normalize_object(object_text)
        object_embedding = self.encode(normalized_object)
        object_projection = self.project(object_embedding)
        target_vertex = np.asarray(VAD_COORDINATES[emotion], dtype=np.float32)
        correlation = float(np.linalg.norm(object_projection - target_vertex))
        gamma = float(np.exp(-correlation))
        closest, similarities = self.closest_objects(object_embedding, emotion, top_k=top_k)
        return MappingResult(
            emotion=emotion,
            object_text=normalized_object,
            correlation=correlation,
            gamma=gamma,
            object_projection=tuple(float(v) for v in object_projection),
            closest_objects=closest,
            cosine_similarities=similarities,
            object_embedding=object_embedding,
        )


# Backward-compatible alias for the original hl_agent.py API.
SentimentEvaluator = SentimentSemanticMapper
