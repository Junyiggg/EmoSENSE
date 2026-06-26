from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from emosense.config import EMOTION_ORDER
from emosense.mapping import normalize_object


def scan_object_emotion_frequency(source_dir: str | Path) -> dict[str, dict[str, int]]:
    source = Path(source_dir)
    freq: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for emotion in EMOTION_ORDER:
        emotion_dir = source / emotion
        if not emotion_dir.is_dir():
            continue
        for json_path in emotion_dir.glob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "object" not in data:
                continue
            obj = normalize_object(data["object"])
            if obj:
                freq[obj][emotion] += 1
    return freq


def dominant_emotions(freq: dict[str, dict[str, int]]) -> dict[str, str]:
    return {
        obj: max(counts.items(), key=lambda item: item[1])[0]
        for obj, counts in freq.items()
        if counts
    }


def _stratified_split_records(
    records: list[dict[str, object]],
    validation_ratio: float,
    random_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    if validation_ratio == 0.0:
        return records, []

    rng = random.Random(random_seed)
    grouped: dict[str, list[dict[str, object]]] = {emotion: [] for emotion in EMOTION_ORDER}
    for record in records:
        grouped[str(record["emotion"])].append(record)

    train_records: list[dict[str, object]] = []
    val_records: list[dict[str, object]] = []
    for emotion in EMOTION_ORDER:
        values = grouped[emotion][:]
        rng.shuffle(values)
        if len(values) <= 1:
            train_records.extend(values)
            continue
        n_val = max(1, int(round(len(values) * validation_ratio)))
        n_val = min(n_val, len(values) - 1)
        val_records.extend(values[:n_val])
        train_records.extend(values[n_val:])
    return train_records, val_records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def filter_annotations(
    source_dir: str | Path,
    dest_dir: str | Path,
    validation_ratio: float = 0.1,
    random_seed: int = 42,
) -> dict[str, object]:
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dominant = dominant_emotions(scan_object_emotion_frequency(source))
    stats = {emotion: 0 for emotion in EMOTION_ORDER}
    duplicate_pairs = 0
    seen_pairs: set[tuple[str, str]] = set()
    records: list[dict[str, object]] = []

    for emotion in EMOTION_ORDER:
        emotion_dir = source / emotion
        if not emotion_dir.is_dir():
            continue
        out_dir = dest / emotion
        out_dir.mkdir(parents=True, exist_ok=True)
        for json_path in sorted(emotion_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "brightness" not in data or "colorfulness" not in data:
                continue
            obj = normalize_object(data.get("object", ""))
            if dominant.get(obj) != emotion:
                continue
            pair_key = (emotion, obj)
            if pair_key in seen_pairs:
                duplicate_pairs += 1
                continue
            seen_pairs.add(pair_key)

            data["emotion"] = emotion
            data["object"] = obj
            data["brightness"] = float(data["brightness"])
            data["colorfulness"] = float(data["colorfulness"])
            target = out_dir / json_path.name
            if target.exists():
                target = out_dir / f"{json_path.stem}_{stats[emotion]}{json_path.suffix}"
            target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            records.append(
                {
                    "emotion": emotion,
                    "object": obj,
                    "brightness": data["brightness"],
                    "colorfulness": data["colorfulness"],
                    "relative_path": str(Path(emotion) / target.name),
                }
            )
            stats[emotion] += 1

    train_records, val_records = _stratified_split_records(records, validation_ratio, random_seed)
    _write_jsonl(dest / "splits" / "train.jsonl", train_records)
    _write_jsonl(dest / "splits" / "validation.jsonl", val_records)
    metadata = {
        "emotion_order": list(EMOTION_ORDER),
        "source_dir": str(source),
        "sample_count": len(records),
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "validation_ratio": validation_ratio,
        "random_seed": random_seed,
        "duplicate_object_emotion_pairs_removed": duplicate_pairs,
        "per_emotion": stats,
    }
    (dest / "preprocess_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter EmoSet annotations by each object's dominant emotion")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--dest-dir", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    stats = filter_annotations(args.source_dir, args.dest_dir, args.validation_ratio, args.random_seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
