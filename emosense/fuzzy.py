from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import DEFAULT_FUZZY_STATE

BRIGHTNESS_LABELS = ("low brightness", "normal brightness", "high brightness")
COLORFULNESS_LABELS = (
    "monochromatic",
    "muted",
    "pastel",
    "normal colorfulness",
    "colorful",
    "vibrant",
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def sorted_thresholds(values: Sequence[float], expected_len: int) -> tuple[float, ...]:
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} thresholds, got {len(values)}")
    return tuple(sorted(clamp01(v) for v in values))


def split_state(state: Sequence[float]) -> tuple[tuple[float, float], tuple[float, float, float, float, float]]:
    if len(state) != 7:
        raise ValueError(f"Fuzzy state must contain 7 thresholds, got {len(state)}")
    brightness = sorted_thresholds(state[:2], 2)
    colorfulness = sorted_thresholds(state[2:], 5)
    return brightness, colorfulness


def clamp_sorted_state(state: Sequence[float]) -> tuple[float, ...]:
    brightness, colorfulness = split_state(state)
    return brightness + colorfulness


def _shift_thresholds_to_value(value: float, thresholds: Sequence[float]) -> tuple[float, ...]:
    """Move the uniform fuzzy partition so the selected reference value seeds Fi."""
    value = clamp01(value)
    thresholds = tuple(float(v) for v in thresholds)
    bounds = (0.0,) + thresholds + (1.0,)
    interval_idx = 0
    for idx, right in enumerate(bounds[1:]):
        if value <= right or idx == len(bounds) - 2:
            interval_idx = idx
            break

    center = (bounds[interval_idx] + bounds[interval_idx + 1]) / 2.0
    delta = value - center
    return tuple(clamp01(v + delta) for v in thresholds)


def initial_state_from_attributes(brightness: float, colorfulness: float) -> tuple[float, ...]:
    """Build the low-level initial fuzzy subset Fi from a reference B/C matrix.

    The paper's high-level action retrieves brightness-colorfulness attributes
    from the top-K emotion-specific references and passes them to the low-level
    agent as Fi. The low-level state still consists of fuzzy rule thresholds, so
    the retrieved crisp attributes seed the default partitions before Actor-
    Critic refinement starts.
    """
    brightness_state = _shift_thresholds_to_value(brightness, DEFAULT_FUZZY_STATE[:2])
    colorfulness_state = _shift_thresholds_to_value(colorfulness, DEFAULT_FUZZY_STATE[2:])
    return clamp_sorted_state(brightness_state + colorfulness_state)


def _labels_from_thresholds(value: float, thresholds: Sequence[float], labels: Sequence[str]) -> str:
    value = clamp01(value)
    bounds = (0.0,) + tuple(thresholds) + (1.0,)
    memberships: dict[str, float] = {}
    for label, left, right in zip(labels, bounds[:-1], bounds[1:]):
        center = (left + right) / 2.0
        spread = max((right - left) / 2.0, 1e-3)
        memberships[label] = math.exp(-((value - center) ** 2) / (2.0 * spread**2))
    return max(memberships, key=memberships.get)


def label_brightness(value: float, thresholds: Sequence[float]) -> str:
    return _labels_from_thresholds(value, sorted_thresholds(thresholds, 2), BRIGHTNESS_LABELS)


def label_colorfulness(value: float, thresholds: Sequence[float]) -> str:
    return _labels_from_thresholds(value, sorted_thresholds(thresholds, 5), COLORFULNESS_LABELS)


def labels_for_attributes(
    brightness: float,
    colorfulness: float,
    fuzzy_state: Sequence[float] = DEFAULT_FUZZY_STATE,
) -> tuple[str, str]:
    brightness_thresholds, colorfulness_thresholds = split_state(fuzzy_state)
    return (
        label_brightness(brightness, brightness_thresholds),
        label_colorfulness(colorfulness, colorfulness_thresholds),
    )


def stability_reward(states: Iterable[Sequence[float]]) -> float:
    """Paper Eq. 13: exp(-sqrt(mean(||s_t - s_{t-1}||_2^2)))."""
    vectors = [tuple(float(v) for v in state) for state in states]
    if len(vectors) < 2:
        return 1.0

    squared_norms = []
    for prev, cur in zip(vectors[:-1], vectors[1:]):
        if len(prev) != len(cur):
            raise ValueError("All fuzzy states must have the same dimensionality")
        squared_norms.append(sum((c - p) ** 2 for p, c in zip(prev, cur)))

    rms = math.sqrt(sum(squared_norms) / len(squared_norms))
    return math.exp(-rms)


@dataclass(frozen=True)
class FuzzyAttributeSet:
    object_text: str
    brightness: float
    colorfulness: float
    similarity: float = 0.0

    def labels(self, state: Sequence[float] = DEFAULT_FUZZY_STATE) -> tuple[str, str]:
        return labels_for_attributes(self.brightness, self.colorfulness, state)
