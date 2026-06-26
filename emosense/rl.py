from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from .config import Paths, RuntimeConfig, resolve_device, validate_emotion
from .fuzzy import (
    FuzzyAttributeSet,
    clamp_sorted_state,
    initial_state_from_attributes,
    labels_for_attributes,
    stability_reward,
)
from .generation import OmniGenImageGenerator
from .mapping import MappingResult, SentimentSemanticMapper
from .prompting import QwenPromptGenerator, TemplatePromptGenerator
from .rewards import RewardEvaluator, low_level_reward


class CategoricalActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(state), dim=-1)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, action_scale: float = 0.08):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_scale * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass
class StepRecord:
    high_episode: int
    low_step: int
    high_action_index: int
    selected_reference: str
    reference_similarity: float
    brightness: float
    colorfulness: float
    initial_fuzzy_state: list[float]
    brightness_label: str
    colorfulness_label: str
    prompt: str
    image_path: str | None
    clip_reward: float
    emotion_reward: float
    stable_reward: float
    low_reward: float
    correlation: float
    gamma: float
    fuzzy_state: list[float]


@dataclass
class GenerationResult:
    emotion: str
    object_text: str
    output_dir: str
    correlation: float
    gamma: float
    best_prompt: str
    best_image_path: str | None
    cumulative_reward: float
    records: list[StepRecord]


class EmoSENSEPipeline:
    """Hierarchical fuzzy RL pipeline described in the EmoSENSE paper."""

    def __init__(
        self,
        paths: Paths = Paths(),
        config: RuntimeConfig = RuntimeConfig(),
        device: str | torch.device | None = None,
        mapper: SentimentSemanticMapper | None = None,
        prompt_generator: QwenPromptGenerator | TemplatePromptGenerator | None = None,
        image_generator: OmniGenImageGenerator | None = None,
        reward_evaluator: RewardEvaluator | None = None,
    ):
        self.paths = paths
        self.config = config
        self.device = resolve_device(device)
        self.mapper = mapper or SentimentSemanticMapper(paths=paths, device=self.device)
        self.prompt_generator = prompt_generator or QwenPromptGenerator(device=str(self.device))
        self.image_generator = image_generator or OmniGenImageGenerator(
            height=config.image_height,
            width=config.image_width,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            seed=config.seed,
        )
        self.reward_evaluator = reward_evaluator or RewardEvaluator(paths.emotion_classifier, device=self.device)

        self.high_actor: CategoricalActor | None = None
        self.high_critic: Critic | None = None
        self.high_actor_opt: optim.Optimizer | None = None
        self.high_critic_opt: optim.Optimizer | None = None
        self.low_actor = GaussianActor(7, 7, action_scale=config.low_level_action_scale).to(self.device)
        self.low_critic = Critic(7, hidden_dim=64).to(self.device)
        self.low_actor_opt = optim.Adam(self.low_actor.parameters(), lr=1e-5)
        self.low_critic_opt = optim.Adam(self.low_critic.parameters(), lr=1e-5)

    def _ensure_high_level(self, state_dim: int, action_dim: int) -> None:
        if self.high_actor is not None:
            return
        self.high_actor = CategoricalActor(state_dim, action_dim).to(self.device)
        self.high_critic = Critic(state_dim).to(self.device)
        self.high_actor_opt = optim.Adam(self.high_actor.parameters(), lr=1e-4)
        self.high_critic_opt = optim.Adam(self.high_critic.parameters(), lr=1e-4)

    def _select_reference(self, mapping: MappingResult) -> tuple[int, FuzzyAttributeSet, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = list(mapping.closest_objects)
        if not candidates:
            candidates = [FuzzyAttributeSet(mapping.object_text, 0.5, 0.5, 0.0)]
        while len(candidates) < self.config.top_k:
            candidates.append(candidates[-1])

        high_state = torch.tensor(mapping.object_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._ensure_high_level(high_state.shape[-1], self.config.top_k)
        assert self.high_actor is not None and self.high_critic is not None

        probs = self.high_actor(high_state).squeeze(0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        action_index = int(action.item())
        return action_index, candidates[action_index], high_state, dist.log_prob(action), self.high_critic(high_state)

    def _build_prompt(
        self,
        object_text: str,
        emotion: str,
        reference: FuzzyAttributeSet,
        fuzzy_state: Sequence[float],
    ) -> tuple[str, str, str]:
        brightness_label, colorfulness_label = labels_for_attributes(
            reference.brightness,
            reference.colorfulness,
            fuzzy_state,
        )
        description = self.prompt_generator.expand(object_text, emotion)
        return (
            brightness_label,
            colorfulness_label,
            f"{brightness_label} and {colorfulness_label}. {description}",
        )

    def generate(
        self,
        object_text: str,
        emotion: str,
        output_dir: str | Path = "runs",
        high_episodes: int = 1,
        low_steps: int = 5,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> GenerationResult:
        emotion = validate_emotion(emotion)
        mapping = self.mapper.evaluate(object_text, emotion, top_k=self.config.top_k)
        if verbose:
            print(
                f"[Mapping] object={mapping.object_text} emotion={emotion} "
                f"corr={mapping.correlation:.4f} gamma={mapping.gamma:.4f}",
                flush=True,
            )
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        records: list[StepRecord] = []
        best_prompt = ""
        best_image_path: str | None = None
        best_reward = float("-inf")
        last_cumulative_reward = 0.0

        for high_episode in range(1, high_episodes + 1):
            action_index, reference, high_state, high_log_prob, high_value = self._select_reference(mapping)
            if verbose:
                print(
                    f"[HL {high_episode}] selected reference #{action_index}: "
                    f"{reference.object_text} sim={reference.similarity:.4f} "
                    f"B={reference.brightness:.3f} C={reference.colorfulness:.3f}",
                    flush=True,
                )
            initial_fuzzy_state = initial_state_from_attributes(reference.brightness, reference.colorfulness)
            fuzzy_state = torch.tensor(initial_fuzzy_state, dtype=torch.float32, device=self.device)
            state_history: list[list[float]] = [list(initial_fuzzy_state)]
            cumulative_reward = 0.0

            log_path = output_root / f"episode_{high_episode:02d}_rewards.jsonl"
            with log_path.open("w", encoding="utf-8") as reward_log:
                for low_step in range(1, low_steps + 1):
                    state_input = fuzzy_state.unsqueeze(0)
                    value = self.low_critic(state_input)
                    mu = self.low_actor(state_input).squeeze(0)
                    dist = torch.distributions.Normal(mu, self.config.low_level_sigma)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum()
                    next_state = torch.tensor(
                        clamp_sorted_state((fuzzy_state + action).detach().cpu().tolist()),
                        dtype=torch.float32,
                        device=self.device,
                    )

                    state_history.append(next_state.detach().cpu().tolist())
                    brightness_label, colorfulness_label, prompt = self._build_prompt(
                        mapping.object_text,
                        emotion,
                        reference,
                        next_state.detach().cpu().tolist(),
                    )
                    if verbose:
                        print(f"[LL {high_episode}.{low_step}] prompt: {prompt}", flush=True)

                    image_path: str | None = None
                    clip_reward = 0.0
                    emotion_reward = 0.0
                    if not dry_run:
                        image_file = output_root / f"episode_{high_episode:02d}_step_{low_step:02d}.png"
                        if verbose:
                            print(f"[LL {high_episode}.{low_step}] generating image...", flush=True)
                        image_path = str(self.image_generator.generate(prompt, image_file))
                        if verbose:
                            print(f"[LL {high_episode}.{low_step}] image saved: {image_path}", flush=True)
                        clip_reward = self.reward_evaluator.clip_score(image_path, prompt)
                        emotion_reward = self.reward_evaluator.emotion_confidence(image_path, emotion)

                    stable = stability_reward(state_history)
                    reward = low_level_reward(
                        clip_reward,
                        emotion_reward,
                        stable,
                        mapping.gamma,
                        alpha=self.config.alpha_clip,
                        beta=self.config.beta_emotion,
                    )
                    cumulative_reward += reward
                    if verbose:
                        print(
                            f"[LL {high_episode}.{low_step}] reward: "
                            f"clip={clip_reward:.4f} emo={emotion_reward:.4f} "
                            f"stable={stable:.4f} total={reward:.4f}",
                            flush=True,
                        )

                    with torch.no_grad():
                        next_value = self.low_critic(next_state.unsqueeze(0))
                    delta = reward + self.config.lambda_discount * next_value - value
                    low_actor_loss = -log_prob * delta.detach().squeeze()
                    low_critic_loss = delta.pow(2).mean()
                    self.low_actor_opt.zero_grad()
                    self.low_critic_opt.zero_grad()
                    (low_actor_loss + low_critic_loss).backward()
                    self.low_actor_opt.step()
                    self.low_critic_opt.step()

                    record = StepRecord(
                        high_episode=high_episode,
                        low_step=low_step,
                        high_action_index=action_index,
                        selected_reference=reference.object_text,
                        reference_similarity=float(reference.similarity),
                        brightness=float(reference.brightness),
                        colorfulness=float(reference.colorfulness),
                        initial_fuzzy_state=[float(v) for v in initial_fuzzy_state],
                        brightness_label=brightness_label,
                        colorfulness_label=colorfulness_label,
                        prompt=prompt,
                        image_path=image_path,
                        clip_reward=float(clip_reward),
                        emotion_reward=float(emotion_reward),
                        stable_reward=float(stable),
                        low_reward=float(reward),
                        correlation=float(mapping.correlation),
                        gamma=float(mapping.gamma),
                        fuzzy_state=[float(v) for v in next_state.detach().cpu().tolist()],
                    )
                    records.append(record)
                    reward_log.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

                    if reward > best_reward:
                        best_reward = reward
                        best_prompt = prompt
                        best_image_path = image_path

                    fuzzy_state = next_state.detach()

            high_reward = 1.0 if cumulative_reward >= self.config.high_level_reward_threshold else 0.0
            assert self.high_actor_opt is not None and self.high_critic_opt is not None
            high_delta = high_reward - high_value
            high_actor_loss = -high_log_prob * high_delta.detach().squeeze()
            high_critic_loss = high_delta.pow(2).mean()
            self.high_actor_opt.zero_grad()
            self.high_critic_opt.zero_grad()
            (high_actor_loss + high_critic_loss).backward()
            self.high_actor_opt.step()
            self.high_critic_opt.step()
            last_cumulative_reward = float(cumulative_reward)

        result = GenerationResult(
            emotion=emotion,
            object_text=mapping.object_text,
            output_dir=str(output_root),
            correlation=float(mapping.correlation),
            gamma=float(mapping.gamma),
            best_prompt=best_prompt,
            best_image_path=best_image_path,
            cumulative_reward=last_cumulative_reward,
            records=records,
        )
        (output_root / "summary.json").write_text(
            json.dumps(
                {
                    **{k: v for k, v in asdict(result).items() if k != "records"},
                    "records": [asdict(record) for record in records],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return result
