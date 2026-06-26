from __future__ import annotations

import re
from pathlib import Path

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

from .config import Paths, RuntimeConfig, resolve_device


CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


class QwenPromptGenerator:
    """LLM prompt generator used as the low-level language component.

    The paper uses LLaMA3.2-3B. This implementation intentionally keeps Qwen as
    requested while preserving the short-prompt constraint used in the paper.
    """

    def __init__(
        self,
        model_name: str | None = None,
        system_prompt_path: str | Path | None = None,
        device: str | None = None,
        precision: torch.dtype = torch.float16,
        max_new_tokens: int | None = None,
    ):
        cfg = RuntimeConfig()
        paths = Paths()
        self.model_name = model_name or cfg.qwen_model
        self.system_prompt_path = Path(system_prompt_path or paths.prompt_system)
        self.device = str(resolve_device(device))
        self.precision = precision if self.device != "cpu" else torch.float32
        self.max_new_tokens = max_new_tokens or cfg.qwen_max_new_tokens
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            device_map = "cpu" if self.device == "cpu" else "auto"
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.precision,
                device_map=device_map,
                local_files_only=True,
            )
        return self._model

    def _system_prompt(self) -> str:
        return self.system_prompt_path.read_text(encoding="utf-8").strip()

    def _generate_response(self, messages: list[dict[str, str]], *, do_sample: bool = True) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": 1.2,
        }
        if do_sample:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.9,
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        generated_ids = self.model.generate(**model_inputs, **generate_kwargs)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return " ".join(response.strip().split())

    def expand(self, object_text: str, emotion: str | None = None) -> str:
        object_value = object_text.strip()
        user_prompt = (
            f"Object: {object_value}\n"
            "Write one ASCII English sentence for a text-to-image prompt."
        )
        if emotion:
            user_prompt = (
                f"Object: {object_value}\n"
                f"Emotion: {emotion.strip().lower()}\n"
                "Write one ASCII English sentence for a text-to-image prompt."
            )

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_prompt},
        ]
        response = self._generate_response(messages, do_sample=True)
        if CJK_RE.search(response):
            retry_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "The previous answer used non-English characters. "
                        "Regenerate it in ASCII English only, exactly one sentence."
                    ),
                }
            ]
            response = self._generate_response(retry_messages, do_sample=False)
        return response


class TemplatePromptGenerator:
    """Small deterministic prompt generator for dry runs and tests."""

    def expand(self, object_text: str, emotion: str | None = None) -> str:
        if emotion:
            return f"A {object_text} scene composed to express {emotion} through atmosphere, lighting, and setting."
        return f"A clear scene centered on {object_text} with coherent visual detail."
