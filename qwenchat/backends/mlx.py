"""MLX backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, List

from .base import BaseBackend

class MLXBackend(BaseBackend):
    def __init__(self, model_path: Path):
        from mlx_vlm import load, generate, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self.model, self.processor = load(str(model_path))
        self.config = load_config(str(model_path))
        self._generate = generate
        self._stream_generate = stream_generate
        self._apply_chat_template = apply_chat_template

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self._apply_chat_template(
            self.processor,
            self.config,
            messages,
            add_generation_prompt=True,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        prompt = self._build_prompt(messages)
        result = self._generate(
            self.model,
            self.processor,
            prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            verbose=False,
        )

        return self._extract_text(result)
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[str, None, None]:
        prompt = self._build_prompt(messages)
        for chunk in self._stream_generate(
            self.model,
            self.processor,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            text = self._extract_text(chunk, strip=False)
            if text:
                yield text

    @staticmethod
    def _extract_text(result: Any, strip: bool = True) -> str:
        if result is None:
            return ""
        if hasattr(result, "text"):
            text = str(getattr(result, "text") or "")
        elif isinstance(result, (list, tuple)) and result:
            text = str(result[0])
        else:
            text = str(result)
        return text.strip() if strip else text