"""MLX backend."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, Generator, List

from .base import BaseBackend

def _mlx_max_tokens() -> int:
    return int(os.getenv("QWENCHAT_MAX_TOKENS", "512"))

def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass

    gc.collect()

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
        self.last_usage = None

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self._apply_chat_template(
            self.processor,
            self.config,
            messages,
            add_generation_prompt=True,
        )

    def _cap(self, max_tokens: int) -> int:
        return max(16, min(max_tokens, _mlx_max_tokens()))

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
            max_tokens=self._cap(max_tokens),
            temp=temperature,
            top_p=top_p,
            verbose=False,
        )

        text = self._extract_text(result)
        _clear_mlx_cache()
        return text
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[str, None, None]:
        prompt = self._build_prompt(messages)
        try:
            for chunk in self._stream_generate(
                self.model,
                self.processor,
                prompt,
                max_tokens=self._cap(max_tokens),
                temperature=temperature,
                top_p=top_p,
            ):
                text = self._extract_text(chunk, strip=False)
                if text:
                    yield text
        finally:
            _clear_mlx_cache()

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        _clear_mlx_cache()

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