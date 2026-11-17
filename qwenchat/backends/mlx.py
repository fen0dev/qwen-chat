"""MLX backend with memory caps, cache control, and optional chunked prefill."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .base import BaseBackend

def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _mlx_max_tokens() -> int:
    return max(16, _int_env("QWENCHAT_MLX_MAX_TOKENS", 512))

def _prefill_step() -> Optional[int]:
    step = _int_env("QWENCHAT_MLX_PREFILL_STEP", 0)
    return step if step >= 64 else None

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
        self.last_usage: Optional[Dict[str, int]] = None

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self._apply_chat_template(
            self.processor,
            self.config,
            messages,
            add_generation_prompt=True,
        )

    def _cap(self, max_tokens: int) -> int:
        return max(16, min(max_tokens, _mlx_max_tokens()))

    def _gen_kwargs(self, max_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_tokens": self._cap(max_tokens),
            "temperature": temperature,
            "top_p": top_p,
        }

        step = _prefill_step()
        if step:
            kwargs["prefill_step_size"] = step
        return kwargs

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        prompt = self._build_prompt(messages)
        kwargs = self._gen_kwargs(max_tokens, temperature, top_p)
        result = self._generate(
            self.model,
            self.processor,
            prompt,
            verbose=False,
            **kwargs,
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
        kwargs = self._gen_kwargs(max_tokens, temperature, top_p)
        self.last_usage = None
        try:
            for chunk in self._stream_generate(
                self.model,
                self.processor,
                prompt,
                **kwargs,
            ):
                text = self._extract_text(chunk, strip=False)
                if text:
                    yield text
        finally:
            _clear_mlx_cache()

    def memory_stats(self) -> Dict[str, float]:
        try:
            import mlx.core as mx
            return {
                "active_gb": mx.metal.get_active_memory() / 1e9,
                "peak_gb": mx.metal.get_peak_memory() / 1e9,
            }
        except Exception:
            return {}

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