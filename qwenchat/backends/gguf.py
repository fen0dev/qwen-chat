"""GGUF backend with lower default context and explicit unload."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional

from .base import BaseBackend

def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

class GGUFBackend(BaseBackend):
    def __init__(
        self,
        model_path: Path,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
    ) -> None:
        from llama_cpp import Llama

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        if model_path.stat().st_size < 1_000_000:
            raise ValueError(
                f"Model file looks small ({model_path.stat().st_size} bytes). "
                "Download is probably incomplete."
            )

        self.n_ctx = n_ctx if n_ctx is not None else _int_env("QWENCHAT_N_CTX", 4096)
        self._n_ctx = max(512, self._n_ctx)
        self._n_batch = max(64, _int_env("QWENCHAT_N_BATCH", 512))
        self._n_gpu_layers = (
            n_gpu_layers
            if n_gpu_layers is not None
            else _int_env("QWENCHAT_N_GPU_LAYERS", -1)
        )
        self.last_usage: Optional[Dict[str, int]] = None

        try:
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=self._n_ctx,
                n_batch=self._n_batch,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )

        except ValueError as e:
            raise RuntimeError(
                f"Failed to load GGUF model.\n"
                f"Path: {model_path}\n"
                f"Original error: {e}\n\n"
                "Common causes:\n"
                "  • Incomplete / corrupted download\n"
                "  • llama-cpp-python not compiled with Metal\n"
                "  • Path contains special characters or is too long\n"
                "  • Incompatible quant / architecture"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading GGUF model: {e}") from e

    def _cap(self, max_tokens: int) -> int:
        # Leave room for the prompt inside n_ctx
        hard = max(16, self._n_ctx - 356)
        return max(16, min(max_tokens, hard))

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self._cap(max_tokens),
            temperature=temperature,
            top_p=top_p,
        )
        self.last_usage = output.get("usage")
        return output["choices"][0]["message"]["content"].strip()

    def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[str, None, None]:
        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self._cap(max_tokens),
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

        self.last_usage = None
        for chunk in stream:
            usage = chunk.get("usage")
            if usage:
                self.last_usage = usage
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content

    def memory_stats(self) -> Dict[str, float]:
        return {
            "n_ctx": float(self._n_ctx),
            "n_batch": float(self._n_batch),
        }
        
    def unload(self) -> None:
        if hasattr(self, "llm"):
            del self.llm
        gc.collect()