"""Model backends: MLX and GGUF (llama-cpp)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class BaseBackend(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        ...


class MLXBackend(BaseBackend):
    def __init__(self, model_path: Path):
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self.model, self.processor = load(str(model_path))
        self.config = load_config(str(model_path))
        self._generate = generate
        self._apply_chat_template = apply_chat_template

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        prompt = self._apply_chat_template(
            self.processor,
            self.config,
            messages,
            add_generation_prompt=True,
        )
        result = self._generate(
            self.model,
            self.processor,
            prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            verbose=False,
        )

        if hasattr(result, "text"):
            return str(result.text).strip()
        if isinstance(result, (list, tuple)) and result:
            return str(result[0]).strip()
        return str(result).strip()


class GGUFBackend(BaseBackend):
    def __init__(self, model_path: Path, n_ctx: int = 8192, n_gpu_layers: int = -1):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,  # -1 = offload all layers to Metal
            verbose=False,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return output["choices"][0]["message"]["content"].strip()


def create_backend(backend_type: str, model_path: Path) -> BaseBackend:
    """Factory that returns the correct backend instance."""
    backend_type = backend_type.lower().strip()

    if backend_type == "mlx":
        return MLXBackend(model_path)

    if backend_type == "gguf":
        return GGUFBackend(model_path)

    raise ValueError(f"Unknown backend: '{backend_type}'. Use 'mlx' or 'gguf'.")