"""Model backends: MLX and GGUF (llama-cpp)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


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

    @abstractmethod
    def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Yield text chunks as they are generated"""
        ...


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
            text = self._extract_text(chunk)
            if text:
                yield text

    @staticmethod
    def _extract_text(result: Any):
        if result is None:
            return ""
        if hasattr(result, "text"):
            return str(getattr(result, "text") or "").strip()
        if isinstance(result, (list, tuple)) and result:
            return str(result[0]).strip()
        
        return str(result).strip()


class GGUFBackend(BaseBackend):
    def __init__(self, model_path: Path, n_ctx: int = 8192, n_gpu_layers: int = -1):
        from llama_cpp import Llama

        model_path = Path(model_path).expanduser().resolve()

        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        if model_path.stat().st_size < 1_000_000:
            raise ValueError(
                f"Model file looks small ({model_path.stat().st_size} bytes)."
                "Download is probably incomplete."
                "Try downloading it again!"
            )
        
        try:
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,  # -1 = offload all layers to Metal
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
            raise RuntimeError(f"Unexpected error loading GGFU model: {e}") from e

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
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


def create_backend(backend_type: str, model_path: Path) -> BaseBackend:
    """Factory that returns the correct backend instance."""
    backend_type = backend_type.lower().strip()

    if backend_type == "mlx":
        return MLXBackend(model_path)

    if backend_type == "gguf":
        return GGUFBackend(model_path)

    raise ValueError(f"Unknown backend: '{backend_type}'. Use 'mlx' or 'gguf'.")