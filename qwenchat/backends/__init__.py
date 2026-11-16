"""Backend factory."""

from __future__ import annotations

from pathlib import Path

from .base import BaseBackend
from .gguf import GGUFBackend
from .mlx import MLXBackend

def create_backend(backend_type: str, model_path: Path) -> BaseBackend:
    backend_type = backend_type.lower().strip()
    if backend_type == "mlx":
        return MLXBackend(model_path)
    if backend_type == "gguf":
        return GGUFBackend(model_path)
    raise ValueError(f"Unknown backend: '{backend_type}'. Use 'mlx' or 'gguf'.")

__all__ = ["BaseBackend", "create_backend", "GGUFBackend", "MLXBackend"]