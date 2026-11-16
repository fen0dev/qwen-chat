"""Model backends: MLX and GGUF (llama-cpp)."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
    ) -> Generator[str, None, None]:
        """Yield text chunks as they are generated"""
        ...

    def unload(self) -> None:
        """Optional hook to free memory. Override in backends that hold large objects."""
        pass