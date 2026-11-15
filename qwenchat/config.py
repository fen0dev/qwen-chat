"""Configuration for QwenChat."""

from pathlib import Path
from typing import Dict, Final, Tuple
import os

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent / ".env.local"
load_dotenv(dotenv_path=ENV_FILE)

def _p(key: str, fallback: str = "") -> Path:
    value = os.getenv(key, fallback).strip()
    return Path(value)

# name → (backend, path)
# Set these in your .env.local
# backend must be "mlx" or "gguf"
DEEPSEEK_Q4_PATH: Final[Path] = _p("DEEPSEEK_Q4_PATH")
LFM_Q4_PATH: Final[Path] = _p("LFM_Q4_PATH")
QWEN_4BIT_PATH: Final[Path] = _p("QWEN_4BIT_PATH")
QWEN_OPUS: Final[Path] = _p("QWEN_OPUS")
QWEN_FABLE: Final[Path] = _p("QWEN_FABLE")

MODELS: Final[Dict[str, Tuple[str, Path]]] = {
    # ---------- main MLX and GGUF models ----------
    "Qwen3.8-27B-Uncensored (4-bit MLX)": (
        "mlx",
        QWEN_4BIT_PATH,
    ),

    "Qwen3.6-40B-Claude-4.6-Opus-Uncensored (Q4_K_M)": (
        "gguf",
        QWEN_OPUS,
    ),
    "Qwen3.6-27B-Claude-Fable-Fusion-Uncensored (Q4_K_M)": (
        "gguf",
        QWEN_FABLE,
    ),
    
    # ---------- other GGUF models ----------
    "LFM2.5-24B-A2B-Uncensored (Q4_K_M)": (
        "gguf",
        LFM_Q4_PATH,
    ),

    "DeepSeek-V4-Pro-Qwen3.5-9B-Heretic (Q4_K_M)": (
        "gguf",
        DEEPSEEK_Q4_PATH,
    ),
}

DEFAULT_MODEL_NAME: Final[str] = "Qwen3.8-27B-Uncensored (Q4_K_M)"

# Generation
MAX_TOKENS: Final[int] = 1024
TEMPERATURE: Final[float] = 0.7
TOP_P: Final[float] = 0.9

# GitHub
GITHUB_TOKEN_FILE: Final[Path] = Path.home() / ".qwenchat_github_token"
GITHUB_API_VERSION: Final[str] = "2022-11-28"
GITHUB_TIMEOUT: Final[int] = 15
GITHUB_MAX_REPOS: Final[int] = 30
GITHUB_MAX_ISSUES: Final[int] = 15

# App
APP_NAME: Final[str] = "QwenChat"
APP_VERSION: Final[str] = "0.2.0"

#History
HISTORY_DIR: Final[Path] = Path.home() / ".qwenchat" / "history"