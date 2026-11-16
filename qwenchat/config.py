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

#DEEPSEEK_Q4_PATH: Final[Path] = _p("DEEPSEEK_Q4_PATH")
LFM_Q4_PATH: Final[Path] = _p("LFM_Q4_PATH")
QWEN_4BIT_PATH: Final[Path] = _p("QWEN_4BIT_PATH")
#QWEN_OPUS: Final[Path] = _p("QWEN_OPUS")
QWEN_FABLE: Final[Path] = _p("QWEN_FABLE")

MODELS: Final[Dict[str, Tuple[str, Path]]] = {
    "Qwen3.8-27B-Uncensored (4-bit MLX)": ("mlx", QWEN_4BIT_PATH),
    "Qwen3.6-27B-Claude-Fable-Fusion-Uncensored (Q4_K_M)": ("gguf", QWEN_FABLE),
    "LFM2.5-24B-A2B-Uncensored (Q4_K_M)": ("gguf", LFM_Q4_PATH),
}

DEFAULT_MODEL_NAME: Final[str] = "Qwen3.8-27B-Uncensored (4-bit MLX)"

MAX_TOKENS: Final[int] = 1024
TEMPERATURE: Final[float] = 0.7
TOP_P: Final[float] = 0.9

GITHUB_TOKEN_FILE: Final[Path] = Path.home() / ".qwenchat_github_token"
GITHUB_API_VERSION: Final[str] = "2022-11-28"
GITHUB_TIMEOUT: Final[int] = 15
GITHUB_MAX_REPOS: Final[int] = 30
GITHUB_MAX_ISSUES: Final[int] = 15

APP_NAME: Final[str] = "QwenChat"
APP_VERSION: Final[str] = "0.2.0"

_history = os.getenv("QWENCHAT_HISTORY_DIR", "").strip()
HISTORY_DIR: Final[Path] = (
    Path(_history).expanduser().resolve()
    if _history
    else Path.home() / ".qwenchat" / "history"
)

SESSION_DIR: Final[Path] = HISTORY_DIR / "sessions"
PLANS_DIR: Final[Path] = HISTORY_DIR / "plans"
SUMMARY_PATH: Final[Path] = HISTORY_DIR / "summaries" / "rolling.json"

SESSION_TTL_DAYS: Final[int] = 7
ROLLING_SUMMARY_MAX_CHARS: Final[int] = 4000