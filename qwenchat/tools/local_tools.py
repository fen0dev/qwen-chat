"""Sandboxed local file tools (read-only)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..config import MODELS

_LOCAL_ROOT: Path = Path(os.getenv("QWENCHAT_LOCAL_ROOT", ".")).expanduser().resolve()
MAX_FILE_BYTES = 64_000
MAX_SEARCH_HITS = 40

def get_local_root() -> Path:
    return _LOCAL_ROOT

def set_local_root(path: str) -> tuple[bool, str]:
    global _LOCAL_ROOT
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            return False, f"Not a directory: {p}"
        _LOCAL_ROOT = p
        return True, f"Local root set to {p}"
    except Exception as exc:
        return False, f"Invalid path: {exc}"

def _resolve_safe(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    try:
        p = (_LOCAL_ROOT / path_str).resolve()
        p.relative_to(_LOCAL_ROOT)
        return p
    except (ValueError, OSError):
        return None

def read_file(args: Dict[str, str]) -> str:
    path_str = (args.get("path") or "").strip()
    p = _resolve_safe(path_str)
    if p is None or not p.is_file():
        return "Error: path not found or outside allowed directory."
    try:
        data = p.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            text = data[:MAX_FILE_BYTES].decode("utf-8", errors="replace")
            return text + f"\n\n[truncated — file is {len(data)} bytes]"
        return data.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"

def list_dir(args: Dict[str, str]) -> str:
    path_str = (args.get("path") or ".").strip()
    p = _resolve_safe(path_str)
    if p is None or not p.is_dir():
        return "Error: directory not found or outside allowed root."
    try:
        entries = []
        for child in sorted(p.iterdir()):
            kind = "dir" if child.is_dir() else "file"
            entries.append(f"{kind:4}  {child.name}")
        return "\n".join(entries[:200]) if entries else "(empty)"
    except Exception as exc:
        return f"Error listing directory: {exc}"

def search_local(args: Dict[str, str]) -> str:
    query = (args.get("query") or "").strip()
    path_str = (args.get("path") or ".").strip()
    if not query:
        return "Error: search_local requires a 'query'."
    root = _resolve_safe(path_str)
    if root is None or not root.is_dir():
        return "Error: search root not found or outside allowed directory."
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    hits: List[str] = []
    try:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.startswith(".") or name.endswith((".pyc", ".gguf", ".safetensors")):
                    continue
                fpath = Path(dirpath) / name
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        rel = fpath.relative_to(_LOCAL_ROOT)
                        hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                        if len(hits) >= MAX_SEARCH_HITS:
                            return "\n".join(hits) + "\n[truncated]"
        return "\n".join(hits) if hits else f"No matches for '{query}'."
    except Exception as exc:
        return f"Search error: {exc}"

def list_available_models() -> list[tuple[str, str, Path]]:
    available = []
    for name, value in MODELS.items():
        if isinstance(value, (tuple, list)) and len(value) == 2:
            backend, path = value
        else:
            backend, path = "mlx", value
        path = Path(path).expanduser()
        if path.exists() and path.stat().st_size > 0:
            available.append((name, backend, path))
    return available