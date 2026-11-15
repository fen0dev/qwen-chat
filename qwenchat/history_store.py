"""Session TTL prune + rolling summary + resume context."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .config import (
    SESSION_DIR,
    SUMMARY_PATH,
    SESSION_TTLS_DAYS,
    ROLLING_SUMMARY_MAX_CHARS,
)
from .plan import list_active_plans

RESUME_CONTEXT_MAX_CHARS = 8000

def ensure_history_dirs() -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

def prune_old_sessions(ttl_days: int = SESSION_TTLS_DAYS) -> int:
    """Delete session files older than set ttl_days. Returns count deleted"""

    ensure_history_dirs()
    cutoff = datetime.now() - timedelta(days=ttl_days)
    deleted = 0
    for path in SESSION_DIR.glob("*.json"):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if mtime < cutoff:
                path.unlink()
                deleted += 1
        except Exception:
            continue
    return deleted

def load_rolling_summary() -> str:
    if not SUMMARY_PATH.exists():
        return ""
    
    try:
        data = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))
        return str(data.get("text", "")).strip()
    except Exception:
        return ""
    
def save_rolling_summary(text: str) -> None:
    ensure_history_dirs()
    text = text.strip()
    if len(text) > ROLLING_SUMMARY_MAX_CHARS:
        text = text[-ROLLING_SUMMARY_MAX_CHARS:]
    SUMMARY_PATH.write_text(
        json.dumps(
            {"text": text, "updated_at": datetime.now().isoformat()},
            indent=2,
        ),
        encoding='utf-8',
    )

def append_rolling_note(note: str) -> None:
    note = note.strip()
    if not note:
        return
    current = load_rolling_summary()
    merged = f"{current}\n- {note}.".strip() if current else f"- {note}"
    save_rolling_summary()

def build_resume_context() -> str:
    """
    Compact context for a new session: rolling summary + active plans only.
    """
    parts: List[str] = []

    summary = load_rolling_summary()
    if summary:
        parts.append("Rolling summary of prior work\n" + summary)

    plans = list_active_plans()
    if plans:
        lines = ["Active plans:"]
        for p in plans:
            lines.append(p.render())
        parts.append("\n".join(lines))

    if not parts:
        return ""
    
    text = (
        "Prior context from previous session (do not invent beyond this:)\n\n"
        + "\n\n".join(parts)
    )

    if len(text) > RESUME_CONTEXT_MAX_CHARS:
        text = text[:RESUME_CONTEXT_MAX_CHARS].rstrip() + "\n[...truncated]"
    
    return text