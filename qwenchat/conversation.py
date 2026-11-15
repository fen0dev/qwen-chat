"""Conversation persistence and management."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import HISTORY_DIR

@dataclass
class Conversation:
    """In-mempry conversation with optional persistence."""

    messages: List[Dict[str, str]] = field(default_factory=list)
    title: str = "Untitled"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.updated_at = datetime.now()

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.updated_at = datetime.now()

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = datetime.now()

    @property
    def message_count(self) -> int:
        return len(self.messages)
    
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, name: Optional[str] = None) -> Path:
        """
        Save the conversation to disk.
        Returns the path of the written file.
        """
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        if name:
            # sanitize filename
            safe = re.sub(r"[^\w\-_.]", "", name).strip().replace(" ", "_")
            filename = f"{safe}.json"
        elif self.path:
            filename = self.path.name
        else:
            timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
            filename = f"qwen-chat_{timestamp}.json"

        path = HISTORY_DIR / filename

        data = {
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": self.messages,
        }

        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=True),
            encoding='utf-8',
        )

        self.path = path
        return path
    
    @classmethod
    def load(cls, path: Path) -> "Conversation":
        """Load a conversation from a JSON file."""
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Conversation file not found: {path}")
        
        data = json.loads(path.read_text(encoding='utf-8'))

        created = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        updated = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))

        return cls(
            messages=data.get("messages", []),
            title=data.get("title", path.stem),
            created_at=created,
            updated_at=updated,
            path=path,
        )
    
    @classmethod
    def list_saved(cls) -> List[Path]:
        """Return all saved conversation files, newest first."""
        if not HISTORY_DIR.exists():
            return []
        files = sorted(
            HISTORY_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return files
    
    def summery(self) -> str:
        """Short human-readable summary."""
        return(
            f"{self.title}  ·   {self.message_count} messages   ·   "
            f"Updated {self.updated_at.strftime('%Y-%m-%d %H:%M')}"
        )