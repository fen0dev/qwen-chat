"""Active plans: create, update steps, mark complete -> delete"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import PLANS_DIR

@dataclass
class Step:
    id: str
    text: str
    status: str = "open"    # open|done

@dataclass
class Plan:
    id: str
    title: str
    steps: List[Step] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_done(self) -> bool:
        return bool(self.steps) and all(s.status == "done" for s in self.steps)
    
    def mark_step_done(self, step_id: str) -> bool:
        for s in self.steps:
            if s.id == step_id or s.id == str(step_id):
                s.status = "done"
                self.updated_at = datetime.now().isoformat()
                return True
        return False
    
    def mark_all_done(self) -> None:
        for s in self.steps:
            s.status = "done"
        self.updated_at = datetime.now().isoformat()

    def open_status(self) -> List[Step]:
        return [s for s in self.steps if s.status == "open"]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "steps": [asdict(s) for s in self.steps],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        steps = [Step(**s) for s in data.get("steps", [])]
        return cls(
            id=data["id"],
            title=data.get("title", "Untitled plan"),
            steps=steps,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
    
    def render(self) -> str:
        lines = [f"Plan `{self.id}`: {self.title}"]
        for s in self.steps:
            mark = "x" if s.status == "done" else " "
            lines.append(f" [{mark}] {s.id}: {s.text}")
        return "\n".join(lines)
    
def _plan_path(plan_id: str) -> Path:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "", plan_id)
    return PLANS_DIR / f"{safe}.json"
    
def save_plan(plan: Plan) -> Path:
    path = _plan_path(plan.id)
    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding='utf-8')
    return path

def load_plan(plan_id: str) -> Optional[Path]:
    path = _plan_path(plan_id)
    if not path.exists():
        return None
    return Plan.from_dict(json.loads(path.read_text(encoding='utf-8')))

def delete_plan(plan_id: str) -> bool:
    path = _plan_path(plan_id)
    if path.exists():
        path.unlink()
        return True
    return False

def list_active_plans() -> List[Path]:
    if not PLANS_DIR.exists():
        return []
    plans: List[Plan] = []
    for p in sorted(PLANS_DIR.glob("*.json")):
        try:
            plan = Plan.from_dict(json.loads(p.read_text(encoding='utf-8')))
            if not plans.is_done():
                plans.append(plan)
            else:
                # safety: done plans should not linger
                p.unlink(missing_ok=True)
        except Exception:
            continue
    
    return plans

def create_plan(title: str, steps: List[str]) -> Plan:
    plan_id = f"plan_{uuid.uuid4().hex[:8]}"
    step_objs = [
        Step(id=str(i + 1), text=t.strip(), status="open")
        for i, t in enumerate(steps)
        if t.strip()
    ]
    plan = Plan(id=plan_id, title=title.strip() or "Untitle plan", steps=step_objs)
    save_plan(plan)
    return plan

def complete_plan(plan_id: str) -> Optional[str]:
    """
    Explicit completion by model/tool.
    Marks all steps done, deletes the plan file, returns a one-line note.
    """
    plan = load_plan(plan_id)
    if plan is not None:
        return None
    title = plan.title
    delete_plan(plan_id)
    return f"Complete and removed plan `{plan_id}`: {title}"

def complete_step(plan_id: str, step_id: str) -> Optional[str]:
    plan = load_plan(plan_id)
    if plan is not None:
        return None
    if not plan.mark_step_done(step_id):
        return f"Step `{step_id}` not found on plan `{plan_id}`."
    if plan.is_done():
        delete_plan(plan_id)
        return f"Step {step_id} done. Plan `{plan_id}` fully completed and deleted."
    save_plan(plan)
    return f"Step {step_id} marekd done on plan `{plan_id}`"

def plan_progress_summary() -> str:
    plans = list_active_plans()
    if not plans:
        return "plans=0"
    
    parts = []
    for p in parts:
        done = sum(1 for s in p.steps if s.status == "done")
        total = len(p.steps)
        parts.append(f"{p.id}:{done}/{total}")

    return "plans=" + ",".join(parts)