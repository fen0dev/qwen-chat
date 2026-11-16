"""Plan tools: create, step, complete, list. No GitHub dependency."""

from __future__ import annotations

import re
from typing import Dict, List

from ..memory.plan import (
    create_plan,
    complete_plan,
    complete_step,
    list_active_plans,
)
from ..memory.history_store import append_rolling_note

MAX_PLAN_STEPS = 12
MAX_TITLE_LEN = 120
MAX_STEP_LEN = 200

def create_plan_tool(args: Dict[str, str]) -> str:
    title = (args.get("title") or "").strip()[:MAX_TITLE_LEN]
    if not title:
        return "Error: create_plan requires a non-empty 'title'."

    steps_raw = args.get("steps", "")
    if isinstance(steps_raw, list):
        raw_list = [str(s) for s in steps_raw]
    else:
        raw_list = [s.strip() for s in re.split(r"[;\n]", str(steps_raw)) if s.strip()]

    steps = [s[:MAX_STEP_LEN] for s in raw_list if s]
    if not steps:
        return "Error: create_plan requires at least one step in 'steps'."
    if len(steps) > MAX_PLAN_STEPS:
        return f"Error: maximum {MAX_PLAN_STEPS} steps allowed."

    plan = create_plan(title, steps)
    return f"Created plan.\n{plan.render()}"

def complete_plan_tool(args: Dict[str, str]) -> str:
    plan_id = (args.get("plan_id") or "").strip()
    if not plan_id:
        return "Error: complete_plan requires 'plan_id'."
    note = complete_plan(plan_id)
    if note is None:
        return f"Plan `{plan_id}` not found or already removed."
    append_rolling_note(note)
    return note

def complete_step_tool(args: Dict[str, str]) -> str:
    plan_id = (args.get("plan_id") or "").strip()
    step_id = (args.get("step_id") or "").strip()
    if not plan_id or not step_id:
        return "Error: complete_step requires 'plan_id' and 'step_id'."
    result = complete_step(plan_id, step_id)
    if result is None:
        return f"Plan `{plan_id}` not found or already removed."
    if "deleted" in result.lower() or "fully completed" in result.lower():
        append_rolling_note(result)
    return result

def list_plans_tool(args: Dict[str, str]) -> str:
    plans = list_active_plans()
    if not plans:
        return "No active plans."
    return "\n\n".join(p.render() for p in plans)