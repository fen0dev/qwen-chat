"""Clean terminal UI — refined palette, clear hierarchy."""

from __future__ import annotations

import os
import sys
from typing import Optional

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    # Refined palette
    ACCENT = "\033[38;5;111m"      # soft blue
    ACCENT_DIM = "\033[38;5;67m"
    SUCCESS = "\033[38;5;114m"     # soft green
    WARNING = "\033[38;5;179m"     # warm gold
    ERROR = "\033[38;5;167m"       # soft red
    MUTED = "\033[38;5;245m"       # mid gray
    TEXT = "\033[38;5;252m"        # near-white
    BORDER = "\033[38;5;238m"

    # Keep old names for compatibility
    CYAN = ACCENT
    GREEN = SUCCESS
    YELLOW = WARNING
    MAGENTA = "\033[38;5;176m"
    RED = ERROR
    WHITE = TEXT
    GRAY = MUTED
    BLUE = ACCENT

C = Colors

def supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())

def clear_screen() -> None:
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        print("\n" * 40)

def print_banner(model_name: str | None = None) -> None:
    if not supports_color():
        print(f"QwenChat  ·  {model_name or 'Select a model'}")
        return

    print()
    print(f"  {C.ACCENT}{C.BOLD}QwenChat{C.RESET}")
    print(f"  {C.MUTED}Local coding assistant · Apple Silicon{C.RESET}")
    if model_name:
        short = model_name if len(model_name) <= 48 else model_name[:45] + "…"
        print(f"  {C.TEXT}{short}{C.RESET}")
    print()

def print_separator(char: str = "─", width: int = 64) -> None:
    if supports_color():
        print(f"{C.BORDER}{char * width}{C.RESET}")
    else:
        print(char * width)

def print_status_bar(
    model_name: str | None = None,
    github_user: str | None = None,
    msg_count: int = 0,
    queue: int = 0,
    plans: str = "",
) -> None:
    parts = []
    if model_name:
        short = model_name.split("(")[0].strip()
        if len(short) > 28:
            short = short[:25] + "…"
        parts.append(f"{C.ACCENT}{short}{C.RESET}")
    if github_user:
        parts.append(f"{C.MUTED}gh:{github_user}{C.RESET}")
    parts.append(f"{C.MUTED}msgs:{msg_count}{C.RESET}")
    if queue:
        parts.append(f"{C.WARNING}q:{queue}{C.RESET}")
    if plans and plans != "plans=0":
        parts.append(f"{C.MUTED}{plans}{C.RESET}")
    print(f"  {'  '.join(parts)}")

def print_help() -> None:
    print()
    print(f"  {C.BOLD}Commands{C.RESET}")
    print(f"  {C.MUTED}{'─' * 52}{C.RESET}")
    rows = [
        ("/models", "List models and switch by number"),
        ("/root [path]", "Show or set local sandbox root"),
        ("/preset <name>", "coding · review · debug · concise · architect"),
        ("/status", "Session status"),
        ("/compact", "Compact older turns into memory"),
        ("/export [name]", "Export conversation to Markdown"),
        ("/copy", "Copy last answer"),
        ("retry", "Regenerate last answer"),
        ("save / load / history", "Conversation persistence"),
        ("/prune", "Delete sessions older than 7 days"),
        ("/resume", "Show prior context"),
        ("/params", "Show generation settings"),
        ("clear / help / exit", "Standard controls"),
    ]
    for cmd, desc in rows:
        print(f"  {C.ACCENT}{cmd:<22}{C.RESET} {C.MUTED}{desc}{C.RESET}")
    print()
    print(f"  {C.MUTED}Generation: /temp  /tokens  /top_p  /think  /rounds{C.RESET}")
    print()

def print_error(message: str) -> None:
    print(f"  {C.ERROR}✗ {message}{C.RESET}")

def print_success(message: str) -> None:
    print(f"  {C.SUCCESS}✓ {message}{C.RESET}")

def print_info(message: str) -> None:
    print(f"  {C.MUTED}{message}{C.RESET}")

def print_warning(message: str) -> None:
    print(f"  {C.WARNING}! {message}{C.RESET}")

def user_prompt() -> str:
    return f"\n  {C.SUCCESS}{C.BOLD}You{C.RESET}  "

def assistant_prompt(model_name: str | None = None) -> str:
    label = "Qwen"
    if model_name:
        label = model_name.split("(")[0].strip()
        if len(label) > 16:
            label = label[:14] + "…"
    return f"\n  {C.ACCENT}{C.BOLD}{label}{C.RESET}  "

def print_tool_call(round_idx: int, tool_name: str) -> None:
    print(f"\n  {C.ACCENT_DIM}⚙ [{round_idx}] {tool_name}{C.RESET}")

def print_tool_result(text: str, max_lines: int = 12) -> None:
    lines = text.strip().splitlines()
    shown = lines[:max_lines]
    for line in shown:
        print(f"  {C.MUTED}│ {line}{C.RESET}")
    if len(lines) > max_lines:
        print(f"  {C.MUTED}│ … ({len(lines) - max_lines} more lines){C.RESET}")