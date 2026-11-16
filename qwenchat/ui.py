"""Terminal UI helpers – colors, banners and messaging."""

from __future__ import annotations

import os
import sys
from typing import Optional


class Colors:
    """ANSI escape codes."""
    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"
    DIM: str = "\033[2m"
    ITALIC: str = "\033[3m"

    CYAN: str = "\033[96m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"
    MAGENTA: str = "\033[95m"
    RED: str = "\033[91m"
    WHITE: str = "\033[97m"
    GRAY: str = "\033[90m"
    BLUE: str = "\033[94m"


# Short alias used throughout the project
C = Colors


def supports_color() -> bool:
    """Return True if the current terminal likely supports ANSI colors."""
    if os.getenv("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return bool(sys.stdout.isatty())


def clear_screen() -> None:
    """Clear the terminal screen in a cross-platform way."""
    command = "cls" if os.name == "nt" else "clear"
    try:
        os.system(command)
    except Exception:
        print("\n" * 40)


def print_banner(model_name: str | None = None) -> None:
    """Print the application ASCII banner. optionally, show the selected model name."""
    if not supports_color():
        title = model_name or "QwenChat"
        print(f"QwenChat - {title}")
        return

    banner = f"""
{C.CYAN}{C.BOLD}
     ██████╗ ██╗    ██╗███████╗███╗   ██╗
    ██╔═══██╗██║    ██║██╔════╝████╗  ██║
    ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║
    ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║
    ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║
     ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝
{C.RESET}"""
    print(banner)

    if model_name:
        print(f"{C.MAGENTA}{C.BOLD}          {model_name}{C.RESET}")
    else:
        print(f"{C.MAGENTA}{C.BOLD}          Select a model{C.RESET}")

    print(f"{C.GRAY}        Apple Silicon · MLX / GGUF · GitHub Enabled{C.RESET}")


def print_separator(char: str = "─", width: int = 64) -> None:
    """Print a horizontal separator line."""
    color = C.GRAY if supports_color() else ""
    reset = C.RESET if supports_color() else ""
    print(f"{color}{char * width}{reset}")


def print_help() -> None:
    print(f"""
        {C.YELLOW}Commands:{C.RESET}
        {C.CYAN}clear{C.RESET}          → Reset conversation memory
        {C.CYAN}save{C.RESET}           → Save current conversation
        {C.CYAN}/root [path]{C.RESET}   → Show or set local file sandbox root
        {C.CYAN}/models{C.RESET}        → List models and switch by number
        {C.CYAN}load <file>{C.RESET}    → Load a saved conversation
        {C.CYAN}history{C.RESET}        → List saved conversations
        {C.CYAN}queue{C.RESET}          → Show queued messages
        {C.CYAN}/qclear{C.RESET}        → Clear the message queue
        {C.CYAN}/prune{C.RESET}         → Delete session files older than 7 days
        {C.CYAN}/resume{C.RESET}        → Show prior context (summary + active plans)
        {C.CYAN}github{C.RESET}         → Connect / reconnect GitHub
        {C.CYAN}/think on|off{C.RESET}  → Toggle visible reasoning
        {C.CYAN}exit{C.RESET}           → Quit the chat

        {C.YELLOW}Plan mode (model can use and call tools):{C.RESET}
        create_plan, complete_step, complete_plan, list_plans

        {C.ITALIC}You can also edit and set temperature, top_p, tokens, 
        max tool rounds, or check parameters generation by typing:{C.RESET}
            /temp 0.7
            /tokens 1024
            /top_p 0.9
            /rounds 5
            /params
        """
    )
    
def print_status(model_name: str | None, github_user: str | None, msg_count: int) -> None:
    parts = []
    if model_name:
        parts.append(f"Model: {model_name}")
    if github_user:
        parts.append(f"Github: {github_user}")
    parts.append(f"Messages: {msg_count}")

    print(f"{C.GRAY}{' · '.join(parts)}{C.RESET}")


def print_error(message: str) -> None:
    print(f"{C.RED}{message}{C.RESET}")


def print_success(message: str) -> None:
    print(f"{C.GREEN}{message}{C.RESET}")


def print_info(message: str) -> None:
    print(f"{C.DIM}{message}{C.RESET}")


def print_warning(message: str) -> None:
    print(f"{C.YELLOW}{message}{C.RESET}")


def user_prompt() -> str:
    return f"{C.GREEN}{C.BOLD}You ✦ {C.RESET}"


def assistant_prompt(model_name: str | None = None) -> str:
    label = "Qwen"
    if model_name:
        label = model_name
        if len(label) > 12:
            label = label[:9]
            
    return f"{C.CYAN}{C.BOLD}{label} ✦ {C.RESET}"
