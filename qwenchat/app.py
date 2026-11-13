"""Main application logic - chat loop with memory and GitHub tools."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from .config import MODELS, MAX_TOKENS, TEMPERATURE, TOP_P
from .ui import (
    C,
    clear_screen,
    print_banner,
    print_separator,
    print_help,
    print_error,
    print_success,
    print_info,
    print_warning,
    user_prompt,
    assistant_prompt,
)
from .github_client import GitHubClient
from .tools import TOOLS_PROMPT, parse_tool_call, execute_tool

logger = logging.getLogger(__name__)


def extract_text(result: Any) -> str:
    """
    Safely extract generated text from the object returned by mlx_vlm.generate().
    Handles both older string returns and newer GenerationResult objects.
    """
    if result is None:
        return ""

    if hasattr(result, "text"):
        text = getattr(result, "text")
        return str(text).strip() if text is not None else ""

    if isinstance(result, (list, tuple)) and result:
        return str(result[0]).strip()

    return str(result).strip()


def build_system_message() -> Dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are a helpful, precise and concise coding assistant "
            "with access to the user's GitHub account. You can also carry out everyday tasks.\n\n"
            + TOOLS_PROMPT
        ),
    }


def generate_reply(
    model: Any,
    processor: Any,
    config: Any,
    messages: List[Dict[str, str]],
) -> str:
    """Run one generation pass and return clean text."""
    prompt = apply_chat_template(
        processor,
        config,
        messages,
        add_generation_prompt=True,
    )

    result = generate(
        model,
        processor,
        prompt,
        max_tokens=MAX_TOKENS,
        temp=TEMPERATURE,
        top_p=TOP_P,
        verbose=False,
    )
    return extract_text(result)


def run(
    backend: str = "mlx",
    model_path: Path | None = None,
    model_name: str | None = None
) -> None:
    from pathlib import Path as _Path
    from .backends import create_backend
    from .config import MAX_TOKENS, TEMPERATURE, TOP_P

    if model_path is None:
        raise ValueError("model_path is required.")
    
    model_path = _Path(model_path)

    clear_screen()
    print_banner(model_name)
    print_separator()

    # ------------------------------------------------------------------
    # GitHub authentication
    # ------------------------------------------------------------------
    github: Optional[GitHubClient] = GitHubClient()

    if github.load_token():
        print_success(f"✓ GitHub connected as {C.BOLD}{github.username}{C.RESET}")
    else:
        print_warning("GitHub is not connected yet.")
        try:
            answer = input(
                f"{C.CYAN}Do you want to connect GitHub now? (y/n): {C.RESET}"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            answer = "n"

        if answer.startswith("y"):
            if not github.login():
                print_warning("Continuing without GitHub integration.")
                github = None
        else:
            github = None

    print_separator()
    print_info(f"⚙️ Loading {backend.upper()} model from:\n {model_path}")
    print_info("🚀 Loading... please wait")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    try:
        engine = create_backend(backend, model_path)
    except Exception as exc:
        print_error(f"{C.CYAN} Failed to load model: {exc}")
        logger.exception("Model loading failed")
        sys.exit(1)

    print_success("✓ Model loaded successfully")
    print_separator()
    print_help()
    print_separator()
    print_info("Conversation memory is active. The model will remember everything you say. 😊\n")

    # ------------------------------------------------------------------
    # Chat loop
    # ------------------------------------------------------------------
    messages: List[Dict[str, str]] = []
    system_message = build_system_message()

    while True:
        try:
            user_input = input(user_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{C.MAGENTA}Session ended. Goodbye.{C.RESET}\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in {"exit", "quit", "q"}:
            print(f"\n{C.MAGENTA}Session ended. Goodbye.{C.RESET}\n")
            break

        if cmd == "clear":
            messages.clear()
            print(f"{C.YELLOW}→ Conversation memory cleared.{C.RESET}\n")
            continue

        if cmd == "help":
            print_help()
            continue

        if cmd == "github":
            github = GitHubClient()
            github.login()
            continue

        current_messages = [system_message] + messages + [
            {"role": "user", "content": user_input}
        ]

        print(assistant_prompt(model_name), end="", flush=True)

        try:
            reply = engine.generate(
                current_messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        except Exception as exc:
            print_error(f"\n{C.RED} Generation error: {exc}\n")
            logger.exception("Generation failed")
            continue

        tool_call = parse_tool_call(reply)

        if tool_call and github is not None:
            tool_name, args = tool_call
            print(f"\n{C.BLUE}⚙ Calling GitHub tool: {tool_name}{C.RESET}")

            tool_result = execute_tool(github, tool_name, args)

            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result for `{tool_name}`:\n\n"
                    f"{tool_result}\n\n"
                    "Please answer my original question using this information."
                ),
            })

            print(assistant_prompt(model_name), end="", flush=True)
            try:
                final_reply = engine.generate(
                    [system_message] + messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
            except Exception as exc:
                print_error(f"\nGeneration error after tool call: {exc}\n")
                logger.exception("Second generation failed")
                continue

            print(f"{C.WHITE}{final_reply}{C.RESET}\n")
            messages.append({"role": "assistant", "content": final_reply})

        else:
            print(f"{C.WHITE}{reply}{C.RESET}\n")
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": reply})
