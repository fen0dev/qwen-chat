"""Main application logic – chat loop with memory, streaming and GitHub tools."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

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
from .backends import create_backend, BaseBackend
from .conversation import Conversation
from .session import GenerationParams
from .input_queue import InputQueue
from .config import HISTORY_DIR

logger = logging.getLogger(__name__)

def build_system_message(params: GenerationParams) -> dict:
    base = """
    You are a senior software engineer. Your primary job is to help with real coding work: reading code, debugging, designing, refactoring, and implementing correctly.

    Core principles:
    - Prefer correctness over cleverness.
    - Be precise. Do not invent APIs, files, or behavior you have not verified.
    - When you are unsure, say so explicitly and state what you would need to verify.
    - Prefer small, safe, incremental changes over large speculative rewrites.
    - Match the existing style, abstractions, and conventions of the codebase when possible.
    - Distinguish clearly between facts you know, assumptions you are making, and recommendations.

    How to work on coding tasks:
    1. Restate the actual engineering problem in one or two sentences.
    2. Identify what must be inspected before answering.
    3. Use tools if repository evidence is required.
    4. Identify the root cause before suggesting fixes.
    5. Propose the smallest change that solves the problem cleanly.
    6. Call out risks, edge cases, and testing implications.
    7. When writing code, make it production-quality: clear names, proper error handling, and no placeholders unless explicitly requested.
    8. If the request is underspecified, ask for the missing constraint instead of inventing one.

    Response style:
    - Be direct and technical.
    - Avoid fluff and motivational language.
    - Use short sections or bullet points when they improve clarity.
    - When showing code, show only the relevant parts unless a full file is necessary.
    - When recommending a change, explain why, not just what.

    Tool use:
    - Use GitHub tools when repository context is needed.
    - Do not guess file contents if you can read them.
    - Prefer evidence from the repo over general knowledge when they conflict.
    """

    thinking = ""
    if params.show_thinking:
        thinking = """
        Reasoning policy:
        - For non-trivial coding problems, first reason inside <thinking>...</thinking>.
        - In the thinking block: inspect the problem, consider constraints, evaluate alternatives, and choose an approach.
        - After </thinking>, give the final answer only.
        - Do not put the final answer inside the thinking block.
        """

    return {
        "role": "system",
        "content": base + thinking + "\n\n" + TOOLS_PROMPT,
    }

def handle_runtime_command(cmd: str, params: GenerationParams) -> bool:
    """
    Process runtime parameter commands.
    Returns True if the command was handled (so we skip generation).
    """
    parts = cmd.strip().split()
    if not parts:
        return False

    command = parts[0].lower()

    try:
        if command in {"/params", "params"}:
            print_info(params.summary())
            return True

        if command in {"/temp", "temp"} and len(parts) == 2:
            params.temperature = float(parts[1])
            print_success(f"Temperature set to {params.temperature}")
            return True

        if command in {"/tokens", "tokens", "/token", "token"} and len(parts) == 2:
            params.max_tokens = int(parts[1])
            print_success(f"Max tokens set to {params.max_tokens}")
            return True

        if command in {"/top_p", "top_p"} and len(parts) == 2:
            params.top_p = float(parts[1])
            print_success(f"Top_p set to {params.top_p}")
            return True

        if command in {"/think", "think"} and len(parts) == 2:
            value = parts[1].lower()
            if value in {"on", "true", "1"}:
                params.show_thinking = True
                print_success("Thinking enabled")
            elif value in {"off", "false", "0"}:
                params.show_thinking = False
                print_success("Thinking disabled")
            else:
                print_error("Usage: /think on|off")
            return True

        if command in {"/rounds", "rounds"} and len(parts) == 2:
            params.max_tool_rounds = int(parts[1])
            print_success(f"Max tool rounds set to {params.max_tool_rounds}")
            return True

    except ValueError:
        print_error("Invalid value")
        return True

    return False

def stream_reply(
    engine,
    messages: list,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model_name: str | None = None,
) -> tuple[str, bool]:
    """
    Stream a reply.
    Returns (text, interrupted)
    """
    print(assistant_prompt(model_name), end="", flush=True)
    full: list[str] = []
    interrupted = False

    try:
        for chunk in engine.stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            print(f"{C.WHITE}{chunk}{C.RESET}", end="", flush=True)
            full.append(chunk)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n{C.YELLOW}⚡ Generation interrupted{C.RESET}")
    except Exception as exc:
        print_error(f"\nGeneration error: {exc}")
        return "", False

    if not interrupted:
        print()

    return "".join(full).strip(), interrupted

def generate_with_tools(
    engine,
    conv: Conversation,
    system_message: dict,
    params: GenerationParams,
    github: Optional[GitHubClient],
    model_name: Optional[str] = None,
) -> bool:
    """
    Run generation with multi-step tool calling.
    Returns True if interrupted by Ctrl+C.
    """
    messages = [system_message] + conv.messages

    for round_idx in range(params.max_tool_rounds + 1):
        reply, interrupted = stream_reply(
            engine,
            messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            model_name=model_name,
        )

        if not reply and not interrupted:
            return False

        # Always keep partial output if we got any text
        if reply:
            tool_call = parse_tool_call(reply)

            if interrupted:
                # Save partial answer and stop tool loop
                conv.add_assistant(reply)
                return True

            if not tool_call or github is None:
                conv.add_assistant(reply)
                return False

            # Tool call path
            tool_name, args = tool_call
            print(f"\n{C.BLUE}⚙ [{round_idx+1}] Calling tool: {tool_name}{C.RESET}")

            tool_result = execute_tool(github, tool_name, args)

            conv.add_assistant(reply)
            conv.add_user(
                f"Tool result for `{tool_name}`:\n\n{tool_result}\n\n"
                "Continue. Use another tool if needed, otherwise give the final answer."
            )
            messages = [system_message] + conv.messages
            continue

        # Interrupted with no text
        if interrupted:
            return True

    print_warning(f"Reached maximum tool rounds ({params.max_tool_rounds}). Stopping.")
    return False

def run(
    backend: str = "mlx",
    model_path: Path | None = None,
    model_name: str | None = None,
) -> None:
    if model_path is None:
        raise ValueError("model_path is required.")

    model_path = Path(model_path)

    clear_screen()
    print_banner(model_name)
    print_separator()

    # ------------------------------------------------------------------
    # GitHub
    # ------------------------------------------------------------------
    github: Optional[GitHubClient] = GitHubClient()

    if github.load_token():
        print_success(f"✓ GitHub connected as {C.BOLD}{github.username}{C.RESET}")
    else:
        print_warning("GitHub is not connected yet.")
        try:
            answer = input(f"{C.CYAN}Do you want to connect GitHub now? (y/n): {C.RESET}").strip().lower()
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
    print_info(f"⚙️  Loading {backend.upper()} model from:\n   {model_path}")
    print_info("🚀 Loading... please wait")

    try:
        engine = create_backend(backend, model_path)
    except Exception as exc:
        print_error(f"Failed to load model: {exc}")
        logger.exception("Model loading failed")
        sys.exit(1)

    print_success("✓ Model loaded successfully")
    print_separator()
    print_help()
    print_separator()

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------
    input_q = InputQueue()
    input_q.start()

    params = GenerationParams()
    conv = Conversation(title=model_name or "Chat")
    system_message = build_system_message(params)

    print_info("Input queue active — you can type while the model is generating.\n")

    try:
        while True:
            pending = input_q.pending()
            if pending:
                print(f"{C.GRAY}({pending} queued){C.RESET}")

            try:
                user_input = input_q.get(block=True)
            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{C.MAGENTA}Session ended. Goodbye.{C.RESET}\n")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            cmd = user_input.lower()

            # -------------------- commands --------------------
            if cmd in {"exit", "quit", "q"}:
                print(f"\n{C.MAGENTA}Session ended. Goodbye.{C.RESET}\n")
                break

            if cmd == "clear":
                conv.clear()
                print(f"{C.YELLOW}→ Conversation memory cleared.{C.RESET}\n")
                continue

            if cmd == "help":
                print_help()
                continue

            if cmd == "github":
                github = GitHubClient()
                github.login()
                continue

            if cmd == "save":
                path = conv.save()
                print_success(f"✓ Saved to {path}")
                continue

            if cmd.startswith("load "):
                name = user_input[5:].strip()
                path = HISTORY_DIR / name
                if not path.suffix:
                    path = path.with_suffix(".json")
                try:
                    conv = Conversation.load(path)
                    print_success(f"✓ Loaded: {conv.summary()}")
                except Exception as exc:
                    print_error(f"Failed to load: {exc}")
                continue

            if cmd == "history":
                files = Conversation.list_saved()
                if not files:
                    print_info("No saved conversations yet.")
                else:
                    print(f"{C.YELLOW}Saved conversations:{C.RESET}")
                    for f in files[:15]:
                        print(f"  {f.name}")
                continue

            if cmd in {"/queue", "queue"}:
                items = input_q.snapshot()
                if not items:
                    print_info("Queue is empty.")
                else:
                    print(f"{C.YELLOW}Queued messages:{C.RESET}")
                    for i, msg in enumerate(items, 1):
                        preview = msg if len(msg) < 80 else msg[:77] + "..."
                        print(f"  {i}. {preview}")
                continue

            if cmd in {"/qclear", "qclear"}:
                n = input_q.clear()
                print_success(f"✓ Cleared {n} queued message(s)")
                continue

            if handle_runtime_command(user_input, params):
                if cmd.startswith(("/think", "think")):
                    system_message = build_system_message(params)
                continue

            # -------------------- generation --------------------
            conv.add_user(user_input)

            interrupted = generate_with_tools(
                engine=engine,
                conv=conv,
                system_message=system_message,
                params=params,
                github=github,
                model_name=model_name,
            )

            if interrupted:
                pending = input_q.pending()
                if pending:
                    print_info(f"Continuing with next queued message ({pending} waiting)...")
                else:
                    print_info("No queued messages.")

    finally:
        input_q.stop()