"""Main application logic - chat loop with memory, streaming and GitHub tools."""

from __future__ import annotations

import gc
import logging
import sys
import time
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
    assistant_prompt,
    user_prompt,
    print_tool_call,
    print_status_bar,
)
from .github_client import GitHubClient
from .tools import TOOLS_PROMPT, parse_tool_call, execute_tool
from .backends import create_backend
from .memory.conversation import Conversation
from .session import GenerationParams
from .input_queue import InputQueue
from .config import HISTORY_DIR
from .memory.history_store import (
    ensure_history_dirs,
    prune_old_sessions,
    build_resume_context,
)
from  .tools.local_tools import get_local_root, set_local_root, list_available_models
from .memory.plan import plan_progress_summary

logger = logging.getLogger(__name__)

PRESETS = {
    "coding": """
    You are a senior software engineer. Your job is to produce correct, minimal, production-quality code changes.

    Rules:
    - Prefer the smallest change that solves the problem cleanly.
    - Do not invent APIs, files, or behavior. If you have not verified something, say so.
    - Match existing style, naming, and abstractions.
    - When proposing code, show only the relevant parts unless a full file is necessary.
    - Call out risks, edge cases, and what should be tested.
    - If the request is underspecified, ask for the missing constraint instead of guessing.
    - Use tools when repository evidence is required. Do not guess file contents.
    """,
    "review": """
    You are a careful, senior code reviewer. Your job is to find real problems, not to rewrite for style.

    Focus on:
    - Correctness bugs and logic errors
    - Security and data-handling issues
    - Missing error handling or edge cases
    - Unclear or fragile abstractions
    - Missing tests for non-trivial behavior

    Rules:
    - Be direct and specific. Point to the exact issue.
    - Prefer “this is wrong because…” over vague suggestions.
    - Do not invent problems. If something is fine, say so briefly.
    - Suggest the smallest fix that addresses the issue.
    - If you need more context, ask or use tools instead of assuming.
    """,
    "debug": """
    You are a senior engineer debugging a real problem. Your job is to find the root cause, not to guess.

    Process:
    1. Restate the observed symptom in one sentence.
    2. Identify what must be inspected before concluding anything.
    3. Use tools or ask for evidence when repository state is required.
    4. Form a hypothesis and say what would confirm or refute it.
    5. Only then propose a fix.

    Rules:
    - Do not invent stack traces, logs, or file contents.
    - Prefer evidence over speculation.
    - Propose the smallest change that addresses the actual cause.
    - Call out what could still break after the fix.
    """,
    "concise": """
        You are a precise technical assistant. Answer briefly and directly.

    Rules:
    - Prefer short bullets or short paragraphs.
    - Lead with the answer, then the necessary detail.
    - Skip preamble and motivational language.
    - If the answer is uncertain, say so in one sentence.
    - When showing code, show only what is needed.
    """,
    "architect": """
        You are a senior engineer focused on design and trade-offs.

    Rules:
    - Restate the constraints and goals before proposing a design.
    - Prefer simple designs that can be implemented incrementally.
    - Explicitly name the trade-offs of the approach you recommend.
    - Call out what you would measure or revisit later.
    - Do not invent requirements. If something is missing, ask.
    - When suggesting code structure, keep it concrete and small.
    """
}

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

    Plans:
    - For multi-step work that must survive across sessions, use the create_plan tool.
    - When a step is finished, call complete_step.
    - When the whole plan is finished, call complete_plan (this DELETES the plan from disk to free up space).
    - Do not leave finished plans open. Completion must be explicit via tools.

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
        - After </thinking> at a new line, give the final answer only.
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
    start = time.perf_counter()

    try:
        for chunk in engine.stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            print(f"{C.TEXT}{chunk}{C.RESET}", end="", flush=True)
            full.append(chunk)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n{C.YELLOW}⚡ Generation interrupted{C.RESET}")
    except Exception as exc:
        print_error(f"\nGeneration error: {exc}")
        return "", False
    
    elapsed = time.perf_counter() - start
    text = "".join(full).strip()
    # rough token estimate
    tokens = max(1, len(text.split()))
    tps = tokens / elapsed if elapsed > 0 else 0

    if text and not interrupted:
        print(f"{C.GRAY}  ({tps:.1f} tokens){C.RESET}")

    return text, interrupted

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
            print_tool_call(round_idx + 1, tool_name)

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
        print_success(f"GitHub connected as {C.BOLD}{github.username}{C.RESET}")
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

    print_info("🐣 Running warm-up...")
    t0 = time.perf_counter()
    try:
        warm = engine.generate(
            [
                {
                    "role": "user",
                    "content": "Hi",
                },
            ],
            max_tokens=16,
            temperature=0.0,
            top_p=1.0,
        )

        dt = time.perf_counter() -t0
        print_success(f"✓ Warm-up ok ({dt:.1f}s)")
    except Exception as exc:
        print_warning(f"Warm-up failed (model may still work): {exc}")


    print_success("✓ Model loaded successfully")
    print_separator()
    print_help()
    print_separator()

    ensure_history_dirs()
    pruned = prune_old_sessions()
    if pruned:
        print_info(f"Pruned {pruned} sessions file(s) older than a week 🥵")

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------
    input_q = InputQueue()
    input_q.start()

    params = GenerationParams()
    conv = Conversation(title=model_name or "Chat")

    system_message = build_system_message(params)

    resume = build_resume_context()
    if resume:
        system_message = {
            "role": "system",
            "content": system_message["content"] + "\n\n" + resume,
        }

        print_info("👻 loaded prior context (summary + active plans).")

    print_info("Input queue active — you can type while the model is generating.\n")

    try:
        while True:
            pending = input_q.pending()
            if pending:
                print(f"{C.GRAY}({pending} queued){C.RESET}")

            try:
                print(user_prompt(), end="", flush=True)
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

            if cmd.startswith("/root") or cmd.startswith("root"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    print_info(f"Current local root: {get_local_root()}")
                    continue
                ok, msg = set_local_root(parts[1])
                if ok:
                    print_success(f"✓ {msg}")
                else:
                    print_error(msg)
                continue

            if cmd in {"/models", "models"}:
                available = list_available_models()
                if not available:
                    print_warning("No models found on disk.")
                    continue
                print(f"{C.YELLOW}Available models:{C.RESET}")
                for i, (name, backend, path) in enumerate(available, 1):
                    size = path.stat().st_size / (1024 * 1024)
                    size_str = f"{size:.0f}MB" if size < 1024 else f"{size/1024:.1f}GB"
                    current = " ←" if name == model_name else ""
                    print(f"  {C.CYAN}{i}{C.RESET}. {name}  {C.GRAY}[{backend}] {size_str}{current}{C.RESET}")
                try:
                    choice = input_q.get(block=True).strip()
                except Exception:
                    continue
                if not choice.isdigit():
                    print_info("Cancelled.")
                    continue
                idx = int(choice) - 1
                if not (0 <= idx < len(available)):
                    print_error("Invalid number.")
                    continue
                new_name, new_backend, new_path = available[idx]
                print_info(f"Switching to {new_name}...")
                try:
                    if hasattr(engine, "unload"):
                        engine.unload()
                    del engine
                    gc.collect()
                    engine = create_backend(new_backend, new_path)
                    model_name = new_name
                    print_success(f"✓ Switched to {new_name}")
                except Exception as exc:
                    print_error(f"Failed to switch model: {exc}")
                continue


            if cmd.startswith("/preset") or cmd.startswith("preset"):
                name = user_input.split(maxsplit=1)[-1].strip().lower()
                if name not in PRESETS:
                    print_error(f"Unknown preset. Available: {', '.join(PRESETS.keys())}")
                    continue
                # override the base system content (keep tools + resume)
                system_message = {
                    "role": "system",
                    "content": PRESETS[name] + "\n\n" + TOOLS_PROMPT,
                }
                print_success(f"✓ Preset set to '{name}'")
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

            if cmd in {"/prune", "prune"}:
                n = prune_old_sessions()
                print_success(f"✓ Pruned {n} old session file(s)")
                continue

            if cmd in {"/resume", "resume"}:
                resume = build_resume_context()
                if not resume:
                    print_info("No prior context.")
                else:
                    print(resume)
                continue

            last_user_msg: Optional[str] = None
            if cmd == "retry":
                if not last_user_msg:
                    print_info("Nothing to retry 🤷🏻‍♂️")
                    continue
                # remove last assistant message if present so we regenerate cleanly
                if conv.messages and conv.messages[-1].get("role") == "assistant":
                    conv.messages.pop()
                generate_with_tools(
                    engine=engine,
                    conv=conv,
                    system_message=system_message,
                    params=params,
                    github=github,
                    model_name=model_name,
                )
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

            if cmd in {"/status", "status"}:
                print_status_bar(
                    model_name=model_name,
                    github_user=getattr(github, "username", None) if github else None,
                    masg_count=conv.message_count,
                    queue=input_q.pending(),
                    plans=plan_progress_summary(),
                )
                continue

            if cmd.startswith("/export") or cmd.startswith("export"):
                name = user_input.split(maxsplit=1)[-1] if " " in user_input else "chat.md"
                if not name.endswith(".md"):
                    name += ".md"
                path = Path.home() / ".qwenchat" / name
                lines = [f"# {conv.title}\n"]
                for m in conv.messages:
                    role = m.get("role", "").upper()
                    lines.append(f"## {role}\n{m.get('content', '')}\n")
                path.write_text("\n".join(lines), encoding='utf-8')
                print_success(f"✓ Exported to {path}")
                continue

            if cmd in {"/copy", "copy"}:
                last = next((m["content"] for m in reversed(conv.messages) if m.get("role") == "assistant"), None)
                if not last:
                    print_info("No assistant message to copy.")
                    continue
                try:
                    import subprocess
                    subprocess.run(["pbcopy"], input=last.encode(), check=True)
                    print_success("✓ Copied last answer to clipboard")
                except Exception as exc:
                    print_error(f"Copy failed: {exc}")
                continue

            if cmd in {"/compact", "compact"}:
                if conv.messages < 8:
                    print_info("Conversation is still short - nothing to compact.")
                    continue
                # keep th elast 6 messages, summarize the rest into rolling summary
                keep = conv.messages[-6:]
                older = conv.messages[:-6]
                summary_lines = []
                for m in older:
                    role = m.get("role", "")
                    text = (m.get("content") or "")[:200].replace("\n", "")
                    summary_lines.append(f"{role}: {text}")
                note = "Compacted older turns:\n" + "\n".join(summary_lines[-20:])

                from .memory.history_store import append_rolling_note
                append_rolling_note(note)
                conv.messages = keep
                print_success(f"✓ Compacted. Kept last {len(keep)} messages.")
                continue


            if handle_runtime_command(user_input, params):
                if cmd.startswith(("/think", "think")):
                    system_message = build_system_message(params)
                continue

            # -------------------- generation --------------------
            last_user_msg = user_input
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