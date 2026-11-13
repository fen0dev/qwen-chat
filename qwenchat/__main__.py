"""Entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

from .config import MODELS
from .ui import (
    C, clear_screen, print_banner, print_separator,
    print_error, print_info, print_success, print_warning,
)
from .app import run


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
        stream=sys.stderr,
    )


def find_available_models():
    available = []
    for name, value in MODELS.items():
        if isinstance(value, (tuple, list)) and len(value) == 2:
            backend, path = value
        else:
            backend, path = "mlx", value

        path = Path(path)
        if path.exists():
            available.append((name, backend, path))   # ← 3 items
    return available


def prompt_model_selection():
    """
    Shows menu and returns:
        (display_name, backend, path)
    """
    available = find_available_models()

    print_separator()
    print(f"{C.YELLOW}{C.BOLD}Select a model:{C.RESET}\n")

    if available:
        for idx, (name, backend, path) in enumerate(available, start=1):
            tag = f"[{backend.upper()}]"
            print(f"  {C.CYAN}{idx}{C.RESET}. {name}  {C.GRAY}{tag}{C.RESET}")
            print(f"     {C.GRAY}{path}{C.RESET}")
    else:
        print_warning("  No pre-configured models found on disk.")

    print(f"\n  {C.CYAN}c{C.RESET}. Enter a custom path")
    print(f"  {C.CYAN}q{C.RESET}. Quit\n")

    while True:
        try:
            choice = input(f"{C.GREEN}Your choice: {C.RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)

        if choice == "q":
            print(f"\n{C.MAGENTA}Goodbye.{C.RESET}\n")
            sys.exit(0)

        if choice == "c":
            try:
                custom = input(f"{C.CYAN}Path to model: {C.RESET}").strip()
                backend = input(f"{C.CYAN}Backend (mlx / gguf): {C.RESET}").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print()
                continue

            custom_path = Path(custom).expanduser().resolve()
            if not custom_path.exists():
                print_error(f"Path does not exist: {custom_path}")
                continue
            if backend not in ("mlx", "gguf"):
                print_error("Backend must be 'mlx' or 'gguf'")
                continue

            display_name = custom_path.name
            print_success(f"✓ Using custom model ({backend}): {custom_path}")
            return display_name, backend, custom_path   # ← 3 valori

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                name, backend, path = available[idx]
                print_success(f"✓ Selected: {name}")
                return name, backend, path              # ← 3 values
            print_error("Invalid number.")
            continue

        print_error("Invalid choice.")


def main() -> None:
    configure_logging()
    clear_screen()
    print_banner()

    display_name, backend, model_path = prompt_model_selection()

    try:
        run(backend=backend, model_path=model_path, model_name=display_name)
    except KeyboardInterrupt:
        print(f"\n\n{C.MAGENTA}Interrupted. Goodbye.{C.RESET}\n")
        sys.exit(0)
    except Exception as exc:
        print_error(f"\nFatal error: {exc}")
        logging.exception("Unhandled exception")
        sys.exit(1)


if __name__ == "__main__":
    main()