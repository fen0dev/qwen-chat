"""Tool definitions, parsing and execution."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from .github_client import GitHubClient


TOOLS_PROMPT = """
You have access to the following GitHub tools.
When you need information from GitHub, reply with a special tool call in this exact format:

TOOL_CALL: tool_name
ARG: key: value
ARG: key: value

Available tools:

1. list_repos
   - Lists the user's GitHub repositories
   - No arguments needed

2. get_file
   - Reads a file from a repository
   - Arguments:
     repo: owner/repo (example: torvalds/linux)
     path: path/to/file (example: README.md)

3. list_issues
   - Lists issues of a repository
   - Arguments:
     repo: owner/repo
     state: open | closed | all  (optional, default=open)

Only use a TOOL_CALL when you actually need GitHub data. Otherwise just answer normally.
""".strip()


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Parse a TOOL_CALL block from model output.

    Returns
    -------
    (tool_name, args) or None if no valid tool call is found.
    """
    if not text or "TOOL_CALL:" not in text:
        return None

    tool_name: Optional[str] = None
    args: Dict[str, str] = {}

    # Accept both "ARG: key: value" and "ARG: key = value"
    arg_pattern = re.compile(
        r"^ARG:\s*([a-zA-Z0-9_]+)\s*[:=]\s*(.+)$",
        re.IGNORECASE,
    )

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()

        if line.upper().startswith("TOOL_CALL:"):
            tool_name = line.split(":", 1)[1].strip().lower()
            continue

        match = arg_pattern.match(line)
        if match:
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            args[key] = value

    if tool_name:
        return tool_name, args
    return None


def execute_tool(
    github: GitHubClient,
    tool_name: str,
    args: Dict[str, str],
) -> str:
    """
    Execute a GitHub tool and return a human-readable result string.
    Never raises – all errors are turned into descriptive messages.
    """
    tool_name = tool_name.lower().strip()

    try:
        if tool_name == "list_repos":
            return github.list_repos()

        if tool_name == "get_file":
            repo = args.get("repo", "").strip()
            path = args.get("path", "").strip()
            if not repo or not path:
                return "Error: get_file requires both 'repo' and 'path' arguments."
            return github.get_file(repo, path)

        if tool_name == "list_issues":
            repo = args.get("repo", "").strip()
            state = args.get("state", "open").strip().lower()
            if not repo:
                return "Error: list_issues requires the 'repo' argument."
            return github.list_issues(repo, state)

        return (
            f"Unknown tool: '{tool_name}'. "
            "Available tools: list_repos, get_file, list_issues."
        )

    except Exception as exc:
        return f"Unexpected error while executing tool '{tool_name}': {exc}"
