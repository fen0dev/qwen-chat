"""Tool definitions, parsing and execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List

from .github_client import GitHubClient

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable[[GitHubClient, Dict[str, str]], str]

    def usage(self) -> str:
        lines = [f"{self.name}", f"   {self.description}"]
        if self.parameters:
            lines.append("  Arguments:")
            for k, v in self.parameters.items():
                lines.append(f"    - {k}: {v}")
        else:
            lines.append("  Arguments: none")
        return "\n".join(lines)

# ----------------------------------------------------------------------
# Tool implementations
# ----------------------------------------------------------------------

def _list_repos(github: GitHubClient, args: Dict[str, str]) -> str:
    return github.list_repos()

def _get_file(github: GitHubClient, args: Dict[str, str]) -> str:
    repo = args.get("repo", "").strip()
    path = args.get("path", "").strip()

    if not repo or not path:
        return "Error: get_file requires both 'repo' (owner/name) and 'path'."
    return github.get_file(repo, path)

def _list_issues(github: GitHubClient, args: Dict[str, str]) -> str:
    repo = args.get("repo", "").strip()
    state = args.get("state", "open").strip().lower() or "open"

    if not repo:
        return "Error: list_issues requires 'repo' argument."
    return github.list_issues(repo, state)

def _search_code(github: GitHubClient, args: Dict[str, str]) -> str:
    query = args.get("query", "").strip()

    if not query:
        return "Error: search_code requires a 'query' argument."
    return github.search_code(query)

def _list_prs(github: GitHubClient, args: Dict[str, str]) -> str:
    repo = args.get("repo", "").strip()
    state = args.get("state", "open").strip().lower() or "open"

    if not repo:
        return "Error: list_pull_requests requires the 'repo' argument."
    return github.list_pull_requests(repo, state)

# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

TOOLS: Dict[str, Tool] = {
    "list_repos": Tool(
        name="list_repos",
        description="Lists the authenticated user's GitHub repositories (most recently updated first).",
        parameters={},
        handler=_list_repos,
    ),
    "get_file": Tool(
        name="get_file",
        description="Reads the content of a file from a repository.",
        parameters={
            "repo": "owner/repo (example: torvalds/linux)",
            "path": "path/to/file (example: README.md)",
        },
        handler=_get_file,
    ),
    "list_issues": Tool(
        name="list_issues",
        description="Lists issues of a repository.",
        parameters={
            "repo": "owner/repo",
            "state": "open | closed | all  (optional, default=open)",
        },
        handler=_list_issues,
    ),
    "search_code": Tool(
        name="search_code",
        description="Searches code across the user's repositories.",
        parameters={
            "query": "GitHub code search query (example: language:python filename:main.py)",
        },
        handler=_search_code,
    ),
    "list_prs": Tool(
        name="list_prs",
        description="Lists pull requests of a repository.",
        parameters={
            "repo": "owner/repo",
            "state": "open | closed | all  (optional, default=open)",
        },
        handler=_list_prs,
    ),
}

def build_tools_prompt() -> str:
    """Generate the system prompt section that describes available tools."""
    lines = [
        "You have access to the following GitHub tools.",
        "When you need information from GitHub, reply with a special tool call in this exact format:",
        "",
        "TOOL_CALL: tool_name",
        "ARG: key: value",
        "ARG: key: value",
        "",
        "Available tools:",
        "",
    ]
    for i, tool in enumerate(TOOLS.values(), start=1):
        lines.append(f"{i}. {tool.usage()}")
        lines.append("")
    lines.append(
        "Only use a TOOL_CALL when you actually need GitHub data. Otherwise answer normally."
    )
    return "\n".join(lines)

TOOLS_PROMPT = build_tools_prompt()

# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

_ARG_PATTERN = re.compile(
    r"^ARG:\s*([a-zA-Z0-9_]+)\s*[:=]\s*(.+)$",
    re.IGNORECASE,
)

def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Parse a TOOL_CALL block from model output.
    Returns (tool_name, args) or None.
    """
    if not text or "TOOL_CALL:" not in text.upper():
        return None

    tool_name: Optional[str] = None
    args: Dict[str, str] = {}

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith("TOOL_CALL:"):
            tool_name = line.split(":", 1)[1].strip().lower()
            continue

        match = _ARG_PATTERN.match(line)
        if match:
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            args[key] = value

    if tool_name and tool_name in TOOLS:
        return tool_name, args
    return None

def execute_tool(
    github: GitHubClient,
    tool_name: str,
    args: Dict[str, str],
) -> str:
    """Execute a tool. Never raises — all errors become readable strings."""
    tool = TOOLS.get(tool_name.lower().strip())
    if tool is None:
        available = ", ".join(TOOLS.keys())
        return f"Unknown tool: '{tool_name}'. Available tools: {available}."

    try:
        return tool.handler(github, args)
    except Exception as exc:
        return f"Unexpected error while executing tool '{tool_name}': {exc}"