"""Tool registry and execution. GitHub tools stay here; local and plan tools are imported."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ..github_client import GitHubClient
from .local_tools import read_file, list_dir, search_local
from .plan_tools import (
    create_plan_tool,
    complete_plan_tool,
    complete_step_tool,
    list_plans_tool,
)

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable
    needs_github: bool = True

# ---------------------------------------------------------------------------
# GitHub tool handlers (keep the ones you already have)
# ---------------------------------------------------------------------------

def _list_repos(github: GitHubClient, args: Dict[str, str]) -> str:
    repos = github.list_repos(limit=30)
    if not repos:
        return "No repositories found."
    lines = []
    for r in repos:
        visibility = "private" if r.get("private") else "public"
        lines.append(f"- {r['full_name']} ({visibility}) — {r.get('description') or 'no description'}")
    return "\n".join(lines)

def _get_file(github: GitHubClient, args: Dict[str, str]) -> str:
    repo = args.get("repo", "").strip()
    path = args.get("path", "").strip()
    if not repo or not path:
        return "Error: get_file requires 'repo' and 'path'."
    return github.get_file_content(repo, path)

def _list_issues(github: GitHubClient, args: Dict[str, str]) -> str:
    repo = args.get("repo", "").strip()
    if not repo:
        return "Error: list_issues requires 'repo'."
    issues = github.list_issues(repo, limit=15)
    if not issues:
        return f"No open issues in {repo}."
    lines = [f"- #{i['number']} {i['title']}" for i in issues]
    return "\n".join(lines)

def _search_code(github: GitHubClient, args: Dict[str, str]) -> str:
    query = args.get("query", "").strip()
    repo = args.get("repo", "").strip()
    if not query:
        return "Error: search_code requires 'query'."
    return github.search_code(query, repo or None)

def _list_prs(github: GitHubClient, args: Dict[str, str]) -> str:
    repo = args.get("repo", "").strip()
    if not repo:
        return "Error: list_prs requires 'repo'."
    prs = github.list_pull_requests(repo, limit=15)
    if not prs:
        return f"No open pull requests in {repo}."
    lines = [f"- #{p['number']} {p['title']}" for p in prs]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS: Dict[str, Tool] = {
    # GitHub
    "list_repos": Tool(
        name="list_repos",
        description="List the authenticated user's repositories.",
        parameters={},
        handler=_list_repos,
        needs_github=True,
    ),
    "get_file": Tool(
        name="get_file",
        description="Read a file from a GitHub repository.",
        parameters={"repo": "owner/repo", "path": "path/to/file"},
        handler=_get_file,
        needs_github=True,
    ),
    "list_issues": Tool(
        name="list_issues",
        description="List open issues in a repository.",
        parameters={"repo": "owner/repo"},
        handler=_list_issues,
        needs_github=True,
    ),
    "search_code": Tool(
        name="search_code",
        description="Search code in a repository or across GitHub.",
        parameters={"query": "search string", "repo": "owner/repo (optional)"},
        handler=_search_code,
        needs_github=True,
    ),
    "list_prs": Tool(
        name="list_prs",
        description="List open pull requests in a repository.",
        parameters={"repo": "owner/repo"},
        handler=_list_prs,
        needs_github=True,
    ),
    # Local
    "read_file": Tool(
        name="read_file",
        description="Read a local file (sandboxed to the allowed root).",
        parameters={"path": "Relative path under the allowed root"},
        handler=read_file,
        needs_github=False,
    ),
    "list_dir": Tool(
        name="list_dir",
        description="List contents of a local directory (sandboxed).",
        parameters={"path": "Relative directory path (default '.')"},
        handler=list_dir,
        needs_github=False,
    ),
    "search_local": Tool(
        name="search_local",
        description="Search for a text string in local files under the allowed root.",
        parameters={"query": "Text to search for", "path": "Directory to search (default '.')"},
        handler=search_local,
        needs_github=False,
    ),
    # Plans
    "create_plan": Tool(
        name="create_plan",
        description="Create an active multi-step plan that survives across sessions until explicitly completed.",
        parameters={
            "title": "Short plan title (max 120 chars)",
            "steps": "Semicolon-separated step descriptions (max 12 steps)",
        },
        handler=create_plan_tool,
        needs_github=False,
    ),
    "complete_step": Tool(
        name="complete_step",
        description="Mark one plan step done. If it was the last open step, the plan is deleted.",
        parameters={"plan_id": "Plan id", "step_id": "Step id"},
        handler=complete_step_tool,
        needs_github=False,
    ),
    "complete_plan": Tool(
        name="complete_plan",
        description="Explicitly mark a plan completed and DELETE it from disk.",
        parameters={"plan_id": "Plan id to complete and remove"},
        handler=complete_plan_tool,
        needs_github=False,
    ),
    "list_plans": Tool(
        name="list_plans",
        description="List all active (not yet completed) plans.",
        parameters={},
        handler=list_plans_tool,
        needs_github=False,
    ),
}

# ---------------------------------------------------------------------------
# Prompt fragment injected into the system message
# ---------------------------------------------------------------------------

TOOLS_PROMPT = """
You can call tools by emitting a block in this exact format:

TOOL_CALL: tool_name
ARG: key=value
ARG: key2=value2
END_TOOL

Available tools:
""" + "\n".join(
    f"- {t.name}: {t.description} (params: {', '.join(t.parameters.keys()) or 'none'})"
    for t in TOOLS.values()
) + """

Rules:
- Only call a tool when you need information you do not already have.
- After receiving a tool result, continue and either call another tool or give the final answer.
- Do not invent tool results.
"""

# ---------------------------------------------------------------------------
# Parsing and execution
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"TOOL_CALL:\s*(\w+)\s*(.*?)\s*END_TOOL",
    re.DOTALL | re.IGNORECASE,
)

def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip().lower()
    body = m.group(2)
    args: Dict[str, str] = {}
    for line in body.splitlines():
        line = line.strip()
        if line.upper().startswith("ARG:"):
            payload = line[4:].strip()
            if "=" in payload:
                k, v = payload.split("=", 1)
                args[k.strip()] = v.strip()
    return name, args

MAX_TOOL_RESULT_CHARS = 8000

def execute_tool(
    github: Optional[GitHubClient],
    tool_name: str,
    args: Dict[str, str],
) -> str:
    tool = TOOLS.get(tool_name.lower().strip())
    if tool is None:
        available = ", ".join(TOOLS.keys())
        return f"Unknown tool: '{tool_name}'. Available tools: {available}."

    try:
        if tool.needs_github:
            if github is None:
                return f"Tool '{tool_name}' requires a GitHub connection."
            result = tool.handler(github, args)
        else:
            result = tool.handler(args)
    except Exception as exc:
        result = f"Unexpected error while executing tool '{tool_name}': {exc}"

    if len(result) > MAX_TOOL_RESULT_CHARS:
        result = result[:MAX_TOOL_RESULT_CHARS].rstrip() + "\n\n[tool result truncated]"
    return result

__all__ = ["TOOLS", "TOOLS_PROMPT", "parse_tool_call", "execute_tool"]