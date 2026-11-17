"""GitHub API client with robust error handling."""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from .config import (
    GITHUB_TOKEN_FILE,
    GITHUB_API_VERSION,
    GITHUB_TIMEOUT,
    GITHUB_MAX_REPOS,
    GITHUB_MAX_ISSUES,
)
from .ui import C, print_success, print_error, print_info, print_warning

logger = logging.getLogger(__name__)


class GitHubError(Exception):
    """Base exception for GitHub-related failures."""


class AuthenticationError(GitHubError):
    """Raised when authentication fails."""


class GitHubClient:
    """Thin wrapper around the GitHub REST API."""

    def __init__(self) -> None:
        self.token: Optional[str] = None
        self.username: Optional[str] = None
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "QwenChat/0.1.0",
        })

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def load_token(self) -> bool:
        """Load a previously saved token and validate it."""
        if not GITHUB_TOKEN_FILE.exists():
            return False

        try:
            self.token = GITHUB_TOKEN_FILE.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Could not read token file: %s", exc)
            return False

        if not self.token:
            return False

        self._apply_token()
        return self._validate()

    def login(self) -> bool:
        """Interactive login flow. Returns True on success."""
        print(f"\n{C.YELLOW}GitHub Authentication{C.RESET}")
        print_info("We need a Personal Access Token to access your repositories.")
        print_info("Create one here: https://github.com/settings/tokens")
        print_info("Recommended scope: repo\n")

        try:
            token = input(f"{C.CYAN}Paste your GitHub token: {C.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False

        if not token:
            print_error("No token provided.")
            return False

        self.token = token
        self._apply_token()

        if self._validate():
            try:
                GITHUB_TOKEN_FILE.write_text(self.token, encoding="utf-8")
                GITHUB_TOKEN_FILE.chmod(0o600)
            except OSError as exc:
                print_warning(f"Could not save token to disk: {exc}")

            print_success(f"Authenticated as {C.BOLD}{self.username}{C.RESET}")
            return True

        print_error("Invalid token or network error.")
        return False

    def _apply_token(self) -> None:
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    def _validate(self) -> bool:
        try:
            data = self._get("/user")
            self.username = data.get("login")
            return bool(self.username)
        except GitHubError:
            return False

    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"https://api.github.com{endpoint}"
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=GITHUB_TIMEOUT,
            )
        except Timeout:
            raise GitHubError("GitHub API request timed out.")
        except ConnectionError:
            raise GitHubError("Could not connect to GitHub. Check your internet connection.")
        except RequestException as exc:
            raise GitHubError(f"Network error while contacting GitHub: {exc}")

        if response.status_code == 401:
            raise AuthenticationError("Authentication failed. Token may be invalid or expired.")
        if response.status_code == 403:
            raise GitHubError("Access forbidden. Check token scopes or rate limits.")
        if response.status_code == 404:
            raise GitHubError("Resource not found.")
        if response.status_code >= 400:
            raise GitHubError(f"GitHub API error {response.status_code}: {response.text[:200]}")

        try:
            return response.json()
        except ValueError as exc:
            raise GitHubError(f"Invalid JSON received from GitHub: {exc}")

    # ------------------------------------------------------------------
    # Public high-level methods
    # ------------------------------------------------------------------

    def list_repos(self, limit: int = GITHUB_MAX_REPOS) -> str:
        try:
            repos: List[Dict[str, Any]] = self._get(
                "/user/repos",
                params={"sort": "updated", "per_page": limit},
            )
        except GitHubError as exc:
            return f"Failed to list repositories: {exc}"

        if not repos:
            return "You have no repositories."

        lines = [f"Found {len(repos)} repositories:\n"]
        for repo in repos:
            icon = "🔒" if repo.get("private") else "🌐"
            name = repo.get("full_name", "unknown")
            language = repo.get("language") or "N/A"
            lines.append(f"{icon} {name}  ({language})")

        return "\n".join(lines)

    def get_file(self, repo: str, path: str) -> str:
        if not repo or not path:
            return "Error: both 'repo' (owner/name) and 'path' are required."

        endpoint = f"/repos/{repo}/contents/{path.lstrip('/')}"

        try:
            data = self._get(endpoint)
        except GitHubError as exc:
            return f"Failed to fetch {repo}/{path}: {exc}"

        if isinstance(data, list):
            entries = [
                f"- {item.get('name', '?')} ({item.get('type', 'unknown')})"
                for item in data
            ]
            return f"Directory listing of {repo}/{path}:\n" + "\n".join(entries)

        encoding = data.get("encoding")
        content_b64 = data.get("content")

        if encoding != "base64" or not content_b64:
            return f"Cannot read file {repo}/{path}: unsupported encoding or empty content."

        try:
            raw = base64.b64decode(content_b64)
            text = raw.decode("utf-8", errors="replace")
        except Exception as exc:
            return f"Failed to decode file content: {exc}"

        return f"File: {repo}/{path}\n\n{text}"

    def list_issues(
        self,
        repo: str,
        state: str = "open",
        limit: int = GITHUB_MAX_ISSUES,
    ) -> str:
        if not repo:
            return "Error: 'repo' (owner/name) is required."

        state = state.lower()
        if state not in {"open", "closed", "all"}:
            state = "open"

        try:
            issues: List[Dict[str, Any]] = self._get(
                f"/repos/{repo}/issues",
                params={"state": state, "per_page": limit},
            )
        except GitHubError as exc:
            return f"Failed to list issues for {repo}: {exc}"

        if not issues:
            return f"No {state} issues found in {repo}."

        lines = [f"{state.title()} issues in {repo}:\n"]
        for issue in issues:
            number = issue.get("number", "?")
            title = issue.get("title", "No title")
            lines.append(f"#{number} – {title}")

        return "\n".join(lines)
    
    def search_code(self, query: str, limit: int = 30) -> str:
        try:
            data = self._get(
                "/search/code",
                params={"q": query, "per_page": limit},
            )
        except GitHubError as exc:
            return f"Failed to search code: {exc}"
        
        items = data.get("items", [])
        if not items:
            return f"No results found for query: {query}"
        
        lines = [f"Found {data.get('total_count', len(items))} results (showing {len(items)}):\n"]
        for item in items:
            repo = item.get("repository", {}).get("full_name", "?")
            path = item.get("path", "?")
            lines.append(f"- {repo}/{path}")
        return "\n".join(lines)
    
    def list_pull_requests(
        self,
        repo: str,
        state: str = "open",
        limit: int = 15,
    ) -> str:
        if not repo:
            return "Error: 'repo' (owner/name) is required."
        
        state = state.lower()
        if state not in {"open", "closed", "all"}:
            state = "open"

        try:
            prs: List[Dict[str, Any]] = self._get(
                f"/repos/{repo}/pulls",
                params={"state": state, "per_page": limit},
            )
        except GitHubError as exc:
            return f"Failed to list pull requests for {repo}: {exc}"
        
        if not prs:
            return f"No {state} pull requests found in {repo}"
        
        lines = [f"{state.title()} pull requests in {repo}:\n"]
        for pr in prs:
            number = pr.get("number", "?")
            title = pr.get("title", "No title")
            user = pr.get("user", {}).get("login", "?")
            lines.append(f"#{number} - {title} (by: {user})")

        return "\n".join(lines)
