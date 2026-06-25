"""
executor_tools.py — sandboxed tool implementations for agentic Executor/Evaluator.

All file paths are checked against the project root before any I/O.
run_bash runs in the project root with a per-command timeout.

Tool schemas follow the OpenAI function-calling format; LiteLLM translates
them for each provider (Anthropic, Zhipu, Gemini, etc.).
"""

import subprocess
from pathlib import Path


# ── Tool schemas (OpenAI function-calling format) ─────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file. Parent directories are created "
                "automatically. Path must be relative to the project root "
                "(e.g. 'workspace/detectors/fvg.py')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text content of a file. Path is relative to the project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a shell command in the project root. Returns combined stdout "
                "and stderr. Use 'uv run pytest' to run tests, 'uv run python -c ...' "
                "for quick checks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at a path relative to the project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to project root. Defaults to '.'.",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Toolkit implementation ────────────────────────────────────────────────────

class ExecutorToolkit:
    """
    Implements the tools agents can call during implementation and review.

    Shared by both the Executor and Evaluator — both need file I/O and pytest.
    Instantiated once per Orchestrator.run() call so the root path is stable.
    """

    BASH_TIMEOUT = 120  # seconds per command

    def __init__(self, root: Path):
        self.root = root.resolve()

    # ── Public dispatch entry point ────────────────────────────────────────────

    def dispatch(self, name: str, arguments: dict) -> str:
        """Route a tool call by name. Returns result string for the LLM."""
        handlers = {
            "write_file": self.write_file,
            "read_file": self.read_file,
            "run_bash": self.run_bash,
            "list_directory": self.list_directory,
        }
        fn = handlers.get(name)
        if fn is None:
            return f"ERROR: unknown tool '{name}'"
        try:
            return fn(**arguments)
        except TypeError as exc:
            return f"ERROR: bad arguments for '{name}': {exc}"
        except Exception as exc:
            return f"ERROR: {exc}"

    # ── Tool implementations ───────────────────────────────────────────────────

    def write_file(self, path: str, content: str) -> str:
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"

    def read_file(self, path: str) -> str:
        target = self._safe_path(path)
        if not target.exists():
            return f"ERROR: {path} does not exist"
        return target.read_text(encoding="utf-8")

    def run_bash(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.root),
                timeout=self.BASH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {self.BASH_TIMEOUT}s"

        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            parts.append(f"[exit {result.returncode}]")
        return "\n".join(parts) if parts else "(no output)"

    def list_directory(self, path: str = ".") -> str:
        target = self._safe_path(path)
        if not target.exists():
            return f"ERROR: {path} does not exist"
        if not target.is_dir():
            return f"ERROR: {path} is not a directory"
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{'DIR ' if e.is_dir() else 'FILE'} {e.name}" for e in entries]
        return "\n".join(lines) if lines else "(empty directory)"

    # ── Path safety ────────────────────────────────────────────────────────────

    def _safe_path(self, rel_path: str) -> Path:
        """Resolve and verify that the path stays within the project root."""
        resolved = (self.root / rel_path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Path escape: '{rel_path}' resolves outside project root")
        return resolved
