"""Tests for executor_tools.py — the write-boundary guard.

The Executor/Evaluator toolkit may READ and run bash anywhere under the repo
root, but WRITES are fenced to an allowlist of directories. These tests pin the
guard that stops an agent from rewriting infrastructure (config/prompts.yaml,
main.py, src/, .kyros_state.json) while still letting it write the sanctioned
surface (workspace/ and tests/).

Writes go through ``dispatch`` — the same path the LLM's tool calls take — which
turns a boundary violation into an ``ERROR:`` string handed back to the model,
rather than an exception that aborts the run.
"""

import pytest

from kyros.core.executor_tools import ExecutorToolkit


@pytest.fixture
def fenced_toolkit(tmp_path):
    """Toolkit rooted at a tmp repo, writes fenced to workspace/ + tests/."""
    (tmp_path / "workspace" / "detectors").mkdir(parents=True)
    (tmp_path / "tests" / "phase2").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "src" / "kyros").mkdir(parents=True)
    (tmp_path / "config" / "prompts.yaml").write_text("real: content\n")
    return ExecutorToolkit(tmp_path, write_roots=["workspace", "tests"])


def _write(toolkit, path, content):
    """Invoke write_file the way the LLM does — through dispatch."""
    return toolkit.dispatch("write_file", {"path": path, "content": content})


# ── Allowed writes ────────────────────────────────────────────────────────────

def test_write_into_workspace_allowed(fenced_toolkit, tmp_path):
    result = _write(fenced_toolkit, "workspace/detectors/fib.py", "x = 1\n")
    assert result.startswith("OK")
    assert (tmp_path / "workspace" / "detectors" / "fib.py").read_text() == "x = 1\n"


def test_write_into_tests_allowed(fenced_toolkit, tmp_path):
    result = _write(fenced_toolkit, "tests/phase2/test_fib.py", "def test_x(): pass\n")
    assert result.startswith("OK")
    assert (tmp_path / "tests" / "phase2" / "test_fib.py").exists()


def test_write_creates_nested_dirs_within_root(fenced_toolkit, tmp_path):
    _write(fenced_toolkit, "workspace/trading/new/mod.py", "y = 2\n")
    assert (tmp_path / "workspace" / "trading" / "new" / "mod.py").exists()


# ── Denied writes (the actual Phase 2B incident) ──────────────────────────────

def test_write_to_config_prompts_denied(fenced_toolkit, tmp_path):
    result = _write(fenced_toolkit, "config/prompts.yaml", "[UNCHANGED — gutted]\n")
    assert result.startswith("ERROR")
    assert "Write denied" in result
    # File on disk is untouched.
    assert (tmp_path / "config" / "prompts.yaml").read_text() == "real: content\n"


def test_write_to_main_py_denied(fenced_toolkit):
    result = _write(fenced_toolkit, "main.py", "print('hijacked')\n")
    assert result.startswith("ERROR")
    assert "Write denied" in result


def test_write_to_src_denied(fenced_toolkit):
    result = _write(fenced_toolkit, "src/kyros/core/orchestrator.py", "pass\n")
    assert result.startswith("ERROR")
    assert "Write denied" in result


def test_write_to_state_file_denied(fenced_toolkit):
    result = _write(fenced_toolkit, ".kyros_state.json", "{}\n")
    assert result.startswith("ERROR")
    assert "Write denied" in result


def test_write_escape_via_traversal_denied(fenced_toolkit, tmp_path):
    # workspace/../config still resolves outside the write roots.
    result = _write(fenced_toolkit, "workspace/../config/prompts.yaml", "bad\n")
    assert result.startswith("ERROR")
    assert (tmp_path / "config" / "prompts.yaml").read_text() == "real: content\n"


def test_write_outside_root_denied(fenced_toolkit):
    result = _write(fenced_toolkit, "../../etc/evil", "bad\n")
    assert result.startswith("ERROR")


# ── Reads stay repo-wide ──────────────────────────────────────────────────────

def test_read_outside_write_roots_still_allowed(fenced_toolkit):
    """The Evaluator must still be able to read infrastructure it cannot write."""
    content = fenced_toolkit.dispatch("read_file", {"path": "config/prompts.yaml"})
    assert content == "real: content\n"


def test_list_directory_outside_write_roots_still_allowed(fenced_toolkit):
    out = fenced_toolkit.dispatch("list_directory", {"path": "config"})
    assert "prompts.yaml" in out


# ── Backwards-compatible default (no fence) ───────────────────────────────────

def test_default_write_roots_unrestricted_within_repo(tmp_path):
    (tmp_path / "config").mkdir()
    tk = ExecutorToolkit(tmp_path)  # write_roots=None
    result = tk.dispatch("write_file", {"path": "config/anything.txt", "content": "ok\n"})
    assert result.startswith("OK")
