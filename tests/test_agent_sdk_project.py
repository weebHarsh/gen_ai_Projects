"""Offline tests for project 09 (Claude Agent SDK): --dry-run builds real ClaudeAgentOptions
and must never touch the network or need ANTHROPIC_API_KEY.

    LLM_PROVIDER=fake pytest -q tests/test_agent_sdk_project.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ["LLM_PROVIDER"] = "fake"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "projects"


def load(path: Path, name: str):
    """Import a project file by path (project folders start with digits, so not packages)."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load(P / "09_claude_agent_sdk" / "main.py", "p09")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def boom(*a, **k):
        raise AssertionError("network/API call attempted in dry-run")

    import socket
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(asyncio, "run", boom)  # run() is only reached on a real call


@pytest.mark.parametrize("cmd,model,tools", [
    (["ask", "How does retrieval work?"], "claude-sonnet-5-5", "['Read', 'Grep', 'Glob']"),
    (["review"], "claude-opus-5-5", "'Write'"),
    (["custom-tool"], "claude-sonnet-5-5", "mcp__acme__search_acme_policies"),
    (["team", "where is bm25"], "claude-opus-5-5", "'Agent'"),
])
def test_dry_run(mod, capsys, cmd, model, tools):
    assert mod.main([*cmd, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "no API call made" in out and f"model:           {model}" in out and tools in out
    assert "permission_mode:" in out


def test_dry_run_flag_before_subcommand(mod, capsys):
    assert mod.main(["--dry-run", "ask", "x"]) == 0
    assert "no API call made" in capsys.readouterr().out


def test_team_has_haiku_subagent(mod):
    agents = mod.build_options("team").agents
    assert agents["file-finder"].model == "claude-haiku-5-5"


def test_ask_is_read_only(mod):
    o = mod.build_options("ask")
    assert set(o.allowed_tools) == {"Read", "Grep", "Glob"} and o.permission_mode == "dontAsk"


def test_review_write_guard(mod):
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    run = asyncio.new_event_loop().run_until_complete
    o = mod.build_options("review")
    assert "Write" not in o.allowed_tools  # else can_use_tool would be bypassed
    ok = run(o.can_use_tool("Write", {"file_path": str(ROOT / "REVIEW.md")}, None))
    bad = run(o.can_use_tool("Write", {"file_path": str(ROOT / "main.py")}, None))
    bash = run(o.can_use_tool("Bash", {"command": "ls"}, None))
    assert isinstance(ok, PermissionResultAllow)
    assert isinstance(bad, PermissionResultDeny) and isinstance(bash, PermissionResultDeny)
    hook = o.hooks["PreToolUse"][0].hooks[0]
    denied = run(hook({"tool_name": "Write", "tool_input": {"file_path": "/etc/passwd"}}, "id", None))
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert run(hook({"tool_name": "Write", "tool_input": {"file_path": "REVIEW.md"}}, "id", None)) == {}
