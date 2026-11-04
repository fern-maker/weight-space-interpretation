"""Isolated tool process for interpretation sessions.

Server side (``python -I -m weightspace.tool_server``): reads one export JSON
line from stdin, then answers line-delimited JSON tool requests. It runs with
an empty environment in an empty temporary working directory, never receives
a path, and does not import torch or any module that knows about checkpoints
or ground truth. Optionally wrapped in macOS ``sandbox-exec`` that denies
network access and reads of the private artifact and results trees.

Client side: :class:`ToolServer`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FORBIDDEN_MODULES = ("torch", "weightspace.checkpointing", "weightspace.evaluation",
                     "weightspace.training", "weightspace.experiment", "anthropic")


def serve() -> None:  # pragma: no cover - exercised through ToolServer in tests
    from weightspace.tools import run_tool

    loaded = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    first = sys.stdin.readline()
    export = json.loads(first)
    hello = {"ok": True, "result": "ready", "forbidden_loaded": loaded, "cwd": os.getcwd(),
             "env_keys": sorted(os.environ)}
    sys.stdout.write(json.dumps(hello) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = run_tool(export, req["name"], req.get("args") or {})
            resp = {"ok": True, "result": res}
        except Exception as e:  # report errors to the agent as tool errors
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


def _sandbox_profile(deny_dirs: list[Path]) -> str:
    rules = "\n".join(f'(deny file-read* (subpath "{d.resolve()}"))' for d in deny_dirs)
    return f"(version 1)\n(allow default)\n(deny network*)\n{rules}\n"


class ToolServer:
    def __init__(self, export: dict, deny_dirs: list[Path] | None = None,
                 os_sandbox: bool = True):
        self.workdir = tempfile.mkdtemp(prefix="ws_tools_")
        cmd = [sys.executable, "-I", "-m", "weightspace.tool_server"]
        self.os_sandbox_active = False
        if os_sandbox and shutil.which("sandbox-exec") and deny_dirs:
            cmd = ["sandbox-exec", "-p", _sandbox_profile(deny_dirs), *cmd]
            self.os_sandbox_active = True
        self.proc = self._spawn(cmd)
        try:
            self.hello = self._handshake(export)
        except RuntimeError:
            if not self.os_sandbox_active:
                raise
            # nested sandboxes are not always permitted; fall back to plain isolation
            self.proc.kill()
            self.os_sandbox_active = False
            self.proc = self._spawn(cmd[3:])
            self.hello = self._handshake(export)
        if self.hello.get("forbidden_loaded"):
            self.close()
            raise RuntimeError(f"tool server loaded forbidden modules: {self.hello}")

    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=self.workdir,
                                env={"PATH": "/usr/bin:/bin"}, text=True)

    def _handshake(self, export: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        try:
            self.proc.stdin.write(json.dumps(export) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError("tool server died at start") from e
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"tool server failed to start: {err[-500:]}")
        return json.loads(line)

    def call(self, name: str, args: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps({"name": name, "args": args}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("tool server terminated")
        return json.loads(line)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self) -> ToolServer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":  # pragma: no cover
    serve()
