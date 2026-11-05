"""Minimal MCP stdio server exposing the analysis tools to Claude Code.

Run as ``python -I -m weightspace.mcp_server <session_mcp_dir>`` with an empty
environment (the backend launches it through ``/usr/bin/env -i``). The
directory holds only ``export.json`` (absent for architecture-only sessions)
and ``spec.json`` (protocol, condition, candidate order). The server writes
``server_info.json``, ``calls.jsonl``, ``submit_rejections.jsonl``,
``submission.json`` and, on a leak, ``leak.json`` into the same directory.

Protocol: JSON-RPC 2.0, one message per line (MCP stdio transport). Methods:
``initialize``, ``notifications/initialized``, ``ping``, ``tools/list``,
``tools/call``. Same import restrictions as ``tool_server``: no torch, no
checkpoint, training, evaluation or ground-truth code.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .anonymization import scan_for_leaks
from .tool_server import FORBIDDEN_MODULES
from .tools import ANALYSIS_NAMES, run_tool, tool_set
from .utils import sha256_bytes

SERVER_NAME = "weights"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"
SUBMIT_OK = "Submission received. The session is complete; no further tool calls are accepted."
AFTER_SUBMIT = "Session ended: a submission was already received. No further tool calls."


def schema_problems(schema: dict, value: Any, path: str = "$") -> list[str]:
    """JSON-schema check for the subset used by tools.py (what strict tools guarantee)."""
    t = schema.get("type")
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: must be one of {schema['enum']}"]
    if t == "object":
        if not isinstance(value, dict):
            return [f"{path}: must be an object"]
        props = schema.get("properties", {})
        out = [f"{path}.{k}: missing" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            out += [f"{path}.{k}: not allowed" for k in value if k not in props]
        for k, v in value.items():
            if k in props:
                out += schema_problems(props[k], v, f"{path}.{k}")
        return out
    if t == "array":
        if not isinstance(value, list):
            return [f"{path}: must be an array"]
        items = schema.get("items")
        return [p for i, v in enumerate(value) for p in
                (schema_problems(items, v, f"{path}[{i}]") if items else [])]
    if t == "string" and not isinstance(value, str):
        return [f"{path}: must be a string"]
    if t == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return [f"{path}: must be a number"]
    if t == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        return [f"{path}: must be an integer"]
    return []


def mcp_tools(spec: dict) -> list[dict]:
    """The condition's tool set in MCP form (same names, descriptions, schemas)."""
    return [{"name": t["name"], "description": t["description"],
             "inputSchema": t["input_schema"]}
            for t in tool_set(spec["protocol"], spec["condition"], spec.get("candidate_order"))]


class Session:
    def __init__(self, directory: Path):
        self.dir = directory
        self.spec = json.loads((directory / "spec.json").read_text())
        ep = directory / "export.json"
        self.export = json.loads(ep.read_text()) if ep.exists() else None
        self.tools = mcp_tools(self.spec)
        self.names = {t["name"] for t in self.tools}
        self.submit_schema = next(t["inputSchema"] for t in self.tools
                                  if t["name"] == "submit_interpretation")
        self.submitted = (directory / "submission.json").exists()

    def _append(self, fname: str, obj: dict) -> None:
        with open(self.dir / fname, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns (text, is_error)."""
        if self.submitted:
            self._append("calls.jsonl", {"name": name, "args": args, "ok": False,
                                         "after_submit": True})
            return AFTER_SUBMIT, True
        if name == "submit_interpretation":
            problems = schema_problems(self.submit_schema, args)
            if problems:
                self._append("submit_rejections.jsonl", {"problems": problems})
                return "Submission rejected (schema): " + "; ".join(problems), True
            (self.dir / "submission.json").write_text(json.dumps(args))
            self.submitted = True
            return SUBMIT_OK, False
        if name not in self.names or name not in ANALYSIS_NAMES or self.export is None:
            text, ok = f"ValueError: tool not available: {name}", False
        else:
            try:
                text, ok = run_tool(self.export, name, args), True
            except Exception as e:  # reported to the model as a tool error
                text, ok = f"{type(e).__name__}: {e}", False
        if ok and scan_for_leaks(text):
            (self.dir / "leak.json").write_text(json.dumps({"tool": name}))
            text, ok = "output withheld", False
        self._append("calls.jsonl", {"name": name, "args": args, "ok": ok,
                                     "result_sha256": sha256_bytes(text.encode()),
                                     "result_chars": len(text)})
        return text, not ok


def handle(session: Session, msg: dict) -> dict | None:
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:  # notification
        return None
    if method == "initialize":
        version = (msg.get("params") or {}).get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
        result: Any = {"protocolVersion": version, "capabilities": {"tools": {}},
                       "serverInfo": {"name": SERVER_NAME, "version": "1.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": session.tools}
    elif method == "tools/call":
        p = msg.get("params") or {}
        text, is_error = session.call(str(p.get("name")), p.get("arguments") or {})
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(directory: Path) -> None:  # pragma: no cover - exercised via subprocess in tests
    session = Session(directory)
    info = {"forbidden_loaded": [m for m in FORBIDDEN_MODULES if m in sys.modules],
            "env_keys": sorted(os.environ), "cwd": os.getcwd(), "pid": os.getpid()}
    (directory / "server_info.json").write_text(json.dumps(info))
    if info["forbidden_loaded"]:
        sys.exit(3)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp: dict | None = {"jsonrpc": "2.0", "id": None,
                                 "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = handle(session, msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    serve(Path(sys.argv[1]))
