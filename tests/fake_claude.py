"""Stand-in for the `claude` CLI (tests only).

Speaks the same stream-json protocol as `claude -p --input-format stream-json
--output-format stream-json`, starts the MCP server from --mcp-config exactly
as configured, and follows a script from the FAKE_CLAUDE_SCENARIO env var:

{"steps": [{"call": name, "args": {...}} | {"submit": {...}} | {"end_turn": true}
           | {"refuse": true}, ...],          # each optional "model": "..."
 "after_nudge": [...],                         # steps after the runner's nudge
 "init": {...},                                # overrides merged into the init event
 "extra_tools": ["Read"],                      # appended to the init tool list
 "error_result": "text"}                       # emit an error result right away
"""

import json
import os
import subprocess
import sys


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


class MCP:
    def __init__(self, cfg):
        (name, srv), = cfg["mcpServers"].items()
        self.name = name
        self.p = subprocess.Popen([srv["command"], *srv["args"]], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, env=srv.get("env") or {})
        self.i = 0
        self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "fake", "version": "0"}})
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0",
                                       "method": "notifications/initialized"}) + "\n")
        self.tools = [t["name"] for t in self.rpc("tools/list", {})["tools"]]

    def rpc(self, method, params):
        self.i += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.i, "method": method,
                                       "params": params}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())["result"]


def main():
    sc = json.loads(os.environ.get("FAKE_CLAUDE_SCENARIO", "{}"))
    model = arg("--model")
    mcp = MCP(json.load(open(arg("--mcp-config"))))
    first = sys.stdin.readline()
    if not first:
        return
    init = {"type": "system", "subtype": "init", "cwd": os.getcwd(), "session_id": "fake-1",
            "tools": [f"mcp__{mcp.name}__{t}" for t in mcp.tools] + sc.get("extra_tools", []),
            "mcp_servers": [{"name": mcp.name, "status": "connected"}], "model": model,
            "permissionMode": arg("--permission-mode"), "slash_commands": [], "skills": [],
            "plugins": [{"name": "cc-plugin-telemetry", "path": "builtin"}],
            "apiKeySource": "none", "claude_code_version": "fake"}
    init.update(sc.get("init", {}))
    out(init)
    if "error_result" in sc:
        out({"type": "result", "subtype": "success", "is_error": True,
             "result": sc["error_result"], "total_cost_usd": 0})
        sys.stdin.readline()
        return
    n = 0

    def message(content, stop, step):
        nonlocal n
        n += 1
        out({"type": "assistant", "message": {
            "id": f"msg_{n}", "model": step.get("model", model), "role": "assistant",
            "content": content, "stop_reason": stop,
            "usage": {"input_tokens": 1000, "output_tokens": 50}}})

    def run(steps):
        for step in steps:
            if "call" in step or "submit" in step:
                name = step.get("call", "submit_interpretation")
                args = step.get("args", step.get("submit", {}))
                tid = f"tu_{n + 1}"
                message([{"type": "thinking", "thinking": "..."},
                         {"type": "tool_use", "id": tid, "name": f"mcp__{mcp.name}__{name}",
                          "input": args}], "tool_use", step)
                r = mcp.rpc("tools/call", {"name": name, "arguments": args})
                out({"type": "user", "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tid, "is_error": r["isError"],
                     "content": r["content"]}]}})
            elif step.get("refuse"):
                message([{"type": "text", "text": "no"}], "refusal", step)
                return "done"
            elif step.get("end_turn"):
                message([{"type": "text", "text": "done"}], "end_turn", step)
                out({"type": "result", "subtype": "success", "is_error": False,
                     "result": "done", "total_cost_usd": 0.01 * n})
                return "ended"
        message([{"type": "text", "text": "finished"}], "end_turn", {})
        out({"type": "result", "subtype": "success", "is_error": False,
             "result": "finished", "total_cost_usd": 0.01 * n})
        return "ended"

    state = run(sc.get("steps", []))
    while state == "ended":
        line = sys.stdin.readline()
        if not line:
            break
        state = run(sc.get("after_nudge", []))
        sc["after_nudge"] = []
    mcp.p.kill()


if __name__ == "__main__":
    main()
