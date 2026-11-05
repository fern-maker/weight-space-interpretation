"""Interpretation sessions through headless Claude Code (ARCHITECTURE §10).

``run_session_cc`` has the same inputs and record schema as
``interpreter.run_session``. Claude Code runs the agent loop; the only tools
it has are the analysis tools served by ``mcp_server`` over stdio. This module
builds the locked-down command, checks Claude Code's ``init`` event before
trusting the session, enforces the turn cap and the single nudge, and checks
the served model on every turn. Infrastructure failures raise
``HarnessError`` and produce no prediction record.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from .interpreter import NUDGE, LeakError, check_prompt, validate_submission
from .llm_client import redact
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from .tool_server import _sandbox_profile
from .tools import tool_schema_hash, tool_set
from .utils import sha256_bytes

SUBMIT = "submit_interpretation"
# reasons that are not the session's fault and never count toward harness_failure
UNCOUNTED_REASONS = {"init_check", "auth", "usage_limit"}


class HarnessError(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail

    @property
    def counted(self) -> bool:
        return self.reason not in UNCOUNTED_REASONS


def backend_ready(icfg: dict) -> tuple[bool, str]:
    """Offline readiness check for the configured backend (login is checked live)."""
    if icfg.get("backend", "anthropic_api") == "claude_code":
        exe = icfg.get("claude_code", {}).get("executable", "claude")
        if shutil.which(exe):
            return True, "claude CLI found"
        return False, f"Claude Code CLI '{exe}' not found on PATH"
    from .llm_client import has_credentials

    if has_credentials():
        return True, "API credentials found"
    return False, "no API credentials (set ANTHROPIC_API_KEY)"


def os_sandbox_available() -> bool:
    if not shutil.which("sandbox-exec"):
        return False
    try:
        r = subprocess.run(["sandbox-exec", "-p", "(version 1)\n(allow default)\n",
                            "/usr/bin/true"], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def tool_names(protocol: str, condition: str, candidates: list[str] | None,
               server: str = "weights") -> set[str]:
    return {f"mcp__{server}__{t['name']}" for t in tool_set(protocol, condition, candidates)}


def build_command(icfg: dict, mcp_config: Path, executable: list[str] | None = None) -> list[str]:
    """The locked-down headless command (ARCHITECTURE §10.2)."""
    cc = icfg.get("claude_code", {})
    exe = executable or [cc.get("executable", "claude")]
    server = cc.get("mcp_server_name", "weights")
    return [*exe, "-p",
            "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--model", icfg["model"], "--effort", icfg["effort"],
            "--system-prompt", SYSTEM_PROMPT,
            "--tools", "",
            "--strict-mcp-config", "--mcp-config", str(mcp_config),
            "--allowedTools", f"mcp__{server}",
            "--permission-mode", "dontAsk",
            "--setting-sources", "",
            "--disable-slash-commands", "--no-session-persistence", "--include-hook-events"]


def recorded_args(cmd: list[str]) -> list[str]:
    """Command for the record: prompt text and temp paths replaced by markers."""
    out, skip = [], None
    for a in cmd:
        if skip:
            out.append(skip)
            skip = None
            continue
        out.append(a)
        if a == "--system-prompt":
            skip = f"<SYSTEM_PROMPT sha256={sha256_bytes(SYSTEM_PROMPT.encode())[:16]}>"
        elif a == "--mcp-config":
            skip = "<session mcp.json>"
    return out


def passing_probe_versions(repo: Path) -> set[str]:
    """Claude Code versions for which the live isolation probe passed."""
    out = set()
    for f in sorted((repo / "results" / "probes").glob("claude_code_probe_*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if r.get("passed") and r.get("claude_code_version"):
            out.add(str(r["claude_code_version"]))
    return out


def init_problems(ev: dict, expected_tools: set[str], model: str, cwd: Path,
                  server: str = "weights", probed_versions: set[str] | None = None) -> list[str]:
    """Init guard (ARCHITECTURE §10.3.1). Empty list = pass."""
    p = []
    if probed_versions is not None and ev.get("claude_code_version") not in probed_versions:
        p.append(f"Claude Code version {ev.get('claude_code_version')!r} has no passing "
                 "isolation probe; run scripts/probe_claude_code.py")
    tools = set(ev.get("tools") or [])
    if tools != expected_tools:
        p.append(f"tools {sorted(tools ^ expected_tools)} differ from the allowed set")
    servers = [(s.get("name"), s.get("status")) for s in ev.get("mcp_servers") or []]
    if servers != [(server, "connected")]:
        p.append(f"mcp_servers {servers}")
    if ev.get("model") != model:
        p.append(f"model {ev.get('model')!r} != {model!r}")
    if ev.get("skills"):
        p.append(f"skills loaded: {ev.get('skills')}")
    if ev.get("slash_commands"):
        p.append(f"slash commands loaded: {ev.get('slash_commands')}")
    extra = [pl.get("name") for pl in ev.get("plugins") or [] if pl.get("path") != "builtin"]
    if extra:
        p.append(f"non-builtin plugins: {extra}")
    if ev.get("permissionMode") != "dontAsk":
        p.append(f"permissionMode {ev.get('permissionMode')!r}")
    if ev.get("cwd") and Path(ev["cwd"]).resolve() != cwd.resolve():
        p.append(f"cwd {ev.get('cwd')!r}")
    return p


def _classify_error(text: str, ev: dict) -> str:
    low = (text or "").lower()
    if ev.get("error") == "authentication_failed" or "login" in low or "logged in" in low:
        return "auth"
    if ev.get("api_error_status") == 429 or "limit" in low or ev.get("error") == "rate_limit":
        return "usage_limit"
    return "result_error"


def _user_event(text: str) -> str:
    return json.dumps({"type": "user", "parent_tool_use_id": None,
                       "message": {"role": "user", "content": [{"type": "text", "text": text}]}})


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] \
        if p.exists() else []


class CCSession:
    """One Claude Code session in its own temp directory."""

    def __init__(self, *, icfg: dict, protocol: str, condition: str, user_msg: str,
                 export: dict | None, candidate_order: list[str] | None,
                 deny_dirs: list[Path] | None, os_sandbox: bool,
                 executable: list[str] | None = None, cwd_files: dict[str, str] | None = None,
                 extra_env: dict[str, str] | None = None,
                 probed_versions: set[str] | None = None):
        self.icfg, self.protocol, self.condition = icfg, protocol, condition
        self.user_msg, self.candidates = user_msg, candidate_order
        cc = icfg.get("claude_code", {})
        self.server = cc.get("mcp_server_name", "weights")
        self.timeout = float(cc.get("session_timeout_s", 2400))
        self.max_turns = int(icfg["max_turns"])
        self.base = Path(tempfile.mkdtemp(prefix="ws_cc_"))
        self.cwd, self.mcp = self.base / "cwd", self.base / "mcp"
        self.cwd.mkdir()
        self.mcp.mkdir()
        for name, text in (cwd_files or {}).items():  # canaries for the probe only
            (self.cwd / name).write_text(text)
        if export is not None:
            (self.mcp / "export.json").write_text(json.dumps(export))
        (self.mcp / "spec.json").write_text(json.dumps(
            {"protocol": protocol, "condition": condition, "candidate_order": candidate_order}))
        self.os_sandbox_active = bool(os_sandbox and deny_dirs and os_sandbox_available())
        server_cmd = [sys.executable, "-I", "-m", "weightspace.mcp_server", str(self.mcp)]
        if self.os_sandbox_active:
            server_cmd = ["/usr/bin/sandbox-exec", "-p", _sandbox_profile(deny_dirs or []),
                          *server_cmd]
        mcp_cfg = {"mcpServers": {self.server: {
            "type": "stdio", "command": "/usr/bin/env",
            "args": ["-i", "PATH=/usr/bin:/bin", *server_cmd], "env": {}}}}
        self.mcp_config = self.base / "mcp.json"
        self.mcp_config.write_text(json.dumps(mcp_cfg))
        self.cmd = build_command(icfg, self.mcp_config, executable)
        self.env = {**os.environ, "ENABLE_TOOL_SEARCH": "false", **(extra_env or {})}
        self.expected_tools = tool_names(protocol, condition, candidate_order, self.server)
        self.probed_versions = probed_versions

    def close(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        """Run to completion. Returns raw outcome; raises HarnessError / LeakError."""
        stderr_path = self.base / "stderr.txt"
        with open(stderr_path, "w") as err:
            proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=err, cwd=self.cwd, env=self.env, text=True,
                                    bufsize=1)
        q: queue.Queue = queue.Queue()

        def reader() -> None:
            assert proc.stdout
            for line in proc.stdout:
                q.put(line)
            q.put(None)

        threading.Thread(target=reader, daemon=True).start()
        events: list[dict] = []
        raw_lines: list[str] = []
        msg_order: list[str] = []
        msgs: dict[str, dict] = {}
        submit_seen_in: int | None = None
        results: list[dict] = []
        hook_events: list[str] = []
        init: dict | None = None
        status: str | None = None
        nudged = False
        thinking_blocks = 0
        t0 = time.perf_counter()

        def send(text: str) -> None:
            assert proc.stdin
            proc.stdin.write(text + "\n")
            proc.stdin.flush()

        def submitted() -> bool:
            return (self.mcp / "submission.json").exists()

        def abort(reason: str, detail: str = "") -> None:
            proc.kill()
            raise HarnessError(reason, detail)

        try:
            send(_user_event(self.user_msg))
            while True:
                remaining = self.timeout - (time.perf_counter() - t0)
                if remaining <= 0:
                    if submitted():
                        status = "submitted"
                        break
                    abort("timeout", f"{self.timeout:.0f}s")
                try:
                    line = q.get(timeout=min(remaining, 5.0))
                except queue.Empty:
                    continue
                if line is None:
                    if status is None and submitted():
                        status = "submitted"
                    if status is None:
                        tail = stderr_path.read_text()[-300:]
                        abort("process_exit", f"code {proc.poll()}: {redact(tail)}")
                    break
                raw_lines.append(line)
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(ev)
                et, st = ev.get("type"), ev.get("subtype")
                if et == "system" and st == "init":
                    init = ev
                    probs = init_problems(ev, self.expected_tools, self.icfg["model"],
                                          self.cwd, self.server, self.probed_versions)
                    if probs:
                        abort("init_check", "; ".join(probs))
                elif et == "system" and st and "hook" in st:
                    hook_events.append(str(ev.get("hook_name") or ev.get("hook_event") or st))
                elif et == "assistant":
                    if init is None:
                        abort("init_check", "assistant output before init event")
                    m = ev.get("message") or {}
                    model = m.get("model")
                    if model == "<synthetic>" or ev.get("error"):
                        if submitted():
                            continue
                        text = " ".join(b.get("text", "") for b in m.get("content") or []
                                        if isinstance(b, dict))
                        abort(_classify_error(text, ev), redact(text[:200]))
                    if model != self.icfg["model"]:
                        if submitted():
                            continue
                        abort("model_mismatch", f"{model!r}")
                    mid = m.get("id") or f"anon{len(msg_order)}"
                    if mid not in msgs:
                        msg_order.append(mid)
                        if len(msg_order) > self.max_turns and submit_seen_in is None:
                            msg_order.pop()  # the over-cap message is discarded unseen
                            status = "turn_cap"
                            proc.kill()
                            break
                    msgs[mid] = {"message_id": mid, "model_served": model,
                                 "stop_reason": m.get("stop_reason"),
                                 "usage": m.get("usage") or {}}
                    for b in m.get("content") or []:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") in ("thinking", "redacted_thinking"):
                            thinking_blocks += 1
                        if b.get("type") == "tool_use" and \
                                str(b.get("name", "")).endswith("__" + SUBMIT) and \
                                submit_seen_in is None:
                            submit_seen_in = len(msg_order)
                    if m.get("stop_reason") == "refusal" and not submitted():
                        status = "refusal"
                        proc.kill()
                        break
                elif et == "result":
                    results.append(ev)
                    if ev.get("is_error") and not submitted():
                        text = str(ev.get("result") or "")
                        abort(_classify_error(text, ev), redact(text[:200]))
                    if submitted():
                        status = "submitted"
                        break
                    if not nudged:
                        nudged = True
                        events.append({"type": "user", "_sent_by_runner": True,
                                       "message": {"role": "user", "content": NUDGE}})
                        send(_user_event(NUDGE))
                        continue
                    status = "no_submission"
                    break
        finally:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        leak = self.mcp / "leak.json"
        if leak.exists():
            raise LeakError(f"tool output leak: {leak.read_text()}")
        sub_path = self.mcp / "submission.json"
        submission = json.loads(sub_path.read_text()) if sub_path.exists() else None
        if status == "submitted" and submission is None:
            status = "no_submission"
        info_p = self.mcp / "server_info.json"
        return {
            "status": status or "no_submission",
            "submission": submission if status == "submitted" else None,
            "init": init or {},
            "turns": [msgs[i] for i in msg_order],
            "results": results,
            "calls": _read_jsonl(self.mcp / "calls.jsonl"),
            "submit_rejections": len(_read_jsonl(self.mcp / "submit_rejections.jsonl")),
            "server_info": json.loads(info_p.read_text()) if info_p.exists() else {},
            "hook_events": hook_events,
            "thinking_blocks": thinking_blocks,
            "nudged": nudged,
            "events": events,
            "raw_stdout": "".join(raw_lines),
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }


def _sum_usage(turns: list[dict]) -> dict[str, int]:
    tot: dict[str, int] = {}
    for t in turns:
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                  "cache_creation_input_tokens"):
            tot[k] = tot.get(k, 0) + int((t.get("usage") or {}).get(k) or 0)
    return tot


def _notional_cost(results: list[dict]) -> float:
    costs = [float(r.get("total_cost_usd") or 0) for r in results]
    if not costs:
        return 0.0
    # Claude Code reports a running total per process; fall back to a sum if not
    return costs[-1] if all(b >= a for a, b in zip(costs, costs[1:], strict=False)) \
        else sum(costs)


def run_session_cc(
    *,
    network_id: str,
    protocol: str,
    condition: str,
    icfg: dict,
    export: dict | None = None,
    candidate_order: list[str] | None = None,
    deny_dirs: list[Path] | None = None,
    os_sandbox: bool = True,
    executable: list[str] | None = None,
    probed_versions: set[str] | None = None,
) -> dict:
    """Same contract and record schema as interpreter.run_session.

    ``probed_versions``: Claude Code versions with a passing live probe; the
    session is refused (init_check) on any other version. None disables the
    check (tests only)."""
    tools = tool_set(protocol, condition, candidate_order)
    user_msg = build_user_message(protocol, condition, export=export, network_id=network_id,
                                  candidate_order=candidate_order)
    check_prompt(protocol, user_msg, export, candidate_order)
    s = CCSession(icfg=icfg, protocol=protocol, condition=condition, user_msg=user_msg,
                  export=export, candidate_order=candidate_order, deny_dirs=deny_dirs,
                  os_sandbox=os_sandbox, executable=executable,
                  probed_versions=probed_versions)
    try:
        out = s.run()
    finally:
        s.close()
    if out["server_info"].get("forbidden_loaded"):
        raise HarnessError("mcp_server", f"forbidden modules: {out['server_info']}")
    submission = out["submission"]
    problems = validate_submission(protocol, submission, candidate_order) if submission else []
    init = out["init"]
    tool_log = [c for c in out["calls"] if not c.get("after_submit")]
    return {
        "network_id": network_id,
        "protocol": protocol,
        "condition": condition,
        "candidate_order": candidate_order,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode()),
        "user_prompt_sha256": sha256_bytes(user_msg.encode()),
        "tool_schema_sha256": tool_schema_hash(tools),
        "request_params": {"backend": "claude_code", "model": icfg["model"],
                           "effort": icfg["effort"], "max_turns": int(icfg["max_turns"]),
                           "claude_code_args": recorded_args(s.cmd)},
        "models_served": sorted({t["model_served"] for t in out["turns"] if t["model_served"]}),
        "status": out["status"],
        "valid": out["status"] == "submitted" and not problems,
        "validation_problems": problems,
        "submission": submission,
        "n_turns": len(out["turns"]),
        "n_tool_calls": len(tool_log),
        "tool_calls": tool_log,
        "calls_after_submit": sum(1 for c in out["calls"] if c.get("after_submit")),
        "turns": [{**t, "request_id": None} for t in out["turns"]],
        "usage_total": _sum_usage(out["turns"]),
        "cost_usd": round(_notional_cost(out["results"]), 5),
        "wall_time_s": out["wall_time_s"],
        "os_sandbox_active": s.os_sandbox_active,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": "claude_code",
        "claude_code_version": init.get("claude_code_version"),
        "session_id": init.get("session_id"),
        "init_check": {"passed": True, "tools": sorted(init.get("tools") or []),
                       "mcp_servers": init.get("mcp_servers"), "model": init.get("model"),
                       "plugins": [p.get("name") for p in init.get("plugins") or []],
                       "api_key_source": init.get("apiKeySource")},
        "hook_events": out["hook_events"],
        "nudged": out["nudged"],
        "submit_rejections": out["submit_rejections"],
        "thinking_blocks_seen": out["thinking_blocks"],
        "mcp_server_env_keys": out["server_info"].get("env_keys"),
        "_transcript": json.loads(redact(json.dumps(out["events"]))),
    }

