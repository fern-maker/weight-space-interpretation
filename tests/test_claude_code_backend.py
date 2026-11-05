"""Claude Code backend: MCP server, locked-down command, init guard, full sessions.

No model is called: full sessions run against tests/fake_claude.py, which
speaks Claude Code's stream-json protocol and drives the real MCP server.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from weightspace.claude_code_backend import (
    CCSession,
    HarnessError,
    build_command,
    init_problems,
    recorded_args,
    run_session_cc,
    tool_names,
)
from weightspace.mcp_server import AFTER_SUBMIT, mcp_tools, schema_problems
from weightspace.prompts import SYSTEM_PROMPT, build_user_message
from weightspace.tools import tool_set
from weightspace.weight_exporter import model_to_export

FAKE = [sys.executable, str(Path(__file__).with_name("fake_claude.py"))]
CANDS = ["diagonal", "square", "circle"]
REPO = Path(__file__).resolve().parents[1]
ICFG = {"backend": "claude_code", "model": "claude-opus-5-5", "effort": "high",
        "max_turns": 16, "claude_code": {"executable": "claude", "mcp_server_name": "weights",
                                         "session_timeout_s": 60}}
SUB_B = {"network_id": "network_017", "predicted_function": "square", "confidence": 0.8,
         "mathematical_explanation": "m", "supporting_evidence": ["e"], "uncertainties": []}


@pytest.fixture
def export(rand_model):
    return model_to_export(rand_model, "network_017")


def run(export, scenario, protocol="B", condition="weights_tools", icfg=ICFG, **kw):
    os.environ["FAKE_CLAUDE_SCENARIO"] = json.dumps(scenario)
    try:
        return run_session_cc(network_id="network_017", protocol=protocol, condition=condition,
                              icfg=icfg, export=export if condition == "weights_tools" else None,
                              candidate_order=CANDS if protocol == "B" else None,
                              deny_dirs=None, os_sandbox=False, executable=FAKE, **kw)
    finally:
        os.environ.pop("FAKE_CLAUDE_SCENARIO", None)


# ------------------------------------------------------------------ command
def test_command_is_locked_down(tmp_path):
    cmd = build_command(ICFG, tmp_path / "mcp.json")
    joined = " ".join(cmd)
    i = cmd.index("--tools")
    assert cmd[i + 1] == ""
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__weights"
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    assert cmd[cmd.index("--system-prompt") + 1] == SYSTEM_PROMPT
    for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in cmd
    for banned in ("--fallback-model", "--dangerously-skip-permissions", "--add-dir",
                   "--resume", "--continue", "bypassPermissions", "--allow-dangerously"):
        assert banned not in joined
    rec = recorded_args(cmd)
    assert SYSTEM_PROMPT not in rec and str(tmp_path) not in " ".join(rec)


def test_session_dirs_hold_only_public_data(export):
    s = CCSession(icfg=ICFG, protocol="B", condition="weights_tools", user_msg="x",
                  export=export, candidate_order=CANDS, deny_dirs=None, os_sandbox=False)
    try:
        assert REPO not in s.base.parents and s.base != REPO
        assert list(s.cwd.iterdir()) == []
        assert sorted(p.name for p in s.mcp.iterdir()) == ["export.json", "spec.json"]
        spec = json.loads((s.mcp / "spec.json").read_text())
        assert set(spec) == {"protocol", "condition", "candidate_order"}
        cfg = json.loads(s.mcp_config.read_text())
        srv = cfg["mcpServers"]["weights"]
        assert srv["command"] == "/usr/bin/env" and srv["args"][:2] == ["-i", "PATH=/usr/bin:/bin"]
        assert str(REPO / "artifacts") not in json.dumps(cfg)
        assert s.env["ENABLE_TOOL_SEARCH"] == "false"
    finally:
        s.close()
    assert not s.base.exists()


# ------------------------------------------------------------------ init guard
def _good_init(tmp_path):
    return {"tools": sorted(tool_names("B", "weights_tools", CANDS)),
            "mcp_servers": [{"name": "weights", "status": "connected"}],
            "model": "claude-opus-5-5", "skills": [], "slash_commands": [],
            "plugins": [{"name": "cc-plugin-telemetry", "path": "builtin"}],
            "permissionMode": "dontAsk", "cwd": str(tmp_path)}


@pytest.mark.parametrize("change", [
    {"tools": ["Read"]}, {"tools": ["Bash"]}, {"tools": ["WebFetch"]}, {"tools": ["Agent"]},
    {"model": "claude-sonnet-5-5"},
    {"mcp_servers": [{"name": "weights", "status": "failed"}]},
    {"mcp_servers": [{"name": "weights", "status": "connected"},
                     {"name": "other", "status": "connected"}]},
    {"skills": ["deep-research"]}, {"slash_commands": ["/init"]},
    {"plugins": [{"name": "context-mode", "path": "/Users/x/.claude/plugins/cm"}]},
    {"permissionMode": "bypassPermissions"}, {"cwd": "/"},
])
def test_init_guard_rejects(tmp_path, change):
    ev = _good_init(tmp_path)
    expected = set(ev["tools"])
    if "tools" in change:
        ev["tools"] = ev["tools"] + change["tools"]
    else:
        ev.update(change)
    assert init_problems(ev, expected, "claude-opus-5-5", tmp_path)


def test_init_guard_accepts_clean(tmp_path):
    ev = _good_init(tmp_path)
    assert init_problems(ev, set(ev["tools"]), "claude-opus-5-5", tmp_path) == []


def test_init_guard_rejects_missing_tool(tmp_path):
    ev = _good_init(tmp_path)
    expected = set(ev["tools"])
    ev["tools"] = ev["tools"][1:]
    assert init_problems(ev, expected, "claude-opus-5-5", tmp_path)


# ------------------------------------------------------------------ MCP server
def _mcp(tmp_path, export, protocol="B", condition="weights_tools"):
    if export is not None:
        (tmp_path / "export.json").write_text(json.dumps(export))
    (tmp_path / "spec.json").write_text(json.dumps(
        {"protocol": protocol, "condition": condition,
         "candidate_order": CANDS if protocol == "B" else None}))
    p = subprocess.Popen(["/usr/bin/env", "-i", "PATH=/usr/bin:/bin", sys.executable, "-I", "-m",
                          "weightspace.mcp_server", str(tmp_path)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, cwd=tmp_path)
    n = [0]

    def rpc(method, params=None):
        n[0] += 1
        p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": n[0], "method": method,
                                  "params": params or {}}) + "\n")
        p.stdin.flush()
        return json.loads(p.stdout.readline())

    def call(name, args=None):
        r = rpc("tools/call", {"name": name, "arguments": args or {}})["result"]
        return r["content"][0]["text"], r["isError"]

    rpc("initialize", {"protocolVersion": "2025-06-18"})
    return p, rpc, call


def test_mcp_server_serves_exactly_the_api_tool_set(tmp_path, export):
    p, rpc, call = _mcp(tmp_path, export)
    try:
        listed = rpc("tools/list")["result"]["tools"]
        api = tool_set("B", "weights_tools", CANDS)
        assert [t["name"] for t in listed] == [t["name"] for t in api]
        for m, a in zip(listed, api, strict=True):
            assert m["inputSchema"] == a["input_schema"]
            assert m["description"] == a["description"]
        text, err = call("get_bias_vector", {"layer": "layer3"})
        assert not err and len(json.loads(text)) == 1
        info = json.loads((tmp_path / "server_info.json").read_text())
        assert info["forbidden_loaded"] == []
        allowed_env = {"PATH", "LC_CTYPE", "__CF_USER_TEXT_ENCODING", "PWD", "SHLVL", "_"}
        assert set(info["env_keys"]) <= allowed_env
        assert rpc("resources/list").get("error")
    finally:
        p.kill()


def test_mcp_server_rejects_forbidden_calls(tmp_path, export):
    p, rpc, call = _mcp(tmp_path, export)
    try:
        for name, args in [("forward", {"x": [0, 0]}), ("evaluate", {}),
                           ("read_file", {"path": "/"}), ("Bash", {"command": "ls"})]:
            text, err = call(name, args)
            assert err and "not available" in text
        text, err = call("get_weight_matrix", {"layer": "../artifacts/private"})
        assert err
        text, err = call("get_bias_vector", {"layer": "layer1", "x": [0.1, 0.2]})
        assert err
    finally:
        p.kill()


def test_mcp_submit_schema_and_lockout(tmp_path, export):
    p, rpc, call = _mcp(tmp_path, export)
    try:
        text, err = call("submit_interpretation", dict(SUB_B, predicted_function="hexagon"))
        assert err and "rejected" in text
        text, err = call("submit_interpretation", {k: v for k, v in SUB_B.items()
                                                   if k != "confidence"})
        assert err
        assert not (tmp_path / "submission.json").exists()
        text, err = call("submit_interpretation", SUB_B)
        assert not err
        assert json.loads((tmp_path / "submission.json").read_text()) == SUB_B
        text, err = call("get_architecture")
        assert err and text == AFTER_SUBMIT
        text, err = call("submit_interpretation", dict(SUB_B, predicted_function="circle"))
        assert err and json.loads((tmp_path / "submission.json").read_text()) == SUB_B
        rej = (tmp_path / "submit_rejections.jsonl").read_text().splitlines()
        assert len(rej) == 2
    finally:
        p.kill()


def test_mcp_architecture_only_has_submit_only(tmp_path):
    p, rpc, call = _mcp(tmp_path, None, protocol="A", condition="architecture_only")
    try:
        assert [t["name"] for t in rpc("tools/list")["result"]["tools"]] == [
            "submit_interpretation"]
        text, err = call("get_weight_matrix", {"layer": "layer1"})
        assert err
    finally:
        p.kill()


def test_schema_problems_matches_strict_subset():
    schema = mcp_tools({"protocol": "B", "condition": "weights_tools",
                        "candidate_order": CANDS})[-1]["inputSchema"]
    assert schema_problems(schema, SUB_B) == []
    assert schema_problems(schema, dict(SUB_B, extra=1))
    assert schema_problems(schema, dict(SUB_B, confidence="high"))
    assert schema_problems(schema, dict(SUB_B, confidence=True))
    assert schema_problems(schema, dict(SUB_B, supporting_evidence=[1]))


# ------------------------------------------------------------------ full sessions
def test_session_submitted(export):
    rec = run(export, {"steps": [{"call": "get_architecture"},
                                 {"call": "first_layer_hyperplanes"}, {"submit": SUB_B}]})
    assert rec["status"] == "submitted" and rec["valid"]
    assert rec["submission"] == SUB_B
    assert [c["name"] for c in rec["tool_calls"]] == ["get_architecture",
                                                      "first_layer_hyperplanes"]
    assert all(c["ok"] for c in rec["tool_calls"])
    assert rec["models_served"] == ["claude-opus-5-5"]
    assert rec["backend"] == "claude_code" and rec["init_check"]["passed"]
    assert rec["n_turns"] == 4 and rec["thinking_blocks_seen"] == 3
    assert rec["usage_total"]["input_tokens"] == 4000
    assert rec["mcp_server_env_keys"] is not None
    assert "ANTHROPIC_API_KEY" not in rec["mcp_server_env_keys"]
    assert rec["request_params"]["backend"] == "claude_code"
    msg = build_user_message("B", "weights_tools", export=export, network_id="network_017",
                             candidate_order=CANDS)
    first_user = next(e for e in rec["_transcript"] if e.get("type") == "system")
    assert first_user["subtype"] == "init"
    assert rec["user_prompt_sha256"] and msg  # same builder as the API backend


def test_session_rejected_then_resubmitted(export):
    bad = dict(SUB_B, predicted_function="hexagon")
    rec = run(export, {"steps": [{"submit": bad}, {"submit": SUB_B}]})
    assert rec["status"] == "submitted" and rec["valid"] and rec["submit_rejections"] == 1


def test_session_out_of_range_confidence_is_invalid(export):
    rec = run(export, {"steps": [{"submit": dict(SUB_B, confidence=1.7)}]})
    assert rec["status"] == "submitted" and not rec["valid"]
    assert "confidence not in [0,1]" in rec["validation_problems"]


def test_session_nudge_then_submit(export):
    rec = run(export, {"steps": [{"end_turn": True}], "after_nudge": [{"submit": SUB_B}]})
    assert rec["nudged"] and rec["status"] == "submitted" and rec["valid"]


def test_session_no_submission_after_nudge(export):
    rec = run(export, {"steps": [{"end_turn": True}], "after_nudge": [{"end_turn": True}]})
    assert rec["nudged"] and rec["status"] == "no_submission" and not rec["valid"]


def test_session_turn_cap(export):
    icfg = dict(ICFG, max_turns=3)
    rec = run(export, {"steps": [{"call": "get_architecture"}] * 5 + [{"submit": SUB_B}]},
              icfg=icfg)
    assert rec["status"] == "turn_cap" and not rec["valid"] and rec["n_turns"] == 3


def test_session_refusal(export):
    rec = run(export, {"steps": [{"refuse": True}]})
    assert rec["status"] == "refusal" and not rec["valid"]


def test_architecture_only_session(export):
    sub_a = {"network_id": "network_000", "behavior_summary": "s", "boundary_geometry": "g",
             "symmetries": [], "proposed_rule": "x1 > 0", "rule_confidence": 0.1,
             "supporting_evidence": [], "alternative_explanations": [], "uncertainties": []}
    rec = run(export, {"steps": [{"call": "get_weight_matrix", "args": {"layer": "layer1"}},
                                 {"submit": sub_a}]}, protocol="A",
              condition="architecture_only")
    assert rec["status"] == "submitted" and rec["valid"]
    assert rec["tool_calls"][0]["ok"] is False


@pytest.mark.parametrize("scenario,reason", [
    ({"extra_tools": ["Read"], "steps": [{"submit": SUB_B}]}, "init_check"),
    ({"init": {"model": "claude-sonnet-5-5"}, "steps": [{"submit": SUB_B}]}, "init_check"),
    ({"init": {"plugins": [{"name": "context-mode", "path": "/p"}]},
      "steps": [{"submit": SUB_B}]}, "init_check"),
    ({"steps": [{"call": "get_architecture", "model": "claude-sonnet-5-5"},
                {"submit": SUB_B}]}, "model_mismatch"),
    ({"error_result": "Not logged in · Please run /login"}, "auth"),
    ({"error_result": "Claude usage limit reached"}, "usage_limit"),
    ({"error_result": "Overloaded"}, "result_error"),
])
def test_harness_errors_produce_no_record(export, scenario, reason):
    with pytest.raises(HarnessError) as ei:
        run(export, scenario)
    assert ei.value.reason == reason
    assert ei.value.counted == (reason not in {"init_check", "auth", "usage_limit"})


def test_model_switch_after_submission_is_ignored(export):
    rec = run(export, {"steps": [{"submit": SUB_B}, {"end_turn": True,
                                                     "model": "claude-sonnet-5-5"}]})
    assert rec["status"] == "submitted" and rec["valid"]


# ------------------------------------------------------------------ stage integration
def test_stage_pass_with_fake_claude(tmp_path, monkeypatch):
    """interpret_stage under backend claude_code: records, aborts log, 3-attempt rule."""
    from weightspace import experiment as ex
    from weightspace.config import load_config

    cfg = load_config(REPO / "configs" / "smoke.yaml")
    cfg.root = tmp_path
    cfg.interpretation = dict(cfg.interpretation, backend="claude_code", os_sandbox=False)
    validated = ex.train_stage(cfg)
    ex.build_units(cfg, validated)
    P = ex.Paths(cfg)
    sub_b = dict(SUB_B)

    def scen(s):
        monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", json.dumps(s))

    # usage limit: pass stops, nothing recorded, abort not counted
    scen({"error_result": "usage limit reached"})
    r = ex.interpret_stage(cfg, "B", cc_executable=FAKE)
    assert not r["complete"] and "usage_limit" in r["stopped"]
    assert not list((P.preds / cfg.stage).glob("B/*.json"))
    # counted failure 3x on the same session -> recorded as harness_failure
    scen({"error_result": "Overloaded"})
    for _ in range(2):
        r = ex.interpret_stage(cfg, "B", cc_executable=FAKE)
        assert not r["complete"] and r["done"] == 0
    r = ex.interpret_stage(cfg, "B", cc_executable=FAKE)
    recs = [json.loads(p.read_text()) for p in (P.preds / cfg.stage).glob("B/*.json")]
    assert len(recs) == 1 and recs[0]["status"] == "harness_failure" and not recs[0]["valid"]
    aborts = (P.preds / cfg.stage / "_harness_aborts.jsonl").read_text().splitlines()
    assert len(aborts) == 5  # 1 uncounted + 3 counted on one unit + 1 on the next unit
    # healthy harness: the rest completes
    sub_b["network_id"] = "x"
    scen({"steps": [{"call": "get_architecture"}, {"submit": sub_b}]})
    r = ex.interpret_stage(cfg, "B", cc_executable=FAKE)
    assert r["complete"], r
    recs = [json.loads(p.read_text()) for p in (P.preds / cfg.stage).glob("*/*.json")]
    assert all(x["backend"] == "claude_code" for x in recs)


def test_unprobed_claude_code_version_is_refused(export, tmp_path):
    os.environ["FAKE_CLAUDE_SCENARIO"] = json.dumps({"steps": [{"submit": SUB_B}]})
    try:
        with pytest.raises(HarnessError) as ei:
            run_session_cc(network_id="network_017", protocol="B", condition="weights_tools",
                           icfg=ICFG, export=export, candidate_order=CANDS, os_sandbox=False,
                           executable=FAKE, probed_versions={"2.1.318"})
        assert ei.value.reason == "init_check" and "probe" in ei.value.detail
        rec = run_session_cc(network_id="network_017", protocol="B", condition="weights_tools",
                             icfg=ICFG, export=export, candidate_order=CANDS, os_sandbox=False,
                             executable=FAKE, probed_versions={"fake"})
        assert rec["valid"]
    finally:
        os.environ.pop("FAKE_CLAUDE_SCENARIO", None)


def test_passing_probe_versions(tmp_path):
    from weightspace.claude_code_backend import passing_probe_versions

    d = tmp_path / "results" / "probes"
    d.mkdir(parents=True)
    (d / "claude_code_probe_a.json").write_text(json.dumps(
        {"passed": True, "claude_code_version": "2.1.318"}))
    (d / "claude_code_probe_b.json").write_text(json.dumps(
        {"passed": False, "claude_code_version": "2.1.319"}))
    assert passing_probe_versions(tmp_path) == {"2.1.318"}

