"""Run one builder: a fresh agent writes a Doom network's weights, then we check and score it.

    .venv/bin/python -m builder.run_builder --scenario basic --mode objects --budget 0

Steps: make an isolated workspace (the brief, the harness source, a numpy-only Python),
start the agent there with no user instructions loaded, save its whole transcript, run
its build.py again in a clean folder, audit the rules, then score the weights on the
held-out games and record three of them.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from builder import audit
from harness import env as E

REPO = Path(__file__).resolve().parent.parent
WS_ROOT = Path(os.environ.get("DOOM_WS_ROOT", "/private/tmp/doom-weights-ws"))
PY = REPO / ".venv" / "bin" / "python"

MODE_TEXT = {
    "objects": ("You see objects, not pixels. Per grid cell, channel 0 is the share of the cell's pixels that "
                "show a monster, channel 1 the share that show a pickup (a medkit, armour, ammo), and channel 2 the "
                "mean depth (distance to whatever is drawn there: 0 touching the camera, 1 far)."),
    "pixels": ("You see the screen itself: per grid cell, the mean red, green and blue of its 64 pixels, each "
               "from 0 to 1. Nothing tells you what is a monster or a wall; that is in the colours."),
}

ALLOWED = ["Read", "Write", "Edit", "Glob", "Grep", "Bash(python:*)", "Bash(python3:*)",
           "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(wc:*)", "Bash(./practice:*)"]
PI_MODEL = "mlxserve/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
MLX_SERVE = "http://127.0.0.1:11234/v1/chat/completions"


def model_answers(model):
    """Ask mlx-serve for two tokens, so a model it will not load fails here, not 90 minutes later.

    Pi waits silently when the server refuses (out of memory: 2026-11-05, five runs lost)."""
    import urllib.error
    import urllib.request
    body = json.dumps({"model": model.split("/", 1)[1], "max_tokens": 2,
                       "messages": [{"role": "user", "content": "Say ok."}]}).encode()
    req = urllib.request.Request(MLX_SERVE, body, {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            json.loads(r.read())
        return None
    except urllib.error.HTTPError as e:
        return f"{e.code}: {e.read().decode(errors='replace')[:300]}"
    except (urllib.error.URLError, TimeoutError) as e:
        return str(e)
DENIED = ["WebFetch", "WebSearch", "Agent", "Task", "Skill", "NotebookEdit"]


def brief(scenario, mode, budget):
    spec = E.SCENARIOS[scenario]
    acts = "\n".join(f"- {i}: {' + '.join(a)}" for i, a in enumerate(spec["actions"]))
    if budget:
        rule = (f"- **Practice: {budget} rounds.** `./practice` plays your current `weights.npz` for 5 practice "
                "games (not the held-out ones) and writes the scores, the actions taken, and a few screens with "
                "the exact inputs your network saw into `practice_runs/<round>/`. You may look at all of it and adjust "
                "your weights by reasoning, but never fit weights to it (the first rule).")
    else:
        rule = ("- **No practice.** Your network never plays before it is scored. You have never seen this "
                "game run and will not.")
    return (REPO / "builder" / "BRIEF.md").read_text().format(
        scenario=scenario, rules=spec["rules"], actions=acts, mode=mode, mode_text=MODE_TEXT[mode],
        d=E.input_size(scenario), a=len(spec["actions"]), budget_rule=rule)


def make_workspace(run, ws, scenario, mode, budget):
    ws.mkdir(parents=True)
    (ws / "BRIEF.md").write_text(brief(scenario, mode, budget))
    (ws / "harness_src").mkdir()
    for f in ("env.py", "net.py"):
        shutil.copy(REPO / "harness" / f, ws / "harness_src" / f)
    subprocess.run(["uv", "venv", "-q", "-p", "3.12", str(ws / ".venv")], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "-p", str(ws / ".venv" / "bin" / "python"), "numpy"], check=True)
    if budget:
        p = ws / "practice"
        p.write_text(f"#!/bin/sh\ncd '{REPO}' && exec '{PY}' -m harness.practice '{run}' 2>/dev/null\n")
        p.chmod(0o755)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=sorted(E.SCENARIOS))
    ap.add_argument("--mode", required=True, choices=sorted(E.MODES))
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--agent", default="claude", choices=["claude", "pi"],
                    help="claude: Claude Code; pi: Pi on a local model (no budget)")
    ap.add_argument("--model", default=None,
                    help="default claude-opus-5-5, or for pi the local Qwen on mlx-serve")
    ap.add_argument("--effort", default=None, help="claude --effort, if set")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    a.model = a.model or (PI_MODEL if a.agent == "pi" else "claude-opus-5-5")
    if a.agent == "pi" and a.model.startswith("mlxserve/"):
        err = model_answers(a.model)
        if err:
            sys.exit(f"model server will not run {a.model}: {err}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = f"{a.scenario}-{a.mode}-b{a.budget}-{stamp}{('-' + a.tag) if a.tag else ''}"
    run = REPO / "runs" / rid
    ws = WS_ROOT / rid / "workspace"
    run.mkdir(parents=True)
    cfg = {"id": rid, "agent": a.agent, "scenario": a.scenario, "mode": a.mode, "budget": a.budget, "model": a.model,
           "effort": a.effort, "workspace": str(ws), "started": stamp}
    (run / "run.json").write_text(json.dumps(cfg, indent=1))
    make_workspace(run, ws, a.scenario, a.mode, a.budget)
    shutil.copy(ws / "BRIEF.md", run / "BRIEF.md")

    if a.agent == "pi":
        # A clean Pi config: the user's model list and settings only, so no personal
        # instructions, skills or extensions reach the builder. Pi's bash has no
        # command allowlist; the numpy-only PATH and the audit keep the rules.
        pidir = WS_ROOT / rid / "pi-agent"
        pidir.mkdir(parents=True)
        for f in ("models.json", "settings.json"):
            shutil.copy(Path.home() / ".pi" / "agent" / f, pidir / f)
        cmd = [shutil.which("pi") or "pi", "-p", "--model", a.model, "--mode", "json", "--no-context-files",
               "--no-skills", "--no-extensions", "--no-session", "--tools", "read,bash,edit,write",
               "Read BRIEF.md in this folder and do what it asks."]
        extra_env = {"PI_CODING_AGENT_DIR": str(pidir)}
        node_dir = os.path.dirname(shutil.which("node") or "/usr/bin/node")
    else:
        extra_env, node_dir = {}, None
        allowed = [t for t in ALLOWED if a.budget or t != "Bash(./practice:*)"]
        cmd = [shutil.which("claude") or "claude", "-p", "Read BRIEF.md in this folder and do what it asks.", "--model", a.model,
               "--setting-sources", "project", "--output-format", "stream-json", "--verbose",
               "--permission-mode", "default", "--allowedTools", *allowed, "--disallowedTools", *DENIED]
        if a.effort:
            cmd += ["--effort", a.effort]
    path = f"{ws / '.venv' / 'bin'}:/usr/bin:/bin" + (f":{node_dir}" if node_dir else "")
    env = {**os.environ, **extra_env, "PATH": path, "VIRTUAL_ENV": str(ws / ".venv")}
    env.pop("PYTHONPATH", None)
    print(f"[{rid}] builder started", flush=True)
    t = time.time()
    with open(run / "transcript.jsonl", "w") as tr:
        try:
            proc = subprocess.run(cmd, cwd=ws, env=env, stdout=tr, stderr=subprocess.STDOUT,
                                  timeout=a.max_minutes * 60)
            cfg["exit"] = proc.returncode
        except subprocess.TimeoutExpired:
            cfg["exit"] = "timeout"
    cfg["builder_minutes"] = round((time.time() - t) / 60, 1)
    cfg["usage"] = audit.usage(run / "transcript.jsonl")
    print(f"[{rid}] builder finished in {cfg['builder_minutes']} min, exit {cfg['exit']}", flush=True)

    for f in ("build.py", "explanation.md", "weights.npz"):
        if (ws / f).exists():
            shutil.copy(ws / f, run / f)
    if (ws / "practice_runs").is_dir():
        shutil.copytree(ws / "practice_runs", run / "practice_runs", dirs_exist_ok=True)
    cfg["audit"] = audit.check(run, ws, a.scenario)
    (run / "run.json").write_text(json.dumps(cfg, indent=1))
    if not cfg["audit"]["rebuilt"]:
        print(f"[{rid}] no valid rebuilt weights: {cfg['audit']['notes']}", flush=True)
        return 1
    subprocess.run([str(PY), "-m", "harness.evaluate", "--scenario", a.scenario, "--mode", a.mode,
                    "--weights", str(run / "rebuilt.npz"), "--record", str(run / "videos"),
                    "--out", str(run / "eval.json")], cwd=REPO, check=True)
    ev = json.loads((run / "eval.json").read_text())
    cfg["score"] = {k: ev[k] for k in ("mean", "se", "median", "kills_mean", "action_share")}
    (run / "run.json").write_text(json.dumps(cfg, indent=1))
    print(f"[{rid}] SCORE {ev['mean']:.1f} ± {ev['se']:.1f}  audit flags: {cfg['audit']['flags']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
