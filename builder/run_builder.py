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
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default=None, help="claude --effort, if set")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = f"{a.scenario}-{a.mode}-b{a.budget}-{stamp}{('-' + a.tag) if a.tag else ''}"
    run = REPO / "runs" / rid
    ws = WS_ROOT / rid / "workspace"
    run.mkdir(parents=True)
    cfg = {"id": rid, "scenario": a.scenario, "mode": a.mode, "budget": a.budget, "model": a.model,
           "effort": a.effort, "workspace": str(ws), "started": stamp}
    (run / "run.json").write_text(json.dumps(cfg, indent=1))
    make_workspace(run, ws, a.scenario, a.mode, a.budget)
    shutil.copy(ws / "BRIEF.md", run / "BRIEF.md")

    allowed = [t for t in ALLOWED if a.budget or t != "Bash(./practice:*)"]
    cmd = [shutil.which("claude") or "claude", "-p", "Read BRIEF.md in this folder and do what it asks.", "--model", a.model,
           "--setting-sources", "project", "--output-format", "stream-json", "--verbose",
           "--permission-mode", "default", "--allowedTools", *allowed, "--disallowedTools", *DENIED]
    if a.effort:
        cmd += ["--effort", a.effort]
    env = {**os.environ, "PATH": f"{ws / '.venv' / 'bin'}:/usr/bin:/bin", "VIRTUAL_ENV": str(ws / ".venv")}
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
