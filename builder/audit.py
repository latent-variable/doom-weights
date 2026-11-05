"""Check a builder's work against the rules, and read its token use.

A flag is a lead for a person to read, not a verdict: `np.linalg.solve` can be a
reasoned construction or a fit. Every flag names the file or command it came from.
"""
import ast
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from harness import env as E
from harness import net as N

OK_IMPORTS = {"numpy", "math", "json", "itertools", "functools", "collections", "typing", "dataclasses",
              "os", "sys", "pathlib", "struct", "__future__", "argparse", "textwrap"}
CODE_FLAGS = {
    r"\blstsq\b|\bpinv\b|polyfit": "least squares or a pseudo-inverse (a fit?)",
    r"\bgrad|backprop|optimi[sz]": "gradient or optimizer words",
    r"\bloss\b": "a loss",
    r"torch|jax|tensorflow|sklearn|scipy": "a learning library",
    r"vizdoom|\.wad\b": "the game itself",
    r"np\.random|default_rng|random\.": "randomness (random features are fine; random search is not)",
    r"for .* in range\(\s*\d{3,}": "a long loop (search?)",
}
CMD_FLAGS = {
    r"\b(pip|uv|conda|brew|apt)\b.*install": "installing a package",
    r"curl|wget|git clone|http": "network access",
    r"vizdoom|freedoom|\.wad": "looking for the game",
    r"find /|ls /(?!private/tmp)|\.\./": "looking outside the workspace",
}


def _events(transcript):
    for line in Path(transcript).read_text().splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def usage(transcript):
    """Totals from the agent's final result line (Claude Code) or its messages (Pi)."""
    pi = [m for m in _events(transcript) if m.get("type") == "message_end"
          and m.get("message", {}).get("role") == "assistant"]
    if pi:
        u = [m["message"].get("usage", {}) for m in pi]
        last = "".join(c.get("text", "") for c in pi[-1]["message"].get("content", []) if c.get("type") == "text")
        return {"turns": len(pi), "cost_usd": 0.0, "input": sum(x.get("input", 0) for x in u),
                "output": sum(x.get("output", 0) for x in u), "cache_read": sum(x.get("cacheRead", 0) for x in u),
                "cache_write": 0, "denials": [], "result": last[:2000],
                "errors": [m["message"].get("errorMessage") for m in pi if m["message"].get("errorMessage")][:5]}
    for line in reversed(Path(transcript).read_text().splitlines()):
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("type") == "result":
            u = m.get("usage", {})
            return {"turns": m.get("num_turns"), "cost_usd": m.get("total_cost_usd"),
                    "input": u.get("input_tokens"), "output": u.get("output_tokens"),
                    "cache_read": u.get("cache_read_input_tokens"),
                    "cache_write": u.get("cache_creation_input_tokens"),
                    "denials": [d.get("tool_name") for d in m.get("permission_denials", [])],
                    "result": (m.get("result") or "")[:2000]}
    return {}


def tool_calls(transcript):
    calls = []
    for m in _events(transcript):
        if m.get("type") == "message_end" and m.get("message", {}).get("role") == "assistant":  # Pi
            for c in m["message"].get("content", []):
                if c.get("type") == "toolCall":
                    name = {"bash": "Bash", "read": "Read", "write": "Write", "edit": "Edit"}.get(c.get("name"), c.get("name"))
                    calls.append({"tool": name, "input": c.get("arguments", {})})
            continue
        if m.get("type") != "assistant":
            continue
        for c in m.get("message", {}).get("content", []):
            if c.get("type") == "tool_use":
                calls.append({"tool": c["name"], "input": c.get("input", {})})
    return calls


def check(run, ws, scenario):
    run, ws = Path(run), Path(ws)
    flags, notes = [], []
    d, a = E.input_size(scenario), len(E.SCENARIOS[scenario]["actions"])
    out = {"rebuilt": False, "matches_submitted": None, "flags": flags, "notes": notes}

    for py in sorted(ws.rglob("*.py")):
        if ".venv" in py.parts or "harness_src" in py.parts:
            continue
        src = py.read_text(errors="replace")
        for pat, why in CODE_FLAGS.items():
            for i, line in enumerate(src.splitlines(), 1):
                if re.search(pat, line):
                    flags.append(f"{py.name}:{i}: {why}: {line.strip()[:120]}")
        if py.name == "build.py":
            try:
                for node in ast.walk(ast.parse(src)):
                    mods = ([n.name for n in node.names] if isinstance(node, ast.Import)
                            else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                    for m in mods:
                        if m.split(".")[0] not in OK_IMPORTS:
                            flags.append(f"build.py imports {m}")
            except SyntaxError as e:
                notes.append(f"build.py does not parse: {e}")

    calls = tool_calls(run / "transcript.jsonl")
    (run / "tool_calls.json").write_text(json.dumps(calls, indent=1))
    out["tool_counts"] = {}
    for c in calls:
        out["tool_counts"][c["tool"]] = out["tool_counts"].get(c["tool"], 0) + 1
        path = c["input"].get("file_path") or c["input"].get("path") or ""
        if c["tool"] in ("Write", "Edit") and path and not str(Path(path).resolve()).startswith(str(ws.resolve())):
            flags.append(f"{c['tool']} outside the workspace: {path}")
        if c["tool"] == "Bash":
            cmd = c["input"].get("command", "")
            for pat, why in CMD_FLAGS.items():
                if re.search(pat, cmd):
                    flags.append(f"command: {why}: {cmd[:160]}")

    rounds = ws / "practice_runs"
    out["practice_rounds_seen"] = len([d for d in rounds.iterdir() if d.is_dir()]) if rounds.is_dir() else 0
    used = run / "practice_used"
    if used.exists() and int(used.read_text()) != out["practice_rounds_seen"]:
        flags.append(f"practice counter {used.read_text()} does not match {out['practice_rounds_seen']} round folders")

    build = ws / "build.py"
    if not build.exists():
        notes.append("no build.py")
        return out
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(build, Path(tmp) / "build.py")
        try:
            p = subprocess.run([str(ws / ".venv" / "bin" / "python"), "-I", "build.py"], cwd=tmp,
                               capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            notes.append("build.py took over 120 s")
            return out
        if p.returncode != 0 or not (Path(tmp) / "weights.npz").exists():
            notes.append(f"build.py failed: {p.stderr[-800:]}")
            return out
        try:
            w = N.load(Path(tmp) / "weights.npz", d, a)
        except ValueError as e:
            notes.append(f"rebuilt weights invalid: {e}")
            return out
        shutil.copy(Path(tmp) / "weights.npz", run / "rebuilt.npz")
    out["rebuilt"] = True
    out["nonzero_weights"] = int(sum((v != 0).sum() for v in w.values()))
    out["total_weights"] = int(sum(v.size for v in w.values()))
    if (ws / "weights.npz").exists():
        try:
            sub = N.load(ws / "weights.npz", d, a)
            out["matches_submitted"] = all(np.allclose(sub[k], w[k]) for k in N.KEYS)
        except ValueError as e:
            notes.append(f"submitted weights invalid: {e}")
    return out
