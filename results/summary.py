"""The results table: every builder run beside random play, the all-zero network and PPO.

    .venv/bin/python -m results.summary            # scores PPO checkpoints not yet scored, writes results/RESULTS.md

PPO checkpoints are scored with the same harness.evaluate on the same held-out games,
so "as good as PPO after N steps" compares like with like.
"""
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python"
SCEN = ["basic", "defend_the_center", "health_gathering"]
MODES = ["objects", "pixels"]


def evaluate(scenario, mode, out, weights=None, player="weights"):
    if not out.exists():
        cmd = [str(PY), "-m", "harness.evaluate", "--scenario", scenario, "--mode", mode, "--player", player,
               "--out", str(out)] + (["--weights", str(weights)] if weights else [])
        subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    return json.loads(out.read_text())


def fmt(e):
    return f"{e['mean']:.1f} ± {e['se']:.1f}" if e else "-"


def main():
    base = REPO / "results" / "baselines"
    rows = ["| scenario | view | random | all-zero | PPO (steps: score) | written by | practice | score | kills/game |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in SCEN:
        for m in MODES:
            rnd = evaluate(s, m, base / f"{s}-{m}-random.json", player="random")
            zer = evaluate(s, m, base / f"{s}-{m}-zeros.json", player="zeros")
            ppo_dir = REPO / "results" / "ppo" / f"{s}-{m}"
            curve = []
            if (ppo_dir / "final.npz").exists():
                for ck in sorted(ppo_dir.glob("steps*.npz")) + [ppo_dir / "final.npz"]:
                    e = evaluate(s, m, ppo_dir / f"{ck.stem}.eval.json", weights=ck)
                    label = "final" if ck.stem == "final" else f"{int(ck.stem[5:]) / 1e6:g}M"
                    curve.append(f"{label}: {e['mean']:.1f}")
            runs = []
            for rj in sorted((REPO / "runs").glob(f"{s}-{m}-b*/run.json")):
                r = json.loads(rj.read_text())
                runs.append(r)
            ppo_txt = ", ".join(curve) if curve else "training"
            if not runs:
                rows.append(f"| {s} | {m} | {fmt(rnd)} | {fmt(zer)} | {ppo_txt} | - | - | - | - |")
            for r in runs:
                sc = r.get("score")
                score = f"{sc['mean']:.1f} ± {sc['se']:.1f}" if sc else f"failed ({r.get('exit')})"
                kills = f"{sc['kills_mean']:.2f}" if sc else "-"
                who = r["model"] + (" (smoke)" if r["id"].endswith("smoke") else "")
                rows.append(f"| {s} | {m} | {fmt(rnd)} | {fmt(zer)} | {ppo_txt} | {who} | {r['budget']} | {score} | {kills} |")
    text = "# Results\n\nHeld-out games: basic 300, the others 100. Score is the game's own reward per game, mean ± standard error.\n\n" + "\n".join(rows) + "\n"
    (REPO / "results" / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
