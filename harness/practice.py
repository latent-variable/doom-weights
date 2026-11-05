"""One practice round for a builder with a budget above 0: plays the builder's current
weights.npz on PRACTICE_SEED games and writes what it saw into the workspace.

Called through the workspace's `./practice` script. The count of rounds used lives in
the run folder, outside the workspace, and a round past the budget is refused.
"""
import json
import sys
from pathlib import Path

import numpy as np

from harness import env as E
from harness import evaluate as V
from harness import net as N

GAMES = 5
SHOTS = 6   # frames saved per round


def main():
    run = Path(sys.argv[1])
    cfg = json.loads((run / "run.json").read_text())
    ws = Path(cfg["workspace"])
    used_file = run / "practice_used"
    used = int(used_file.read_text()) if used_file.exists() else 0
    if used >= cfg["budget"]:
        print(f"Refused: all {cfg['budget']} practice rounds are used.")
        return 1
    sc, mode = cfg["scenario"], cfg["mode"]
    d, a = E.input_size(sc), len(E.SCENARIOS[sc]["actions"])
    try:
        w = N.load(ws / "weights.npz", d, a)
    except Exception as e:  # a bad file does not spend a round
        print(f"weights.npz is not valid: {e}")
        return 1
    used_file.write_text(str(used + 1))
    out = ws / "practice_runs" / f"{used + 1:02d}"
    out.mkdir(parents=True, exist_ok=True)
    trace = {"episodes": 1, "steps": []}
    results = V.play(sc, mode, lambda x: N.act(w, x), GAMES, V.PRACTICE_SEED + used, trace=trace)
    game = E.Doom(sc, mode, V.PRACTICE_SEED + used)   # replay game 0 to save screens
    x, shots, step, done = game.reset(), [], 0, False
    picks = set(np.linspace(0, len(trace["steps"]) - 1, SHOTS).astype(int).tolist())
    from PIL import Image
    while not done:
        q, h1, h2 = N.forward(w, x)
        if step in picks:
            Image.fromarray(game.last_screen).save(out / f"screen_step{step:03d}.png")
            np.save(out / f"input_step{step:03d}.npy", x)
            shots.append({"step": step, "q": q.round(3).tolist(), "action": int(np.argmax(q)),
                          "h1_active": int((h1 > 0).sum()), "h2_active": int((h2 > 0).sum())})
        x, _, done = game.step(int(np.argmax(q)))
        step += 1
    game.close()
    rep = {"round": used + 1, "rounds_left": cfg["budget"] - used - 1, **V.summarise(results),
           "games": results, "game0_snapshots": shots}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("round", "rounds_left", "mean", "min", "max", "action_share")}))
    print(f"Screens, inputs and the full report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
