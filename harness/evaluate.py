"""Score a player over fixed held-out games, and optionally record some as video.

    python -m harness.evaluate --scenario basic --mode objects --weights w.npz --out result.json
    python -m harness.evaluate --scenario basic --mode objects --player random --out r.json

The held-out games use EVAL_SEED, which the builder never sees played: its practice
games (budget above 0) use PRACTICE_SEED.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from harness import env as E
from harness import net as N

EVAL_SEED = 20261104
PRACTICE_SEED = 7
EPISODES = {"basic": 300, "defend_the_center": 100, "health_gathering": 100, "deadly_corridor": 100}


def player(kind, scenario, weights=None, seed=0):
    d, a = E.input_size(scenario), len(E.SCENARIOS[scenario]["actions"])
    if kind == "random":
        rng = np.random.default_rng(seed)
        return lambda x: int(rng.integers(a)), None
    w = N.zeros(d, a) if kind == "zeros" else N.load(weights, d, a)
    return (lambda x: N.act(w, x)), w


def play(scenario, mode, choose, episodes, seed, record=None, record_n=0, trace=None):
    """Play `episodes` games; return per-game results. `record` is a folder for mp4s."""
    game = E.Doom(scenario, mode, seed)
    n_act = len(E.SCENARIOS[scenario]["actions"])
    results = []
    for ep in range(episodes):
        frames = [] if (record and ep < record_n) else None
        x, score, done, counts = game.reset(), 0.0, False, [0] * n_act
        if frames is not None:
            frames.append(game.last_screen.copy())
        while not done:
            a = choose(x)
            counts[a] += 1
            if trace is not None and ep < trace["episodes"]:
                trace["steps"].append((ep, x, a))
            x, r, done = game.step(a, on_tic=(lambda s: frames.append(s.copy())) if frames is not None else None)
            score += r
        st = game.stats()
        results.append({"score": score, "steps": sum(counts), "actions": counts, **st})
        if frames is not None:
            write_mp4(frames, Path(record) / f"game{ep:02d}.mp4")
    game.close()
    return results


def write_mp4(frames, path):
    """35 fps, each frame doubled to 640 x 480 (nearest neighbour, so the pixels stay sharp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", "35", "-i", "-", "-vf", "scale=iw*2:ih*2:flags=neighbor",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def summarise(results):
    s = np.array([r["score"] for r in results], float)
    acts = np.sum([r["actions"] for r in results], axis=0)
    return {"episodes": len(results), "mean": float(s.mean()),
            "se": float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0,
            "median": float(np.median(s)), "min": float(s.min()), "max": float(s.max()),
            "kills_mean": float(np.mean([r["kills"] for r in results])),
            "tics_mean": float(np.mean([r["tics"] for r in results])),
            "action_share": (acts / max(acts.sum(), 1)).round(4).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=sorted(E.SCENARIOS))
    ap.add_argument("--mode", required=True, choices=sorted(E.MODES))
    ap.add_argument("--player", default="weights", choices=["weights", "random", "zeros"])
    ap.add_argument("--weights")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--record", help="folder for mp4s of the first games")
    ap.add_argument("--record-n", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.player == "weights" and not a.weights:
        ap.error("--weights is required for --player weights")
    episodes = a.episodes or EPISODES[a.scenario]
    choose, _ = player(a.player, a.scenario, a.weights, seed=a.seed)
    t = time.time()
    results = play(a.scenario, a.mode, choose, episodes, a.seed, a.record, a.record_n)
    out = {"scenario": a.scenario, "mode": a.mode, "player": a.player, "weights": a.weights,
           "seed": a.seed, "seconds": round(time.time() - t, 1), **summarise(results), "games": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{a.scenario} {a.mode} {a.player}: mean {out['mean']:.1f} ± {out['se']:.1f} "
          f"over {episodes} games ({out['seconds']} s)", file=sys.stderr)


if __name__ == "__main__":
    main()
