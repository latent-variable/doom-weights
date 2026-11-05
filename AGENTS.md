# AGENTS.md

doom-weights: a fresh agent writes (never trains) a neural network that plays Doom; we audit and score it.
The README is the human story; this file is the agent's.

## Where things are

| Path | What |
|---|---|
| `harness/env.py` | The game wrapper and the exact input encoding. Given to builders as source: change it and every past run's brief is stale |
| `harness/net.py` | The fixed network and weight checks. Also given to builders |
| `harness/evaluate.py` | Scores a player on held-out games (`EVAL_SEED`), records mp4s |
| `harness/practice.py` | One practice round for a builder with a budget (`PRACTICE_SEED + round`) |
| `builder/BRIEF.md`, `builder/run_builder.py` | The brief template and the launcher: workspace, agent, rebuild, audit, score |
| `builder/audit.py` | Rule flags from code and transcript; token use from the transcript's result line |
| `baselines/ppo.py` | PPO on the same inputs and shape, exported to the same `.npz` |
| `results/summary.py` | `results/RESULTS.md` |

## Traps

- **Builder workspaces live outside every repo** (`/private/tmp/doom-weights-ws`, `DOOM_WS_ROOT`), so no CLAUDE.md is discovered.
  Everything worth keeping is copied into `runs/<id>/` when the builder ends.
- **The builder's PATH is only its numpy venv plus `/usr/bin:/bin`**, so the launcher calls `claude` by its full path.
- **Never let the eval seed reach a builder.** Practice uses `PRACTICE_SEED + round`; held-out games use `EVAL_SEED`.
- **`basic` objects is nearly solved by a five-unit network** (a Haiku smoke run scored 77.5 with two practice rounds):
  the hard tests are `pixels` and the two longer scenarios.
- Weights, videos and PPO checkpoints are gitignored; `build.py` and the PPO seed regenerate them.
