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
| `builder/queue.py`, `builder/jobs-*.txt` | Runs a list of builder jobs one at a time, paced against `~/.agents/bin/usage`; stops on a job that never started |
| `results/floor_check.py`, `results/predictions.md` | The acid-floor colour check on real frames; every run's bet beside its score |
| `docs/` | The README's pictures and GIFs, made from the video's frames and recordings |

- Security gate: `bash scripts/audit-deps.sh && bash scripts/audit-code.sh` (run it before every merge; there is no CI).
  Fix a finding first; carry one only with its reason (`osv-scanner.toml`, or a `nosemgrep` / `gitleaks:allow` comment at the line).

## Traps

- **Builder workspaces live outside every repo** (`/private/tmp/doom-weights-ws`, `DOOM_WS_ROOT`), so no CLAUDE.md is discovered.
  Everything worth keeping is copied into `runs/<id>/` when the builder ends.
- **The builder's PATH is only its numpy venv plus `/usr/bin:/bin`**, so the launcher calls `claude` by its full path.
- **Never let the eval seed reach a builder.** Practice uses `PRACTICE_SEED + round`; held-out games use `EVAL_SEED`.
- **`basic` objects is nearly solved by a five-unit network** (a Haiku smoke run scored 77.5 with two practice rounds):
  the hard tests are `pixels` and the two longer scenarios.
- Weights, videos, PPO checkpoints and the raw run records (transcripts, tool calls, practice inputs, `runs-failed/`) are
  gitignored. `build.py` and the PPO seed regenerate the weights; the weights and raw records are published as GitHub
  release assets, never committed.
- **Pi on a local model waits silently when the model server refuses to load** (out of memory, 2026-11-05: five 90-minute
  runs produced nothing). `run_builder.py` asks mlx-serve for two tokens first and exits non-zero; the queue then stops.
- **PPO logs moves, not games.** Any "games of training" figure is an estimate (moves over each checkpoint's game length);
  log episode counts in the next PPO run if a count matters.
- **Level 4 (`deadly_corridor`) is the hard one**: PPO never beat holding forward (231.8) in 3M moves.
