# doom-weights

Can a language model **write** a neural network that plays Doom, without training it?

In November 2026 an MIT team showed GPT-6 Astra writing game-playing networks from scratch: every
weight starting at zero, the game's rules and source code, no data, no gradients, no practice games
(Takehi, Teoh, Zhang and Khattab, "Sparks of Weight-Space Programming", Experiment 5). Its Breakout
network scored about twenty times a PPO network trained for 500 million steps. This repo asks the same
of Claude, on real Doom.

A fresh agent gets a brief, the harness's source, and Python with numpy only. It hands back `build.py`, a
program that writes the weights. We run that program again ourselves, check the rules, and score the
network on held-out games it never saw.

## The test

| | |
|---|---|
| Game | Real Doom through [ViZDoom](https://vizdoom.farama.org) 1.3.2, headless, Freedoom graphics |
| Scenarios | `basic` (shoot one monster), `defend_the_center` (turn and shoot monsters closing in), `health_gathering` (find medkits on acid) |
| Network | input → 128 → 128 → actions, ReLU, argmax; no memory between steps; every weight starts at zero |
| What it sees | `objects`: a 30 × 40 grid of monster, pickup and depth (like the paper's object grids); `pixels`: the same grid of mean RGB (the "plays Doom" claim) |
| Practice | budget 0 (never plays before it is scored, the paper's Experiment 5) or N rounds of 5 practice games each, with screens and the inputs the network saw |
| Forbidden | training of any kind, gradients, fitting, search over weights, installing anything, the internet, the game files |
| Score | the scenario's own reward, mean over 300 (`basic`) or 100 held-out games |
| Yardsticks | random play, the all-zero network, PPO trained on the same inputs and network shape, scored at checkpoints |

Exact encoding: `harness/env.py`. Exact network: `harness/net.py`. The brief: `builder/BRIEF.md`.

## Run it

```
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt -r requirements-ppo.txt pytest
.venv/bin/python -m pytest -q tests

# one builder: a fresh agent writes the weights, then we audit and score them
.venv/bin/python -m builder.run_builder --scenario basic --mode objects --budget 0
.venv/bin/python -m builder.run_builder --scenario basic --mode pixels --budget 20 --model claude-opus-5-5

# the trained yardstick (CPU, about 15 minutes per 2 million steps)
.venv/bin/python -m baselines.ppo --scenario basic --mode objects --steps 1000000

# score a weights file, record three games as mp4
.venv/bin/python -m harness.evaluate --scenario basic --mode objects --weights runs/<id>/rebuilt.npz \
    --record /tmp/vids --out /tmp/eval.json

# the table
.venv/bin/python -m results.summary
```

`claude` must be signed in. Each builder run is a full agent session and spends the plan's budget.

## What a run leaves

`runs/<scenario>-<view>-b<budget>-<time>/`:

| File | What |
|---|---|
| `BRIEF.md` | Exactly what the builder was told |
| `transcript.jsonl` | The builder's whole session, every tool call and reply |
| `build.py`, `explanation.md` | Its program and its account of what each layer computes, with a predicted score |
| `run.json` | Model, minutes, tokens, the audit (flags, rebuilt, matches submitted) and the score |
| `eval.json` | Every held-out game |
| `videos/` | Three held-out games as mp4 (not committed) |
| `practice_runs/` | What practice showed it, when it had a budget |

## How the rules are kept

- The builder starts in a folder outside every repo, with `--setting-sources project`, so it sees no user
  instructions or memory (checked: a Haiku probe reported none).
- Its tools: read, write, edit, and Bash limited to `python`, `ls`, `cat`, `head`, `wc` and `./practice`.
  The web, subagents and skills are denied. Its PATH holds only a numpy-only Python.
- Doom is not installed where it works. Practice runs in our own Python, outside its folder, and counts
  rounds in a file it cannot reach.
- `builder/audit.py` re-runs `build.py` in a clean folder and flags fitting, learning libraries, randomness,
  long loops, installs, network access and looking for the game, in the code and in every command. A flag
  is a lead for a person to read, not a verdict.

## What is yours to decide

- Whether a flagged run counts. Read its `explanation.md` and `build.py`.
- How much budget a batch of builder runs may spend (`run.json` records each run's tokens and cost).
- Whether a result goes public (a ShipShow video, a post). The repo is private.
