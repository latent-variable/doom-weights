# defend_the_center, `objects` input: a hand-written network

## Prediction

**Average score on held-out games: about 13** (roughly 14 kills, minus 1 for the death that ends
almost every game).

Confidence is low. I have never seen the game run, and the score depends mostly on things I had to
recall or guess (room size, monster speeds and damage, sprite sizes). My 50% interval is 8 to 18 and
my 90% interval is 4 to 23. The hard ceiling is 25–26 (26 bullets).

What I am fairly confident of: the network computes exactly the program below (checked on 3,000
random frames against a plain-Python version), it very rarely wastes a bullet, and it has no way to
get permanently stuck. What I am not confident of is how many monsters get through and bite before it
turns to face them.

## The program the weights implement

Let `C[c]` be the amount of monster in screen column `c`, in grid cells (channel 0 summed over the 30
rows). The crosshair is the line between columns 19 and 20. Each step:

1. **ATTACK** if a monster straddles the crosshair with a margin:
   - `C[19] > 0.25` and `C[20] > 0.25` (monster on both sides of the line), and
   - `left6 ≥ ρ·right6` and `right6 ≥ ρ·left6`, where `left6`/`right6` are the monster cells in the
     6 columns left/right of the line. This keeps the crosshair off the outer edge of the sprite by a
     fraction of the monster's own width, whatever its distance.
   - `ρ = 0.15` normally (about 13% of the width as margin). It drops to `ρ = 0.04` once
     `ammo/26 − 1.5·health/100 + 0.4 > 0`, i.e. once health has fallen behind ammo.
2. otherwise **TURN_LEFT** if `Σ_{c<20} P[c] − Σ_{c≥20} P[c] > 0.3`, where
   `P[c] = relu(C[c−1] + C[c] + C[c+1] − 5.5)`: a big (close) monster is on the left and outweighs
   whatever is on the right.
3. otherwise **TURN_RIGHT**: the default sweep.

Only the monster channel and the two game variables are read. The pickup and depth channels have zero
weight.

## What each layer computes

**Layer 1 (54 of 128 units used)**
- Units 0–39: `P[c]`, one per column. A unit is on only when three adjacent columns hold more than
  5.5 cells of monster, which a far, small monster never does.
- Units 40–53: seven tests, each as a pair `(relu(10·t), relu(10·t − 1))` whose difference is
  `clip(10·t, 0, 1)`, a 0-to-1 switch. The tests are `C[19] − 0.25`, `C[20] − 0.25`, the two balance
  tests with strict ρ, the same two with loose ρ, and the health-versus-ammo gate.

**Layer 2 (4 units used)**
- Unit 0: `relu(t0 + t1 + t2 + t3 − 3)`, the AND of the four strict fire tests.
- Unit 3: `relu(t0 + t1 + t4 + t5 + t6 − 4)`, the AND of the loose fire tests and the gate.
- Units 1, 2: `relu(10·s)` and `relu(10·s − 1)` with `s = left pull − right pull − 0.3`; their
  difference is a 0-to-1 "go left" switch.

**Layer 3**
- `q_left = go-left switch` (0 to 1), `q_right = 0.5` (bias only), `q_attack = 10 × (unit 0 + unit 3)`.
- So ATTACK wins whenever a fire AND holds, LEFT wins when its switch is above 0.5, RIGHT otherwise.

## Why this design

**Fire only when the shot is near-certain.** There are 26 bullets and each can be worth one point, so
a miss costs a point. The pistol has no horizontal auto-aim; the bullet goes where the crosshair is.
Requiring monster pixels on both sides of the crosshair, with a width-proportional margin, covers the
sideways drift of a zig-zagging monster during the few tics between the decision and the bullet.

**Aiming is coarse, so far monsters are skipped rather than chased.** From what I remember of the
engine, a held turn key moves about 14° per 4-tic step (about 5 grid columns), and only releasing the
turn keys, which here means pressing ATTACK, resets it to the slow 7° first step. A network with no
memory therefore cannot fine-aim: if a thin monster falls between two 14° positions, turning back
just flips between the same two views. So the rule only turns back (left) for monsters big enough to
be hit within a few steps; small far ones are fired at only when the sweep happens to land on them,
and are otherwise left to come closer. The 3-column window makes wide monsters count from slightly
farther out than slim ones, which is the right direction if the fast monsters are the bulky ones.

**The sweep finds what is behind.** The player sees 90° of 360°. With nothing worth turning back for,
it keeps turning right, which shows every direction about every 3 seconds.

**Loosening when health falls behind ammo.** Bullets are only worth protecting if the player lives to
use them. Falling health is the evidence that monsters are getting through and time, not ammo, is the
scarce thing, so the margin is relaxed to shoot sooner.

**No dependence on screen rows.** I could not be sure whether the status bar is drawn (which moves the
horizon), so everything uses column sums.

## How it was checked (no game, no fitting)

- `test_frames.py` pushes hand-made frames through the harness's real `encode` and `forward`: empty
  screen, monsters of four sizes on, near and off the crosshair, both sides, corpses (ignored by
  `encode`), the health/ammo gate. All pass, and the network matches a plain-Python statement of the
  program on 3,000 random frames (largest difference in any score 0.002).
- `sim_check.py` is a crude stand-in for the game that I wrote from memory of Doom's rules (turn
  speeds, pistol timing, 8-direction monster movement). I used it to look for logic failures in
  closed loop: endless left-right flipping, firing loops, wasted bullets. It showed about 0.1 to 1
  wasted bullets per game and no stuck states. It also showed that left-right flipping on a
  slow, thin monster can last 20+ steps, which is the main time sink.
- The stand-in is where my number comes from, and it is not reliable. Across ten guessed
  configurations (room radius 350 to 1,000 units, different monster mixes and respawn delays) the
  network averaged between 6.7 and 17.4, always ending in death with ammo left. A random policy
  dies there in 130 to 300 tics; I believe random play in the real game lasts longer than that, so I
  placed my estimate in the upper half of that range.
- I compared a handful of hand-picked variants in the stand-in early on (no turn-back at all, nearer
  and farther turn-back thresholds, a smaller margin). The results confirmed the reasoning above
  (fast monsters favour turning back early, slow ones favour skipping) and prompted the health/ammo
  gate. The final constants are round values chosen by that reasoning, not the best-scoring ones in
  the stand-in.

## Known weaknesses

- If two monsters stand either side of the crosshair with a gap under about 2 columns, it fires into
  the gap.
- A monster that reaches the player from behind bites until the sweep comes round (up to about
  3 seconds).
- A second target within about 20° of the last kill is shot while the pistol is still cycling, which
  the engine makes less accurate; this only matters for far, thin targets.
- If the engine labels corpses with the live monster's name (the harness says it does not), the
  network would shoot at corpses.
- There is no clock input, so it cannot pace its ammo to outlast the 2,100-tic limit.

## Files

- `build.py`: writes `weights.npz` (numpy only, reads nothing, runs in well under a second).
- `weights.npz`: its output.
- `test_frames.py`, `sim_check.py`: the checks described above; not needed by the harness.
