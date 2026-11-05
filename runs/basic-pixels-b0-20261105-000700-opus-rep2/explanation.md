# A hand-written Doom policy for `basic` (pixels)

The policy is: **find the red blob; strafe toward it; shoot only when the crosshair is on it.**
Every weight in `build.py` comes from the reasoning below. Nothing was fitted or searched.

## Layer 1: where is the red? (120 of 128 units)

Each cell's colour is reduced to a **redness** `f = 2R − G − B`. This is exactly 0 for any gray at
any brightness, so gray walls and lighting changes don't register. Saturated red scores about 1,
and dim red (0.35, 0.08, 0.08) still scores about 0.5. Purple-ish red also stays positive.

Rows 2–22 are split into three overlapping horizontal bands (rows 2–9, 8–15, 14–22). The bottom
rows are left out because the pistol and the hand holding it are reddish. There is one unit per
(band, column):

```
E[b,c] = S[b,c] − (n_b/n)·T[c]          S = band's redness sum, T = column's redness sum over all rows
E[b,c] −= mean over columns of E[b,·]
h1[b,c] = relu(E[b,c] − 0.35)
```

Both subtractions are linear, so they live in W1:
- The **vertical surround** cancels any tint that runs evenly down a whole column, such as a
  coloured stripe in a wall texture.
- The **horizontal surround** cancels anything that spans the screen, such as a brownish floor or
  ceiling band.

What's left is a vertically compact, horizontally local patch of red, which is how a Cacodemon
looks against the far wall. The 0.35 threshold sits well above the noise I tested and well below
the signal from a dim, small monster (about 1 to 2).

## Layer 2: aim geometry (4 units)

Let `P_c = Σ_b h1[b,c]` be the red evidence in column c. The crosshair is at x = 160, which is the
boundary between columns 19 and 20.
- `m = Σ_c P_c·(19.5 − c)` is a signed moment. It is positive when the red mass is left of the
  crosshair. The units are `relu(m)` and `relu(−m)`.
- `relu(P_19)` and `relu(P_19 − P_20)`. Their difference is **min(P_19, P_20)**, which is large
  only when there is red on *both* sides of the crosshair, meaning the crosshair is inside the
  silhouette.

## Output

- `q_left = relu(m)` and `q_right = relu(−m)`: strafe toward the red mass.
- `q_attack = 1000·(min(P_19, P_20) − 0.15)`: once the crosshair is on the monster, this beats any
  move score. Otherwise it is negative.
- If nothing is red, both move scores are 0 and attack is negative, so the tie goes to MOVE_LEFT.
  The network never fires blind, so no shot (−5) is wasted on an empty screen.

## Checks I ran (synthetic frames only, built with the harness's own `encode`)

- `test_synth.py` and `stress_synth.py` place a disc on a gray room with a dark ceiling, a brownish
  floor, pixel noise, the weapon, and optional per-column hue stripes on the wall. The discs vary
  in radius (12–28 px), colour (bright red, dark red, purple-red, washed-out red) and position.
  With the wall hue at ≤ 2% the network moves the right way or shoots correctly in more than 99% of
  frames. The only misses are tiny, dim discs about 5 px off-centre, where it takes one more step
  instead of firing.
- `window_synth.py`: the network fires only when the crosshair is within about 55–80% of the
  monster's on-screen radius of its centre, so the shots it takes should hit. There were zero false
  fires in 300 monster-free noisy frames (min(P_19, P_20) stayed at exactly 0).
- `closed_loop_synth.py` is a toy loop with my guess at Doom strafe momentum, not the game. It got
  60/60 kills, each with exactly one shot, with no oscillation and no stalls.

## Prediction

If the Cacodemon renders clearly redder than its surroundings, as the brief says, each episode
should be: strafe for roughly 0–40 tics to centre on it, then one shot and a kill. That gives about
106 − 5 − (10–50 tics) per episode.

**Predicted mean score on held-out games: about 70** (likely range 55–85).

Confidence: moderate, about 65% that the mean lands in 55–85. Main risks:
1. Momentum overshoot is larger than I modelled. That would cost extra steps, not kills.
2. The monster's real colours or size are weaker than my tests (dim, very small, or not very red).
   Shots would come late, and in the worst case the agent would drift left without firing (about
   −300 per episode).
3. A reddish element of the real room that I never saw pulls the moment the wrong way.

Risks 2 and 3 are what would make the mean much lower. I'd put that at about a 15–20% chance.
