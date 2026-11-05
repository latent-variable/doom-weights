# `basic`, objects mode: hand-set weights

No weight was fitted. `build.py` writes every nonzero weight from a rule, and `weights.npz` is its output.
The input is the 30 x 40 x 3 object grid plus ammo (index 3600), read as `(r * 40 + c) * 3 + k` from
`harness_src/env.py`. Only channel 0 (monster share) is used. Channels 1 and 2 and the ammo input get zero weight.

## What each layer computes

The crosshair is the boundary between columns 19 and 20 (pixel 160 of 320). A hitscan shot hits only if
the monster covers that pixel, so the network fires when the monster covers both of those columns, and
otherwise steers toward the side with more monster.

**Layer 1 (128 units; five are used, the rest are zero).** Each unit is a ReLU of a linear sum of channel 0.
- `Pl` = monster share summed over all rows in column 19. `Pr` = same for column 20.
- `relu(Pl - Pr)`, used to get the minimum below.
- `L` = monster mass in columns 0..19, `R` = monster mass in columns 20..39. Units `relu(L - R)` and `relu(R - L)`.

**Layer 2 (4 units used).**
- `m = min(Pl, Pr)`, computed as `Pl - relu(Pl - Pr)`. It is exact because both are nonnegative.
- Two ramps on `m`: `g_A = relu(10 (m - 0.8))` and `g_B = relu(10 (m - 1.2))`. Their difference runs from 0 to 4 as `m` goes from 0.8 to 1.2. That is the step: "crosshair is on the monster".
- `g_DP = relu(L - R)` and `g_DN = relu(R - L)`, passed through.

**Output layer.** With `G = 250` and `g_A - g_B` in [0, 4]:
- `q2 (ATTACK) = G (g_A - g_B)`, which is 0 to 1000.
- `q0 (MOVE_LEFT) = g_DP - G (g_A - g_B)`.
- `q1 (MOVE_RIGHT) = g_DN - G (g_A - g_B)`.

So when the monster covers the crosshair, ATTACK is about 1000 and both moves are pushed below zero. Otherwise ATTACK is 0 and the moves are the side masses, so the player strafes toward the bigger side. With nothing visible all scores are 0, and the tie goes to MOVE_LEFT (action 0).

## Checks

`test_synthetic.py` builds frames by hand from real column profiles (taken from the practice inputs) and checks the action. All 12 cases pass:
- Monster well left or right: moves that way.
- Monster centred on columns 18..21, or off by one column: attacks.
- Monster two columns off centre: moves toward it.
- Monster on only one of columns 19 or 20: moves toward it, because the shot would miss.
- Ammo value does not change the decision.

## Practice results (from `./practice`)

- Round 1 (first weights, attack gate on the sum of columns 19 and 20): mean 66.4, min 19, max 95, 1 kill in 5 games. One game needed 16 attack steps, so the gate was firing on partial overlap of one central column.
- Round 2 (current weights, gate on `min(Pl, Pr)`): mean 75.0, min 67, max 95, median 67, SE about 5.5. Five of five games ended in a kill.

Both rounds are the same 5 practice games, so round 2 is not held-out evidence. The change from round 1 came from reading the round-1 report and reasoning about the crosshair geometry. I did not fit any number to the practice output.

## Prediction

Expected average on held-out games: about **70** (plausible range 55 to 85).

Confidence: **moderate** on the mechanism, **low** on the exact number, since it rests on 5 games.

The main costs are time and strafing granularity. The score is roughly `106 - 5 * shots - tics`. Each strafe step lasts 4 tics and moved the monster about 3 columns in the practice inputs, so the player often has to overshoot or re-align before the gate opens. That is most of the cost, and it cannot be fixed without a memory of the previous step.

Known failure modes that the held-out games might hit:
- The monster sits outside the 90 degree field of view at the extremes of the room. No signal reaches the network, so it defaults to MOVE_LEFT and may never find it. A fixed default cannot know which side to search.
- The monster flies vertically away from the crosshair row. The shot would then miss, and the network has no vertical correction.
- The monster moves between steps, so an aligned shot can miss.
