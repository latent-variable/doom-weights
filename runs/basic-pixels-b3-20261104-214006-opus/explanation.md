# A hand-written network for `basic` (pixels)

## What the screen looks like (from practice round 1)

The room is brown: brown brick walls and a brown floor, with a dark gray grid ceiling. The Freedoom
Cacodemon is **blue and red**, not plain red. My first guess, "redness" R − (G+B)/2, fired on the
brown walls everywhere, so the agent strafed right for 300 tics in every game (score −300).

Reading the actual cell values gave a better feature: the "magenta" score

    m = R + B − 2G

Brown has R − G ≈ G − B, and gray has R = G = B, so m ≈ 0 (±0.02) on walls, floor and ceiling.
On the monster's cells, m is +0.05 to +0.16, because its blue and red parts both lack green. Muzzle
flash (yellow) and the gun and hand (skin/gray) score 0 or below, and the gun sits below row 19.

## Layers

**Layer 1 (40 of 128 units): one monster detector per screen column.**
`a_c = relu( Σ_{rows 5..19} (R + B − 2G)(r, c) − 0.15 )`.
Over the wall band, a column's sum is 0.3–0.55 where the monster is and |sum| ≤ 0.16 elsewhere
(the largest background values come from side walls seen up close). The 0.15 threshold drops the
background and keeps the monster. The monster is about 3–4 columns wide.

**Layer 2 (3 units): where the monster is relative to the crosshair.**
The crosshair is at pixel x = 160, i.e. column coordinate 19.5. With `off_c = 19.5 − (c + 0.5)`:
- `P = relu(Σ a_c · off_c)`: the monster's mass times its offset, when it is left of the crosshair
- `N = relu(−Σ a_c · off_c)`: the same, when it is right of the crosshair
- `Cm = Σ_{c=17..22} a_c`: the monster's mass in the six columns around the crosshair

**Layer 3: actions.**
`q_left = P − N`, `q_right = N − P`, `q_attack = 2 · Cm`.
The agent strafes toward the monster. When the monster is near the centre, Cm is roughly its whole
mass M and |P − N| ≈ M·|offset|, so ATTACK wins once the centroid is within about 2 columns (16 px)
of the crosshair. That is about the monster's half-width. With nothing visible, every score is 0 and
the tie goes to MOVE_LEFT.

The 2-column window is set wider than "dead centre" on purpose. The pistol fires about 4 tics after
ATTACK is pressed, and strafe momentum keeps carrying the player toward the monster during that
delay. Firing slightly early therefore lands the bullet near the centre and saves a step (about 4
points). In round 2 I used 1.5 columns; the per-step frames showed the shot landing one step after
the press with plenty of margin, so I widened it.

## Practice results

| round | network | mean | games |
|---|---|---|---|
| 1 | redness detector (wrong colour model) | −300 | never found the monster |
| 2 | magenta detector, window 1.5 cols | 75.0 | 79, 95, 67, 67, 67 |
| 3 | magenta detector, window 2.0 cols (final) | 79.8 | 79, 75, 75, 95, 75 |

In rounds 2 and 3 every game was a one-bullet kill. The second ATTACK step in each game is the
pistol's fire delay, not a second shot: 95 = 106 − 5 − 6 tics. All of the remaining loss is the
travel time to line up with the monster, which depends on where it spawns.

## Prediction

Held-out average: **about 77** (my 80% range is 70–83). I'm fairly confident about the basic
behaviour: the colour feature separated cleanly in every frame I looked at, and all 10 games in
rounds 2–3 were one-shot kills. Most of the uncertainty is in spawn positions, since monsters
spawned near the side walls take longer to reach (around 60–65 points). There is also a small risk
of an occasional miss when the monster spawns about 2 columns off-centre and the agent fires from a
standstill. A miss costs about 5 points plus the re-aim time.
