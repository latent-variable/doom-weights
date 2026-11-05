# Hand-written network for `basic` (pixels)

## What the screen looks like (from the one practice round)

The practice screens (`practice_runs/01`) showed my first guess about the room was wrong. My first network
looked for red and scored -300: it saw nothing and walked left the whole game. What the screens actually show:

- The walls and floor are **brown brick**. In every cell R > G > B, with R - G ≈ G - B (e.g. (0.30, 0.22, 0.15)).
  The ceiling is a neutral gray grid.
- The Freedoom Cacodemon is **dark blue and dark red**, e.g. cells (0.11, 0.07, 0.16) and (0.27, 0.12, 0.09).
  It is small: about 3 grid columns (≈24 px) wide, in rows 13–17, floating at eye level against the far wall.
- The weapon sprite only reaches up to about row 22, in columns 17–22.

The feature **f = R + B − 2G** ("how far green sits below the mean of red and blue") separates them cleanly.
It is within ±0.02 on every brick, floor, ceiling and gun cell, and 0.05–0.16 on monster cells. Yellow muzzle
flash and gray bullet puffs come out negative. It also mostly ignores brightness, because the brick's
colour ratios stay the same when it's lit differently.

## Layers

**Layer 1: monster column detectors (units 0–39).**
`m_c = relu( Σ_{rows 10..19} (R + B − 2G)[r, c] − 0.10 )`.
On the practice frame the monster's columns give 0.29, 0.36 and 0.04, and every other column gives 0.
The 0.10 threshold is about twice the largest background column sum I saw (≈0.05).

**Layer 2: where the monster is relative to the screen centre.** With `d_c = c − 19.5` (column offset from
the centre line, x = 160 px) and K = 1.0 column:
- `P = relu(Σ m_c (d_c − K))` is positive iff the m-weighted centroid is more than K columns right of centre.
- `N = relu(Σ m_c (−d_c − K))` is the same, to the left.
- `M = relu(Σ m_c)` is the total monster mass.

This compares the centroid to a window without dividing: centroid > K ⇔ Σ m_c d_c > K Σ m_c.

**Layer 3: action scores.** `q_left = N`, `q_right = P`, `q_attack = 0.001·M`.
Outside the window, the strafe toward the monster wins. Inside it, both strafe scores are 0, so ATTACK wins.
If no monster is visible at all, every score is 0 and the tie goes to MOVE_LEFT; that should not happen here.

Units 40–127 are unused (all zero).

## Checks I ran (on inputs I built, not on the game)

- On all 6 practice inputs the network now picks MOVE_RIGHT, toward the monster. That includes the frames
  where the player was pressed against the left wall and the monster was in column 39.
- I took the real monster patch, slid it across the real background with sub-cell interpolation, and recorded
  the action at every position. Result: MOVE_LEFT if the monster's centre is more than 6 px left of the screen
  centre, ATTACK from −6 to +8 px, MOVE_RIGHT beyond that.

## Why it should score well

The monster's hitbox is 62 units wide and the sprite is ~24 px, so a hit needs the centre ray within roughly
±12 px of the monster's centre. The firing window (±7 px) sits inside that with room for the drift that
continues after the player stops strafing. The pistol fires about 4 tics after the press. Each step moves at
most ~12 px, so the window (~14 px wide) is rarely skipped over. A rough 1-D model of Doom strafe physics
(acceleration, friction 0.906, shot delay) puts the typical episode at a 30–150 px approach, about 3–12 steps,
then one shot. That works out to a score of about 106 − 5 − (15…55 tics) ≈ 50–90, averaging around 80.
In that model, a monster spawned already in the window gives ~95.

Risks:
- If the sprite centre and the hitbox centre differ by more than ~5 px, the player can stop inside the window
  and miss. Held fire in Doom then adds random spread, so it would eventually hit, but it would cost shots and time.
- The monster may drift a little ("stays roughly there"). The policy reacts to each frame, so it just re-aims.
- Momentum can carry the player past the monster after a miss. Then it strafes back, losing ~10–20 points.

## Prediction

Average held-out score: **about 75** (my 80% interval is 55 to 88). I'm moderately confident in the perception
part, which is checked against the real screens. I'm less sure about the physics: strafe speed, shot timing and
the exact hit width are estimates. There is a small chance (~10%) of a systematic miss problem, which would
pull the average down toward 30–50.
