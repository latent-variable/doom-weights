# deadly_corridor, pixels: a hand-written network

`build.py` writes every weight explicitly. Nothing was trained, fitted or searched. The colour
coefficients, thresholds and gains came from reading cell colours off the practice inputs and
choosing numbers that separate them. One constant `FIGHT` picks between the two policies I built.
**The submitted weights use `FIGHT = False`** ("run and gun"), for the reasons in the last section.

## What the input looks like (from the practice frames)

* The whole corridor (walls, floor, ceiling) is brown: blue is about half of green and red is
  about 1.8× green. Ceiling and floor are brighter than the walls.
* The zombies have blue helmets, purple/blue armour and red helmets. In those cells blue ≥ green,
  or red is very high while green is near zero.
* Bullet puffs (grey smoke where our shots hit the wall) have blue just below green. They look a lot
  like grey armour. My first detector fired on them and the agent shot a wall for 1,000 tics.
* When the player is hit, big red blood squares cover the screen.

## Layer 1 (128 units, 50 used)

| units | computes |
|---|---|
| 0–39 | **Monster evidence for screen column c**: `relu(10·(Σ_{rows 8..15} (0.656·R − 1.63·G + B) − 0.015))`. I picked the colour direction so it is exactly 0 on the wall colour and on the puff colour. It is negative on ceiling and floor, and +0.05 to +0.27 per cell on helmets and armour. Rows 8–15 are at or above eye level, where a standing zombie's head and torso are; corpses lie below the horizon. On every saved frame, puff columns score ≤ +0.01 and visible near zombies score +0.03 to +0.4. |
| 40, 41 | **Heading cue**. `H = Σ_{rows 0..8} (c − 19.5)·G`. The bright ceiling band is centred on the corridor's vanishing point, so H > 0 means the corridor lies to the right. The units are `relu(±0.1·H − deadband)`. I calibrated it by shifting real frames sideways: about 6.5 units of H per column. When the player drifts toward one wall, the centroid also shifts, so following the cue steers back toward the centre line. |
| 42–45 | Saturating 0..1 flags for strong misalignment (\|H\| beyond about 14 columns), built as `relu(x) − relu(x−1)`. |
| 46, 47 | Saturating low-health flag. **Disabled**: round 5 showed that it made the agent abandon the opening fight. |
| 48, 49 | Saturating "facing a wall at point-blank" flag. It uses brightness `v = G + 0.244·R`, which puts blood midway between wall and ceiling values, and computes `mean v(mid-screen) − mean v(top rows) − mean v(floor beside the gun)`. On the 52 saved frames, corridor views score ≤ −0.194 and wall views ≥ −0.167, so the threshold is −0.180. It fires on exactly the 10 wall frames. |

## Layer 2 (decision logic)

* **u0, centre target.** Monster evidence in columns 16–23, weighted 1,1,2,3,3,2,1,1. The middle 8 columns (±4, about ±9°) cover the pistol's ±5.6° autoaim plus a zombie's half-width.
* **u1, u2: targets left and right of centre.** Each column is weighted `exp(−distance from centre / 6)`, so the target needing the smallest turn wins. With plain sums, round 7 swung back and forth between the two opening zombies until it died.
* **u3, u4: heading turns.** In fight mode, any monster evidence (×100) gates these off.
* **u5, go forward.** On when nothing is seen, and off when a wall is ahead (fight mode).
* **u6, stand and shoot.** A centred target while facing far off the corridor axis. The output weight is set to 0, so this is disabled.
* **u7, wall turn.** Turns left at a wall; the heading cue overrides it when the cue is decisive.

## Output layer

* `FWD+ATTACK = 5.1·u0 + 0.5·u5 + 0.01`
* `ATTACK = 5.0·u0`
* `TURN_LEFT = u1 + 10·u4 + 2·u7`
* `TURN_RIGHT = u2 + 10·u3`

All the turn terms toward monsters (u1, u2) and the wall term (u7) apply only in fight mode.

## The two policies, and what practice showed

**Fight mode (`FIGHT = True`).** Turn toward the visible zombie nearest the crosshair. When it is
centred, advance and shoot (FWD+ATTACK). Standing still to shoot did worse in the opening fight.
When nothing is visible, re-align with the corridor using the ceiling cue (or turn off a wall), then
advance while firing.

**Run and gun (`FIGHT = False`, submitted).** Always MOVE_FORWARD+ATTACK, so anything straight
ahead gets shot. The heading cue has a wide deadband (about 6 columns, 14°) and acts only as a
safety net. The player spawns facing down the corridor and nothing in this mode rotates it, so the
cue should almost never fire. The wall rule is off, so the end wall in front of the vest can't make
the agent turn away.

Practice scores (5 games per round; the same 5 seeds each round, though games diverge as soon as
actions differ):

| round | policy | mean | per game |
|---|---|---|---|
| 1 | always forward (zero weights) | 282 | 92, 84, 385, 585, 266 |
| 2 | first detector (locked onto puffs) | 2 | |
| 3 | fight, stand to shoot | 329 | 712, −116, −96, 795, 348 |
| 4 | fight, advance to shoot | 280 | 306, 140, 366, 380, 206 |
| 5 | + low-health rush (bug) | 80 | |
| 6 | + stand when misaligned | 140 | 206, −65, 303, 328, −71 |
| 7 | + wall rule | 210 | 631, −116, −91, 341, 286 |
| 8 | + nearest-target weighting | 117 | 63, 305, −56, 330, −56 |
| 9 | fight, final form | 207 | 83, −38, 379, 349, 259 |
| 10 | run and gun (heading deadband 1.2) | 229 | 1, 344, 14, 64, 723 |

Fighting works on some seeds: 4 kills and 630–800 points. But on this skill level the zombies fire
continuously, and a pistol needs 2–7 hits per zombie. In about 40% of games the agent dies during the
opening fight with −100 to +60. Across 30 games of usable fight variants the mean was about 213. The
two never-fight rounds averaged about 256 over 10 games. That difference is within the noise
(standard error around 50–90), so it is no proof that running is better. But nothing I measured
says fighting is better, and running has fewer ways to fail. In round 10, about 20% of steps were
spurious heading turns triggered by blood and monster asymmetry, each costing 4 tics of movement.
The final runner widens the deadband to stop that, which makes it behave like round 1's
always-forward policy, the best never-fight round.

## Prediction

Held-out average score: **about 250**, with a likely range of 150–350 (about 70% confidence for
that range). Most games should end in death after 50–170 tics, having covered 150–800 units. An
occasional game may reach 700+, and reaching the vest (+1,000) will be rare. I don't expect a mean
above 400.

I'm only moderately confident in the choice between the two modes. If the held-out set rewards the
seeds where fighting pays off, `FIGHT = True` could do about as well; in practice it averaged
somewhat lower.
