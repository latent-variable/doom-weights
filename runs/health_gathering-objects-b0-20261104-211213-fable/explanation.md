# health_gathering, objects mode: what the hand-written network computes

## Prediction

**Average score on held-out games: about 1,850** (the maximum is 2,100, reached by surviving the whole episode).

Confidence is moderate. I have never seen the game run, so the estimate is split three ways:

| Case | My probability | Score I expect |
|---|---|---|
| The policy works as designed and medkit supply outpaces the acid | about 70% | 1,950 to 2,100 |
| It works, but loses time (targets slipping out of view, slow searches) or the acid is harsher than I assume | about 20% | 900 to 1,700 |
| Something I assumed about the game is wrong (see "What could make this wrong") | about 10% | 200 to 500 |

## The policy

Walk to the nearest medkit that is roughly ahead; if no medkit is in sight, turn left on the spot until one is.
The network uses only channel 1 (pickup share). Monster share, depth and HEALTH have all-zero weights.

There is no wall avoidance because none is needed: the room is an empty square, so the player only walks forward
when a medkit is ahead, and there is never a wall between the player and a visible medkit.

## Layer 1: six units (the other 122 are zero)

Units 0 to 4 each hold the weighted amount of medkit visible in one screen region, and the regions correspond
one-to-one to the five actions. Each cell's pickup share is multiplied by a row weight and a column weight and
added to exactly one unit.

**Regions** (offset = columns from the screen centre, 20 columns = 45 degrees):

| Region | Offset | Bearing | Action it votes for |
|---|---|---|---|
| Centre | up to 3.5 | within about 10 degrees | MOVE_FORWARD |
| Arc, far medkit (rows 0 to 21) | 4.5 to 10.5 | about 13 to 28 degrees | MOVE_FORWARD + TURN toward it |
| Arc, near medkit (rows 22 to 29) | 4.5 to 6.5 | about 13 to 18 degrees | MOVE_FORWARD + TURN toward it |
| Side | beyond the arc | larger | TURN toward it, on the spot |

- The centre band is 20 degrees wide, wider than the largest turn I expect in one 4-tic step (about 14 degrees), so a
  turn cannot hop over it and oscillate.
- Turning on the spot for large bearings prevents orbiting a medkit that sits inside the walking turn circle.
- The arc is narrower for near medkits (base drawn in the bottom 8 rows, closer than roughly 120 units) because a
  near medkit off to the side slides out of view quickly if the player keeps walking.

**Row weight** = (rows below the horizon) cubed, minimum 1. A medkit lies on the floor, so the nearer it is, the
lower it is drawn. Its pixel count grows like 1/distance squared and the cubed row adds 1/distance cubed, so its
total weight falls off roughly like 1/distance to the fifth power. The nearest medkit therefore dominates the sums,
and comparing region sums approximates "which region holds the nearest medkit".

**Column weight** = exp(-|offset| / 6). This prefers what is already ahead. It decays faster than perspective
inflates a medkit sliding toward the screen edge, so turning toward a medkit always increases its weight and
decreases the weight of medkits on the other side. The choice of target is self-reinforcing, which is what stops a
memoryless network from dithering left-right between two medkits.

**Unit 5** = relu(1 - 1000 x total pickup share). One pickup pixel is 1/64 of a cell, so this unit is exactly 1 when
nothing is in sight and exactly 0 otherwise.

## Layer 2

Identity on those six units. They are all non-negative, so the relu changes nothing. The policy needs only one
layer of features; the second layer is a pass-through.

## Layer 3

Each action's score is its region's sum. Unit 5 adds 1 to TURN_LEFT, so an empty view spins left. All biases are
zero, so the comparison between regions is scale-free: one far pixel steers as decisively as a large near medkit.
Ties go to the lowest index, which is only reached in the empty-view case.

The search turn is always in the same direction on purpose. Any rule that picks the search direction from the
current view (for example "turn toward the deeper side") can lock a memoryless network onto a room corner.

## Why I expect it to score well

- In objects mode the medkits are handed over exactly, so perception is not the hard part; the risk is control.
- Each control failure I could think of has a specific guard: oscillation (wide centre band), orbiting (turn on the
  spot at large bearings), dithering between targets (self-reinforcing column weight), search lock-up (constant
  spin direction). A full spin sees the whole room because the room is convex.
- A greedy collector reaches a medkit every few seconds, which should outpace periodic acid damage if the scenario
  is survivable at all.

## How it was checked

No game was run. Two scripts in this folder test the network on inputs I constructed:

- `check_synthetic.py`, single frames: an empty view gives TURN_LEFT; a medkit drawn at bearings from -40 to +40
  degrees and distances from 60 to 1,500 units gives the intended action; mirroring 2,000 random multi-medkit frames
  swaps left and right actions every time.
- `check_synthetic.py` and `check_deadends.py`, a toy closed loop I wrote from first principles (a point player in
  a square room, medkits drawn as rectangles in perspective). It is not Doom, and its damage, spawn, speed and room
  numbers are my guesses, so its survival times are not evidence about the real score. I used it only to look for
  dead ends. Across 30 seeds per setting, with walk or run speed, turn rates from half to triple my estimate, sparse
  and crowded rooms, and health pinned near 100 (medkits cannot be picked up at full health), the longest left-right
  alternation was 2 steps and the longest stretch without forward motion was 18 steps.
- In that toy the player died only when I made the damage exceed what the medkits could restore. That is a limit
  no policy could avoid in the toy.

No constant was fitted or searched in the toy. The constants in `build.py` were set from the geometry above and
were not changed after the checks.

## What could make this wrong

- **Acid versus medkit supply.** I do not know the real damage rate or spawn rate. If medkits restore less than the
  acid takes over an episode, every policy dies early. This one also wastes part of a medkit when it picks one up
  near full health; I left out a "wait until health is low" rule because it adds a holding pattern that could fail.
- **Turn and walk speed.** I assumed about 7 to 14 degrees of turn per step and about 33 units of walking per step.
  The toy checks suggest the logic tolerates a factor of two or three either way, but that is my model, not the game.
- **A medkit lost off the right edge** at close range is found again only after most of a full left spin (roughly
  90 tics). This costs time but does not get stuck.
- **Screen layout.** I assumed no status bar and the horizon at mid-screen. If a status bar is drawn, the horizon is
  two rows higher; the row weights still increase downward, so distance ranking is a little coarser but steering is
  unaffected.
