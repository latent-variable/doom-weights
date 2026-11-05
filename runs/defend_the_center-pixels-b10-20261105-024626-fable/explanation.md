# defend_the_center / pixels — hand-written network

## Prediction

**Average score on held-out games: about 23** (kills − 1; the ceiling is 25 because there are 26 bullets).
I would put roughly 70% on the held-out mean landing between 21.5 and 24.5.

What that rests on: the final weights scored 24, 22, 24, 24, 21 (mean 23.0) in practice round 8. Two
further rounds played the same design with every threshold nudged by about 4% (down, then up), only to
get fresh game trajectories; they scored 24.0 and 24.2. That is 15 games of this design, mean 23.7,
worst game 21, no early death. I shade the prediction down from 23.7 because the practice games reuse
the same five seeds, and because earlier designs died early (9–15 kills) in roughly a third of their
games; I believe that failure is fixed but 15 games cannot rule out a rate of a few percent, and each
such game costs about 12 points.

## What the screen looks like (measured from the practice screens)

- Sky: rows 0–2, exactly R = G = 0. Wall: one flat colour (0.439, 0.351, 0.273), ±0.004 per cell,
  rows 8–15 everywhere. Floor: gray, rows 17+. The horizon is at the row 14/15 boundary, so **every
  standing monster covers rows 14–15 in its own columns at any distance**; that is where the network looks.
- Two monsters: a dark-red/pink "demon" (fast, hit box twice as wide as its torso) and a gray marine
  with a blue helmet and a chainsaw (slow, thin, and by far the more dangerous once it is adjacent).
- The pistol is drawn at the bottom centre and is the policy's only memory: its animation frame tells
  the network whether a shot is in progress.

## Layer 1 (58 units): detectors, each `relu(100 · (linear colour test − threshold))`

- **Monster on the aim line** (cells rows 14–15, columns 19 and 20, either side of the screen centre):
  per column, "less red than wall" (marine body), "bluer than wall" (helmet / demon's pink head),
  "less green than wall" (demon body), "darker in R+G" (thin far bodies), "redder than wall" (marine's
  arm, demon's head). Wall colour is either the constant above or the mean of 16 always-wall cells in
  row 8 (which cancels the muzzle-flash brightening).
- **Wider aim rules**, because hit boxes are wider than sprites: demon colour covering ≥ ~4.5 px of
  the two line columns; one strong column of dark body; a big dark body across columns 18–21; and a
  saturated set (line columns "not wall" + two adjacent columns of gray body nearby) for a near marine
  whose arm, not its body, is on the line.
- **Heads**: "bluer than wall" summed over bands of rows 9–13 in the right half (and rows 9–11 in the
  left half). A head that high means the monster is near; rows 9–11 mean it is about to reach me.
  Blood has no blue, so it cannot fake a head.
- **Demon to the right**: demon colour in rows 14–15, columns 21–30 (a demon within roughly 400 units).
- **Blood flying**: any red in the sky rows. Blood squares have the demon's colour, so this switches
  off the rules they could fake.
- **Pistol busy**: two tests on weapon cells for the raised-hand frame and the slide-back frame,
  written as differences so that brightness changes cannot flip them.

## Layer 2 (19 units): logic

`min(col19, col20)` as `relu(a) − relu(a − b)` (a monster must be on both sides of the line), the same
with the blood-sensitive features included but vetoed by blood, pass-throughs for the wider rules, a
three-way AND for the near-marine rule, and saturated 0..1 flags for busy, near-right, demon-right,
very-near-right and very-near-left.

## Output

- `TURN_LEFT = 1 + 4·very_near_left`
- `TURN_RIGHT = 2·near_right + 1.5·demon_right + 4·very_near_right − 1000·busy`
- `ATTACK = 10·(any aim rule) − 10⁶·busy`

So: shoot when something is on the line and the pistol is idle; otherwise turn toward a near monster
on the right; otherwise scan left. While the pistol recoils the network always turns left, which
sweeps about 33° past the kill and stops it from firing a second, inaccurate shot into a dying monster.

## Why I expect it to work, and what I learned on the way

- Aiming at columns 19|20 with both columns required puts the aim line inside the sprite, so nearly
  every shot hits (25–26 kills from 26 bullets in most practice games).
- Turning moves about 14° per step, so a thin monster is often stepped over. The network therefore
  locks on to near monsters and alternates around them until the line lands inside; near ones are wide
  enough that this ends quickly.
- Three bugs found from practice screens mattered most: the busy veto was too small, so a large
  adjacent monster out-voted it and bullets were wasted on refires; blood squares corrupted the
  wall-colour reference and made the turn logic dither while being hit; and a marine in melee with its
  arm on the aim line was never shot while the turn rules flipped left/right around it. Fixing the
  last one, and accepting arm/helmet cells as "monster on the line", is what removed the early deaths.

## Caveats

- No learning was used. Practice rounds were used to look at screens, find bugs and compare whole
  designs; the three near-identical final rounds were for measuring, not for picking the best of three
  (the submitted weights are the middle setting, which happened to score lowest of the three).
- I also wrote a crude geometric model of the game (`tools/sim.py`) to think about lock-on range and
  aim strictness. It is not the game, it disagreed with practice in places, and no weight was fitted
  to it.
- Unverified assumptions: turn speed and pistol timing come from my memory of the engine's source, not
  from measurement; the far-demon lock-on is supported by the model but practice was too noisy to
  confirm it.
- After the 26th bullet the network can do nothing, so every game ends in death (−1).
