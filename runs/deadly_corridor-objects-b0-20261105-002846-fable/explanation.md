# deadly_corridor, objects mode: a hand-written policy

`build.py` writes every weight from named constants; nothing is fitted, searched or trained.
92 of the 128 first-layer units and 41 of the 128 second-layer units are used, the rest stay zero.

## What the policy does

1. **Walk at the vest.** The vest is the farthest pickup, so the pickup highest on the screen wins.
2. **Stop and kill any monster that gets near** (roughly 350-460 map units): turn to it, sidestep
   until the crosshair is on it, then hold ATTACK standing still until it is gone.
3. **Ignore far monsters**, except to shoot one that happens to be under the crosshair while walking
   (FORWARD+ATTACK).
4. **Below 40 health, stop duelling and run.** Distance already covered is kept and death costs only
   100, so with little health left a run banks more than a fight I would probably lose.
5. **When the vest is out of view** (after turning to a side monster), turn towards the side of the
   picture that is much deeper than straight ahead, which is where the corridor runs.

The untouched all-zero network always picks FORWARD, i.e. "run straight for the vest", which the
brief says usually dies. This policy is that run plus the fights that make the run survivable.

## Layer 1 (92 units): detectors read straight off the grid

| units | what each computes |
|---|---|
| 40 `E[c]` | `relu(monster share summed down column c - threshold)`: a *near* monster stands in column c. A column a standing zombie fills is about 1200/distance cells tall, so a threshold of 2.6-3.4 cells means "closer than ~350-460 units". |
| 2 `A19, A20` | `relu(1 - 16*monster share in column 19 / 20)`: 1 when the column just left / right of the crosshair holds no monster. |
| 1 `LH` | `relu(0.40 - health)`: low health. |
| 27 | `relu(1 - 64*pickup share)` for 9 row bands x 3 zones (left, centre = columns 17-22, right): 1 when no pickup pixel is there. Rows are the distance ranking: a pickup on the floor appears higher the farther it is. |
| 1 `MA` | no monster at all in columns 12-27. |
| 3 | depth straight ahead (mean of 18 cells at the horizon): a clipped ramp at 0.06 ("no wall right in front") and `relu(depth - 0.14)` ("no wall within ~250 units"). |
| 16 | for four side regions of the depth picture: "this region is more than 1.4x as deep as straight ahead" and "this region is deeper than 0.10", each as a clipped ramp (two ReLUs). |
| 2 | "more than 20 cells of pickup around the crosshair": a vest-sized item under ~100 units away. |

Two details matter in the monster detector:

- **Perspective correction.** The engine draws a sprite 1/cos(a) taller and wider at angle a from the
  view axis. Column thresholds are raised by 1/cos and masses scaled by cos^2, so "near mass" depends
  on true distance and does not change as the player turns.
- **Thresholds are lowest at the crosshair and rise towards the edges.** A monster near enough to
  make the player turn towards it is then still "near", with margin, once it is in front.

## Layer 2 (41 units): combine detectors into drives

- `NE` = total near-monster mass. `TR` / `TL` = near mass right minus left of the aim zone, weighted
  by `exp(-columns/3)` so the monster closest to the crosshair wins and each turn step towards it
  multiplies its lead over a rival.
- `SR` / `SL` = near mass just right / left of the crosshair (columns 17-22).
- `COV = relu(1 - A19 - A20)`: a monster straddles the crosshair. `CN`: it does, and it is near.
- All six duelling drives get `-5000*LH`, which switches them off at low health.
- 27 pickup presences `relu(1 - absence)`, exactly 0 when nothing is there. Near-band pickups off to
  one side are also switched off unless a wall is close ahead.
- `MONC` (a monster in the middle of the view), `OPEN` (no wall right in front), four "open side"
  units (the AND of the two depth conditions per region) and `BIG` (big pickup dead ahead).

## Output layer: priorities by magnitude

Duelling terms are worth hundreds to millions, navigation terms at most 85, so a near monster always
wins.

| action | score |
|---|---|
| TURN_L / TURN_R | `30000*TL` / `30000*TR` + navigation |
| MOVE_L / MOVE_R | `-1 + 120000*SL` / `-1 + 120000*SR` |
| ATTACK | `-10 + 100*COV + 1e6*CN` |
| FWD+ATTACK | `100*COV - 500*NE` |
| FORWARD | navigation `- 500*NE` |

Navigation is a ladder in which each rung is a third of the one above, so a rung outranks
everything below it:

1. a big pickup dead ahead: walk onto it;
2. far pickups (rows 15 and above), farthest band first: walk if in the centre zone, else turn to it;
3. one side of the view is much deeper than straight ahead: turn that way;
4. nearer pickups: walk onto one straight ahead; turn to one only when a wall is close ahead (this is
   the vest at the end wall, where rule 3 is silent);
5. a monster in the middle of the view: walk; 6. no wall right in front: walk; 7. turn right.

## Why I expect it to score well

- Reward is distance plus 1,000 for the vest. The policy only stops walking for monsters that are
  about to become dangerous, and it banks distance before each fight.
- It fights at 350-460 units, where a pistol aimed at a zombie's body mostly hits and the monsters'
  wide spread mostly misses. It holds fire on one target until it is dead.
- Bullets are spent only when a monster is under the crosshair.
- Target choice and aiming are built so they cannot oscillate (see below).

## What I checked, and how

Doom was never run. `check.py` is a toy renderer I wrote (box corridor, optional side alcoves,
rectangles for sprites) that produces input vectors in the harness's format, plus a toy movement loop
with ZDoom-like turning, friction and a 19-tic pistol. I used it to check that the logic does what I
meant and cannot get stuck, not to tune for score.

- Static frames: 30 hand-made situations at each of the two possible horizon heights (status bar
  drawn or not), 60 frames in all; every one gives the intended action.
- Closed loop, box corridor: 190 of 190 runs reach the vest across 19 variants (turn rate halved,
  doubled, sprites 0.7x-1.3x, corridor width 96-320, walking monsters, with and without drops).
- Vest out of view, random start pose: 144 of 144 runs find it, at three assumed depth scales.
- Corridor with side alcoves: 123 of 128 runs reach the vest inside the time limit. Three of the five
  failures were long detours back down the corridor; two were sidestepping into a wall at a
  half-hidden monster that never moves.

The toy found three real bugs that I fixed by redesign, not by tuning: endless left/right flapping
between two near monsters (cause: sprites grow towards the screen edge, hence the perspective
correction), flapping caused by grid-alignment noise in the near detector (hence the steep weights
and centre-low thresholds), and flapping between two navigation rules (hence the single ladder).

## Known weaknesses

- **Detours.** With nothing in view and open space ahead it walks, and that can be back towards the
  start. In deep-alcove toy layouts this cost up to ~1,000 tics per episode.
- **Dropped clips and shotguns look like the vest.** It can walk towards one when the vest is out of
  view.
- **No memory.** A monster outside the 90 degree view is not known to it.
- **Ammo.** If the pistol runs dry it keeps "attacking" at range until health falls under 40, then runs.
- **Things I assumed and could not check**: turn rate (7 degrees for the first 4-tic step, 14 after);
  sprite sizes; that depth 1.0 is roughly 1,800 map units; whether the status bar is drawn; that
  corpses are not labelled as monsters; the corridor's real layout.

## Prediction

**Average score on held-out games: about 750.**

I am not confident: my 80% range for the average is 250 to 1,500. How I get there:

- If the fights work as designed (I put that at about 75%), I expect to reach the vest in roughly a
  quarter to a third of games (about 2,300 each) and otherwise die after one or two fights with a few
  hundred points banked: an average near 900-1,000.
- If something systematic is wrong (a threshold off so it never engages, or engages from the start
  line and dies there, or a stuck state the toy did not show), the average falls to roughly 0-600.
- My estimate of the monsters' damage says no policy survives most games on this skill level, which
  caps the upside; if they are weaker than I estimate, the average could exceed 1,500.
