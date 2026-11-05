# Hand-written network for ViZDoom `basic` (pixels mode)

## Prediction

**Average score on held-out games: about 79** (I expect it to land between 77 and 81).

- My own model of the game (uniform monster position, the policy below, no misses) gives 80.2.
- The 25 practice games played with the final weights averaged 77.2 (rounds 4–8: 74.2, 79.8, 78.2,
  74.2, 79.8), every one a kill with a single shot. Those games happened to have the monster far
  from the centre more often than a uniform draw would give, which is why they sit below 80; a
  5-game mean has a standard error of about 4, so the two numbers are compatible.
- Confidence: high that it is in 77–81. The score of a game is fixed by how far away the monster
  spawns (95 if it is already in front of the gun, minus 4 per 4-tic step needed to get there, down to
  67), so the average can only move through the spawn distribution or through missed shots. A miss
  costs roughly 20 points; I have seen none in 30 practice games with the speed-aware window, but I
  cannot rule out a rate of a few percent.

## What the game looks like (measured from practice frames, not assumed)

- The monster is a small blue-and-red creature about 24 px wide that always stands in screen rows
  13–17 (it never changes distance or height). Walls and floor are brown, the ceiling is gray, the
  hand is tan. There is no status bar.
- Geometry: monster line 384 map units from the player, far wall 448, room half-width 224. With a 90°
  field of view the focal length is 160 px, so 1 px at the monster is 2.4 units and one grid column is
  19.2 units. The monster's hit box is ±31 units (±12.9 px), which matches the sprite's visible width.
- Movement is ZDoom walking strafe: each tic `v += 0.75; x += v; v *= 0.90625`. Positions at
  successive decisions when strafing from rest: 0, 6.8, 21.9, 42.4, 66.7, 93.5, 121.9, 151.6 units.
- The pistol fires 4 tics after ATTACK is chosen, i.e. at the start of the next step, and during those
  4 tics the player keeps sliding (3.48 × current speed, up to about 24 units). The game ends one tic
  after the kill, so a kill whose ATTACK was chosen at decision `n` scores `95 − 4n`.

My simulator built from these rules reproduced real practice games step for step (same monster
offsets within about 1 unit, same scores).

## The policy

Strafe toward the monster at full thrust and pull the trigger at the first decision from which the
slide will carry the gun inside the hit box by the time the bullet leaves. Because of the slide, the
right moment to fire is *before* the monster reaches the crosshair, and how much before depends on
the current speed: the window is ±26 units at rest and grows to ±49 units at top speed
(31 hit-box half-width − 5 safety margin + slide). This is the earliest shot that still hits, so it
is time-optimal apart from the margin.

The network has no memory, so it reads its speed off the screen: in every game the player strafes
from the same spot in the same way, so speed is a fixed function of how far it has travelled, and
that is visible in the side walls.

## Layer 1 (128 units, 125 used)

1. **Monster-colour detectors (112 units).** Each computes `relu(R + B − 2.1·G − 0.01)` on grid cells
   in rows 13–17. That expression is below zero on every brown, gray or tan cell (brown is
   R:G:B ≈ 1:0.73:0.5, so R + B ≈ 2.05·G; gray gives −0.1·brightness) and positive on the
   monster's blue rim and red tentacles. For columns 8–31, where the monster can be during an
   approach, there is one unit per cell for rows 14, 15, 16 and one for rows 13+17 together; for the
   outer columns there is one unit per column. Per-cell rectification keeps wall texture from adding
   up into false detections.
2. **Travel gates (12 units).** The cue is the sum of `R − B` over rows 0–10, right half minus left
   half. Those rows show only gray ceiling (R − B = 0) and the two brown side walls (R − B ≈ 0.14).
   Each side wall's top edge is a line through the screen centre whose slope depends on the distance
   to that wall, so the cue grows steadily as the player moves right and shrinks as it moves left.
   Gate `k` is `relu(±cue − edge_k)`: it switches on once the player has passed decision `k` in that
   direction. The edges sit between the cue values computed from the geometry at consecutive
   decisions (including ZDoom's view bobbing, which changes the eye height by a known amount at each
   decision). Real frames gave 0.35, 1.04, 2.07, 3.90, 7.73, 12.4 against edges at 0.19, 0.67, 1.58,
   3.01, 5.74, 9.74.
3. **"Not fired yet" (1 unit).** `relu(ammo/50 − 0.99)`: 0.01 with full ammo, 0 after any shot.

## Layer 2 (17 units used)

With `S` = sum of detections and `M` = sum of detections × (column − 19.5), `M/S` is the monster's
offset from the crosshair in columns (accurate to about ±1.4 units on test frames).

- `right[n] = relu(M − T_n·S − 2000·gate_right[n])` for speed bins n = 0…6, and the mirror image
  `left[n]`. Unit `n` is on when the monster is to the right of window `T_n`, and is switched off
  once the player has travelled past decision `n`. Since the windows widen with `n`, the tightest
  window still switched on is the one for the current speed.
- `vis = S`.
- Fallback pair: the same test with a narrow ±14-unit window, held off by the "not fired yet" unit.
  If a shot has been fired and the game is still running, that shot missed, so the player re-centres
  properly before firing again. Without this, a miss from rest would repeat until the time limit; in
  simulation with a deliberately shrunken hit box this turns scores of −380 into ordinary ones.

## Layer 3

- MOVE_LEFT = 1000 × (sum of left units), MOVE_RIGHT = 1000 × (sum of right units), ATTACK = `vis`.
- So the player strafes toward the monster while it is outside the current window and shoots as soon
  as no move unit is on.

## How the numbers were chosen

Everything is derived from the geometry and physics above; nothing was fitted or searched. Practice
rounds were used to look at frames (colours, sizes, wall edges), to confirm the movement and pistol
timing, and to check that shots hit. Eight of the ten practice rounds were used: round 1 was a blind
colour guess that never found the monster (−300), round 2 used a fixed window (75.8), rounds 3–8 the
speed-aware window. The one judgement call is the 5-unit safety margin: a practice shot landed 24.4
units off centre and killed, the sprite is symmetric about its hit-box centre, and each unit of
margin costs only about 0.14 points, so I kept it rather than shaving it.

## Files

- `build.py` — writes `weights.npz` (numpy only, reads nothing, runs in about 0.1 s).
- `weights.npz` — its output.
- `tools/` — my analysis helpers (PNG reader, frame probes, the simulator used for the checks above).
  Not needed by `build.py`.
