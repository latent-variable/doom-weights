# Hand-written network for `health_gathering` (pixels)

## Prediction

**Average score on held-out games: about 1500.** I hold this loosely: roughly a 50%
chance the true average lands between 1200 and 2100, and the outcome is two-humped
rather than bell-shaped.

- If the screen looks the way I assumed (see "What I assumed"), I expect most
  episodes to run the full 2,100 tics, for an average near 1900–2100.
- If one of the visual assumptions is wrong (for example a status bar is drawn, or the
  floor is not the palette green), the medkit detectors will misfire and the average
  will be closer to 300–700.
- If medkits are much scarcer than I remember, the average falls to roughly 1000–1400
  even with perception working.

I never saw the game. The only evidence is reasoning plus runs in a crude stand-in
room I wrote myself (`sim.py`), which shares my assumptions and so cannot test them.

## What I assumed about the screen

- No status bar and no weapon sprite, so the horizon is at y = 120: grid rows 0–14 are
  above it, rows 15–29 below.
- Horizontal focal length 160 px; vertical about 192 px (Doom's 1.2 pixel stretch);
  eye height 41 units. A point on the floor `d` units ahead appears `984/d` grid rows
  below the horizon. A medkit (about 28 wide, 19 tall) has its foot at `984/d` rows, its
  top at `528/d` rows, and is `560/d` columns wide.
- The acid floor is drawn from the Doom palette's green ramp, whose R:G:B ratio is
  about 0.47 : 1 : 0.40 at every brightness. So `R − 0.66·G + 0.5·B` is almost exactly
  zero on floor at any light level, and clearly positive on white, grey, red or brown.
- Medkits are mostly white with red; walls are not green.
- Health falls by roughly 10 every 32 tics and a medkit restores 25.

## What each layer computes

### Layer 1: 127 feature detectors (one unit spare)

Everything above the horizon is ignored. Each unit is a ReLU of a linear filter over
cells, with a threshold of `0.03 + 0.10 × (mean green of rows 20–29)` so that
sensitivity scales with scene brightness.

| Units | What it detects | How |
|---|---|---|
| 58 "guarded" detectors, 4 distance bands | A medkit 35–180 units away standing on open floor | Mean non-green in a window, minus 1.25–1.4 × non-green in the row directly above it. A wall is non-green from the horizon down, so it always covers the guard row and cancels itself; a medkit has floor above it. Nearly zero-sum, so floor colour errors and the red pain flash cancel. |
| 55 "bump" detectors, 4 distance bands | A medkit 80–600 units away, including one standing in front of a wall | Mean non-green in a block, minus the same rows a few columns left, minus the same rows a few columns right. A wall's base is a straight line on screen, so at any slope the block is never more covered than both flanks: the response is ≤ 0. A medkit hanging below that line, or alone on the floor, is positive. |
| 1 hard-wall unit | Wall within about 70 units ahead | Fires when the block of floor at the player's feet (rows 25–29, middle 24 columns) is not floor-coloured, using `0.8·G − 0.25·R − 0.9·B`, whose sign separates floor from wall at any brightness. |
| 1 soft-wall unit | Wall within about 150 units ahead or to the right | Mid-distance rows (19–21): centre and right columns less green than the left columns of the same rows. Comparing within the same rows makes it immune to distance shading. |
| 12 health units | Health inside 62–72, 32–42 or 7–17 | Steep ramps on the health input; layer 2 turns them into three flat-topped bumps. |

Window sizes grow with nearness (2 columns for far bands, 8 for the nearest), matching
how wide a medkit appears at that distance.

### Layer 2: 8 units

- **Four vote units** (forward, forward-left, forward-right, turn-left). Each detector
  belongs to a zone by its column: a central zone votes forward, the sides vote
  forward-left or forward-right, and for the nearer bands the far-left edge votes for a
  pure left turn. The central zone is wider for near bands, since a close medkit is
  picked up without precise aim. Nearer bands carry larger weights so the closest
  medkit dominates.
  The votes are opponent: left = ReLU(left − right), right = ReLU(right − left),
  forward = ReLU(centre − a fraction of the sides), each with a dead zone of 0.3. A
  horizontal stripe in a wall texture lights every column equally and cancels.
- **Hard wall** (pass-through), **soft wall** (two units forming a 0–1 clip), **scan**
  (sum of the health bumps).

### Layer 3: priorities

| Priority | Condition | Action |
|---|---|---|
| 1 | Hard wall (gain 100,000) | Turn left |
| 2 | Any medkit vote | The winning vote |
| 3 | Soft wall, or health in a scan band (+0.05) | Turn left |
| 4 | Nothing (bias +0.02) | Forward |

## Why I expect it to work

- **It walks to the nearest medkit.** On my constructed frames a single medkit is
  steered toward correctly from 45 to about 500 units at bearings up to ±40°, and from
  about 250 units when it stands close in front of a wall.
- **It cannot dither in place.** There is no pure right turn anywhere in the network.
  Every pure turn is to the left, so no two-step left/right cycle exists; any other
  cycle contains a forward step, which changes the view.
- **It does not push into walls.** The hard-wall turn overrides everything inside about
  70 units. If a wall is somehow not recognised, the scan bands still force a left turn
  of roughly 100° each time health passes through one, which frees a stuck agent.
- **It explores when nothing is visible.** It walks straight, veers left off walls at
  about 150 units so it crosses the room rather than hugging the edge, and stops to
  look around at three health levels.

## What I checked, and its limits

`sim.py` renders a square room (palette-green textured floor, textured walls, medkit
sprites, red pain flash) through the same 8×8 cell averaging as `encode`, and moves a
player with Doom-like speeds. `checks.py` probes single frames; `stress.py` plays
whole episodes. Five episodes per variant, final weights:

| Stand-in variant | Mean score |
|---|---|
| Nominal (16 medkits at start, one more every 30 tics, 10 damage per 32 tics) | 2100 |
| Medium density (10 at start, one per 45 tics) | 2100 |
| Scarce (6 at start, one per 70 tics: supply barely exceeds drain) | 1280 |
| Very scarce (3 at start, one per 120 tics: not survivable) | 353 |
| 20 damage per 32 tics | 443 |
| Vertical focal 160 or 220; room 700 or 1800 wide; turn speed doubled | 2100 each |
| Brown, bright high-contrast, dark, or greenish walls | 2100 each |
| Dim light, fast distance fade, pain flash doubled | 2100 each |
| Medkit taller, smaller, or grey | 2100 each |
| Floor more saturated, yellowish, bluish or pale than the palette ramp | 2100 each |

On 400 medkit-free frames at random poses, 4% produced a spurious medkit vote with
grey or brown walls, 13% with an exaggerated high-contrast wall; these cause a brief
wrong steer, not a stall.

These runs show the network does what I designed it to do under my assumptions. They
say nothing about whether the assumptions hold in the real game.

## Known weaknesses

- A medkit beyond about 500 units is under one cell and is not seen.
- A medkit within about 40 units of a wall cannot be reached head-on: the hard-wall
  turn fires first.
- A medkit at a wide angle on the right at close range can be missed, because the
  network curves toward it instead of turning on the spot.
- The bump detectors go blind for the few frames of a red pain flash; the guarded
  detectors keep working.
- If a status bar or weapon is drawn, the horizon moves and the near detectors see a
  permanent blob; I did not hedge against this.

## No learning

Every number in `build.py` was set by hand from geometry and palette arithmetic. I
revised several constants after watching the network fail in my own stand-in room
(for example adding the bump detectors when it missed medkits in front of walls);
each change was a reasoned redesign, not a fit or a search. `build.py` uses only
numpy, reads no files and runs in well under a second.
