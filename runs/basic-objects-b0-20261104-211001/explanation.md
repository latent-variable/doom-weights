# Hand-written network for `basic` (objects mode)

## The idea
The policy aims by centring the monster horizontally on screen. The pistol fires along the
screen's centre ray, so the only thing that matters is where the Cacodemon's pixels sit relative to
the centre column. Pixel 160 sits on the border between grid columns 19 and 20, so the centre is
at column coordinate 19.5.

## Layer 1: three real units (the other 125 are zero)
Let `m[r,c]` be the monster share of cell (r,c) (channel 0), `M = Σ m` (total monster mass) and
`S = Σ m·(c − 19.5)` (mass-weighted horizontal offset). Then `S/M` is the centroid offset in columns.
Since a network can't divide, I compare `S` against `TOL·M` directly. Each weight is linear in the cell's column:

- `h1[0] = relu(S − TOL·M)`: positive only when the centroid is more than TOL columns **right** of centre.
  Weight on cell (r,c) = `(c − 19.5) − TOL`.
- `h1[1] = relu(−S − TOL·M)`: the same for **left**. Weight = `−(c − 19.5) − TOL`.
- `h1[2] = relu(M − 0.3)`: "a monster is visible" (more than about a third of a cell of monster pixels).

TOL = 1.2 columns (about 10 px). At the distances in this room the Cacodemon is several columns wide,
so a centroid within ±1.2 columns puts the centre ray well inside its body. The extra margin covers the
strafe momentum that carries on while the pistol's fire animation plays.

## Layer 2: identity on those three units.

## Output
- MOVE_LEFT = `h2[1]`, MOVE_RIGHT = `h2[0]`, ATTACK = `0.02·h2[2]`.

When the monster is off-centre, one movement score is roughly `M·(|offset| − TOL)`. That is far larger
than the tiny attack score, so the player strafes toward it. Inside the tolerance both movement
scores are exactly 0 and ATTACK wins. The two movement units can never both be positive. If no
monster is visible, all scores are 0 and the tie goes to MOVE_LEFT. That case shouldn't occur in
this scenario, because the monster spawns in view on the far wall.

The game-variable input (ammo) is not used.

## Checks (on frames I built myself, `check.py`)
I slid a 4×4-cell monster blob across every column. Centroids from 1.5 to 17.5 give LEFT, 18.5 to 20.5
give ATTACK, and 21.5 to 37.5 give RIGHT. Partial blobs at either screen edge give the correct
direction, an empty frame gives LEFT, and a small faint blob at the centre gives ATTACK.

## Expected score
In an episode the player strafes until the monster is centred (from 0 to roughly 40 tics depending
on where it spawned), then fires one shot, which takes a few tics to fire. That scores about
106 − 5 − (tics elapsed). Some risks: a miss caused by momentum drift during the fire delay, which
costs −5 plus some tics, and later shots that may get pistol spread; or a little left/right
oscillation if the strafe step overshoots the ±1.2-column window.

**Prediction: average about 75 per episode (plausible range 60–85). I'm moderately confident (~65%)
it lands in that range.** The main uncertainties are strafe speed versus window width and the
on-screen size of the monster. I have never seen either.
