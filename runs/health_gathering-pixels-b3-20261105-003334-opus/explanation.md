# A hand-written network for `health_gathering` (pixels)

Every weight is set by hand in `build.py`. The network does three things: it finds health
pickups by their colour, turns away from walls, and otherwise walks forward.

## What the screen looks like (from the round-1 screens)

- **Floor:** green acid. Cell means are about R:G:B = 0.49 : 1 : 0.29 (G ≈ 0.17).
- **Walls:** olive stone. Red is close to green and blue is lower: R/G ≈ 0.85–1.0, B/G ≈ 0.65–0.84
  (0.84 is the highest seen in any wall cell).
- **Sky:** dark blue. It is only ever above the horizon (row 15), and the detectors never look there.
- **Pickups:** grey canisters with a green band. Grey means R = G = B (B/G = 1). Neither the floor
  nor the walls are ever that blue relative to their green.

## Layer 1 (3601 → 128)

**Units 0–119 are medkit detectors**, one per (screen column, distance band). Each unit sums one
colour feature over the cells of its column in a band of rows:

- far: rows 15–18
- mid: rows 19–23
- near: rows 24–29

Rows below the horizon are floor, and how far down an object's base sits tells its distance.
The colour feature per cell is `30 · (−0.55 R + 0.13 G + 0.48 B)`. I chose this direction to be
orthogonal to the floor colour (0.49, 1, 0.29), which gives about 0 on floor, and orthogonal to the
bluest wall colour (0.97, 1, 0.84), which gives ≤ 0 on every wall cell. On a grey cell of
brightness 0.55 it gives about +1.

Floor texture still adds some noise, so each band gets its own threshold (bias): far 0.0, mid 0.6,
near 1.0. I set these from the noise I measured on real frames, and they sit well below what a
pickup produces:
- At 60–120 units away, a pickup gives a raw sum of 2–6 in the near and mid bands.
- At about 250 units, it gives roughly +0.3 in the far band, against background of −0.7 or lower.

**Units 120–123 are wall detectors.** Each one averages `10 · (R − 0.7 G + 0.3 B)` over a region.
This gives about −0.2 on floor and +1 to +2 on wall, and each unit has a threshold of 0.3. The
regions are:
- wall right ahead: rows 24–29, centre columns 13–26
- wall ahead, further off: rows 20–29, centre columns
- wall low on the left: rows 21–29, columns 0–13
- wall low on the right: rows 21–29, columns 26–39

**Unit 124 is a constant 1.**

## Layer 2 (128 → 128; 10 units used)

**Medkit units:**
- **M** is the total medkit evidence. Near bands are weighted ×2, mid ×1.5, far ×1, so closer
  pickups count more.
- **SR / SL** are the pull to the right and to the left: the evidence weighted by the column's
  offset from the screen centre (−1 at the left edge, +1 at the right edge), split by sign with
  ReLUs.
- **SRX / SLX** are the same pull counted only beyond half of the half-screen (about 22° off
  centre). They flag a pickup that is too far to the side to reach by curving.

**Wall units:**
- **WC** passes through the "wall right ahead" detector.
- **WM** passes through the "wall ahead, further off" detector.
- **WLR / WRL** are left wall minus right wall, and the reverse, split by sign.

**Constant:** ONE.

## Layer 3: the action scores

| action | score |
|---|---|
| FORWARD | 0.3 · ONE + 5 · M |
| FWD+LEFT | 20 · SL + 0.8 · WM + 1 · WRL − 2 · WC |
| FWD+RIGHT | 20 · SR + 0.8 · WM + 1 · WLR − 2 · WC (+0.01) |
| TURN_LEFT | 80 · SLX |
| TURN_RIGHT | 80 · SRX + 2 · WC (+0.01) |

How the policy reads:

- **A pickup is visible:** its terms are the largest, so they win.
  - Walk straight when it is roughly centred. A turn takes over once the offset is more than
    about a quarter of the half-screen.
  - Curve toward it while walking when it is moderately off centre.
  - Turn in place when it is more than about 33° to the side, so the agent doesn't circle around
    a pickup it is close to.
  - The pull comes from evidence that is mostly in near columns, so the nearest pickup dominates.
- **A wall is right ahead:** turn right in place, always right. The policy has no memory. In
  round 2, turning toward the more open side made it rock left and right in corners, because the
  open side swaps as the view rotates. One game died that way (score 444). A fixed direction
  always gets out.
- **A wall is approaching, or there is a wall low on one side:** keep walking and curve toward the
  more open side.
- **Nothing is in view:** walk forward. This sweeps the small room, and new pickups come into view
  quickly.

## Checks

- **Synthetic frames:** I pasted a grey and green pickup (colours copied from real sprite pixels)
  onto a real floor frame, at 60–400 units and seven screen positions. The network walked straight
  at centred pickups, curved toward off-centre ones, and turned in place for ones at the screen
  edges. On the real close-wall frames it turns right.
- **Round 1:** the network only walked forward. Mean score 284, the death time when no pickups
  are collected.
- **Round 2:** the network had pickup seeking and wall avoidance, with turns toward the open side.
  Mean 1510: games scored 2100, 2100, 1756, 1148 and 444. The 444 game was the corner-rocking
  failure described above.
- **Round 3:** the current network, with the fixed right turn at walls. All 5 games scored 2100,
  the maximum, ending with 84–100 health.

## Prediction

My prediction for held-out games is a **mean of about 2000** (the maximum is 2100).

I'm fairly confident, about 70%, that the mean will fall between 1850 and 2100. In round 3, every
game ended with high health, so there is plenty of margin. The remaining risks:
- an unlucky layout with few pickups nearby
- getting stuck against a wall at an angle where the "wall ahead" region doesn't fire
- lighting or texture variants I didn't see in practice

Any of these could occasionally cause a death at 500–1500, which would pull the mean down.
