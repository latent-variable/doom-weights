# Hand-written network for `health_gathering` (pixels)

The idea: the acid floor is strongly green, and a medkit is a small non-green object (white box, red cross)
standing on it. So I look for small, isolated, non-green blobs in the floor part of the screen and steer
toward them. When nothing is in view I walk forward. When a wall comes close I turn toward the side with
more open floor.

Geometry used (90° HFOV, 320x240, focal length about 160 px, eye height 41): the horizon sits between
rows 14 and 15. A floor point at distance d is about 6560/d pixels below it, so rows 15-16 are more than
about 400 units away, rows 17-19 about 160-400, rows 20-24 about 90-160 and rows 25-27 about 55-80.

## Layer 1 (100 of 128 units used)

Two per-cell colour features, both linear in the input:
- `s = R + B - G` ("non-greenness"): negative on the green floor, clearly positive on a white or red medkit.
- `g = G - (R + B)/2` ("greenness"): positive on the floor, about 0 or below on grey or brown walls.

**Medkit detectors (48 detectors, 2 units each).** Each one covers a band of rows and three equally wide
groups of columns: left, centre and right. The bands are rows 15-16, 17-19, 20-24 and 25-27, with widths
2, 2, 3 and 4 columns, wider for nearer bands because near medkits look bigger. I want
`min(centre - left, centre - right)`, where each term is the summed `s` difference. This is positive only
when the centre is less green than both neighbours, i.e. a localised blob. A straight wall/floor boundary
(a step or a slanted ramp) makes one of the two differences ≤ 0, so walls do not fire it. A concave room
corner comes out negative. The min is computed exactly with relus:
`P = relu(K + centre - left)` (K = 30 keeps it positive) and `Q = relu(right - left)`. Then layer 2 takes
`relu(P - Q - K - thr) = relu(min(...) - thr)`.

The two lower bands skip the centre columns (15-24 and 13-26). That way a weapon sprite at the bottom
centre, if the config draws one, can't look like a permanent medkit ahead. A medkit straight ahead and
close needs no detection anyway, because the default action is forward.

**Wall units:**
- `Bf = relu(1 - Σg / (0.06·N))` over rows 19-21 in the flank columns 6-14 and 25-33: the wall is fairly
  close ahead.
- `Bn`: the same over rows 22-24: the wall is very close.
- `gL` and `gR`: the summed floor greenness on the left (cols 0-11) and right (cols 28-39), rows 17-21.

## Layer 2
- One unit per detector: `relu(min(centre-left, centre-right) - threshold)`. Thresholds are 0.10, 0.15,
  0.30 and 0.40 by band.
- `tL = relu(Bf + 3·Bn - 0.2 + 0.02·(gL - gR))` and `tR` the mirror image (plus 0.01 to break ties): turn
  because blocked, toward the more open side. The small greenness term also nudges the agent away from a
  wall it is running alongside.
- A copy of `Bn`.

## Output
Each detector votes for an action according to its centre column. Nearer bands vote with more weight:
1, 1.5, 2 and 2.5.

| Columns | Action |
|---|---|
| 0-7 | TURN_LEFT |
| 8-15 | FWD+LEFT |
| 16-23 | FORWARD |
| 24-31 | FWD+RIGHT |
| 32-39 | TURN_RIGHT |

On top of the votes:
- FORWARD gets a bias of 0.02, so it wins when nothing is seen.
- FORWARD gets −20·Bn, so the agent never pushes straight into a wall.
- TURN_LEFT and TURN_RIGHT get +2·tL and +2·tR.

## Checks (synthetic only)
`test_synth.py` ray-casts a crude square room: a green, noisy floor, brown or grey walls, and white
medkit sprites with a red cross. `sweep_synth.py` tries random poses under four colour/lighting variants,
including a 45% dimmer scene and grey or greenish walls.
- **False positives:** with no medkit in view, no pose (0 of 240) produced a false steer toward a "medkit".
- **Steering with one medkit in view:** the agent chose the right direction in about 80-90% of cases.
  Most misses were medkits that were very close or right at a sector boundary.
- **Walls:** a wall 90 units ahead makes it turn. Under about 60 units, forward is vetoed. Corners and
  running alongside a wall turn the agent away from the wall.

Real colours, lighting and the room size are unknown to me. The design relies on contrasts (green floor
versus non-green objects, and centre-surround) rather than on exact values.

## Prediction
For reference, an agent that does nothing dies at roughly 300 tics. A perfect one survives all 2,100 tics
and scores about 2,100.

If the floor is clearly green and medkits read as non-green, I expect most episodes to last a long time.
I predict an **average of about 900 (plausible range 350-1800)**, and my confidence is low. The main
risks are:
- The floor's greenness is below the wall threshold (mean `G - (R+B)/2` < 0.06), which would make the
  agent turn all the time.
- A HUD or status bar drawn in the lower rows.
- Wall textures with small high-contrast features near the floor line.
- The agent may circle the room's perimeter when no medkits are visible.
