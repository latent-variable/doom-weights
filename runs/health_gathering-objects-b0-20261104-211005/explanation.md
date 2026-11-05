# Hand-written health_gathering policy (objects mode)

The policy has no memory. It does three things:
1. **Steer toward medkits.** Every pickup pixel on screen votes for a steering action.
2. **Avoid walls.** If a wall is close ahead, turn right in place.
3. **Otherwise walk forward.**

Only 9 of the 128 hidden units in each layer are used. The rest stay at zero.

## Layer 1 (x → h1)

**Units 0–4: medkit votes, one per action.** Each unit is a weighted sum of the pickup channel (channel 1) over the whole grid. The inputs are never negative, so the ReLU changes nothing. Each cell's weight depends on where the cell is on screen:

- **Centre band** (|column − 19.5| ≤ 2.5, about ±6°): votes `MOVE_FORWARD`. One column further out on each side, the cell splits its vote evenly between forward and the turn.
- **Off to one side:** votes `MOVE_FORWARD + TURN` toward that side.
- **Close to the player and well off to the side:** votes for a turn in place (`TURN_LEFT` or `TURN_RIGHT`) instead. "Close" is judged from the floor distance of the cell's row, `41·160/(y−120)` (eye height 41, focal length 160 px). A point is too close when it lies inside the circle the player traces while moving and turning together. That circle has a radius of about 135 units (about 8.3 units per tic at about 3.5° per tic). A forward+turn toward such a point would just circle it, so the bot turns in place first.
- **Row weight:** each cell's vote is scaled by `1 + 0.15·(row − 14)`, so lower rows (nearer the player) count more. Nearer medkits also cover more pixels, so the nearest one usually decides. If two medkits tie, `argmax` picks the lower index, which is a consistent left. This breaks the symmetry instead of driving between them.

**Units 5 and 6: wall-ahead test.** These look at the depth channel in the centre columns 16–23 and compute `u = 1.9·depth(row 22) − depth(rows 14–15) + 0.004`.
- Rows 14–15 sit at the horizon, so they show whatever is straight ahead.
- Row 22 shows the floor about 109 units away, unless a wall is nearer than that.
- So `u > 0` exactly when the object at the horizon is closer than about 1.9 × 109 ≈ 200 units.

This test compares two depths as a ratio. As long as depth grows in proportion to distance, the test is right whatever the depth buffer's units are. I checked this on synthetic frames with depth ranges from 500 to 5000 units: walls at 190 units or nearer trigger it, walls at 230 or further don't.

The gain is 500. Unit 5 is `relu(500u)` and unit 6 is `relu(500u − 1)`, so their difference is a sharp 0/1 step.

**Units 7 and 8: safety veto.** These form the same kind of 0/1 step, firing when the horizon depth is saturated (≥ 0.97). In that case the depth scale is too short for the ratio test to mean anything, so the veto switches the wall test off. Without it, a short depth scale would make the bot spin in place forever.

## Layer 2 (h1 → h2)

- Copies the 5 vote units unchanged.
- Computes `wall = relu((h5 − h6) − (h7 − h8))`: a clean 0/1 signal for "wall close ahead and the depth reading is trustworthy".

## Output (h2 → q)

- `q[a]` = the medkit vote for action `a`.
- `+ 0.05` bias on `MOVE_FORWARD`: with nothing in view, walk forward.
- `+ 3·wall` on `TURN_RIGHT`: at a wall, turn right in place until the centre of view is clear.
  - The turn direction is always right. A rule that compared left and right openness could flip back and forth in a corner and stall there, because the policy has no memory.
  - A medkit straight ahead that is closer than roughly 250 units has more than 3 votes, so it beats the wall turn. Medkits lying against a wall still get picked up.

## Checks on synthetic frames

| Scene | Action chosen |
|---|---|
| Medkit straight ahead at 400 units | forward |
| Medkit 15° right | forward + right |
| Medkit 30° left | forward + left |
| Medkit 40° left at 100 units | turn left in place |
| Wall at 160 units with a medkit 5° right at 150 | forward (goes for the medkit) |
| Wall at 150 units with only a distant medkit | turn right |
| Medkits 20° left at 300 and 25° right at 200 | goes for the nearer, right one |

## Why it should score well

The bot always moves unless it is facing a wall. It heads for the nearest visible medkit, and when there is none it crosses the room and turns off at the walls. This sweeps the room, and medkits keep spawning. That is the core of what good agents do on this scenario.

Ways it can go wrong:
- **The depth buffer is not proportional to distance.** Then the wall test is miscalibrated:
  - If it fires too rarely, the bot rubs along walls, but sliding still moves it.
  - If it fires too often, the bot turns more than it needs to.
- **Wasted time:** turning right at a wall can sometimes cost a turn of about 270°.
- **Medkits off screen:** with no memory, a medkit that drops out of view is forgotten.

## Prediction

Score = tics survived, minus 100 if the bot dies. A full episode scores about 2,100.

I expect most episodes to survive all or most of the 2,100 tics, with some deaths when medkits are sparse or the bot loses time at walls.

- **Predicted average on held-out games:** about **1,400**.
- **Range:** a plausible spread of 800–2,000, at about 60% confidence.
- **Main uncertainty:** the depth-buffer scaling and the room's actual layout, which I could not observe.
