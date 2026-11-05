# Hand-written Doom `basic` policy (objects mode)

## Idea
The player can't turn, so the shot always goes along the screen's centre column, pixel x = 160. That is
exactly the boundary between grid columns 19 and 20. The monster channel (channel 0) is exactly 1.0 for
monster pixels and 0 elsewhere, so the policy only needs to know how much monster lies left of the
crosshair and how much lies right of it:

- `a` = sum of channel 0 over all rows, columns 0..19 (left of the crosshair)
- `b` = sum of channel 0 over all rows, columns 20..39 (right of the crosshair)

The Cacodemon stays at the far wall, so its size on screen is nearly constant: about 3.5 columns wide and
5 rows tall, so a fully covered column sums to about 3.6. If both `a` and `b` are clearly positive, the
crosshair is on the monster, so fire. Otherwise, strafe toward the side with more monster.

## Layers
- **Layer 1** (2 active units): `h1[0] = a`, `h1[1] = b`. All weights are 1 on the monster channel of the
  matching columns, and the biases are 0. Depth, pickups and ammo are ignored.
- **Layer 2** (3 active units): `h2[0] = a`, `h2[1] = b`, `h2[2] = relu(a - b)`. Together these give
  `min(a, b) = h2[0] - h2[2]` exactly.
- **Output**:
  - `q_left = a - b`
  - `q_right = b - a`
  - `q_attack = 1000 * (min(a, b) - 0.4)`

  The gain of 1000 makes the fire decision a sharp gate. Once the crosshair is at least 0.4 mass units
  inside the monster's edge, firing beats any strafe score, since |a-b| is at most about 12. If no monster
  is visible, every score is at or below 0 and the tie goes to MOVE_LEFT, which acts as a search; in
  practice the monster has always been visible.

## Why the 0.4 threshold
Doom strafing has momentum, and the pistol fires about 4 to 5 tics after ATTACK is pressed. While
approaching, the player keeps sliding about 1 to 1.7 columns before the bullet leaves. My first version
waited for the monster to be balanced, which caused one overshoot and an extra correction. Firing when
the edge is only 0.4 inside (the monster centroid about 1.3 columns from the crosshair) means that:
- when moving, the drift brings the monster onto the crosshair by the time the shot fires;
- when already standing still (the monster spawned near the centre), the crosshair is still inside the
  sprite and hitbox, which have a half-width of about 1.75 columns.

I checked the network on synthetic frames, such as a 4-column monster at different columns, and on the
recorded practice inputs, to confirm it computes this rule.

## Practice
| round | change | mean | kills |
|---|---|---|---|
| 1 | balance-based fire, centre columns only | 76.8 | 5/5 (one overshoot + 2 shots) |
| 2 | fire on min(a,b) > 0.7 | 75.0 | 5/5, all one shot |
| 3 | threshold 0.4 | 77.4 | 5/5, all one shot |

The rounds used different monster positions, so the means differ mostly because of how far the player had
to strafe. Since round 2, every game has been one shot, and the shot comes as soon as the monster reaches
the crosshair.

## Prediction
Score = 106 - 5 - tics. A monster spawned near the centre gives about 95. One at the far edge needs about
8 to 10 strafe steps, which gives about 60 to 65. Averaged over random spawn positions, I predict a
**held-out mean of about 76, likely between 72 and 80**. I am fairly confident (about 80%) that it will be
in that range. The main risk is a rare miss, which costs about 15 to 20 points for that game, or a spawn
that starts off-screen, which I haven't seen happen.
