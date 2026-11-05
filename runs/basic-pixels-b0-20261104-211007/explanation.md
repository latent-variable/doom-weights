# A hand-written Doom `basic` policy, `pixels` mode

## The idea

The monster is the only red object in a gray room, and the player can only strafe. So the policy is:
find the horizontal centre of the red blob, strafe toward it, and shoot once it is in front of the
crosshair (screen centre, x = 160 px, the line between grid columns 19 and 20).

## Layer 1: a red detector for each column (40 of 128 units)

For column `c`, I sum the redness `R − G` over rows 0–18 (pixels 0–151). Rows 17–18 are skipped in
the central columns 14–25 and rows 19–29 are skipped everywhere, because the pistol, hand and
muzzle flash sit there (skin and the yellow-orange flash both have R > G).

- `R − G` is about 0 for any gray or neutral pixel, so walls, floor and ceiling add almost nothing.
  The red Cacodemon gives roughly 0.3–0.6 per covered cell.
- Then I subtract the mean of that sum over all columns. The player faces the far wall straight on,
  so the background looks about the same in every column. Any overall warm tint (brownish floor,
  lighting) cancels out, and only a red blob in one place stays.
- The bias is −0.3, so after the ReLU a column fires only if it is clearly redder than average:
  `a_c = relu(s_c − mean(s) − 0.3)`.

All of these weights are ±1 and ±1/n, with n the number of cells used in that row. A completely
uniform image gives exactly 0 for every unit.

## Layer 2: where the blob is compared with the crosshair (2 units)

Let `o_c = c − 19.5` be the column's offset from the screen centre, measured in columns,
`M = Σ a_c` the red mass and `D = Σ a_c·o_c` its first moment. The centroid is `D/M`. No division is
needed, because comparing the centroid with a tolerance τ is linear:

- `u_right = relu(Σ a_c (o_c − τ)) = relu(D − τM)`: positive only if the centroid is more than τ right of centre.
- `u_left  = relu(Σ a_c (−o_c − τ))`: the same for the left.

τ = 1 column (8 px). Here is the reasoning. The Cacodemon's radius is 31 map units, so at a few
hundred units away the bullet still hits when the centroid is within roughly ±1.5–2 columns. One
4-tic strafe step moves the view by about 1–1.5 columns, so a window 2 columns wide cannot be
skipped over in a single step. Doom momentum carries the player a bit further after strafing stops.
Since we always enter the window moving toward the centre, that slide pushes the aim further
toward the monster rather than away from it.

## Output

`q_left = u_left`, `q_right = u_right`, `q_attack = 1e-4` (a constant bias).

- Monster more than 1 column off centre: strafe toward it. One strafe unit is positive, the other is 0.
- Monster within ±1 column: both strafe units are 0, so ATTACK wins.
- Nothing red visible: ATTACK. The likeliest cause is the monster sitting low and dead centre,
  inside the excluded weapon band, and then shooting is correct. With no memory, a "search" move
  could only swing back and forth.

The pistol's first shot is perfectly accurate and Doom aims vertically for you, so only horizontal
alignment matters.

## Checks I ran (synthetic frames only)

I built frames by hand: a gray ceiling and wall, a brownish floor, pixel noise, a skin-coloured hand,
optionally a muzzle flash and a weapon bob, and a red disc. The disc had radius 8–24 px, brightness
0.5–1.0 and heights from y = 80 to 145, and I swept it across x = 20…300. In every case the output
was LEFT left of centre, ATTACK at x ≈ 152–167 and RIGHT to the right, with no background-only false
alarms. The one marginal case was a disc low enough to sit mostly in the weapon band while a muzzle
flash was showing, which gave a few wrong strafe choices near the centre. That is what led to
skipping rows 17–18 in the centre and to the attack-by-default rule.

## Predicted score

When it works: about 15–45 tics of strafing, depending on where the monster spawns, plus about
10 tics of shooting, and one shot (−5), sometimes two. That would give each game about 106 − 5 − 25 ≈ 70–90.

The main risks come from assumptions I could not check without seeing the game:
- (a) The Freedoom Cacodemon might be darker or less red than I assume, or the lighting very dim,
  so detection is weak. This is partly covered: it still works at half brightness.
- (b) There might be a warm-coloured object in only part of the room, which would pull the centroid.
- (c) The monster might be much smaller or further away than I think, making τ too loose, so more
  shots miss.
- (d) The offset between what the pixels show and the hitbox.

If detection fails completely, the bot fires every step and scores very badly, so the average has a
fat lower tail.

**Prediction: about 75 average on held-out games.** I'm about 60% confident it lands in 60–90, with
roughly a 15% chance of a systematic failure (colour or geometry assumptions wrong) that drags it
below 30.
