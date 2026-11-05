# Hand-written network for `basic` (objects mode)

## Prediction

**Average score on held-out games: about 79.** I would put roughly 70% on the range 74 to 84,
and about 10% on something systematic being wrong (below 60).

Nothing here was run against Doom. The number comes from a stand-in simulator I wrote from memory
of the engine's rules (`sim_check.py`), in which this network averages 80.4 and kills with its
first shot in every episode. The stand-in shares my assumptions, so it checks that the weights do
what I meant, not that what I meant is right. The assumptions that matter most are listed at the
end.

## The policy

1. **Strafe toward the monster.** Whichever half of the screen holds more monster, move that way.
2. **Shoot when the bullet line is inside the monster.** The bullet travels along the vertical
   centre line of the screen (between columns 19 and 20). The Cacodemon's hit box is 62 units
   wide and its sprite is about the same width, so "the silhouette straddles the centre line"
   means "a shot would hit". I require at least 5% of the silhouette's area on each side of the
   line, which for a round sprite is about 6 units inside the edge.
3. **Shoot early while sliding.** The pistol fires 4 tics after ATTACK is pressed, and the player
   keeps coasting during those tics (up to about 24 units at full walking speed). So when the
   player is moving fast, ATTACK is pressed while the monster's edge is still 4 to 14 units short
   of the centre line. This is worth about 2 points per game in the stand-in (80.4 against 78.3
   without it) and makes the policy far more tolerant of faster movement than I assume.
4. **After a miss, play it plain.** Ammo below 50 means a shot has missed. Early shots are
   switched off and the rule becomes "centre line at least 3% inside the silhouette". After five
   misses it tightens to 15%.

### How a memoryless network knows it is moving fast

It cannot see speed, but it can see where it is. The player starts at rest in the middle of the
room and only ever slides toward the monster, so distance from the spawn point bounds speed from
below. Distance from the spawn point shows up in the depth channel: slide right and the right
wall gets nearer than the left. I use

    rho = (SL - SR) / (SL + SR)

where `SL`, `SR` are summed depth over rows 13 and 14 in the 13 outermost columns on each side.
`rho` is exactly 0 at the spawn point by symmetry, needs no knowledge of the depth scale, and
grows with distance slid. Six thresholds on `rho` (about 12, 22, 42, 66, 100, 150 units of
travel) unlock six sizes of early shot (4, 7, 10, 12, 13, 14 units). Each is only unlocked on the
side the player has slid toward.

Each early shot is deliberately smaller than the coasting distance it compensates: it is sized
for the speed the player would have after 1/1.6 of the travel, minus 4 units. In the stand-in the
first bullet lands at least 4.6 units inside the box across room depths from 300 to 448.

## What each layer computes

`L`, `R` are the monster area (in cells) left and right of the centre line; `T = L + R`.

**Layer 1, 45 units used** (all linear sums of the input, then ReLU):

| units | computes | for |
|---|---|---|
| 2 | `relu(L - R)`, `relu(R - L)` | which way to strafe |
| 3 | `relu(L - f*T - 0.02)` for f = 0.05, 0.03, 0.15 | margin on the left side |
| 3 | from ammo: "1+ shots fired", "5+ shots fired", "no shot fired" | switching rules |
| 2 | `relu(±M)`, `M` = sum of (column centre - 160 px) x area | monster centroid offset x `T` |
| 22 | `relu(T - t_i)` at knots 0, 1, 1.25, 1.56, ... | piecewise-linear `T^1.5` |
| 12 | `relu(rho_j*(SL+SR) ∓ (SL-SR))` for 6 thresholds, 2 sides | positive when the player has NOT slid that far |
| 1 | `relu(0.25 - (SL+SR))` | positive when the depth channel is blank |

**Layer 2, 17 units used:**

- 2 pass-through units for strafe direction.
- 3 "aligned" units: `relu(left margin - relu(L - R) - gates)`, which equals
  `min(left margin, right margin)`, so it is positive only when the centre line is inside the
  silhouette with margin `f` on both sides. The 5% rule is gated off once a shot has been
  fired; the 3% rule is on for 1 to 4 shots fired; the 15% rule is always on.
- 12 early-shot units (6 sizes x 2 sides):
  `relu(kappa * width * T - centroid offset * T - 1e5 * (not slid far enough) - 1e5 * (shot fired) - 1e6 * (depth blank))`
  with `kappa = 0.5 + early/62`. The sprite's width in pixels is estimated from its area as
  `1.06 * sqrt(area)`, which is why `T^1.5` is needed: `width * T = 8 * 1.06 * T^1.5`. Using the
  sprite's own size as the ruler means I do not need to know how far away the monster is.

**Output:** `q_left = relu(L-R) - relu(R-L)`, `q_right` the negative of that,
`q_attack = 1000 * (aligned units) + 100 * (early-shot units) - 0.01`. Any rule that fires
outvotes strafing; with no monster visible ATTACK is never chosen.

## Why I expect it to score well

- The score is `106 - 5 * shots - tics`. One shot always; the rest is time. Reaching the monster
  takes up to about 30 tics, so a clean policy averages near 80.
- Strafing is at full thrust the whole way and the shot goes off as early as is safe.
- Failure is bounded. A miss costs about 20 points (the shot plus the pistol's cycle time), then
  the plain rule takes over. From 455 adversarial mid-episode starts with a shot already spent,
  the stand-in never fails to kill under my main assumptions.

## Checks I ran (all on inputs I constructed)

- `unit_check.py`: 24 constructed frames (empty screen, block monsters left/right/straddling,
  at rest just outside and inside the box, sliding with and without a spent shot, blank depth).
  All give the intended action.
- `sim_check.py play`: closed-loop average over 97 monster positions, under 21 variations.

| variation | mean | episodes needing 2+ shots |
|---|---|---|
| main assumptions (walking speed, 4-tic pistol delay) | 80.4 | 0% |
| pistol delay 5 tics / 3 tics | 79.4 / 81.4 | 0% |
| one tic of input lag | 78.6 | 0% |
| room depth 300 / 448 | 80.4 / 80.6 | 0% |
| sprite 72 or 54 wide, boxy, horned, no vertical stretch, status bar on | 79.9 to 81.0 | 0% |
| depth scale about 4x larger or 4x smaller, Euclidean depth | 80.4 | 0% |
| running speed (5/3 the thrust) | 84.0 | 1% |
| running speed and 5-tic delay | 81.1 | 7% |
| running speed and input lag | 78.7 | 13% |

## What could make the real score differ

- **Movement and pistol timing are from memory** (thrust 0.75/tic, friction 0.90625, shot 4 tics
  after the press). If the player is slower than I think, early shots land short; the margins
  above cover about 4 units of error.
- **Sprite versus hit box.** I assume the Freedoom sprite is about 62 units wide, roughly round
  and centred on the hit box. A sprite 10 units wider still worked in the stand-in; much wider
  would cause early misses on far monsters.
- **Depth channel.** I assume wall depth is visible in rows 13 and 14 and the room is symmetric
  about the spawn point (side walls 224 units each way, inferred from the monster's spawn range).
  If depth carries nothing, early shots simply never unlock and the policy is the plain one
  (about 78). If the room were not symmetric, early shots could fire from rest on one side.
- **Weak spot.** If the player both runs and has input lag, the after-miss rule can oscillate
  past the monster and never kill (17 of 455 adversarial starts with three shots spent). I think
  that combination is unlikely (a few percent) and did not find a memoryless fix for it.

## Files

- `build.py` writes `weights.npz` (numpy only, about 0.1 s, reads nothing).
- `sim_check.py`, `unit_check.py`, `margin_check.py`, `fallback_check.py`, `variants_check.py`,
  `probe_check.py`, `trace_check.py` are my own checking tools; none of them fit or search
  weights. Every constant in `build.py` was set by the reasoning above. Three were revised after
  a check exposed a mistake in that reasoning: the gate strength (too weak to shut an early-shot
  rule for a large monster), the depth columns (the monster's own depth leaked in), and the
  after-miss rule (I first tightened it, which made fast overshoots worse).
