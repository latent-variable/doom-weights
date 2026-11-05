# Hand-written policy for `defend_the_center` (objects mode)

Only 5 of the 128 hidden units in each layer are used. The network only looks at
channel 0 (the monster share). The crosshair sits at pixel x = 160, the border
between grid columns 19 and 20.

## The policy in words

1. If a monster covers the crosshair, **shoot**.
2. Otherwise, if a monster is *near* (it draws pixels at row 19 or lower on the
   screen) and the near monsters on the left outweigh those on the right,
   **turn left**.
3. Otherwise **turn right**. The player keeps sweeping in one direction.

## Layer 1 (input → h1)

| unit | computes |
|---|---|
| 0 | `a = m19`: monster share summed over all 30 rows of column 19 |
| 1 | `relu(m19 - m20)` (so that `min(m19, m20) = a - unit1`) |
| 2 | near-left mass: monster share in rows 19–29, columns 0–19, weight 0.5–1.0 rising toward the centre |
| 3 | near-right mass: the same for columns 20–39 |
| 4 | constant 1 (bias) |

## Layer 2 (h1 → h2)

* `h2[0] = relu(min(m19,m20) - 0.1)` and `h2[1] = relu(min(m19,m20) - 0.3)`.
  `(h2[0] - h2[1]) / 0.2` is a 0-to-1 step that means "monster pixels sit on both
  sides of the crosshair". Requiring *both* columns means the sprite actually
  spans x = 160, so the shot hits. A sprite that only touches column 19 or 20 is
  not counted.
* `h2[2..4]` pass near-left, near-right and the constant through unchanged.

## Output

* `q_attack = 1000 · step`. When the crosshair test fires, this beats everything else.
* `q_left = 2 · (nearL − nearR)`
* `q_right = 0.01`, which wins whenever nothing else is positive.

## Why this design (and what practice showed)

* **The turn step is coarse.** This comes from reading how (Z)Doom handles
  keyboard turning, not from measurement. A held turn speeds up: roughly 7° on the first
  step, then up to ~14° per 4-tic step, which is about 5 grid columns. Switching
  between left and right does not reset that speed-up. My first policy steered
  toward the monster closest to the centre. Against a small, far monster
  (1–2 columns wide) the aim can then swing back and forth past it every step
  and never cross it. That policy scored 16.4 and 11.2 in practice, and most
  games ended in death with ammo left over.
* **Sweeping fixes this.** Turning right whenever possible means the crosshair
  passes every monster in turn, with no back-and-forth. Firing stops the turn,
  so the next turn steps after a shot are small again. Far monsters only get
  wider as they approach, so later passes hit them.
* **Only near monsters justify turning left.** A near monster is at least as
  wide as one turn step, so turning toward it can't swing past it, and it is the
  one about to bite.
* **Ammo is spent almost only on hits.** The crosshair test is strict, so a
  bullet is nearly always a kill. With 26 bullets the best possible score is
  about 25: 26 kills, then death once the ammo is gone.

Practice results (5 games per round):

| round | policy | mean |
|---|---|---|
| 1 | steer toward the monster closest to centre | 16.4 |
| 2 | same, weighted toward near monsters | 11.2 |
| 3 | + fire at narrow monsters just off the crosshair | 12.6 (wasted ammo) |
| **4** | **sweep right, turn left only for near monsters (final)** | **24.2** (24, 24, 24, 24, 25) |
| 5 | + only shoot once the monster is past mid-room (to make ammo last to 2100 tics) | 21.6 (one game: 10) |
| 6 | round 4 with a stricter crosshair threshold (0.5) | 23.2 |

Round 5 tried to survive to the time limit, which would turn the −1 for death
into 0. It came close (1,994 tics) but once let monsters overwhelm the player, so
I dropped it. Round 6 was not clearly better than round 4, so the final weights
are exactly the round-4 network.

## Prediction

Average held-out score: **about 22–23** (range of plausible means 20–24.5).
Most games should end with 24–25 (all ammo used, at most one or two misses).
Occasionally the player gets bitten to death early by a monster that comes from
behind while it is sweeping, and a game like that scores 10–20. This pulls the
mean below the round-4 practice score of 24.2. I'm moderately confident (~70%)
the mean will be ≥ 21.
