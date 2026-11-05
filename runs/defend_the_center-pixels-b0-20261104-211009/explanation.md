# A hand-written network for `defend_the_center` (pixels)

## The policy in one paragraph

Spin in place to the left forever. Fire only when the aim line (screen x = 160, the
boundary between grid columns 19 and 20) is covered by something **standing** that does
not have the wall's colour. Never fire while the pistol's muzzle flash is on screen.
TURN_RIGHT is never used. Because the player never reverses, it can't get stuck
swinging back and forth; turning starts slow (Doom's slow-turn for the first 6 tics) and
the bot sweeps the full circle about every 3 seconds.

## Geometry this relies on (reasoned, not observed)

* 90° horizontal FOV over 320 px, so the focal length is 160 px. A thing of world size *s*
  at distance *d* is *s·160/d* px tall or wide.
* The scenario's config, as I remember it, renders the classic 32-px status bar. The 3-D
  view is then 208 px tall and the horizon (eye height, 41 units) is at y = 104. Grid row 12
  (y 96–103) lies just **above** eye level and row 13 (y 104–111) just below.
* Monsters are about 56 units tall, so they always reach above eye level and always
  cover row 12 (about 60% of it at d ≈ 500, all of it from d ≈ 300 in). Corpses lie on
  the floor, below eye level, so **they can never touch row 12**. They reach row 13 only
  when very far away (d > ~570) and only by a sliver. This is what keeps the bot from
  shooting corpses, which would otherwise lock it into firing at the same spot until it
  runs out of ammo.
* The room is a circle with the player exactly at its centre. So the wall is at the same
  distance and under the same light in every direction, and a given screen row shows the
  same strip of wall texture in every column. Apart from horizontal texture detail,
  "wall" looks the same in every column of a row.

## Layer 1 (116 of 128 units used)

**Colour difference, centre vs. reference (96 units).** The centre colour is the mean of
cells (r, 19) and (r, 20). For each pair (row r, reference column j) it is compared with
cell (r, j) through 4 ReLUs, relu(u·(centre − ref)), with u ∈ {(1,1,1), (1,−1,−1),
(−1,1,−1), (−1,−1,1)}. These four directions sum to zero, so the sum of the ReLUs is
zero only when the colours are equal. Divided by 3, the sum is a norm-like colour
difference, roughly the mean absolute channel difference, and it doesn't care which
colour the monster or the wall happens to be.
* Rows 12 and 13 against 8 one-column references spread over the outer screen: columns
  0, 4, 8, 12, 27, 31, 35 and 39 (64 units).
* Rows 9 and 10 against the four outermost columns 0, 3, 36 and 39 (32 units). This is
  the "monster in biting range" detector. Such a monster towers far above the horizon,
  and even at melee range it leaves the screen edges free.

**Muzzle-flash detectors (20 units).** For rows 16–25 and columns 19–20 (above the
pistol), each unit is relu(Y(cell) − mean of Y over the row − 0.7), where the
yellowness is Y = R + 2G − 3B. Gray, pink, brown and white score low on Y; the yellow or
orange flash scores about 1.9. Comparing with the row mean cancels the floor colour,
whatever it is.

## Layer 2 (50 units used): votes

For each (row, reference) pair, two units p = relu(D − T0) and q = relu(D − T0 − ε)
turn the difference into a clipped step, vote = (p − q)/ε ∈ [0, 1], with T0 = 0.07 and
ε = 0.02. A vote means "the centre differs from this reference strip". Two more units
clip 20·Σ(flash units) to [0, 1], giving the flash flag.

## Layer 3

q_attack = 10·(Σ weighted votes − 5.5) − 500·flash. The vote weights are 1 per strip in
row 12, 0.5 in row 13, and 5.5/7.5 per near vote (so 8 near votes are enough on their
own). q_left = 0, q_right = −1.

* Row 13 alone tops out at 4 points, so it can never fire. That is the corpse safety. Row
  12 alone needs at least 6 of its 8 strips. A real hit typically gets close to 12.
* The vote is a robust stand-in for "differs from the wall colour", which the network
  can't know in advance. Other monsters standing in a few reference strips (which agree
  with the centre, or differ from it, at random) cost a few votes but don't flip the
  decision. To fake a target on empty wall, about half the strips have to be covered.
* Requiring both centre columns means the shot is taken only when the sprite covers the
  aim line. A Doom hitscan hits exactly when the aim passes through the monster's width,
  so the bot doesn't fire at monsters it would miss.
* The flash veto works with the timing of Doom's pistol, roughly: press, bullet 4 tics
  later, flash for tics 4–10, refire check at tic 14. The bot holds ATTACK for 2 steps,
  sees the flash on the 3rd step and turns instead, which stops the refire on a monster
  that is already dying.

## What I checked (inputs I built myself)

* `test_synthetic.py` draws synthetic frames: ceiling, a textured wall band, floor,
  pistol, status bar, monster rectangles and corpse rectangles. In the status-bar layout:
  * Fires at a centred monster at every distance from 480 down to 46 (adjacent).
  * Does not fire on empty wall, on monsters 6° or more off the aim line, on corpses at
    any distance, or while the flash is visible.
  * Still fires with one or two other monsters in the reference strips.
  * Known weaknesses: it fires about 35% of the time at empty wall when four monsters fill
    the strips. It misses monsters whose colour is within about 0.07 of the wall.
* `sim_check.py` is a crude 2-D model of my own: assumed speeds, damage, pistol timing
  and turn rates; it is not the game and nothing was fitted to it. It confirmed the loop
  behaves as intended: spin, fire on alignment, the flash veto breaks refire, roughly
  0.6–0.7 kills per bullet. It also showed one remaining failure. When the player is
  surrounded by monsters at both screen edges, the references are spoiled and some
  bullets go into empty wall.

## Risks, roughly in order

1. **The status bar might not be drawn.** Then the horizon is at y = 120, rows 12–13 sit
   17–32 px above eye level, and only monsters within about 140 units get shot. The bot
   would still score, but far less.
2. **Monster/wall contrast below about 0.07**, for example in a dark map, delays shots
   until monsters are close.
3. **Strong vertical wall-texture features** at the centre could cause occasional stray
   shots. The flash veto plus the spin limit each one to about one bullet.
4. **The flash cue may be missing or look different.** Then refire wastes about one
   bullet per kill. If it falsely triggered on something persistent it would suppress
   firing, but nothing in a normal scene should look like the flash after the row-mean
   comparison.

## Prediction

Average score on held-out games: **about 4** (kills minus 1 for the inevitable death).
My 50% interval is 1 to 8 and my 90% interval is −1 to 12. Confidence is low: the
design rests on remembered facts about the scenario (status bar, sprite heights, colours,
pistol timing) that I could not check here.
