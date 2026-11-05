# A hand-written Doom network for `defend_the_center` (pixels)

## What the picture looks like (from the round-1 screens)

The player stands at the centre of a round room, so the background is the same in every direction:
dark-blue sky at the top, a uniform brown wall (cell colour ≈ (0.43, 0.34, 0.27)), and a grey floor
below the horizon at row ~16. The monsters are grey chainsaw marines and pink demons. Both have
**blue ≈ green**, but the wall has blue clearly below green (b−g ≈ −0.075), and demons are also much redder
than the wall. The feature `0.3r − 1.3g + b` is therefore flat on the wall (cell noise ≈ 0.01) and rises
for any monster pixel.

I only look at rows 9–14 (y = 72–119). Those rows are always wall, apart from a monster in front of it.
The sky reaches row 7 at most, and the floor starts at row 16. Because those rows are above eye level,
corpses and blood on the floor (which are shorter than the 41-unit eye height) can never show up there.
Only standing monsters can.

## Layer 1 (43 of the 128 units used)

* **Units 0–39: one monster detector per screen column.**
  `s_c = relu(10·(F_c − mean over all columns of F) − 0.2)`, where `F_c` is the sum of the colour
  feature over rows 9–14 of column c. Subtracting the mean over all columns cancels anything that shifts
  the whole screen at once, such as the red pain tint or the muzzle-flash light. The 0.2 threshold
  (0.02 raw) sits above the wall's texture noise.
* **Unit 40: gun recoiling.** The pistol sprite is drawn on top of everything, so certain cells in
  rows 25–29 only ever see the hand and gun. When the gun kicks, the hand moves into cells (29,21),
  (29,22), (25,19) and (25,20). I compare their brightness with gun cells that never change, and with a
  bright-minus-dark contrast on the gun that sets the threshold. Every term is a difference of equal
  weights, so a red pain tint or a flash of light only scales the value and cannot flip its sign. On
  every saved frame, ready gives −3.5, recoil +4.0 and fired +28.
* **Unit 41: muzzle flash.** The two cells above the barrel turn bright yellow.
* **Unit 42: out of ammo.** `relu(1 − ammo)`, computed from the AMMO2 input.

## Layer 2

* h0 / h1: **left and right monster mass**, `Σ w(c)·s_c`. The weight w(c) is larger near the centre, so
  the target we are already turning toward stays the preferred one (a form of hysteresis without memory).
  Bigger means closer, so the closest monster wins.
* h2: **crosshair coverage**, `s19 + s20`, the two columns on either side of x = 160.
* h3: **near-miss brake**, `relu(mass in columns 15–18 and 21–24 − 3)`. A sizeable monster is just beside
  the crosshair.
* h4: **recoil veto**, `relu(recoil − 30·flash − 1000·out_of_ammo)`.

## Output

`q_left = h0 + 1`, `q_right = h1`, `q_attack = 100·h2 + 20·h3 − 5000·h4`.

* If the crosshair is covered, shoot.
* Otherwise turn toward the heavier side. With nothing on screen, turn left: the +1 scan bias also makes
  the network sweep past tiny, far-away targets on the right instead of fixating on them before they can
  be hit.
* **Brake.** A held turn speeds up after 6 tics, so continuous turning moves about 6 columns per step.
  That overshoots a monster only 2–3 columns wide, and the network can oscillate left/right around it
  forever. Pressing ATTACK for one step resets the turn speed, so the next turn is a slow ~3-column step.
  When the brake shot is fired at a big monster it usually hits anyway.
* **Recoil veto.** Pressing ATTACK while the pistol is recoiling triggers an automatic refire about 2
  tics later. That shot goes into the monster that is still falling over from the previous hit, which
  wastes a bullet. During the flash frames a press does nothing, so attacking (for example as a brake) is
  free then. Once ammo runs out the veto is switched off, so the fist can still punch.

## Practice history (5 games per round)

| round | change | mean |
|---|---|---|
| 1 | all-zero network (turns left forever) | −1.0 |
| 2 | column detectors on rows 7–14, fire on crosshair | 8.0 (the sky dipping into row 7 wasted ammo) |
| 3 | rows 9–14 | 6.4 (efficient with ammo, but kept oscillating and got overrun) |
| 4 | + brake | 13.4 |
| 5 | + scan bias | 20.0 |
| 6, 7 | + recoil veto | 21.4, 21.4 |
| 8 | stronger out-of-ammo gate (final weights) | 20.6 |

In the last rounds most games end the same way: the network spends all 26 bullets for 20–25 kills, then
gets killed bare-handed. The score is close to its ceiling of 26 kills − 1 for death, plus any fist kills.

## Prediction

Average held-out score **≈ 21** (15 practice games with the final design: mean 21.1, range 16–24).
I'm about 80% confident the held-out mean lands between 19 and 23. Risks that could pull it lower:
lighting or texture situations the practice screens didn't show, such as a large red tint while badly
hurt, or blood gibs flying through rows 9–14 and drawing brief turns or shots.
