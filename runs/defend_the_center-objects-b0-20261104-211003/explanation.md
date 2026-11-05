# Hand-written network for `defend_the_center` (objects mode)

## The idea

The player can only turn and shoot, and the pistol's first shot goes exactly through screen
x = 160, the boundary between grid columns 19 and 20. So the policy is:

1. **Shoot** if a monster straddles the crosshair, meaning it has pixels in both column 19 and column 20.
2. Otherwise **turn toward the side of the screen with more monster mass**. Mass grows with
   apparent size, so this goes after the nearest, most dangerous monster first. Mass is weighted
   up a little near the centre, so a target that is almost lined up wins a tie.
3. With nothing in view, **turn left** to scan. That is the tie-break toward action 0.

**Only rows 0–14 count**, i.e. everything above the horizon (y < 120). The player's eye is 41 units
above the floor. Every standing Doom monster is taller than that, so at any distance some of it
shows above the horizon line. A corpse lying on the floor never does, even near the wall, where a
flat corpse projects only a few pixels below y = 120 (into row 15). This keeps the network from
shooting corpses or aiming at them, whatever the labels buffer does with dead actors.

## Layer by layer

Below, `m[r,c]` is the monster share (channel 0) and `s_c = sum over rows 0..14 of m[r,c]`.

- **Layer 1** (6 of 128 units used)
  - Units 0–3 are saturating presence detectors for columns 19 and 20:
    `relu(60·s_c)` and `relu(60·s_c − 1)`. Their difference is ≈1 once a single monster
    pixel is in that column's band, and 0 when there is none.
  - Unit 4 is the left mass: Σ over columns ≤ 19 of `w_c · m[r,c]`, where
    `w_c = 2 − |c − 19.5|/20` is a centre weight between 1 and 2.
  - Unit 5 is the right mass, the same sum over columns ≥ 20.
- **Layer 2**
  - Unit 0 is FIRE = `relu(sat19 + sat20 − 1)`, an AND of the two column detectors.
  - Units 1 and 2 pass the left and right mass through unchanged (they are never negative).
- **Output**
  - `q_left = left mass` and `q_right = right mass`.
  - `q_attack = 5000·FIRE − 1`. A centred monster beats any turn score (masses are at most
    a few hundred). The −1 means an empty or tied screen never fires.

I checked it on synthetic frames:

| Frame | Action |
|---|---|
| empty | left |
| far monster straddling x = 160 | fire |
| monster starting exactly at x = 160 | right |
| monster ending at x = 159 | left |
| monster on the right / on the left | right / left |
| corpse-shaped blob at the centre below the horizon | ignored (left) |
| big close monster on the left plus a small far one on the right | left |
| close monster filling the centre | fire |
| a 1-pixel sliver on each side of x = 160 | fire |

## Known weaknesses

- **Refire.** Pressing attack fires 4 tics later. The pistol's `A_ReFire` check comes about
  14 tics after the press. A memoryless policy still sees the dying monster in the crosshair
  (its first death frame is roughly upright), so it is likely to hold ATTACK through the refire
  check. That fires a second, spread shot into an actor that is no longer shootable. I could not
  find a single-frame cue that tells "just shot" apart from "new target". The only difference is
  ammo parity, and that can't be used consistently. Expect up to about 2 bullets per kill. If the
  labels buffer drops monsters as they die, this problem goes away.
- **Turn quantum.** Turning comes in steps of about 7° (just after a shot) up to about 14°
  (held). A small, far monster that sits a few columns off-centre can make the network flip
  between left and right until the monster comes close enough to be wider than one step. It
  is approaching the whole time, so this always ends, but it costs time.
- Monsters behind the player (outside the 90° view) are only found when the scan reaches them.

## Prediction

The ammo limit is the main cap. 26 bullets at about 1.3–2 bullets per kill gives roughly 13–20
kills at most. Turning time, monsters biting from behind, and the −1 for death take that down.
**I predict an average score of about 8**, with a plausible range of 4–12. My confidence is low
to moderate (about 50% that the true mean is within ±3 of 8). The biggest unknowns are:

- how the labels buffer treats dying monsters;
- the real turn speed;
- whether the horizon is exactly at y = 120 for these monsters' heights. If the monsters are
  shorter than eye height, far ones would only be seen once they come close.
