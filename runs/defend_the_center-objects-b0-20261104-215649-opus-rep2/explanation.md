# Hand-written network for `defend_the_center` (objects mode)

## The policy in one sentence
Shoot only when a **standing** monster covers the crosshair. Otherwise turn toward
whichever half of the screen holds more standing-monster mass. If no monster is
visible, keep turning right to scan the room.

## The key geometric fact it relies on
The player's eye is 41 map units above the floor. A standing monster is about 56
units tall, so its sprite **always reaches above the horizon** at any distance. A
corpse, and the late frames of a death animation, are shorter than 41 units, so they
**always stay below the horizon**. The labels buffer still tags a corpse with its
monster class name. So if the network shot at any monster pixel, a corpse left at
the crosshair could keep it firing until the ammo ran out. Looking only at pixels
*above* the horizon prevents that, and also stops corpses from pulling the aim.

The scenario's cfg turns the status bar on (`render_hud`). That leaves a 3D view of
about 208 px with its horizon at pixel row 104, which is exactly the top edge of
grid row 13. So the "live band" is **grid rows 4–12** (pixels 32–103). The view uses
ZDoom's 1.2× vertical scaling for 4:3, so a monster 1000 units away still pokes about
3 px above the horizon.

## Layers
Let `m[r,c]` = channel 0 (monster share) of cell (r, c), and `s_c = Σ_{r=4..12} m[r,c]`.

**Layer 1 (42 of 128 units used)**
- Units 0–39: `relu(s_c − 0.15)` for each column, a per-column dead zone so slivers
  such as the top pixel of a distant corpse or a dying sprite don't steer the aim.
- Unit 40: `s_19`. Unit 41: `relu(s_19 − s_20)`. Their difference is
  `min(s_19, s_20)`. The crosshair (pixel x = 160) lies on the boundary between
  columns 19 and 20, so a blob with mass in *both* columns covers the aim line.

**Layer 2 (4 units)**
- `L = Σ_{c<20}` of units 0–19 and `R = Σ_{c≥20}` of units 20–39: live mass in each half.
- `Fa = relu(50·(min − 0.25))` and `Fb = relu(50·(min − 0.25) − 1)`. `Fa − Fb` is a
  steep 0→1 step: the "monster on the crosshair" detector.

**Layer 3 (action scores)**
- TURN_LEFT = `L − R`, TURN_RIGHT = `R − L + 0.01` (the bias makes right the default scan
  direction), ATTACK = `1000·(Fa − Fb)`. ATTACK outweighs every turn score whenever
  the detector is on, because `|L − R|` is at most about 360.

Closer monsters are bigger, so they carry more mass. That makes the turn choice
favour the nearest (most dangerous) threat. Ammo is only spent when the hit-scan
pistol's first, accurate shot will land on a live monster.

## Checks on synthetic frames (`test_synthetic.py`)
I rendered monsters as rectangles using the projection maths above. With the horizon
at 104 the network:
- fires at centred monsters 70–600 units away and at one 3 px off centre;
- turns the right way for monsters off to either side;
- scans when the room is empty;
- does not fire at corpses (200 and 800 units away) or at a 40-unit death frame;
- turns toward a live monster while ignoring a corpse at the centre.

## Known weaknesses
- **The HUD assumption is the big bet.** If the status bar were off, the horizon
  would be at 120. The network would then see only monsters closer than about 180
  units, which means fewer kills and more damage taken. It would still not waste
  ammo on corpses.
- If the status bar were 38 px instead of 32 px (horizon at 101), a frame partway
  through a death animation could trigger one extra wasted shot.
- Monsters far away (about 1000 units) are not shot until they come closer.
- Keyboard turning moves about 2.5 columns in the first step and about 5 columns per
  step after that. A small, far target can make the aim swing back and forth across
  it for a few steps. Monster movement and growing size usually end that.
- The network has no memory. While a monster is still on the crosshair, holding
  ATTACK through the pistol's refire point (about 19 tics) fires a second shot with
  spread.

## Prediction
Average score on held-out games: **about 6** (roughly 7 kills minus 1 for the
inevitable death). I'd expect most games to land between 3 and 10. Confidence is
moderate-to-low, about 50% that the mean falls in 4–9. The biggest uncertainty is
whether the horizon really sits where I assumed. If it does not, expect about 1–3.
