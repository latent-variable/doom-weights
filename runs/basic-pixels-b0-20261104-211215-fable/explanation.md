# Hand-written network for ViZDoom `basic` (pixels input)

## The policy in one sentence

Find the red blob in a band of screen rows that contains only wall and monster; strafe towards
whichever side of the aim line (screen centre) holds more of it; shoot as soon as the aim line cuts
through the blob with a margin on both sides.

Nothing was trained or fitted. Every constant is in `build.py` with the reason next to it.

## What each layer computes

### Layer 1: 40 units, "how much monster-red is in screen column j"

For each of the 40 grid columns `j`:

    r_j = relu( band_red(j) - mean band_red(columns 6..9 away on both sides) - theta )

- **`band_red(j)`** is the sum over grid rows 11..15 (pixel rows 88..127) of `R - 1.25 G + 0.25 B`.
- **Why that colour mix.** The coefficients sum to zero, so any gray of any brightness scores exactly
  0, which removes gray walls, lighting changes and texture brightness. Brown, tan and skin (G about
  halfway between R and B) score little, a yellow-white muzzle flash scores about 0, and a red body
  scores close to its full `R - G`.
- **Why those rows.** I assumed the status bar is drawn, which puts the horizon at pixel row 104. The
  monster stands on the floor at the far wall, with its body from below eye level to a little above,
  so it straddles the horizon at roughly pixel rows 91..125. The far wall's floor line is at about
  row 124 and the idle pistol starts lower, so this band should hold only far wall and monster. The
  status bar (red digits), floor, ceiling and weapon get zero weight. If there is no status bar
  (horizon at 120), the band still covers the top 60% of the monster.
- **Why subtract the surround.** If the walls are not perfectly gray, a tint that is uniform or
  changes smoothly across the screen cancels. The offsets 6..9 clear a monster up to about 6 columns
  wide; I expect about 3.
- **Thresholds.** `theta` is 0.05 (a fully covered monster column is about 1.0), plus 0.10 for the
  six columns at each screen edge, whose surround is one-sided.

The other 88 hidden units and the ammo input are unused (zero).

### Layer 2: 43 units

With `S = sum_j r_j` (all redness found):

- **Units 0..39:** `s_j = relu(r_j - 0.08 S)`. Columns holding under 8% of all redness are dropped as
  texture speckle; each main monster column holds 20-40%.
- **Units 40..42:** `relu(r_19 - 0.05 S)`, `relu(0.05 S - r_19)`, `relu(r_19 - r_20)`. The first minus
  the other two equals `M = min(r_19, r_20) - 0.05 S`. Columns 19 and 20 are the two columns touching
  the aim line (x = 160).

### Layer 3

    q[LEFT]   = sum of s_j for j <= 19
    q[RIGHT]  = sum of s_j for j >= 20
    q[ATTACK] = 1000 * M - 1

So the network attacks if and only if both centre columns hold more than 5% of all the redness, and
otherwise strafes towards the side with more red. With nothing detected it outputs LEFT.

## Why the shoot test is built this way

- **It is scale-free.** "5% of the red mass in the centre column on the minor side" means the aim
  line is inside the sprite by about 23% of its half-width, whatever the monster's distance or
  brightness. I do not know the room's depth; I guessed about 384 map units, which makes the monster
  about 26 pixels wide.
- **It respects Doom's inertia.** The player coasts after releasing a strafe key, and the pistol
  fires 4 tics after the press. By my arithmetic (walking strafe: 0.75 units/tic² thrust, friction
  0.90625, top speed 8 units/tic) the player coasts up to 25 units before the shot leaves. The
  monster's hitbox is ±31 units.
  - From rest, the shoot zone must be narrower than ±31.
  - Arriving at full speed, it must be wider than about ±23, or one 4-tic step can carry the player
    through the zone and out of the hitbox before the shot.
  - The test gives about ±24 units if the sprite is as wide as the hitbox, and stays inside that
    window for sprites from 0.85× to 1.25× the hitbox width.
- **The first pistol shot is perfectly accurate**, so being inside the hitbox horizontally is enough.

## What I checked

`check.py` draws my own guess of the room and feeds it through the real `encode` and `forward`, with
a crude model of strafing and pistol timing. It is not the game, and nothing in it sets the weights.

- **Static frames:** a clean LEFT | ATTACK | RIGHT split with the attack zone at ±20 to ±30 units
  across every variation I drew. These covered distance 256 to 640, gray or brown walls, dull, orange
  or dark monsters, with and without status bar, a raised weapon, and the player near a side wall.
  The one exception is a frame with a muzzle flash drawn at the centre, where the zone widens to
  about ±42. That frame only occurs for one decision after a missed shot, and attacking then does
  not fire an extra bullet by my timing model.
- **Closed loop in my model:** mean score 77.5 with 100% one-shot kills in the nominal case.
- **Stress cases** (the stain rows come from `stress3.py`, the rest from `check.py`):

| Case | Mean score in my model | One-shot kills |
|---|---|---|
| Nominal | 77.5 | 100% |
| Sprite 0.85× hitbox width | 75.9 | 96% |
| Running strafe speed (13.3 units/tic) | 73.4 | 71% |
| Wall colour stains, per-channel std 0.03 | 77.2 | 99% |
| Wall colour stains, per-channel std 0.06 | 67.2 | 87% |

- At stain level 0.06 some episodes never kill (score -300): the stains are then about as red as the
  monster, and the detector follows one.
- I tried one alternative hand design (requiring a column to be redder than the wall on both sides
  separately). It was no better on these checks, so I kept the simpler one.

## Prediction

**About 74 average on held-out games.** If my picture of the scene is right, I expect 72-79: about
18 tics of strafing on average, about 5 tics for the shot, one bullet, so 106 - 5 - 23 or so.

Confidence is moderate:

- **About 70%** that the average lands between 65 and 82.
- **About 15-20%** that something I could not see breaks the detector and the average is far lower,
  possibly negative. The candidates are:
  - the monster is not red-dominant in the way I assumed;
  - the pistol sprite or something else reddish reaches into pixel rows 88..127 at the screen centre,
    which would cause repeated wasted shots;
  - the monster sits outside that row band, in which case the network strafes left forever (-300).
- **A smaller risk** is that the strafe is faster than walking speed. That costs some second shots
  (my model says about 4 points), not a collapse.
