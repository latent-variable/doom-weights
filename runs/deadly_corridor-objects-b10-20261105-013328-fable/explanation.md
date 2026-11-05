# Hand-written network for `deadly_corridor` (objects mode)

## Prediction

**Average score on held-out games: about 1,600.** I am not very confident: my 80% range is roughly 1,250 to 1,950.

- A game is close to all-or-nothing: about 2,277 for reaching the vest, a few hundred for dying on the way.
- The final network won 3 of its 5 practice games (mean 1,649). The four near-identical versions before it also won 3 of 5 each (means 1,491 to 1,571).
- So the estimate is "wins about 60 to 65% of games, about 500 when it loses". Five games is a small sample; a true win rate anywhere from 45% to 80% is consistent with what I saw.

## How the network is used

The weights are a small logic program, written by `build.py`:

| Layer | What it computes | Units used |
|---|---|---|
| 1 | "Predicates": linear measurements of the screen, each clipped to 0..1. One ReLU when the measurement cannot be negative, two (`relu(z) - relu(z-1)`) otherwise. | 30 of 128 |
| 2 | ANDs of predicates and their negations: `relu(1 - sum of the literals that fail)`. | 20 of 128 |
| 3 | Each AND votes for one action with a fixed weight; the weights set the priority (shoot 128, turn 32, side-step or walk on 8, navigation 1). | — |

Health is not used. `MOVE_FORWARD + ATTACK` is never chosen.

## What layer 1 measures

All monster measurements are column sums of the monster channel over the whole screen height. The crosshair lies on the line between columns 19 and 20.

- **On the crosshair:** at least 2 monster pixels in columns 19–20.
- **Solidly on it:** those two columns hold at least 12% of the monster pixels within 4 cells of the crosshair. A monster's hit box is wider than its body but narrower than an outstretched gun or leg; when only such a fringe touches the crosshair the shot misses.
- **Which side to deal with:** left mass against right mass, each column's weight halving every 6 columns from the centre. Turning towards a monster makes its side win by more on the next step, so the choice cannot flip between two equal monsters. Ties go left.
- **Left monster is far out:** more mass beyond 4 cells left than within, or one monster lying across that line.
- **A much closer monster off to one side:** mass beyond 4 cells is over 4 times the mass near the crosshair.
- **Close monster:** monster pixels in rows 19 and below near the centre (legs that low mean under about 250 units).
- **Door frame beside us:** mean depth at the left or right screen edge under about 55 units.
- **Pickups:** one within 3 cells of straight ahead; one further to a side; a distant one 3 to 8 cells off; which side has more (a distant pickup counts 1,000 times a nearby dropped item, so the vest wins).
- **Way ahead:** mean depth in the middle just above the horizon, over about 200 units ("open") and over about 550 ("deep").

## What it does

1. **Shoot** when a monster is solidly on the crosshair, unless a much closer one stands far to a side.
2. **Turn** the heavier side onto the crosshair when nothing touches it.
   - Right side: turn right.
   - Left side: turn left only if the monster reaches more than 4 cells out.
3. **Otherwise** side-step towards a close monster, or walk on if it is far.
4. **No monster in view:** walk to a pickup that is ahead; keep walking if the vest is only a little to one side and the way is open; turn towards a pickup that is further off; with nothing in view, walk if the corridor is deep ahead, else turn right.

## Why these rules

Every one answers something seen in practice.

- **One shot kills, and monsters sleep until they can see the player's centre.** A monster half-hidden behind a door frame shows a sliver of sprite, cannot see the player, and can be shot for free. So the network stops and fires at anything on the crosshair, at any range. Held fire scatters a few degrees, which is what hits a sliver.
- **Turns are coarse and speed up.** A held turn is about 7° on the first step, then 12°, then 14° (2.5, 4.4 and 5 cells). With turns allowed on both sides of the crosshair, a narrow monster could flip left-right-left forever. The asymmetry in rule 2, and no turn once something touches the crosshair, rule that out. A crude closed-loop check (`aim_sim.py`) on monster shapes cut from saved frames found the loops in earlier rule sets and finds no turn loop now. (For a far sliver it reports about half the start positions as "no attack": those end in walking on, which the check does not follow further.)
- **The first pair is the same size.** With plain "turn to the heavier side" the choice flipped each step and two games were lost in the first 50 tics. The centre-weighted comparison fixed it.
- **The player is on a rail.** Saved frames showed the player can only move about 16 units either side of the centre line; the wider rooms cannot be entered and nothing in the depth picture marks the limit. Side-stepping to aim at far monsters lost two games against that invisible wall, so it is now used only against close ones.
- **Never wait.** Round 2 refused to advance while any monster was visible and timed out against sleeping monsters. Every branch now shoots, turns safely, or walks.

## Practice record

| Round | Mean | Wins | What changed |
|---|---|---|---|
| 1 | 1,968 | 4/5 | First version: shoot when centred and big, otherwise walk |
| 2 | 660 | 1/5 | Never advance while a monster is visible (wrong) |
| 3 | 1,321 | 3/5 | Shoot slivers; safe turn rules; two losses to the opening flip |
| 4 | 1,852 | 4/5 | Centre-weighted side choice; one loss side-stepping |
| 5 | 1,064 | 2/5 | Tighter side-step gate; a turn loop and an opening loss |
| 6 | 1,491 | 3/5 | Whole-height sums; "solid" test; no turn while touching |
| 7 | 1,524 | 3/5 | Looser side-step gate; one turn loop against the vest |
| 8 | 1,546 | 3/5 | Softer vest re-centring; one loss side-stepping |
| 9 | 1,571 | 3/5 | Tried keeping left of centre (this revealed the rail) |
| 10 | 1,649 | 3/5 | **Final network**: side-step only when close |

Rounds 1 and 4 scored highest, but the difference from the later rounds is within what five games can show. I kept the last version because it has no known way to get stuck.

## Known weaknesses

- **The opening costs about 25 health on average, sometimes over 40.** Both first monsters fire before the weapon has finished rising, and nothing in the rules can prevent that.
- **A far monster 1 to 4 cells left of the crosshair is walked towards, not shot.** In round 10 one such approach cost 42 health. I found no turn rule for that band that is safe from loops, and the last practice round was spent before I could test an alternative.
- **Usually only one monster of each later pair is shot while asleep.** The other wakes as the player closes in.
- **Ammunition is not visible.** Long fire at a sliver that never dies would empty the gun and stall; this did not happen in practice.

## Method note

No weight was fitted. Thresholds were set from geometry (hit box 40 units, eye height 41, 8-pixel cells) and from reading saved frames. The one number taken most directly from frames is the 12% share: I tabulated the share for about ten monster positions against whether a shot there would land, and chose a round value between the clear misses (under 3%) and the hits (over 14%).

## Files

- `build.py` writes `weights.npz` (numpy only, reads nothing, runs in under a second).
- `synth_test.py`, `aim_sim.py`, `show.py` are my checks on constructed and saved frames; the harness does not need them.
- My test scripts left a `harness_src/__pycache__` folder behind; it is safe to delete.
