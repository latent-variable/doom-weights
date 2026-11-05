# A hand-written Doom policy for `deadly_corridor` (objects mode)

The network is a fixed priority list of rules, written as weights. `build.py` defines
13 linear features of the input. Layer 1 turns each one into a clamped step. Layer 2
finishes the steps and builds a few ANDs. Layer 3 adds the steps into action scores, and
the score weights are spaced so that whenever a higher-priority rule fires, it wins the
argmax.

## Layer 1: two ramps per feature
For each feature `f` (a weighted sum over grid cells or the health variable) with threshold
`t` and width `s`, there are two units: `relu((f-t)/s)` and `relu((f-t)/s - 1)`.
That makes 26 units. The other 102 are zero.

"Upper" monster pixels below means monster pixels in rows 0..12, i.e. above y = 104.

| feature | what it measures |
|---|---|
| ON | upper monster pixels in the crosshair columns 19-20 (about ±3°) |
| FINE_L / FINE_R | upper monster mass in columns 17-18, minus the mass in columns 21-22, minus the centre mass (and the mirror image): the target touches the crosshair but its middle is off to one side |
| STR_L / STR_R | upper mass in columns 15-18 versus 21-24: a target close to the crosshair |
| OUT_L / OUT_R | upper mass in columns 0-14 versus 25-39: a target far off-centre |
| LO_C | monster pixels in rows 13-14 near the centre |
| NAV_L / NAV_R | depth balance: Σ_c (w_c ∓ 0.1)·depth_c over rows 9-17, with w_c = (19.5-c)/19.5. This fires when the left/right depth imbalance is more than 10 % of the total depth. Because it is a ratio, it does not depend on the unknown depth scale. |
| PICK_L / PICK_R | pickup pixels near the horizon (rows 11-15, so only far items such as the vest) on the left versus the right |
| LOW | health < 30 |

**Why rows 0..12 for live monsters.** In the labels buffer a corpse is probably still a
`Zombieman`, so the network has to tell corpses from standing monsters itself. The eye is
41 units up and a standing zombie is 56 tall, so a live monster always reaches above the
horizon line. A corpse is about 15 units tall and never does. I assume the Doom status bar
is drawn (`render_hud = true`). The 3-D view is then 208 px high, so the horizon is at
y = 104, which is exactly the boundary between rows 12 and 13. As a result, "upper" pixels
are live monsters and never corpses. If this rule were broken the policy could deadlock:
it would shoot a corpse forever and never move.

## Layer 2: steps and ANDs
- `step_i = relu(rampA_i - rampB_i)` = clamp((f-t)/s, 0, 1), one per feature (13 units).
- `relu(LOW + X - 1)` for X in {NAV_L, NAV_R, PICK_L, PICK_R}: "low health AND heading off" (4 units).

## Layer 3: the priority list
These are the scores per step. The bias on FORWARD is 1.

| priority | condition | action | weight |
|---|---|---|---|
| 1 | low health and off the corridor axis / vest off to one side | TURN toward the corridor / vest | 70 |
| 2 | low health | MOVE_FORWARD+ATTACK (run for the vest, shooting) | 50 |
| 3 | FINE | strafe toward the target's centre | 25 |
| 4 | ON | ATTACK | 20 |
| 5 | STR | strafe toward the target | 10 |
| 6 | OUT | turn toward the heavier side | 5 |
| 7 | PICK | turn toward the vest | 2 |
| 8 | NAV | turn back to the corridor axis | 1.5 |
| 9 | LO_C | MOVE_FORWARD+ATTACK | 1.2 |
| 10 | otherwise | MOVE_FORWARD | 1 |

Each weight is larger than the sum of all the lower rules that could stack on the same
action, so the order is strict.

**Why these choices.**
- **Turn rate.** Doom's keyboard turn starts slow: about 1.8°/tic for the first 6 held tics,
  then about 3.5°/tic. One 4-tic step is 7-14°, and the counter keeps running when you
  switch between left and right. Coarse aim therefore uses turning only outside a ±14°
  dead zone, which is wider than the largest step, so it cannot oscillate.
- **Fine aim.** Close to the crosshair the policy strafes instead. Strafing changes the
  bearing of a target 500 units away by only about 3° per step. Accuracy matters because
  held fire already spreads by about ±5.6°.
- **Getting back on course.** With no memory, the network finds the corridor direction
  again after a fight from the depth balance. The far end of the corridor is the deep side.
  In my toy model of the corridor this points the right way for headings up to about 70°
  and also steers the player away from the side walls.
- **Low health.** The reward is mostly distance, the death penalty is only -100, and the
  vest is worth +1000. Once health is low, running for the vest is worth more than fighting.

## Checks I ran
These are all on inputs I constructed myself; the network never played. The scripts are in
`tools/`. There is a toy ray-cast corridor for depth and rectangles of the right
angular size for monsters, corpses and the vest.
- **Corpses.** A standing monster at the centre at 150, 400 or 1000 units → ATTACK. A
  corpse at the centre → walk on.
- **Aiming.** A closed loop using the Doom turn model, starting from a target at ±9° to
  ±40° and 300 to 1000 units away, always ends in ATTACK with the target within about 3°
  of the crosshair, after 2-9 steps.
- **Navigation.** Turned ±15° or more → turns back. Within ±6° → goes forward. Next to a
  wall → turns away from it. A dropped clip on the floor nearby is ignored. The vest off
  to one side → turns toward it.

## Prediction
**Predicted average score: about +250, with low confidence.** I would put an 80 % interval at
roughly -50 to +800.

The key uncertainties:
- **The status bar.** If there is no status bar (horizon at y = 120), the upper-rows rule
  only sees monsters closer than about 150 units. The policy then degrades to "walk forward,
  shooting at whatever is centred", which probably scores about +100 to +300.
- **The real map and timing.** The map geometry, the depth scale, and how fast nightmare
  zombies wear 100 HP down are all unmodelled. Some episodes should reach the vest
  (+1000 plus about 1300 of distance), and many will die partway (a few hundred minus 100).
