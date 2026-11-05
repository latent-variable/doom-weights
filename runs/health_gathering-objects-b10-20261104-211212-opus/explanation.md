# Hand-written health_gathering policy (objects mode)

All weights are set by hand in `build.py`. Nothing was trained, fitted or searched.

## Layer 1: what is on screen (43 of 128 units used)
- **Units 0–39: pickup mass per column.** Unit *c* adds up the pickup channel (channel 1) over all 30 rows of column *c*. Each row gets weight `0.5 + r/30`, so lower rows count more. Lower on screen means closer on the floor, and a closer medkit also covers more pixels, so the nearest medkit dominates.
- **Units 40 / 41: near-side pickups.** These add up pickup mass in the bottom rows (24–29) of the 12 left-most and the 12 right-most columns. A medkit there is close but well off-centre. Moving forward while turning would circle past it, so the network turns in place instead.
- **Unit 42: wall ahead.** `relu(0.12 − mean depth)` over the centre patch (rows 13–17, columns 14–25), which is the horizon straight ahead. It is positive only when a wall is close in front.

## Layer 2: grouping (6 units, identity-like)
These units sum the columns into left (0–16), centre (17–22) and right (23–39) pickup mass. They pass the near-left and near-right units through unchanged, and scale the wall unit to the range 0–1.

## Output
- Left mass → FORWARD+LEFT, centre mass → FORWARD (gain ×1.3), right mass → FORWARD+RIGHT. The gain is 100 per unit of mass.
- Near-left → TURN_LEFT, near-right → TURN_RIGHT (gain 200).
- Wall closeness → TURN_LEFT (gain 5). Any visible medkit (mass above about 0.05) outweighs it, so the network still goes for medkits that are next to a wall.
- Bias of 0.5 on FORWARD+LEFT. With nothing in view, the agent walks a wide left-hand circle to sweep the room, and turns in place when a wall is close.

The policy keeps walking toward the biggest and closest medkit cluster. Medkits respawn, so it collects health faster than the acid drains it.

## Practice evidence
Round 1 (the only round used): all 5 games scored **2100**, the maximum. Every game survived all 2,100 tics and ended at **100 health**. Actions were spread across all five, and the network spent most steps moving forward toward medkits.

## Prediction
The expected average on held-out games is **about 2050–2100**, and my point estimate is **~2090**. Every practice game ended at full health, so there is a lot of margin. The main risk is a rare game where the agent gets stuck against a wall or a corner with no medkit in view. I'm fairly confident (~80%) that the mean will be above 2000.
