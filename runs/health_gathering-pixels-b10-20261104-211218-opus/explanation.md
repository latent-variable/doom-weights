# Hand-written medkit-seeking network for `health_gathering` (pixels)

## What it sees
Three things are on screen: a dark-blue sky, brownish-grey walls, a dark-green acid floor, and the
medkits, which are grey/white boxes with a green cross. I measured the cell colours by hand in the
practice inputs:

| surface | R : G : B (roughly) |
|---|---|
| floor | 1 : 2.1 : 0.6 (dark, G dominant) |
| wall (near or far) | 1 : 1.03 : 0.75 |
| medkit (grey body) | 1 : 1 : 1, bright (~0.6) |
| sky | 0 : 0 : 0.1, but only above row 12 |

I solved three equations to get one colour vector `f = -4.8 R + 0.9 G + 4.9 B`. It gives 0 for floor
colour, -0.2 per unit brightness for wall colour, and +1 per unit brightness for grey. A medkit cell
scores about +0.3 to +0.6, floor cells about 0 ± 0.05, and wall cells -0.05 to -0.5. The wall had to be
negative: my first try had walls come out slightly positive, and a close-up wall then looked like a huge
medkit. In practice round 2 the agent pressed itself into a wall for the rest of the game.

## Layer 1: medkit evidence per column (units 0–39)
Unit `c` sums `f` over rows 13–29 of column `c`. Those are the rows where floor (and so medkits) can
appear, and the sky never reaches them. Lower rows get slightly more weight (×1.0 up to ×1.6) because
they hold nearer medkits. The bias is -0.3, so floor noise and dark walls give 0, and any visible
medkit in that column gives a positive value. Units 40–127 are unused.

## Layer 2: where the medkit mass is
- unit L = sum of columns 0–16 (left of centre)
- unit C = sum of columns 17–22 (about ±7° around straight ahead)
- unit R = sum of columns 23–39

## Layer 3: actions
- MOVE_FORWARD = 1.3·C + 0.005
- MOVE_FORWARD+TURN_LEFT = L + 0.010
- MOVE_FORWARD+TURN_RIGHT = R

So the agent always moves forward. It turns toward whichever side holds more medkit evidence and goes
straight when most of it is ahead. If nothing is visible, every score is 0 and the bias picks
forward+left. That walks a wide circle and slides along walls, so new medkits soon come into view.
Walls give 0 evidence, so a wall alone never attracts the agent.

## Practice evidence
| round | network | mean score |
|---|---|---|
| 1 | constant forward+left | 348 |
| 2 | first colour feature (walls slightly positive) | 463 (stuck on walls) |
| 3 | current network | **2100 in all 5 games** (the maximum), health 92–100 at the end |

I checked the final weights on the saved practice inputs. Far-away medkits at the horizon (rows 15–16
next to the wall base) are sometimes below threshold. The wandering left circle covers that case, and
nearby medkits are always detected.

## Prediction
Held-out average score: **about 2050** (most games reach the 2,100-tic limit). It could drop if some
seeds leave the agent in a medkit-poor corner for a long time. I'm fairly confident (~75%) that the
mean will be at least 1,900, and about 90% confident it will be at least 1,500.
