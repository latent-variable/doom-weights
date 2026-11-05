# A hand-written network for `basic` (pixels)

## What the network sees, and the one colour trick

On the first practice screen the room's colours were: ceiling a neutral gray (R = G = B), walls and floor
brown (R > G > B, with G sitting almost exactly halfway between R and B), and the monster in this Freedoom
build red and blue/magenta. The linear feature

    f = R + B - 2G

is therefore about 0 (|f| ≤ 0.01) on every background cell and 0.1–0.2 on monster cells. It's a "distance
from the gray/brown line" detector that one linear layer can compute exactly. Rows 20–29 (gun and hand) are
left out.

## Layer 1: a per-column monster profile (40 of 128 units)

    m_c = relu( sum_{r=0..19} f(r, c) - 0.10 ),   c = 0..39

Background column sums never went above 0.09 on any practice frame (the corner seams of the room are the
worst case), so the 0.10 bias zeroes them. Monster columns sum to 0.3–0.55. The monster covers about 3–4
columns and gives a total mass of about 0.7 after the threshold. Since it always sits at the far wall, its
size on screen barely changes.

## Layer 2: where is it relative to the crosshair (4 units)

The crosshair is at x = 160, between columns 19 and 20. With S = sum_c m_c (c − 19.5), the mass-weighted
offset:

- P = relu(1000 · (S − 1.5·0.7)): positive only if the monster's centroid is more than about 1.5 columns
  to the right.
- N = relu(1000 · (−S − 1.5·0.7)): the same for the left.
- G1 − G2 = min(Σm/0.1, 1): a saturating "a monster is visible" gate.

I used a constant threshold (1.5 columns × typical mass) on purpose, not one that scales with Σm. The
pistol's muzzle flash is orange and also scores on f. It sits symmetrically around the crosshair, so it adds
nothing to S, but it would have widened a mass-proportional dead zone.

## Output

    q_left = N,  q_right = P,  q_attack = (G1 − G2) − 0.5 + 0.001

- Monster outside the ±1.5-column dead zone: the strafe score is in the hundreds or thousands, so the player
  strafes toward it.
- Monster inside the dead zone: both strafe scores are 0 and attack wins at 0.501.
- No monster visible: attack scores −0.5, so the player strafes left rather than wasting ammo.

The dead zone is 3 columns wide, wider than the roughly 1.1 columns the view moves per step at full strafe
speed. So the policy can't jump over it and oscillate.

## Why ±1.5 columns

The player keeps sliding after it stops strafing, and the pistol's bullet leaves a few tics after ATTACK is
pressed. In practice the monster drifted about 1 more column toward the centre after the attack started.
Triggering at 1.5 columns while moving means the shot lands near the centre. Triggering from standing still
at spawn means at most 12 px off-centre, inside the monster's roughly ±14 px body. I tried ±2.0 for one round
and it also hit everything, but at rest a 16 px offset is too close to the edge, and a miss costs about 20
points (−5 for the shot, plus about 14 tics waiting to refire).

I checked the logic on synthetic frames: I moved the real monster patch across every column, and the network
chose LEFT for centres at columns 3–18, ATTACK at 19–21 and RIGHT at 22–37.

## Practice results

| Round | Change | Scores |
|---|---|---|
| 1 | Bad first guess (R − (G+B)/2 picks up the brown room) | Just shot constantly |
| 2 | f = R+B−2G, ±1 column dead zone | 74.2, 5/5 kills, all one shot |
| 3 | Threshold 0.10, ±1.5 column dead zone (**final**) | 76.6, 5/5 kills, all one shot |
| 4 | ±2.0 column dead zone | 72.6, 5/5 one-shot kills (different seeds, so this is noise) |

Every episode is a single shot. The score is 106 − 5 − tics, and the tics are almost entirely the time
needed to strafe across to the monster, which the policy does at full speed in a straight line.

## Prediction

Average held-out score of about **75** (likely range 70–80). The best possible is about 85–90 given the
strafe time, and spawns near the centre score around 91.

Confidence: fairly high (about 80%) that the mean is between 65 and 85. The main risks:

- an occasional miss (roughly −20 each);
- a monster spawning so close to a side wall that it is partly off-screen or blends into the corner seams.
  There is no sign of this in 20 practice games.
