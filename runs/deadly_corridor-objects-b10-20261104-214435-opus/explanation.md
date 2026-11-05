# A hand-written Doom policy for `deadly_corridor` (objects mode)

The network is a priority list of rules: shoot whatever is on the crosshair, otherwise turn
toward the monster nearest the crosshair, otherwise head for the vest, otherwise walk forward.
Every weight was set by hand in `build.py`. There was no fitting of any kind.

## Layer 1: 12 linear image statistics (units 0–11; the other 116 are zero)

All are weighted sums over the 30×40 grid. `o = c − 19.5` is a cell's column offset from
the crosshair, which sits between columns 19 and 20.

| unit | name | weights | meaning |
|---|---|---|---|
| 0 | xhair | monster channel, columns 18–21, weight 1 | monster on the crosshair strip |
| 1 | xwide | monster channel, columns 15–24 | monster roughly ahead |
| 2 | mtotal | monster channel, all cells | how much monster is on screen (≈ how close) |
| 3, 4 | far± | monster channel, weight `±sign(o)·(22−|o|)` | which side the monster *nearest the crosshair* is on |
| 5, 6 | vest± | pickup channel, rows 13–16, 3.5 < |o| ≤ 10.5, weight `±sign(o)` | distant vest off-centre left/right |
| 7 | vestmass | pickup channel, rows 13–16, |o| ≤ 10.5 | vest visible near the centre of the horizon |
| 8, 9 | depth± | depth channel, rows 10–16, weight `±o/19.5/7` | which side of the view is deeper (corridor direction) |
| 10 | open | depth channel, rows 8–21, columns 12–27 | total depth ahead; near 0 means we are facing a wall |

Two signed features are each split into a `relu(+x)` / `relu(−x)` pair: the monster moment
(units 3, 4) and the depth asymmetry (units 8, 9). The vest pair (units 5, 6) is a pair of
mirror-image features that are each clipped at 0 by the ReLU.

Why the monster weight is largest *near* the centre: a plain centroid sits between the two
monsters you see at the start (one in each alcove), and the policy flipped between them
forever. That happened in practice round 1. With weight `22−|o|`, the monster closest to
the crosshair dominates. Turning toward it moves it closer to the centre and pushes the
other one further out, so the choice locks in.

Why the vest window is only the central horizon band: dead zombies drop clips and
shotguns, which also show up in the pickup channel. They lie on the floor nearby (lower
rows) or in the alcoves (screen edges). Steering toward them caused oscillation (round 4).

## Layer 2: near-binary gates

Each rule is a gate `clip(k·(f − thr), 0, 1)`. It takes two layer-2 units,
`relu(k(f−thr))` and `relu(k(f−thr) − 1)`, and layer 3 subtracts the second from the first.
That uses 18 units for 9 gates:

- on-target: xhair > 0.01
- ahead: xwide > 0.01
- small target: mtotal < 2
- monster right / left: far± > 1
- vest right / left: vest± > 0.005
- vest visible: vestmass > 0.005
- deeper right / left: depth± > 0.4
- blocked: open < 6

## Layer 3: priorities (action scores)

- MOVE_FORWARD: base score 1.
- ATTACK: `40·on-target`. This outranks all the turn rules put together (12+8+4+3).
- MOVE_FORWARD+ATTACK: `40·ahead + 40·small − 39.5`. It wins only when a *small (distant)*
  monster is roughly ahead.
  - Reason: a 4-tic turn moves the view about 3–6 columns, but a far monster is about one
    column wide. Turning toward it overshot back and forth for the whole episode (rounds 2
    and 5 timed out at 2,100 tics).
  - Advancing while firing makes the target bigger and resets Doom's turn acceleration, so
    the aim converges. It also earns distance reward.
- TURN toward the monster nearest the crosshair: 12.
- TURN toward the vest when it is visible but more than 3.5 columns off-centre: 8.
- Vest visible: −4 on both turns. This cancels the depth and blocked rules (4 and 3) while
  the vest is in view, so a centred vest means walking straight at it.
- TURN toward the deeper side of the view (corridor heading after a side fight): 4.
- Facing a wall (all depth ahead near 0, e.g. after walking into an alcove): turn right, 3.
  Round 2 had a game stuck against a wall for 1,700 tics.

`tools/synth.py` builds synthetic frames and checks the network gives the intended action in
each case. All 17 cases pass: empty corridor → forward; corridor end left → turn left; wall
→ turn right; big centred monster → attack; tiny monster at column 17 → forward+attack;
two monsters → the one nearer the centre; and so on.

## Practice history (5 games per round, held-out games never seen)

| round | change | mean |
|---|---|---|
| 1 | centroid aim, depth heading | 842 |
| 2 | nearest-to-centre aim | 1023 |
| 3 | strafe for fine aim | −116 (strafing pinned the player against the wall; removed) |
| 4 | vest steering, blocked rule | −56 (chased dropped items) |
| 5 | vest restricted to the horizon band | 690 (oscillation on tiny far monsters) |
| 6 | wider and more sensitive crosshair strip, advance-and-shoot on small targets | 1452 |
| 7 | attack priority fix, vest window limited to the centre, wide band for far targets | 1116 |
| 8 | walk over dropped weapons | 158 (oscillated around wide items; reverted) |
| 9 | **final = round-7 network, unchanged** | 1175 |

Over 10 games, the final network scored a mean of about 1,145. Games come out two ways:

- About 40% end at the vest, with 6 kills and a score of about 2,280.
- The rest die after 2–4 kills, scoring about 150–600 from the distance covered first.

There were no timeouts in the 10 games with the final network.

## Prediction

The expected held-out average is about **1,100**. I'd put a 70% interval at roughly
**750–1,450**.

- The score is almost entirely the probability of reaching the vest (about 2,280 vs about
  300).
- 10 practice games put that probability at about 0.4, with wide uncertainty (roughly
  0.2–0.6).

Known weaknesses:

- Aim is coarse because of the 4-tic turn quantum.
- On the hardest skill the pistol often loses the race against two shooters at once.
- There is no attempt to dodge.
