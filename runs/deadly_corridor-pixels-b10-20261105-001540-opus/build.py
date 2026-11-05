"""Hand-written weights for deadly_corridor, pixels mode.

See explanation.md for the full story. Summary:

Layer 1 (feature extraction from the colour grid):
  units 0..39  monster evidence per screen column c:
               M_c = relu(K * (sum_{rows 8..15} (0.656 r - 1.63 g + b) + OFF))
               (zero on brown walls and on grey bullet puffs, positive on the
               zombies' blue / purple / red parts; rows 8..15 = at/above eye level)
  40/41        heading cue H = sum_{rows 0..8} (c - 19.5) * green  (the bright ceiling
               band is centred on the corridor's vanishing point), with deadband
  42..45       strong-misalignment flags (saturating), 46/47 low-health flag
               (saturating, disabled), 48/49 "facing a wall up close" flag
Layer 2 (decision logic):
  0 centre monster, 1/2 left/right monsters (nearer the crosshair weighs more),
  3/4 heading turns, 5 "go forward", 6 stand-and-shoot when misaligned, 7 wall turn
Output: argmax over actions.

FIGHT selects between the two policies tested in practice (see explanation.md).
"""
import numpy as np

D, H, A = 3601, 128, 7
FWD, MLEFT, MRIGHT, TLEFT, TRIGHT, ATK, FWDATK = range(7)
# FIGHT = True: turn toward monsters seen off-centre and shoot them.
# FIGHT = False: "run and gun" - always advance while firing straight ahead,
# steering only to keep facing down the corridor and away from walls.
FIGHT = False


def idx(r, c, k):
    return (r * 40 + c) * 3 + k


def build():
    W1 = np.zeros((D, H)); b1 = np.zeros(H)
    W2 = np.zeros((H, H)); b2 = np.zeros(H)
    W3 = np.zeros((H, A)); b3 = np.zeros(A)

    # ---- layer 1: monster evidence per column
    # colour direction f = 0.656 r - 1.63 g + b: ~0 on brown walls and on grey
    # bullet puffs, negative on the brighter ceiling/floor, positive on the
    # zombies' blue/purple/red parts
    K, OFF = 10.0, -0.015
    CR, CG, CB = 0.656, -1.63, 1.0
    for c in range(40):
        for r in range(8, 16):
            W1[idx(r, c, 0), c] += CR * K
            W1[idx(r, c, 1), c] += CG * K
            W1[idx(r, c, 2), c] += CB * K
        b1[c] = K * OFF

    # ---- layer 1: heading cue (scaled so 1 unit ~ 1.5 columns of misalignment)
    # deadband: ~2 columns when fighting (we turn a lot and must re-align);
    # ~6 columns (~14 deg) when running, where only a real misalignment should turn us
    HS = 0.1
    DB = 1.2 if FIGHT else 4.0
    for r in range(0, 9):
        for c in range(40):
            W1[idx(r, c, 1), 40] += HS * (c - 19.5)
            W1[idx(r, c, 1), 41] += -HS * (c - 19.5)
    b1[40] = -DB
    b1[41] = -DB
    # strong misalignment (|H| beyond ~14 columns ~ 30 degrees), saturating to 0..1:
    # (u42 - u43) for "corridor far to the right", (u44 - u45) for far left
    MIS = 9.0
    for r in range(0, 9):
        for c in range(40):
            for u, sg in ((42, 1), (43, 1), (44, -1), (45, -1)):
                W1[idx(r, c, 1), u] += sg * HS * (c - 19.5)
    b1[42] = -MIS
    b1[43] = -MIS - 1
    b1[44] = -MIS
    b1[45] = -MIS - 1
    # low health, saturating to 0..1: u46 - u47 = 1 when health < LOWHP - 0.05
    LOWHP = -1.0   # disabled: practice round 5 showed it abandons the opening fight
    W1[3600, 46] = -20.0
    b1[46] = 20.0 * LOWHP
    W1[3600, 47] = -20.0
    b1[47] = 20.0 * LOWHP - 1.0

    # facing a wall up close: the middle of the screen is as bright as the top
    # (no ceiling band) and the floor beside the gun is not visible.
    # S = mean_g(rows 8-15, cols 12-27) - mean_g(rows 0-2) - mean_g(floor beside gun) + 0.133
    # corridor views give S <= -0.015, wall views S >= -0.001; u48 - u49 saturates to 0..1.
    # Brightness here is v = green + 0.244 red, which puts the red blood spray
    # (seen when the player is hit) midway between wall and ceiling values.
    # On v: corridor views S <= -0.194, wall views S >= -0.167; threshold -0.180.
    KW = 80.0
    for u, off in ((48, 0.0), (49, -1.0)):
        for k, cw in ((1, 1.0), (0, 0.244)):
            for r in range(8, 16):
                for c in range(12, 28):
                    W1[idx(r, c, k), u] += cw * KW / 128.0
            for r in range(0, 3):
                for c in range(40):
                    W1[idx(r, c, k), u] += -cw * KW / 120.0
            for r in range(19, 24):
                for c in list(range(0, 13)) + list(range(27, 40)):
                    W1[idx(r, c, k), u] += -cw * KW / 130.0
        b1[u] = KW * 0.180 + off

    # ---- layer 2
    wc = {16: 1, 17: 1, 18: 2, 19: 3, 20: 3, 21: 2, 22: 1, 23: 1}
    for c in range(40):
        if c in wc:
            W2[c, 0] = wc[c]
        elif c < 16:
            # nearer the crosshair counts more, so the controller commits to the
            # target needing the smallest turn instead of flipping between two
            W2[c, 1] = np.exp(-(15 - c) / 6.0)
        else:
            W2[c, 2] = np.exp(-(c - 24) / 6.0)
    # low health: stop turning toward side monsters (just push on and shoot ahead)
    W2[46, 1] = W2[46, 2] = -100.0
    W2[47, 1] = W2[47, 2] = 100.0
    # centred target while facing well away from the corridor axis (an alcove):
    # shoot standing instead of walking into the wall
    for c, wgt in wc.items():
        W2[c, 6] = wgt
    W2[42, 6] = W2[44, 6] = 100.0
    W2[43, 6] = W2[45, 6] = -100.0
    b2[6] = -100.0
    # gate strength: any monster evidence shuts off the heading/forward units
    # (fight mode only; the runner keeps steering by the corridor regardless)
    G = 100.0 if FIGHT else 0.0
    for c in range(40):
        W2[c, 3] = -G
        W2[c, 4] = -G
        W2[c, 5] = -G
    W2[40, 3] = 1.0          # corridor to the right -> turn right
    W2[41, 4] = 1.0          # corridor to the left -> turn left
    b2[5] = 1.0
    # wall ahead: no walking forward; turn (left by default, the heading cue overrides)
    # (fight mode only: the runner never turns away from the corridor, and the
    # far end wall in front of the vest must not make it turn)
    WALL = 1.0 if FIGHT else 0.0
    W2[48, 5] = -100.0 * WALL
    W2[49, 5] = 100.0 * WALL
    W2[48, 7] = 1.0
    W2[49, 7] = -1.0
    # wall ahead: whatever looks like a monster is a bullet puff on the wall; ignore it
    for u in (0, 1, 2, 6):
        W2[48, u] = -100.0
        W2[49, u] = 100.0

    # ---- output
    W3[0, FWDATK] = 5.1
    W3[0, ATK] = 5.0
    W3[6, ATK] = 0.0   # "stand and shoot when misaligned": disabled, see explanation.md
    W3[1, TLEFT] = 1.0 if FIGHT else 0.0
    W3[2, TRIGHT] = 1.0 if FIGHT else 0.0
    W3[3, TRIGHT] = 10.0
    W3[4, TLEFT] = 10.0
    W3[5, FWDATK] = 0.5
    W3[7, TLEFT] = 2.0 * WALL
    b3[FWDATK] = 0.01

    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    w = build()
    np.savez("weights.npz", **{k: v.astype(np.float32) for k, v in w.items()})
