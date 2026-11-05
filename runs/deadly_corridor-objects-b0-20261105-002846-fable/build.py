"""Hand-written weights for deadly_corridor, objects mode.

Every number below is chosen by reasoning about what the unit should compute.
Nothing is fitted, searched or trained. Run:  python build.py  -> weights.npz

Input layout (harness_src/env.py): index (r*40 + c)*3 + k
    k=0 monster share, k=1 pickup share, k=2 depth (0 near .. 1 far); x[3600] = health/100.
Actions: 0 FWD, 1 MOVE_LEFT, 2 MOVE_RIGHT, 3 TURN_LEFT, 4 TURN_RIGHT, 5 ATTACK, 6 FWD+ATTACK
"""
import numpy as np

ROWS, COLS, D, H, A = 30, 40, 3601, 128, 7
MON, PICK, DEPTH = 0, 1, 2
HEALTH = 3600
FWD, MOVE_L, MOVE_R, TURN_L, TURN_R, ATT, FWD_ATT = range(7)


def idx(r, c, k):
    return (r * COLS + c) * 3 + k


# ------------------------------------------------------------------ design constants
TAU_MID, TAU_EDGE = 2.6, 3.4   # a column holds a "near" monster if its monster shares sum to more than
                   # this (a standing zombie fills ~1200/dist cells of a column: ~460 units when it is
                   # in front of the crosshair, ~350 units when it is well off to the side)
K_MON = 16.0       # monster "present" saturates at 4 pixels
K_PICK = 64.0      # pickup "present" saturates at 1 pixel
RUSH_HEALTH = 0.40  # below this health stop duelling, just run (distance is banked, death costs only 100)
RHO = 5000.0       # how hard low health switches the duelling units off
MID = range(17, 23)            # columns within ~8.5 deg of the crosshair: fine-aim (strafe) zone
# Pickups are ranked by how high on the screen they are (higher = farther; the vest is the farthest).
PICK_BANDS = [range(0, 14), [14], [15], [16], [17], [18], [19], [20, 21], range(22, 30)]
PICK_ZONES = [range(0, 17), range(17, 23), range(23, 40)]      # left, centre, right
FAR_BANDS = 3      # bands 0..2 (rows <= 15): far pickups, they outrank the "open side" rule below
LAM = 1.0 / 3.0    # each priority level is a third of the one above: a level outranks all below it
S_C, S_L, S_R = 30.0, 20.0, 19.0   # inside a band: centre > left > right
DEPTH_ROWS, DEPTH_COLS = (12, 13, 14), range(17, 23)   # rows at the horizon, columns straight ahead
DEPTH_GAIN = 50.0
DEPTH_NEAR = 0.06  # depth straight ahead below this = a wall right in front (~110 map units, if depth
                   # 1.0 is ~1800 units as I believe; the rules only need this to be roughly right)
WALL_AHEAD = 0.14  # depth straight ahead below this = a wall within ~250 units ahead
SIDE_REGIONS = {"L": (range(0, 10), range(7, 17)), "R": (range(30, 40), range(23, 33))}
SIDE_RATIO, SIDE_MARGIN = 1.4, 0.02   # a side region counts as "open" if its mean depth exceeds 1.4x
                   # the depth straight ahead (+0.02): true when looking at a side wall with the
                   # corridor running off to that side, never true when looking along the corridor
SIDE_DEEP = 0.10   # ... and only if that side is deep in absolute terms (more than ~180 units)
BIG_PICKUP = 20.0  # cells of pickup around the crosshair: only a vest-sized item < ~100 units away


def sec2(c):
    """1/cos^2 of the viewing angle of column c (90 deg field of view over 40 columns).
    The engine draws a sprite at angle a from the view axis 1/cos(a) taller and 1/cos(a) wider than
    the same sprite dead ahead, so a monster seems to grow as it slides to the screen edge. Column
    thresholds are raised by 1/cos and column masses scaled by cos^2 so that "near mass" depends on
    the true distance only and does not change while the player turns."""
    u = c - 19.5
    return 1.0 + (u / 20.0) ** 2


def near_threshold(c):
    """Column-sum threshold for "near", lowest around the crosshair and rising towards the edges.
    A monster that is near enough to make the player turn towards it therefore stays "near" (with
    ~25% margin) once it is in front: turning towards a target can only strengthen the case for it."""
    u = abs(c - 19.5)
    core = TAU_MID + (TAU_EDGE - TAU_MID) * min(1.0, max(0.0, (u - 2.5) / 8.0))
    return core * sec2(c) ** 0.5


def turn_weight(c):
    """Signed pull of a near monster in column c on the turn decision (0 inside the fine-aim zone).
    Falls off steeply (e-fold every 3 columns) so that each turn step towards a target multiplies
    its lead over a rival on the other side: the choice of target cannot flip back and forth."""
    u = c - 19.5
    if abs(u) < 3.0:
        return 0.0
    w = float(np.exp(-(abs(u) - 3.5) / 3.0))
    return w if u > 0 else -w


def build():
    W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

    # ================================================================ layer 1
    # E[c] = relu(monster share summed down column c - threshold): "a near monster stands in column c".
    E = list(range(0, 40))
    for c in range(COLS):
        for r in range(ROWS):
            W1[idx(r, c, MON), E[c]] = 1.0
        b1[E[c]] = -near_threshold(c)

    # A19, A20 = 1 when no monster pixel in the column just left / right of the crosshair, 0 when present.
    A19, A20 = 40, 41
    for unit, c in ((A19, 19), (A20, 20)):
        for r in range(ROWS):
            W1[idx(r, c, MON), unit] = -K_MON
        b1[unit] = 1.0

    # LH = relu(RUSH_HEALTH - health): positive only when health is low.
    LH = 42
    W1[HEALTH, LH] = -1.0
    b1[LH] = RUSH_HEALTH

    # Pickup absence units: 1 when no pickup pixel in that (row band, zone), 0 when any.
    n = 43
    PA = {}
    for j, rows in enumerate(PICK_BANDS):
        for z, cols in enumerate(PICK_ZONES):
            for r in rows:
                for c in cols:
                    W1[idx(r, c, PICK), n] = -K_PICK
            b1[n] = 1.0
            PA[(j, z)] = n
            n += 1

    # Any monster, however far, in the middle of the view (columns 12..27): 1 when absent.
    MA = n
    n += 1
    for r in range(ROWS):
        for c in range(12, 28):
            W1[idx(r, c, MON), MA] = -K_MON
    b1[MA] = 1.0

    # Depth straight ahead d (mean of 18 cells at the horizon): D1 - D2 = clip(50*(d - 0.06), 0, 1).
    cells = [(r, c) for r in DEPTH_ROWS for c in DEPTH_COLS]
    D1, D2 = n, n + 1
    n += 2
    for unit, off in ((D1, 0.0), (D2, 1.0)):
        for r, c in cells:
            W1[idx(r, c, DEPTH), unit] = DEPTH_GAIN / len(cells)
        b1[unit] = -DEPTH_GAIN * DEPTH_NEAR - off
    # DG = relu(d - 0.14): positive when there is no wall within ~250 units ahead.
    DG = n
    n += 1
    for r, c in cells:
        W1[idx(r, c, DEPTH), DG] = 1.0 / len(cells)
    b1[DG] = -WALL_AHEAD

    # Is a region to one side much deeper than what is straight ahead?  For each side, two overlapping
    # regions (outer and inner, together covering everything but the centre), each as a pair:
    #   G1 - G2 = clip(50 * (mean_region - 1.4 * d - 0.02), 0, 1)
    # Looking along the corridor the centre is the deepest part of the picture (never fires); looking
    # at a side wall or into an alcove, the side that looks along the corridor is far deeper.
    # ... and is that region deep in absolute terms (mean depth > 0.10, ~180 units)?  Second pair:
    #   G3 - G4 = clip(50 * (mean_region - 0.10), 0, 1)
    # Both must hold (layer 2 takes the AND), so that once the player has turned to face the open
    # side there really is open space ahead and the "walk if open" rule takes over.
    G = {"L": [], "R": []}
    for side, regions in SIDE_REGIONS.items():
        for cols in regions:
            quad = []
            for ratio, margin in ((SIDE_RATIO, SIDE_MARGIN), (0.0, SIDE_DEEP)):
                for off in (0.0, 1.0):
                    for r in DEPTH_ROWS:
                        for c in cols:
                            W1[idx(r, c, DEPTH), n] += DEPTH_GAIN / (len(cols) * len(DEPTH_ROWS))
                        for c in DEPTH_COLS:
                            W1[idx(r, c, DEPTH), n] += -ratio * DEPTH_GAIN / len(cells)
                    b1[n] = -DEPTH_GAIN * margin - off
                    quad.append(n)
                    n += 1
            G[side].append(quad)

    # A big pickup right in front: B1 - B2 = clip((pickup cells in columns 14..25 - 20) / 5, 0, 1).
    B1, B2 = n, n + 1
    n += 2
    for unit, off in ((B1, 0.0), (B2, 1.0)):
        for r in range(ROWS):
            for c in range(14, 26):
                W1[idx(r, c, PICK), unit] = 1.0 / 5.0
        b1[unit] = -BIG_PICKUP / 5.0 - off
    assert n <= H

    # ================================================================ layer 2
    NE, TR, TL, SR, SL, COV, CN = 0, 1, 2, 3, 4, 5, 6
    for c in range(COLS):
        W2[E[c], NE] = 1.0 / sec2(c)            # NE: total near-monster mass in view
        s = turn_weight(c) / sec2(c)
        W2[E[c], TR] = s                        # TR: near mass right of the aim zone minus left
        W2[E[c], TL] = -s                       # TL: the mirror image
    for c in MID:                               # SR/SL: near monster just right / left of the crosshair
        sgn = 1.0 if c >= 20 else -1.0
        W2[E[c], SR] = sgn
        W2[E[c], SL] = -sgn
    W2[A19, COV] = -1.0; W2[A20, COV] = -1.0; b2[COV] = 1.0   # COV: monster on both sides of the crosshair
    W2[E[19], CN] = 1.0; W2[E[20], CN] = 1.0                  # CN: ... and that monster is near
    W2[A19, CN] = -100.0; W2[A20, CN] = -100.0
    for unit in (NE, TR, TL, SR, SL, CN):       # low health: no duelling, run
        W2[LH, unit] = -RHO

    m = 7
    PP = {}
    for key, a in PA.items():                   # presence = 1 - absence (exactly 0 when nothing is there)
        W2[a, m] = -1.0; b2[m] = 1.0
        if key[0] >= FAR_BANDS and key[1] != 1:
            W2[DG, m] = -50.0                   # a nearer pickup off to one side only counts when a
        PP[key] = m                             # wall is close ahead (the end of the corridor)
        m += 1
    MONC, OPEN, BIG = m, m + 1, m + 2
    m += 3
    W2[MA, MONC] = -1.0; b2[MONC] = 1.0         # MONC: some monster in the middle of the view
    W2[D1, OPEN] = 1.0; W2[D2, OPEN] = -1.0     # OPEN: no wall right in front (0..1)
    GS = {"L": [], "R": []}                     # the open corridor is off to the left / right:
    for side in ("L", "R"):                     # one unit per region = (much deeper than ahead) AND (deep)
        for g1, g2, g3, g4 in G[side]:
            W2[g1, m] = 1.0; W2[g2, m] = -1.0
            W2[g3, m] = 1.0; W2[g4, m] = -1.0
            b2[m] = -1.0
            GS[side].append(m)
            m += 1
    W2[B1, BIG] = 1.0; W2[B2, BIG] = -1.0       # BIG: a big pickup dead ahead (0..1)
    assert m <= H

    # ================================================================ output
    # Navigation, by priority (each level a third of the one above, so a level outranks all below):
    #   1 far pickups (bands 0..2), farthest band first: walk at it if in the centre columns, else
    #     turn to it. The vest is always the farthest pickup.
    #   2 the open corridor is off to one side (and nothing far is in view): turn that way
    #   3 nearer pickups (bands 3..8): walk onto it if straight ahead; turn to it only if a wall is
    #     close ahead - true for the vest at the end wall, where rule 2 is silent
    #   4 a monster somewhere in the middle of the view: walk (they stand between player and vest)
    #   5 no wall right in front: walk
    #   6 otherwise: turn right
    zone_action = {0: (TURN_L, S_L), 1: (FWD, S_C), 2: (TURN_R, S_R)}
    for (j, z), unit in PP.items():
        act, s = zone_action[z]
        W3[unit, act] = s * LAM ** (j if j < FAR_BANDS else j + 1)
    for unit in GS["L"]:
        W3[unit, TURN_L] = 19.5 * LAM ** FAR_BANDS
    for unit in GS["R"]:
        W3[unit, TURN_R] = 19.5 * LAM ** FAR_BANDS
    lvl = len(PICK_BANDS) + 1
    W3[MONC, FWD] = S_C * LAM ** lvl
    W3[OPEN, FWD] = S_C * LAM ** (lvl + 1)
    b3[TURN_R] = 3e-6
    W3[BIG, FWD] = 40.0                         # a big pickup dead ahead is the vest: just walk onto it

    # Duelling (all of it is zero when no near monster is in view, or health is low).
    W3[NE, FWD] = -500.0                        # do not walk on while a near monster is alive in view
    W3[NE, FWD_ATT] = -500.0
    W3[TR, TURN_R] = 30000.0                    # face the near monster closest to the crosshair
    W3[TL, TURN_L] = 30000.0
    W3[SR, MOVE_R] = 120000.0                   # nearly on target: sidestep until the crosshair is on it
    W3[SL, MOVE_L] = 120000.0
    b3[MOVE_L] = -1.0; b3[MOVE_R] = -1.0
    W3[COV, ATT] = 100.0                        # something under the crosshair: shoot ...
    W3[COV, FWD_ATT] = 100.0                    # ... on the move if it is far,
    W3[CN, ATT] = 1.0e6                         # standing still if it is near
    b3[ATT] = -10.0

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


if __name__ == "__main__":
    w = build()
    np.savez("weights.npz", **{k: v.astype(np.float32) for k, v in w.items()})
    print({k: v.shape for k, v in w.items()})
