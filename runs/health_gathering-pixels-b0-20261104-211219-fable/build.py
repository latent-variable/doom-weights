"""Hand-written weights for ViZDoom `health_gathering`, `pixels` mode.

No learning: every number below is chosen by reasoning about screen geometry
and the Doom palette.  See explanation.md.

    python build.py   ->   weights.npz  (W1, b1, W2, b2, W3, b3; float32)
"""
import numpy as np

ROWS, COLS, HID, NACT = 30, 40, 128, 5
D = ROWS * COLS * 3 + 1
HORIZON = 15                     # first grid row below the horizon (y = 120)

# Action indices
TL, TR, FWD, FL, FR = 0, 1, 2, 3, 4

# ---------------------------------------------------------------- colour forms
# (weights on R, G, B of one cell)
# "non-green": zero on the Doom palette's green ramp (R:G:B ~ 0.47:1:0.40 at
# every brightness), strongly positive on white / grey / red / brown.
NG = (1.0, -0.66, 0.5)
NG_LAT = (1.0, -0.64, 0.5)       # a hair conservative for the lateral filters
# "green": positive on the acid floor, ~0 on grey / white / brown.
GR = (-0.5, 1.0, -0.5)
# sign form: positive on floor, negative on anything grey / white / brown / tan;
# little red in it so the red pain flash barely moves it.
FLOORSIGN = (-0.25, 0.8, -0.9)

THETA0, TAU = 0.03, 0.10         # detector threshold = THETA0 + TAU * mean G of rows 20-29

# rho = rows below the horizon (grid row = 15 + rho).  A medkit d units away has
# its foot at rho = 984/d and its top at rho = 528/d and is 560/d columns wide.

# "Guarded" detectors: non-green in a window, green in the row directly above it.
#   (guard rho, first rho, last rho, width, stride, vote weight,
#    half-width of the "go straight" zone in columns, pure-turn-left zone or None,
#    guard weight relative to the window)
GUARDED = [
    (2, 3, 5, 2, 2, 35.0, 3.0, None, 1.4),     # d ~  90-180
    (3, 4, 7, 4, 2, 30.0, 4.0, 14.0, 1.25),    # d ~  65-130
    (5, 6, 11, 4, 4, 100.0, 6.0, 13.0, 1.25),  # d ~  45-90
    (8, 9, 14, 8, 4, 80.0, 8.0, 14.0, 1.25),   # d ~  35-60
]

# "Bump" detectors: non-green in a block, minus the same rows a few columns to
# the left, minus the same rows a few columns to the right.  Never positive on a
# wall's straight base line, whatever its slope; positive on a medkit hanging
# below that line or standing alone on the floor.
#   (first rho, last rho, block width, stride, gap to each flank, flank width,
#    vote weight, straight-zone half-width, pure-turn-left zone or None, gain)
# gain 2 on the farthest band: a medkit 500+ away fills well under one cell.
BUMPS = [
    (1, 2, 2, 2, 1, 2, 20.0, 1.5, None, 2.0),  # foot at d ~ 330-700
    (3, 4, 2, 2, 1, 2, 25.0, 2.0, None, 1.0),  # d ~ 200-330
    (5, 7, 3, 3, 2, 2, 40.0, 3.5, 14.0, 1.0),  # d ~ 120-200
    (8, 11, 4, 4, 2, 2, 60.0, 5.0, 13.0, 1.0), # d ~  80-120
]
VOTE_MARGIN = 0.3                # dead zone on each summed vote

# Hard wall signal: the floor right in front of the feet is not floor-coloured.
HARD_ROWS = range(HORIZON + 10, HORIZON + 15)  # rho 10..14
HARD_COLS = range(8, 32)
HARD_ABS = 0.02
WALL_GAIN = 1.0e5

# Soft wall signal for exploring: mid-distance rows, centre+right vs left reference.
SOFT_ROWS = range(HORIZON + 4, HORIZON + 7)    # rho 4..6
SOFT_COLS = range(14, 40)
SOFT_REF_COLS = range(0, 10)
SOFT_ABS, SOFT_REL, SOFT_GAIN = 0.02, 0.6, 100.0

# Health levels (fractions of 100) at which an idle agent turns to look around.
SCAN_BANDS = [(0.625, 0.725), (0.325, 0.425), (0.075, 0.175)]
SCAN_K = 200.0
EXPLORE_TL = 0.05                # idle turn-left bonus; F_BIAS < this < any real medkit vote
F_BIAS = 0.02


def idx(r, c, k):
    return (r * COLS + c) * 3 + k


def add(W1, unit, r, c, form, w):
    for k in range(3):
        W1[idx(r, c, k), unit] += w * form[k]


def vote_for(x, half, tl_zone):
    """Which action a detector centred x columns right of screen centre votes for."""
    if abs(x) <= half:
        return FWD
    if x < 0:
        return TL if (tl_zone is not None and x <= -tl_zone) else FL
    return FR                     # never a pure right turn: see explanation.md


def bump_columns(c0, width, gap, fw):
    left = [c for c in range(c0 - gap - fw, c0 - gap) if c >= 0]
    right = [c for c in range(c0 + width + gap, c0 + width + gap + fw) if c < COLS]
    return left, right


def build():
    W1 = np.zeros((D, HID)); b1 = np.zeros(HID)
    W2 = np.zeros((HID, HID)); b2 = np.zeros(HID)
    W3 = np.zeros((HID, NACT)); b3 = np.zeros(NACT)

    # layer-2 units: one summed vote per action, the wall signals, the scan signal
    H2 = {FWD: 0, FL: 1, FR: 2, TL: 3}
    H2_WALL, H2_SCAN, H2_SOFT = 4, 5, 6   # H2_SOFT uses two units (a clip)

    def wire_votes(members, k):
        """Left-vs-right opponent votes for one group of detectors.

        h2[FL] += k (L - R), h2[FR] += k (R - L), h2[FWD] += k (C - mu (L + R)),
        h2[TL] += k (1.5 T - 1.5 R - 0.5 (L - T))   [T = pure-turn part of L]
        so anything that lights every column alike (a stripe along a wall) cancels.
        """
        n = {a: sum(1 for _, v in members if v == a) for a in (FWD, FL, FR, TL)}
        n_side = n[FL] + n[FR] + n[TL]
        mu = min(n[FWD] / n_side, 0.4) if n_side else 0.0
        for unit, v in members:
            if v == FWD:
                W2[unit, H2[FWD]] += k
            elif v == FR:
                W2[unit, H2[FR]] += k
                W2[unit, H2[FL]] -= k
                W2[unit, H2[TL]] -= 1.5 * k
                W2[unit, H2[FWD]] -= mu * k
            else:                                   # FL or TL: both are "left"
                W2[unit, H2[FL]] += k
                W2[unit, H2[FR]] -= k
                W2[unit, H2[TL]] += 1.5 * k if v == TL else -0.5 * k
                W2[unit, H2[FWD]] -= mu * k

    def threshold(unit):
        b1[unit] -= THETA0
        rows = range(20, 30)
        for r in rows:
            for c in range(COLS):
                W1[idx(r, c, 1), unit] -= TAU / (len(rows) * COLS)

    u = 0
    # ---- guarded detectors -------------------------------------------------
    for g, t0, t1, width, stride, kvote, half, tlz, gamma in GUARDED:
        members = []
        trows = [HORIZON + p for p in range(t0, t1 + 1)]
        for c0 in range(0, COLS - width + 1, stride):
            cols = range(c0, c0 + width)
            n = len(trows) * width
            for r in trows:
                for c in cols:
                    add(W1, u, r, c, NG, 1.0 / n)
            gcols = [c for c in range(c0 - 1, c0 + width + 1) if 0 <= c < COLS]
            for c in gcols:
                add(W1, u, HORIZON + g, c, NG, -gamma / len(gcols))
            threshold(u)
            x = c0 + (width - 1) / 2.0 - (COLS - 1) / 2.0
            members.append((u, vote_for(x, half, tlz)))
            u += 1
        wire_votes(members, kvote)

    # ---- bump detectors ----------------------------------------------------
    for t0, t1, width, stride, gap, fw, kvote, half, tlz, gain in BUMPS:
        members = []
        rows = [HORIZON + p for p in range(t0, t1 + 1)]
        for c0 in range(0, COLS - width + 1, stride):
            left, right = bump_columns(c0, width, gap, fw)
            if not left or not right:
                continue          # a one-sided unit would fire on a wall at the screen edge
            for r in rows:
                for c in range(c0, c0 + width):
                    add(W1, u, r, c, NG_LAT, gain / (len(rows) * width))
                for c in left:
                    add(W1, u, r, c, NG_LAT, -gain / (len(rows) * len(left)))
                for c in right:
                    add(W1, u, r, c, NG_LAT, -gain / (len(rows) * len(right)))
            threshold(u)
            x = c0 + (width - 1) / 2.0 - (COLS - 1) / 2.0
            members.append((u, vote_for(x, half, tlz)))
            u += 1
        wire_votes(members, kvote)

    # ---- wall right in front (hard): floor at the feet is not floor-coloured
    nblk = len(HARD_ROWS) * len(HARD_COLS)
    for r in HARD_ROWS:
        for c in HARD_COLS:
            add(W1, u, r, c, FLOORSIGN, -1.0 / nblk)
    b1[u] = HARD_ABS
    W2[u, H2_WALL] = 1.0
    u += 1

    # ---- wall within ~150-200 ahead or to the right (soft, exploration) ----
    # mid-distance floor rows: centre+right block less green than the left reference
    n_s = len(SOFT_ROWS) * len(SOFT_COLS)
    n_r = len(SOFT_ROWS) * len(SOFT_REF_COLS)
    for r in SOFT_ROWS:
        for c in SOFT_COLS:
            add(W1, u, r, c, GR, -1.0 / n_s)
        for c in SOFT_REF_COLS:
            add(W1, u, r, c, GR, SOFT_REL / n_r)
    b1[u] = SOFT_ABS
    W2[u, H2_SOFT] = SOFT_GAIN
    W2[u, H2_SOFT + 1] = SOFT_GAIN
    b2[H2_SOFT + 1] = -1.0
    u += 1

    # ---- scan: at a few health levels, look around instead of walking on ---
    hp = D - 1
    for lo, hi in SCAN_BANDS:
        for edge, sign in ((lo, 1.0), (hi, -1.0)):
            W1[hp, u] = SCAN_K; b1[u] = -SCAN_K * edge
            W1[hp, u + 1] = SCAN_K; b1[u + 1] = -SCAN_K * edge - 1.0
            W2[u, H2_SCAN] = sign
            W2[u + 1, H2_SCAN] = -sign
            u += 2
    assert u <= HID, u
    for h in H2.values():
        b2[h] = -VOTE_MARGIN

    # ---- output ------------------------------------------------------------
    for a, h in H2.items():
        W3[h, a] = 1.0
    W3[H2_WALL, TL] = WALL_GAIN
    W3[H2_SOFT, TL] = EXPLORE_TL          # clip(SOFT_GAIN * soft, 0, 1)
    W3[H2_SOFT + 1, TL] = -EXPLORE_TL
    W3[H2_SCAN, TL] = EXPLORE_TL
    b3[FWD] = F_BIAS

    return {k: v.astype(np.float32) for k, v in
            dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3).items()}


if __name__ == "__main__":
    np.savez("weights.npz", **build())
