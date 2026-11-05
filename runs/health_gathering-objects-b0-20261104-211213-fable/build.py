"""Hand-written weights for health_gathering / objects mode.

Nothing here is learned or fitted: every number is set by the reasoning in the
comments (and in explanation.md). Running this file writes weights.npz.

The policy in one sentence: add up the medkit pixels the network can see, weighted
so that near and central medkits count most, separately for five screen regions
that correspond one-to-one to the five actions; the action whose region holds the
most weighted medkit wins. If no medkit pixel is visible at all, turn left on the
spot until one is.
"""
import os

import numpy as np

GRID_H, GRID_W = 30, 40
D = GRID_H * GRID_W * 3 + 1       # 3600 grid values + HEALTH
HIDDEN = 128
N_ACTIONS = 5

TURN_LEFT, TURN_RIGHT, FORWARD, FORWARD_LEFT, FORWARD_RIGHT = range(5)
PICKUP = 1                         # channel 1 of a cell = share of pickup pixels

# ---- design constants (each chosen by geometry, see explanation.md) ---------
HORIZON_ROW = 15      # the eye-level line is the boundary between rows 14 and 15
ROW_POWER = 3         # row weight = (rows below the horizon) ** 3
LATERAL_DECAY = 6.0   # column weight = exp(-|columns from centre| / 6)
CENTRE_HALF = 3.5     # |offset| <= 3.5 columns (about +-10 degrees): walk straight
ARC_HALF_FAR = 10.5   # far medkit, |offset| <= 10.5 (about 28 degrees): walk and turn
ARC_HALF_NEAR = 6.5   # near medkit, |offset| <= 6.5 (about 18 degrees): walk and turn
NEAR_ROW = 22         # rows 22..29 hold the base of medkits closer than ~120 units
SEARCH_GAIN = 1000.0  # one pickup pixel is 1/64 of a cell; 1000/64 > 1 switches search off


def row_weight(r):
    """Nearness. A medkit lies on the floor, so the nearer it is, the lower on the
    screen it is drawn: rows below the horizon ~ 1/distance. Cubing that, and
    multiplying by the pixel count (~ 1/distance^2), makes a medkit's total weight
    fall off like 1/distance^5, so the nearest one dominates the sum."""
    return float(max(r - (HORIZON_ROW - 1), 1)) ** ROW_POWER


def col_weight(offset):
    """Preference for what is already ahead. It decays faster than perspective
    inflates a medkit that slides toward the screen edge, so turning toward a
    medkit always raises its weight: the choice of target is self-reinforcing
    and the network cannot dither between a medkit on the left and one on the right."""
    return float(np.exp(-abs(offset) / LATERAL_DECAY))


def region(r, c):
    """Which action a medkit pixel in cell (r, c) votes for."""
    offset = c - (GRID_W - 1) / 2.0            # columns right of the screen centre
    if abs(offset) <= CENTRE_HALF:
        return FORWARD
    arc_half = ARC_HALF_NEAR if r >= NEAR_ROW else ARC_HALF_FAR
    if abs(offset) <= arc_half:
        return FORWARD_LEFT if offset < 0 else FORWARD_RIGHT
    return TURN_LEFT if offset < 0 else TURN_RIGHT


def build():
    W1 = np.zeros((D, HIDDEN), np.float64)
    b1 = np.zeros(HIDDEN, np.float64)
    W2 = np.zeros((HIDDEN, HIDDEN), np.float64)
    b2 = np.zeros(HIDDEN, np.float64)
    W3 = np.zeros((HIDDEN, N_ACTIONS), np.float64)
    b3 = np.zeros(N_ACTIONS, np.float64)

    SEARCH = 5  # hidden unit index of the "no medkit in sight" detector

    # Layer 1, units 0..4: weighted medkit mass in the region belonging to action 0..4.
    # Layer 1, unit 5: 1 when no pickup pixel is visible anywhere, 0 otherwise.
    for r in range(GRID_H):
        for c in range(GRID_W):
            i = (r * GRID_W + c) * 3 + PICKUP
            offset = c - (GRID_W - 1) / 2.0
            W1[i, region(r, c)] = row_weight(r) * col_weight(offset)
            W1[i, SEARCH] = -SEARCH_GAIN
    b1[SEARCH] = 1.0

    # Layer 2: carry the six values through unchanged (they are all >= 0, so the
    # relu does nothing to them).
    for u in range(6):
        W2[u, u] = 1.0

    # Layer 3: each region's mass is the score of its action; the search detector
    # adds to TURN_LEFT, which is therefore chosen exactly when nothing is in sight.
    for a in range(N_ACTIONS):
        W3[a, a] = 1.0
    W3[SEARCH, TURN_LEFT] = 1.0

    return {k: v.astype(np.float32) for k, v in
            dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3).items()}


if __name__ == "__main__":
    weights = build()
    # Next to this file, and in the working directory if that is somewhere else.
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")
    for out in sorted({here, os.path.abspath("weights.npz")}):
        np.savez(out, **weights)
        print("wrote", out)
