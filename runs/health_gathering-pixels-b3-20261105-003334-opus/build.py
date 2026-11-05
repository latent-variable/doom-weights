"""Hand-written weights for health_gathering, pixels mode.

Layer 1: 120 medkit detectors (40 columns x 3 distance bands) + wall detectors.
Layer 2: medkit mass / left pull / right pull, wall signals, a constant.
Layer 3: action scores.  See explanation.md.
"""
import numpy as np

D, H, A = 3601, 128, 5
GW, GH = 40, 30


def idx(r, c, k):
    return (r * GW + c) * 3 + k


W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

# ---- colour directions (per cell, applied to mean R,G,B) ----
# MED: zero on floor green (R:G:B ~ .49:1:.29), <= 0 on the bluest wall stone
# (.97:1:.84), positive on neutral grey (the medkit box).  Grey 0.55 -> ~1.0.
MED = np.array([-0.55, 0.13, 0.48]) * 30.0
# WAL: floor ~ -0.2, wall stone ~ +1..2 (red close to green; floor has R << G).
WAL = np.array([1.0, -0.7, 0.3]) * 10.0

# ---- layer 1: medkit detectors ----
BANDS = [range(15, 19), range(19, 24), range(24, 30)]   # far, mid, near (rows on the floor)
BAND_W = [1.0, 1.5, 2.0]                                # nearer medkits pull harder
MED_THR = [0.0, 0.6, 1.0]                              # per-band noise floor
for c in range(GW):
    for bi, rows in enumerate(BANDS):
        u = c * 3 + bi
        for r in rows:
            for k in range(3):
                W1[idx(r, c, k), u] = MED[k]
        b1[u] = -MED_THR[bi]


def wall_unit(u, rows, cols, thr):
    n = len(rows) * len(cols)
    for r in rows:
        for c in cols:
            for k in range(3):
                W1[idx(r, c, k), u] = WAL[k] / n
    b1[u] = -thr


U_WCLOSE, U_WL, U_WR, U_WMID, U_ONE = 120, 121, 122, 123, 124
wall_unit(U_WCLOSE, range(24, 30), range(13, 27), 0.3)   # wall right in front
wall_unit(U_WL, range(21, 30), range(0, 14), 0.3)        # wall low on the left
wall_unit(U_WR, range(21, 30), range(26, 40), 0.3)       # wall low on the right
wall_unit(U_WMID, range(20, 30), range(13, 27), 0.3)     # wall ahead, a bit further
b1[U_ONE] = 1.0

# ---- layer 2 ----
M, SR, SL, WC, WLR, WRL, ONE, WM, SRX, SLX = range(10)
for c in range(GW):
    x = (c - 19.5) / 19.5
    for bi in range(3):
        u = c * 3 + bi
        W2[u, M] = BAND_W[bi]
        W2[u, SR] = BAND_W[bi] * x
        W2[u, SL] = -BAND_W[bi] * x
        # pull beyond half of the half-screen (~22 deg): turn in place first
        W2[u, SRX] = BAND_W[bi] * (x - 0.5)
        W2[u, SLX] = BAND_W[bi] * (-x - 0.5)
W2[U_WCLOSE, WC] = 1.0
W2[U_WL, WLR] = 1.0; W2[U_WR, WLR] = -1.0    # more wall on the left
W2[U_WR, WRL] = 1.0; W2[U_WL, WRL] = -1.0    # more wall on the right
W2[U_ONE, ONE] = 1.0
W2[U_WMID, WM] = 1.0

# ---- layer 3: action scores ----
# 0 TURN_LEFT, 1 TURN_RIGHT, 2 FORWARD, 3 FORWARD+LEFT, 4 FORWARD+RIGHT
W3[ONE, 2] = 0.3                 # default: walk forward
W3[M, 2] = 5.0                   # medkit seen: go for it ...
W3[SL, 3] = 20.0                 # ... steering when it is off-centre
W3[SR, 4] = 20.0
W3[SLX, 0] = 80.0
W3[SRX, 1] = 80.0
# Wall right ahead: turn in place, always to the right.  A fixed direction
# cannot rock back and forth in a corner (the policy has no memory).
W3[WC, 1] = 2.0
W3[WC, 3] = -2.0; W3[WC, 4] = -2.0
# Wall approaching but not yet close: keep walking, curving to the more open side.
W3[WM, 3] = 0.8; W3[WM, 4] = 0.8
W3[WLR, 4] = 1.0; W3[WRL, 3] = 1.0
b3[1] = 0.01; b3[4] = 0.01        # tie-break: right

np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
