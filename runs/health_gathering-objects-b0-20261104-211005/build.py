"""Hand-written weights for health_gathering, objects mode.

Policy (stateless):
  * Every pickup pixel below the horizon casts a vote for one steering action,
    chosen by where it sits on screen (left/centre/right, near/far).  Closer
    (lower) cells vote more heavily.  The action with most votes wins.
  * With nothing visible, walk forward (small bias).
  * If a wall is close ahead (scale-free depth test: the horizon depth in the
    centre columns is not much larger than the floor depth at row 22), turn
    right in place - unless a large/near medkit is dead ahead.
No numbers were fitted; see explanation.md.
"""
import numpy as np

GH, GW = 30, 40
D = GH * GW * 3 + 1          # 3601 inputs; last one is HEALTH/100 (unused)
H = 128
TL, TR, F, FL, FR = 0, 1, 2, 3, 4


def idx(r, c, k):
    return (r * GW + c) * 3 + k


# --- geometry (Doom: eye height 41, 90 deg hfov at 320 px -> focal 160 px) ---
EYE = 41.0
FOCAL = 160.0
R_TURN = 135.0               # radius of a forward+turn circle (8.3 u/tic, 3.5 deg/tic)


def floor_dist(r):
    """Distance to the floor seen at the centre of grid row r (inf above horizon)."""
    dy = (8 * r + 4) - 120
    return np.inf if dy <= 0 else EYE * FOCAL / dy


def vote_weights():
    """(30, 40, 5) weight of a pickup cell's vote for each action."""
    V = np.zeros((GH, GW, 5), np.float32)
    for r in range(GH):
        wr = 0.5 if r < 14 else 1.0 + 0.15 * (r - 14)
        rho = floor_dist(r)
        for c in range(GW):
            d = c - 19.5                       # columns from screen centre
            ad = abs(d)
            sin_t = ad / np.hypot(ad, 20.0)    # 20 columns = focal length in cells
            # centre band -> forward, with a soft edge
            f = 1.0 if ad <= 2.5 else (0.5 if ad <= 3.5 else 0.0)
            t = 1.0 - f
            # a target inside the forward+turn circle needs a turn in place
            inside = rho < 0.9 * 2 * R_TURN * sin_t
            side_fwd, side_turn = (FR, TR) if d > 0 else (FL, TL)
            V[r, c, F] += wr * f
            if inside:
                V[r, c, side_turn] += wr * t
            else:
                V[r, c, side_fwd] += wr * t
    return V


def build():
    W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, 5), np.float32); b3 = np.zeros(5, np.float32)

    # ---- layer 1 ----
    # units 0..4: pickup votes for actions 0..4 (inputs >= 0, so relu is identity)
    V = vote_weights()
    for r in range(GH):
        for c in range(GW):
            for a in range(5):
                W1[idx(r, c, 1), a] = V[r, c, a]

    # units 5, 6: wall-ahead test.  u = K*floor22 - horizon (centre columns)
    # is > 0 iff the thing seen at the horizon is within ~K x 109 units, i.e.
    # a wall is close.  Independent of the depth buffer's scale if depth is
    # proportional to distance.  Steep gain + clamp to [0, 1] in layer 2.
    K, G, EPS = 1.9, 500.0, 0.004
    cols = range(16, 24)
    n = len(cols)
    for c in cols:
        for r in (14, 15):
            W1[idx(r, c, 2), 5] += -G / (2 * n)
            W1[idx(r, c, 2), 6] += -G / (2 * n)
        W1[idx(22, c, 2), 5] += G * K / n
        W1[idx(22, c, 2), 6] += G * K / n
    b1[5] = G * EPS
    b1[6] = G * EPS - 1.0

    # units 7, 8: safety veto.  If the horizon depth is saturated (>= ~0.97),
    # the depth scale is too short for the ratio test to mean anything, so
    # the wall test is switched off.  s = clamp(G*(horizon - 0.97), 0, 1).
    for c in cols:
        for r in (14, 15):
            W1[idx(r, c, 2), 7] += G / (2 * n)
            W1[idx(r, c, 2), 8] += G / (2 * n)
    b1[7] = -G * 0.97
    b1[8] = -G * 0.97 - 1.0

    # ---- layer 2 ----
    for a in range(5):                       # copy votes
        W2[a, a] = 1.0
    # wall = relu(clamp(u) - clamp(s))
    W2[5, 5] = 1.0; W2[6, 5] = -1.0; W2[7, 5] = -1.0; W2[8, 5] = 1.0

    # ---- output ----
    for a in range(5):
        W3[a, a] = 1.0
    W3[5, TR] = 3.0                          # wall close -> turn right in place
    b3[F] = 0.05                             # default: walk forward
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    w = build()
    np.savez("weights.npz", **{k: v.astype(np.float32) for k, v in w.items()})
