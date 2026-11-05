"""Hand-written weights for health_gathering, objects mode.

Layer 1: per-column pickup mass (40 units), near-zone side pickup mass (2 units),
         wall closeness ahead (1 unit).
Layer 2: left / centre / right pickup mass, near-left / near-right mass, wall.
Output:  steer toward the pickup mass; with nothing in view, walk forward while
         turning slowly, and turn in place when a wall is close.
"""
import numpy as np

D, H, A = 3601, 128, 5
GH, GW = 30, 40


def idx(r, c, k):
    return (r * GW + c) * 3 + k


# ---- tunable constants (chosen by reasoning, adjusted from practice) ----
CENTER_LO, CENTER_HI = 17, 22     # columns counted as "straight ahead" (inclusive)
NEAR_ROW = 24                     # rows >= this are "close to the feet"
NEAR_SIDE = 12                    # near pickups this far off-centre: turn in place
WALL_T = 0.12                     # mean centre depth below this = wall close
WALL_ROWS = range(13, 18)
WALL_COLS = range(14, 26)
G = 100.0                         # gain on pickup mass
G_CENTER = 1.3                    # straight-ahead bonus
G_NEAR = 2.0                      # near-side pure-turn gain
K_WALL = 5.0                    # wall -> turn-in-place gain
EXPLORE_BIAS = 0.5                # default action bias when nothing seen


def build():
    W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

    # L1 units 0..39: pickup mass in column c (lower rows a bit heavier: nearer)
    for c in range(GW):
        for r in range(GH):
            W1[idx(r, c, 1), c] = 0.5 + r / GH
    # L1 unit 40 / 41: near-zone pickup mass far left / far right
    for r in range(NEAR_ROW, GH):
        for c in range(0, NEAR_SIDE):
            W1[idx(r, c, 1), 40] = 1.0
        for c in range(GW - NEAR_SIDE, GW):
            W1[idx(r, c, 1), 41] = 1.0
    # L1 unit 42: wall closeness = relu(WALL_T - mean depth of centre patch)
    n = len(WALL_ROWS) * len(WALL_COLS)
    for r in WALL_ROWS:
        for c in WALL_COLS:
            W1[idx(r, c, 2), 42] = -1.0 / n
    b1[42] = WALL_T

    # L2: 0 left, 1 centre, 2 right, 3 near-left, 4 near-right, 5 wall
    for c in range(GW):
        if c < CENTER_LO:
            W2[c, 0] = 1.0
        elif c <= CENTER_HI:
            W2[c, 1] = 1.0
        else:
            W2[c, 2] = 1.0
    W2[40, 3] = 1.0
    W2[41, 4] = 1.0
    W2[42, 5] = 1.0 / WALL_T          # 0..1

    # Output
    W3[0, 3] = G                      # left mass -> forward+left
    W3[1, 2] = G * G_CENTER           # centre mass -> forward
    W3[2, 4] = G                      # right mass -> forward+right
    W3[3, 0] = G * G_NEAR             # near-left -> turn left
    W3[4, 1] = G * G_NEAR             # near-right -> turn right
    W3[5, 0] = K_WALL                 # wall close -> turn left
    b3[3] = EXPLORE_BIAS              # default: forward + left (sweep)
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    w = build()
    np.savez("weights.npz", **{k: v.astype(np.float32) for k, v in w.items()})
