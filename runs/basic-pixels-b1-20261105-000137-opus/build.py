"""Hand-written weights for the ViZDoom `basic` scenario, pixels mode.

The room is brown brick (walls R > G > B, roughly R - G ~= G - B), gray ceiling, brown floor.
The Freedoom Cacodemon is dark blue / dark red: its green is far below the mean of red and blue.
So the "magenta" feature  f = R + B - 2G  is ~0 (|f| <= 0.02) on every background cell and
0.05 .. 0.16 on monster cells (measured on the practice screens).

Layer 1: one monster-column detector per screen column c (units 0..39):
    m_c = relu( sum_{r in ROWS} (R + B - 2G)[r, c]  -  T )

Layer 2: with d_c = c - 19.5 (column centre offset from the screen centre, in columns):
    P = relu(sum_c m_c * ( d_c - K))     > 0  iff monster centroid is > K columns right of centre
    N = relu(sum_c m_c * (-d_c - K))     > 0  iff monster centroid is > K columns left of centre
    M = relu(sum_c m_c)                  monster mass (visible at all)

Layer 3:
    q_left = N,  q_right = P,  q_attack = EPS * M
  centroid within +-K columns of the screen centre -> ATTACK, otherwise strafe toward it.
"""
import numpy as np

D_IN, H, A = 3601, 128, 3
GW, GH = 40, 30

ROWS = list(range(10, 20))  # monster flies at rows 13..17; weapon sprite starts at row ~22
T = 0.10                    # per-column threshold on the summed magenta feature
K = 1.0                  # half-width of the shooting window, in columns (8 px each)
EPS = 1e-3                  # attack score per unit of monster mass


def idx(r, c, k):
    return (r * GW + c) * 3 + k


def build():
    W1 = np.zeros((D_IN, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

    for c in range(GW):
        for r in ROWS:
            W1[idx(r, c, 0), c] = 1.0    # +R
            W1[idx(r, c, 1), c] = -2.0   # -2G
            W1[idx(r, c, 2), c] = 1.0    # +B
        b1[c] = -T

    d = np.arange(GW) - 19.5
    W2[:GW, 0] = d - K       # P: centroid right of window
    W2[:GW, 1] = -d - K      # N: centroid left of window
    W2[:GW, 2] = 1.0         # M: mass

    W3[1, 0] = 1.0           # MOVE_LEFT  <- N
    W3[0, 1] = 1.0           # MOVE_RIGHT <- P
    W3[2, 2] = EPS           # ATTACK     <- M
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    np.savez("weights.npz", **build())
