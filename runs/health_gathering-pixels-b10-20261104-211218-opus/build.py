"""Hand-written weights for health_gathering, pixels mode.

Layer 1: one 'medkit evidence' unit per screen column (40 units).
    Per cell, f = 2*B - 1.2*R - 0.1*G is ~0 on the green floor and on the brown
    walls, but ~0.3 on the grey/white medkits.  Each column unit sums f over the
    floor rows (13..29) with a weight that grows toward the bottom of the screen
    (nearer medkits matter more), minus a small threshold that removes noise.
Layer 2: three 'where is the medkit mass' units: left, centre, right.
Layer 3: forward when the mass is centred, forward+turn toward it otherwise;
    with nothing visible, forward+turn-left (a wandering circle).
"""
import numpy as np

H, W_, NIN = 30, 40, 3601
TOP = 13          # first floor row that can hold a medkit
CENTER = 19.5
HALF_CENTRE = 3.0  # columns either side of centre that count as 'straight ahead'


def idx(r, c, k):
    return (r * W_ + c) * 3 + k


def build():
    W1 = np.zeros((NIN, 128), np.float32); b1 = np.zeros(128, np.float32)
    W2 = np.zeros((128, 128), np.float32); b2 = np.zeros(128, np.float32)
    W3 = np.zeros((128, 5), np.float32); b3 = np.zeros(5, np.float32)

    # ---- layer 1: per-column medkit evidence
    # zero on floor (R:G:B ~ 1:2.1:0.6), negative on wall (1:1.03:0.75),
    # about +1 per unit grey brightness on a medkit (1:1:1)
    colour = (-4.8, 0.9, 4.9)  # R, G, B
    for c in range(W_):
        for r in range(TOP, H):
            rw = 1.0 + 0.1 * (r - TOP)       # nearer rows count a bit more
            for k in range(3):
                W1[idx(r, c, k), c] = colour[k] * rw
        b1[c] = -0.3

    # ---- layer 2: left / centre / right mass
    L, C, R = 0, 1, 2
    for c in range(W_):
        d = c - CENTER
        if abs(d) <= HALF_CENTRE:
            W2[c, C] = 1.0
        elif d < 0:
            W2[c, L] = 1.0
        else:
            W2[c, R] = 1.0

    # ---- layer 3: actions 0 TL, 1 TR, 2 F, 3 F+TL, 4 F+TR
    W3[C, 2] = 1.3
    W3[L, 3] = 1.0
    W3[R, 4] = 1.0
    b3[3] = 0.010   # nothing seen: wander forward+left
    b3[2] = 0.005
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    np.savez("weights.npz", **build())
