"""Hand-written weights for the `basic` scenario, pixels mode.

The room is brown (walls, floor) and gray (ceiling); the Freedoom Cacodemon is blue and red.
For a cell with mean colour (R, G, B), the "magenta" score m = R + B - 2G is ~0 on brown and on gray
(brown has R - G ~= G - B) and clearly positive on the monster.

Layer 1 (40 units): a_c = relu( sum_{r=ROW0..ROW1-1} m(r, c) - T ), one per screen column c.
Layer 2 (3 units):  P = relu(I), N = relu(-I), Cm = relu(sum_{c in centre band} a_c)
                    where I = sum_c a_c * (CX - c) is the monster's mass times its offset to the
                    left of the crosshair (CX = 19.5 = screen x 160, in column units).
Layer 3:            q_left = P - N, q_right = N - P, q_attack = K * Cm.
So it strafes toward the monster and shoots once |offset| < ~K columns with the monster near centre.
"""
import numpy as np

GW, GH = 40, 30
D = GW * GH * 3 + 1
H = 128

ROW0, ROW1 = 5, 20      # rows scanned (back wall band; excludes the gun at the bottom)
T = 0.15                # per-column threshold on the summed magenta score
CX = 19.5               # crosshair x (pixel 160) in column units (cell c spans c..c+1)
CENTRE = range(17, 23)  # columns whose mass counts as "on target"
K = 2.0                 # attack when |centroid offset| < ~K columns. The monster is ~4 columns wide,
                        # and strafe momentum keeps closing the gap during the pistol's ~4-tic fire delay.


def idx(r, c, k):
    return (r * GW + c) * 3 + k


def build():
    W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, 3), np.float32); b3 = np.zeros(3, np.float32)

    for c in range(GW):
        for r in range(ROW0, ROW1):
            W1[idx(r, c, 0), c] = 1.0
            W1[idx(r, c, 1), c] = -2.0
            W1[idx(r, c, 2), c] = 1.0
        b1[c] = -T

    for c in range(GW):
        off = CX - (c + 0.5)          # >0: column is left of the crosshair
        W2[c, 0] = off                # P = relu(I)
        W2[c, 1] = -off               # N = relu(-I)
        if c in CENTRE:
            W2[c, 2] = 1.0            # Cm

    W3[0, 0], W3[1, 0] = 1.0, -1.0    # MOVE_LEFT  = P - N
    W3[0, 1], W3[1, 1] = -1.0, 1.0    # MOVE_RIGHT = N - P
    W3[2, 2] = K                      # ATTACK     = K * Cm
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    import os
    w = build()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")
    np.savez(out, **{k: v.astype(np.float32) for k, v in w.items()})
