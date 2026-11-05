"""Hand-written weights for defend_the_center, objects mode. See explanation.md.

m[r, c] = channel 0 (monster share) of grid cell (r, c).
The crosshair is at pixel x=160, the border between columns 19 and 20.
"""
import numpy as np

D_IN, H, A = 3602, 128, 3
ROWS, COLS = 30, 40

T_HIT, D_HIT = 0.10, 0.20   # min(col19 mass, col20 mass) above T_HIT -> monster under crosshair
NEAR_ROW = 19               # a monster drawing pixels at/below this row is close
K = 1000.0                  # attack dominance
G_TURN = 2.0                # left/right comparison gain
EPS = 0.01                  # default: keep sweeping right
GATE_ROW = None               # only shoot once the target reaches this row (None = no gate)
G_LO, G_HI = 0.05, 0.25     # gate ramp on the low-body mass
BIG = 100.0


def idx(r, c, k):
    return (r * COLS + c) * 3 + k


def build():
    W1 = np.zeros((D_IN, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

    for r in range(ROWS):
        # unit 0: a = m19 (column sum) ; unit 1: relu(m19 - m20)  => min = a - relu(a - b)
        W1[idx(r, 19, 0), 0] = 1.0
        W1[idx(r, 19, 0), 1] = 1.0
        W1[idx(r, 20, 0), 1] = -1.0
        if r >= NEAR_ROW:
            # units 2/3: near-monster mass on the left / right, weighted up toward the centre
            for c in range(COLS):
                if c < 20:
                    W1[idx(r, c, 0), 2] = 0.5 + 0.5 * (1.0 + c) / 20.0
                else:
                    W1[idx(r, c, 0), 3] = 0.5 + 0.5 * (40.0 - c) / 20.0
    b1[4] = 1.0  # unit 4: constant 1

    if GATE_ROW is not None:
        # units 5/6: mass in rows >= GATE_ROW of columns 18-21 (the monster's lower body is
        # low on screen only once it is close). gate = (u5 - u6) / (G_HI - G_LO) in 0..1
        for r in range(GATE_ROW, ROWS):
            for c in range(18, 22):
                W1[idx(r, c, 0), 5] = 1.0
                W1[idx(r, c, 0), 6] = 1.0
        b1[5] = -G_LO; b1[6] = -G_HI
        gw = BIG / (G_HI - G_LO)
        W2[5, 0] = gw; W2[6, 0] = -gw; b2[0] -= BIG
        W2[5, 1] = gw; W2[6, 1] = -gw; b2[1] -= BIG

    W2[0, 0] = 1.0; W2[1, 0] = -1.0; b2[0] += -T_HIT            # relu(min - T [- BIG*(1-gate)])
    W2[0, 1] = 1.0; W2[1, 1] = -1.0; b2[1] += -T_HIT - D_HIT
    W2[2, 2] = 1.0; W2[3, 3] = 1.0; W2[4, 4] = 1.0

    W3[0, 2] = K / D_HIT; W3[1, 2] = -K / D_HIT                 # attack: 0..K step
    W3[2, 0] = G_TURN; W3[3, 0] = -G_TURN                        # left if near-left > near-right
    W3[4, 1] = EPS                                               # otherwise right
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    np.savez("weights.npz", **build())
