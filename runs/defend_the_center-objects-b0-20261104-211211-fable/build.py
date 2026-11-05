"""Hand-written weights for defend_the_center, `objects` input.

No training of any kind: every number below is chosen by reasoning about what the
unit should compute.  Run `python build.py` to (re)write weights.npz.

The program the weights implement (see explanation.md for the why):

    C[c]  = monster pixels in screen column c, in cells (sum of channel 0 over the 30 rows)

    FIRE  if a monster straddles the crosshair (the line between columns 19 and 20)
          with a margin:   C[19] > TAU  and  C[20] > TAU
                      and  left6  >= RHO * right6   and   right6 >= RHO * left6
          (left6 / right6 = monster cells in the 6 columns left / right of the crosshair)
          RHO is strict (0.15) normally and loose (0.04) once health has fallen behind
          ammo (ammo/26 - 1.5 * health/100 + 0.4 > 0): bullets we will not live to
          use are not worth protecting, time is.
    else LEFT   if  sum_{c<20} P[c] - sum_{c>=20} P[c] > THETA,
          where P[c] = relu(C[c-1] + C[c] + C[c+1] - KAPPA)
          (a big = close monster on the left that outweighs what is on the right)
    else RIGHT  (default sweep)
"""
import numpy as np

ROWS, COLS, CH = 30, 40, 3
D = ROWS * COLS * CH + 2          # 3602: grid + (ammo/26, health/100)
H = 128
A_LEFT, A_RIGHT, A_ATTACK = 0, 1, 2

# ---- the constants of the program -------------------------------------------------
KAPPA = 5.5    # cells: three adjacent columns only "pull" if they hold > 5.5 cells of
               # monster between them (a monster within very roughly 350-450 units)
THETA = 0.3    # cells: how much the left pull must beat the right pull to turn left
K_LAM = 10.0   # gain turning (left pull - right pull - THETA) into a 0..1 switch
TAU = 0.25     # cells of monster required in column 19 and in column 20
RHO = 0.15     # each side of the crosshair must hold >= RHO x the other side (strict)
RHO_LOOSE = 0.04   # the same test when the loose gate is open
GATE_A, GATE_H, GATE_0 = 1.0, -1.5, 0.4   # loose gate: ammo/26 - 1.5*health/100 + 0.4 > 0
EPS = 0.05     # keeps the balance tests false on an empty screen
G = 10.0       # gain turning each fire test into a 0..1 switch
WIN = 6        # columns on each side of the crosshair used for the balance test
Q_RIGHT = 0.5  # constant score of the default action
Q_FIRE = 10.0  # score of ATTACK per unit of the fire switch


def col_weights(colw):
    """A W1 column that reads sum_c colw[c] * C[c] (channel 0, all rows)."""
    w = np.zeros((ROWS, COLS, CH), np.float32)
    w[:, :, 0] = np.asarray(colw, np.float32)[None, :]
    full = np.zeros(D, np.float32)
    full[: ROWS * COLS * CH] = w.reshape(-1)
    return full


def build():
    W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, 3), np.float32); b3 = np.zeros(3, np.float32)

    # ---------------- layer 1 ----------------
    # units 0..39: P[c] = relu(C[c-1] + C[c] + C[c+1] - KAPPA): "a close monster is at column c"
    for c in range(COLS):
        e = np.zeros(COLS); e[max(c - 1, 0):c + 2] = 1.0
        W1[:, c] = col_weights(e)
        b1[c] = -KAPPA

    # four fire tests, each as a pair (a, b) with a - b = clip(G * test, 0, 1)
    left6 = np.zeros(COLS);  left6[20 - WIN:20] = 1.0
    right6 = np.zeros(COLS); right6[20:20 + WIN] = 1.0
    e19 = np.zeros(COLS); e19[19] = 1.0
    e20 = np.zeros(COLS); e20[20] = 1.0
    gate = np.zeros(D, np.float32)
    gate[-2], gate[-1] = GATE_A, GATE_H    # x[-2] = ammo/26, x[-1] = health/100
    tests = [
        (col_weights(e19), -TAU),                            # 0: C[19] > TAU
        (col_weights(e20), -TAU),                            # 1: C[20] > TAU
        (col_weights(right6 - RHO * left6), -EPS),           # 2: right vs left, strict
        (col_weights(left6 - RHO * right6), -EPS),           # 3: left vs right, strict
        (col_weights(right6 - RHO_LOOSE * left6), -EPS),     # 4: right vs left, loose
        (col_weights(left6 - RHO_LOOSE * right6), -EPS),     # 5: left vs right, loose
        (gate, GATE_0),                                      # 6: loose gate open
    ]
    u = COLS
    pairs = []
    for vec, off in tests:
        W1[:, u] = G * vec;     b1[u] = G * off
        W1[:, u + 1] = G * vec; b1[u + 1] = G * off - 1.0
        pairs.append((u, u + 1))
        u += 2

    # ---------------- layer 2 ----------------
    # unit 0: f_strict = relu(t0 + t1 + t2 + t3 - 3): the four strict tests all hold (AND)
    for i in (0, 1, 2, 3):
        a, b = pairs[i]
        W2[a, 0], W2[b, 0] = 1.0, -1.0
    b2[0] = -3.0
    # unit 3: f_loose = relu(t0 + t1 + t4 + t5 + t6 - 4): loose tests and the gate (AND)
    for i in (0, 1, 4, 5, 6):
        a, b = pairs[i]
        W2[a, 3], W2[b, 3] = 1.0, -1.0
    b2[3] = -4.0
    # units 1, 2: lam = clip(K_LAM * (Lpull - Rpull - THETA), 0, 1) = h2[1] - h2[2]
    for c in range(COLS):
        s = K_LAM if c < 20 else -K_LAM
        W2[c, 1] = s
        W2[c, 2] = s
    b2[1] = -K_LAM * THETA
    b2[2] = -K_LAM * THETA - 1.0

    # ---------------- layer 3 ----------------
    W3[1, A_LEFT] = 1.0
    W3[2, A_LEFT] = -1.0        # q_left  = lam  in [0, 1]
    b3[A_RIGHT] = Q_RIGHT       # q_right = 0.5
    W3[0, A_ATTACK] = Q_FIRE    # q_fire  = 10 * (f_strict + f_loose)
    W3[3, A_ATTACK] = Q_FIRE

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


if __name__ == "__main__":
    np.savez("weights.npz", **build())
