"""Hand-designed weights for the `basic` scenario in objects mode.

Nothing is fitted. Every nonzero weight is set by the rule written next to it.
Input layout (harness_src/env.py): index (r * 40 + c) * 3 + k, row r in 0..29,
column c in 0..39, channel k (0 = monster share, 1 = pickup share, 2 = depth).
The last input (index 3600) is ammo / 50; it is not used.

Idea. The crosshair is the boundary between columns 19 and 20 (screen pixel 160).
A hitscan shot hits only if the monster covers that pixel, so the monster has to
cover BOTH column 19 and column 20. Sums are over all rows: the shot is at the
vertical centre and the monster flies roughly level.

  Layer 1 (monster channel only; each unit is relu of a linear function):
    u_PL  = Pl          Pl = monster share summed over rows in column 19
    u_PR  = Pr          Pr = same for column 20
    u_PLR = relu(Pl - Pr)
    u_L   = relu(L - R) L = monster mass in columns 0..19, R = in columns 20..39
    u_R   = relu(R - L)
  Layer 2:
    m     = min(Pl, Pr) = u_PL - u_PLR             (Pl, Pr >= 0, so exact)
    g_A   = relu(10 * (m - 0.8))                   ramp opens at m = 0.8
    g_B   = relu(10 * (m - 1.2))                   ramp closes at m = 1.2
    step  = (g_A - g_B) / 4 in [0, 1]              "crosshair is on the monster"
    g_DP  = u_L                                    "monster mostly left"
    g_DN  = u_R                                    "monster mostly right"
  Output (argmax; ties go to the lowest index):
    q0 (MOVE_LEFT)  = g_DP - 1000 * step
    q1 (MOVE_RIGHT) = g_DN - 1000 * step
    q2 (ATTACK)     = 1000 * step
  So: fire when the crosshair is on the monster; otherwise strafe toward the side
  with more monster; with nothing visible all scores are 0 and the tie goes to
  MOVE_LEFT.
"""
import numpy as np

GRID_W, GRID_H, CH = 40, 30, 3
D = GRID_W * GRID_H * CH + 1  # 3601
H = 128
A = 3

CENTRE_L, CENTRE_R = 19, 20   # the two columns that straddle the crosshair
RAMP_LO, RAMP_HI = 0.8, 1.2   # ramp for m = min(Pl, Pr), in column-sum units
SLOPE = 10.0
GATE = 250.0                  # 1000 * step = GATE * (g_A - g_B)


def idx(r, c, k):
    return (r * GRID_W + c) * CH + k


def build():
    W1 = np.zeros((D, H), np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32)
    b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, A), np.float32)
    b3 = np.zeros(A, np.float32)

    # Layer 1 unit indices.
    U_PL, U_PR, U_PLR, U_L, U_R = 0, 1, 2, 3, 4
    # Layer 2 unit indices.
    G_A, G_B, G_DP, G_DN = 0, 1, 2, 3

    for r in range(GRID_H):
        for c in range(GRID_W):
            i0 = idx(r, c, 0)  # monster channel of this cell
            if c == CENTRE_L:
                W1[i0, U_PL] = 1.0
                W1[i0, U_PLR] = 1.0      # + Pl
            if c == CENTRE_R:
                W1[i0, U_PR] = 1.0
                W1[i0, U_PLR] = -1.0     # - Pr
            # Left mass = columns 0..19, right mass = columns 20..39, so a
            # monster visible on only one side of the crosshair still steers.
            if c <= CENTRE_L:
                W1[i0, U_L] = 1.0        # L inside relu(L - R)
                W1[i0, U_R] = -1.0       # -L inside relu(R - L)
            else:
                W1[i0, U_L] = -1.0       # -R inside relu(L - R)
                W1[i0, U_R] = 1.0        # R inside relu(R - L)

    # Layer 2.
    W2[U_PL, G_A] = SLOPE
    W2[U_PLR, G_A] = -SLOPE
    b2[G_A] = -SLOPE * RAMP_LO
    W2[U_PL, G_B] = SLOPE
    W2[U_PLR, G_B] = -SLOPE
    b2[G_B] = -SLOPE * RAMP_HI
    W2[U_L, G_DP] = 1.0
    W2[U_R, G_DN] = 1.0

    # Output layer.
    # q0 = g_DP - GATE*(g_A - g_B)   (GATE*(g_A - g_B) = 1000*step)
    # q1 = g_DN - GATE*(g_A - g_B)
    # q2 = GATE*(g_A - g_B)
    W3[G_DP, 0] = 1.0
    W3[G_DN, 1] = 1.0
    for a in (0, 1):
        W3[G_A, a] = -GATE
        W3[G_B, a] = GATE
    W3[G_A, 2] = GATE
    W3[G_B, 2] = -GATE

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


if __name__ == "__main__":
    w = build()
    np.savez("weights.npz", **{k: v.astype(np.float32) for k, v in w.items()})
