"""Hand-written weights for defend_the_center, objects mode.

Policy (see explanation.md):
  * Only monster pixels at or above the horizon (rows 0..14) count. A standing
    monster is taller than the player's eye height (41 units), so it always
    reaches above the horizon line (y = 120); a lying corpse never does.
  * FIRE if a monster covers both column 19 and column 20 in that band, i.e. it
    straddles the crosshair at x = 160.
  * Otherwise turn toward the side with more (centre-weighted) monster mass.
  * Nothing visible: turn left (scan).
"""
import numpy as np

GW, GH, NV = 40, 30, 2
D = GW * GH * 3 + NV
H = 128
BAND = range(0, 15)          # rows 0..14: everything above the horizon
K = 60.0                     # gain: ~1 monster pixel in a cell saturates the detector


def idx(r, c, k=0):
    return (r * GW + c) * 3 + k


def build():
    W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, 3), np.float32); b3 = np.zeros(3, np.float32)

    # ---- layer 1 ----
    # units 0..3: saturating presence detectors for columns 19 and 20
    #   sat(c) = relu(K*s_c) - relu(K*s_c - 1)  in [0, 1]
    for j, c in enumerate((19, 20)):
        for r in BAND:
            W1[idx(r, c), 2 * j] = K
            W1[idx(r, c), 2 * j + 1] = K
        b1[2 * j + 1] = -1.0
    # unit 4: left monster mass, unit 5: right monster mass (centre-weighted)
    for c in range(GW):
        w = 1.0 + 1.0 * (1.0 - abs(c - 19.5) / 20.0)
        for r in BAND:
            if c <= 19:
                W1[idx(r, c), 4] = w
            else:
                W1[idx(r, c), 5] = w

    # ---- layer 2 ----
    # unit 0: FIRE = relu(sat19 + sat20 - 1)  (an AND, in [0, 1])
    W2[0, 0] = 1.0; W2[1, 0] = -1.0
    W2[2, 0] = 1.0; W2[3, 0] = -1.0
    b2[0] = -1.0
    # units 1, 2: pass left / right mass through
    W2[4, 1] = 1.0
    W2[5, 2] = 1.0

    # ---- output ----
    # Masses are at most ~ 20 cols * 15 rows * 2 = 600; fire must dominate.
    W3[1, 0] = 1.0           # TURN_LEFT  ~ left mass
    W3[2, 1] = 1.0           # TURN_RIGHT ~ right mass
    W3[0, 2] = 5000.0        # ATTACK when a monster straddles the crosshair
    b3[2] = -1.0             # never attack on a tie / with nothing in sight
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    np.savez("weights.npz", **build())
