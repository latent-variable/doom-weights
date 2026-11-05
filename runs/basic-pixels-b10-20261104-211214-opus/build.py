"""Hand-written weights for the ViZDoom `basic` scenario, pixels mode.

Layer 1: one unit per screen column c (40 units): "how much monster is in this column"
         m_c = relu( sum_{rows 0..19} (R + B - 2G) - T )
         R + B - 2G is ~0 for the gray ceiling and brown walls/floor, ~0.1-0.2 on the monster.
Layer 2: from the column profile, with S = sum_c m_c (c - 19.5):
         P = relu( K (S  - OFFSET*MASS0) )   monster clearly right of the crosshair
         N = relu( K (-S - OFFSET*MASS0) )   monster clearly left of it
         G1 - G2 = min(sum_c m_c / M, 1)     "a monster is visible" gate in [0, 1]
Output:  q_left = N, q_right = P, q_attack = (G1 - G2) - 0.5 (+0.001)
"""
import numpy as np

D_IN, H = 3601, 128
GW, GH = 40, 30
CX = 19.5            # crosshair sits between columns 19 and 20 (screen x = 160)

# --- tunable constants (chosen by reasoning, see explanation.md) ---
ROW0, ROW1 = 0, 20   # rows looked at
T = 0.10             # column threshold on summed magenta (background columns reach 0.09)
M = 0.10             # column-mass for the presence gate to saturate
MASS0 = 0.7          # typical total column mass of the monster after thresholding (fixed distance)
OFFSET = 1.5       # tolerated centroid offset in columns before strafing

W1 = np.zeros((D_IN, H), np.float32); b1 = np.zeros(H, np.float32)
W2 = np.zeros((H, H), np.float32);    b2 = np.zeros(H, np.float32)
W3 = np.zeros((H, 3), np.float32);    b3 = np.zeros(3, np.float32)

def idx(r, c, k):
    return (r * GW + c) * 3 + k

for c in range(GW):
    for r in range(ROW0, ROW1):
        W1[idx(r, c, 0), c] = 1.0
        W1[idx(r, c, 1), c] = -2.0
        W1[idx(r, c, 2), c] = 1.0
    b1[c] = -T

# Layer 2. Signed position S = sum m_c (c - CX). With total mass Mt, centroid = S / Mt.
# We want strafing when |centroid| > OFFSET, i.e. |S| > OFFSET * Mt.
# P = relu(S - OFFSET*Mt), N = relu(-S - OFFSET*Mt) (scaled), so they are zero inside the dead zone.
SCALE = 1000.0
for c in range(GW):
    W2[c, 0] = SCALE * (c - CX)               # P
    W2[c, 1] = SCALE * (-(c - CX))            # N
    W2[c, 2] = 1.0 / M                        # G1
    W2[c, 3] = 1.0 / M                        # G2
b2[3] = -1.0
b2[0] = b2[1] = -SCALE * OFFSET * MASS0

W3[1, 0] = 1.0          # left  <- N
W3[0, 1] = 1.0          # right <- P
W3[2, 2] = 1.0; W3[3, 2] = -1.0; b3[2] = -0.5 + 1e-3

np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
