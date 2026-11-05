"""Hand-written weights for ViZDoom `basic`, objects mode. numpy only, no learning."""
import numpy as np

D, H, A = 30 * 40 * 3 + 1, 128, 3
ROWS, COLS = 30, 40
CENTER = 19.5     # screen centre (pixel 160) in column-index units (cell c is centred at 8c+4)
TOL = 1.2         # fire when the monster's centroid is within 1.2 columns (~10 px) of centre
EPS = 0.02        # attack score per cell's-worth of monster pixels
MIN_MASS = 0.3    # ignore specks smaller than this (in cells' worth of monster pixels)

W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

for r in range(ROWS):
    for c in range(COLS):
        i = (r * COLS + c) * 3 + 0          # monster-share channel of cell (r, c)
        off = c - CENTER
        W1[i, 0] = off - TOL                # unit 0: relu(S - TOL*M)  -> monster right of centre
        W1[i, 1] = -off - TOL               # unit 1: relu(-S - TOL*M) -> monster left of centre
        W1[i, 2] = 1.0                      # unit 2: relu(M - MIN_MASS) -> a monster is visible
b1[2] = -MIN_MASS
# (S = sum of monster share * column offset, M = total monster share)

# layer 2: pass the three features straight through
for j in range(3):
    W2[j, j] = 1.0

W3[1, 0] = 1.0     # MOVE_LEFT score  = how far left the centroid is beyond tolerance (times mass)
W3[0, 1] = 1.0     # MOVE_RIGHT score = how far right
W3[2, 2] = EPS     # ATTACK score     = small and positive whenever a monster is visible

np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
