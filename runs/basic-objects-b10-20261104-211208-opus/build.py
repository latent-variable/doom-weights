"""Hand-written weights for the `basic` scenario, objects mode. Only numpy.

The crosshair sits at pixel x=160, the boundary between grid columns 19 and 20.
  a = monster mass (channel 0 summed over rows) in columns 0..19  -> left of the crosshair
  b = monster mass in columns 20..39                                -> right of the crosshair
Policy:  fire if min(a, b) > T  (crosshair is inside the monster with some margin),
         otherwise strafe toward the side with more monster mass.
"""
import numpy as np

D, H, A = 3601, 128, 3
GW, GH = 40, 30
W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

# Layer 1: h1[0] = a, h1[1] = b
for r in range(GH):
    for c in range(GW):
        i = (r * GW + c) * 3 + 0          # monster-share channel
        W1[i, 0 if c <= 19 else 1] = 1.0

# Layer 2: h2[0] = a, h2[1] = b, h2[2] = relu(a - b)   so  min(a, b) = h2[0] - h2[2]
W2[0, 0] = 1.0
W2[1, 1] = 1.0
W2[0, 2] = 1.0; W2[1, 2] = -1.0

# Output
#   q_left   = a - b
#   q_right  = b - a
#   q_attack = K * (min(a, b) - T)
# K is large so that once min(a,b) clears T, firing beats any strafe score (|a-b| <= ~12).
# With no monster visible all scores are <= 0 and the tie at 0 goes to MOVE_LEFT (searching).
K, T = 1000.0, 0.4
W3[0, 0] = 1.0;  W3[1, 0] = -1.0
W3[0, 1] = -1.0; W3[1, 1] = 1.0
W3[0, 2] = K;    W3[2, 2] = -K
b3[2] = -K * T

np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
