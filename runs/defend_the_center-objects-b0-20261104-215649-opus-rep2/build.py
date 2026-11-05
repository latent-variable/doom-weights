"""Hand-written weights for defend_the_center, objects mode.

Policy (no learning; every number below is chosen by hand):
  * FIRE  if a *live* monster covers the crosshair (the boundary between grid
          columns 19 and 20) in the band of rows just above the horizon.
  * else TURN toward the side (left/right half of the screen) holding more live
          monster mass in that band.
  * else (nothing visible) keep TURN_RIGHT to scan the room.

"Live" = monster pixels above eye level. With the status bar on (render_hud in the
scenario cfg) the 3D view is ~208 px tall, so the horizon sits at pixel row ~104 =
the top edge of grid row 13. A standing monster (56 units tall, eye level 41) always
pokes above the horizon; corpses and late death frames (< 41 units tall) never do,
so they neither attract the aim nor trigger shots (no ammo burned on corpses).
"""
import numpy as np

D, H, A = 30 * 40 * 3 + 2, 128, 3
ROWS = range(4, 13)          # pixel rows 32..103: above the horizon
DEAD = 0.15                  # per-column dead zone for steering (ignores slivers)
T_FIRE = 0.25                # min monster mass in BOTH centre columns to shoot
K = 50.0                     # steepness of the fire step


def idx(r, c, k=0):
    return (r * 40 + c) * 3 + k


W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

# ---- layer 1 ----
# units 0..39: per-column live monster mass minus a dead zone
for c in range(40):
    for r in ROWS:
        W1[idx(r, c), c] = 1.0
    b1[c] = -DEAD
# unit 40: s19 ; unit 41: relu(s19 - s20)   -> min(s19, s20) = u40 - u41
for r in ROWS:
    W1[idx(r, 19), 40] = 1.0
    W1[idx(r, 19), 41] = 1.0
    W1[idx(r, 20), 41] = -1.0

# ---- layer 2 ----
W2[0:20, 0] = 1.0            # L: live mass in left half
W2[20:40, 1] = 1.0           # R: live mass in right half
# fire step: F = clip(K*(min - T), 0, 1) = u2 - u3
W2[40, 2], W2[41, 2], b2[2] = K, -K, -K * T_FIRE
W2[40, 3], W2[41, 3], b2[3] = K, -K, -K * T_FIRE - 1.0

# ---- layer 3 ----  actions: 0 TURN_LEFT, 1 TURN_RIGHT, 2 ATTACK
W3[0, 0], W3[1, 0] = 1.0, -1.0
W3[0, 1], W3[1, 1] = -1.0, 1.0
b3[1] = 0.01                 # default scan direction when nothing is seen
W3[2, 2], W3[3, 2] = 1000.0, -1000.0

np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
