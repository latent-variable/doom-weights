"""Hand-written weights for the ViZDoom `basic` scenario, `pixels` mode.

Layer 1 (40 used units): unit c = "how much redder than average is screen column c".
    s_c   = sum over rows ROW0..ROW1 of (R - G) in column c   (gray/neutral pixels give ~0)
    a_c   = relu(s_c - mean_c'(s_c') - THETA)
  Subtracting the across-column mean cancels whatever uniform tint the walls,
  floor and ceiling have; only a localised red blob (the Cacodemon) survives.

Layer 2 (2 used units): with offset o_c = c - 19.5 (column centre minus screen centre,
  in columns) and mass M = sum a_c, signed moment D = sum a_c * o_c:
    u_right = relu(D - TAU*M)    > 0 iff centroid is more than TAU columns right of centre
    u_left  = relu(-D - TAU*M)   > 0 iff centroid is more than TAU columns left of centre

Output: q_left = u_left, q_right = u_right, q_attack = EPS (a constant bias).
  Monster off-centre -> strafe toward it; within +-TAU columns -> shoot.
  Nothing red visible -> shoot too: the likeliest reason is that the monster is
  hidden straight ahead in the excluded weapon band.
"""
import numpy as np

D_IN, H = 3601, 128
GW, GH = 40, 30
ROW0, ROW1 = 0, 18       # rows used (inclusive); lower rows hold the weapon / muzzle flash
WEAPON_ROWS, WEAPON_COLS = (17, 18), range(14, 26)   # skipped: where the gun/flash can reach
THETA = 0.3              # column must be this much redder (summed R-G) than average
TAU = 1.0                # aim tolerance, in grid columns (8 px each)
EPS = 1e-4


def idx(r, c, k):
    return (r * GW + c) * 3 + k


def build():
    W1 = np.zeros((D_IN, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32);   b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, 3), np.float32);   b3 = np.zeros(3, np.float32)

    for c in range(GW):              # hidden unit c <- column c
        for r in range(ROW0, ROW1 + 1):
            used = [cc for cc in range(GW) if not (r in WEAPON_ROWS and cc in WEAPON_COLS)]
            if c not in used:        # this column skips this row entirely (self and mean),
                continue             # so a uniform background still gives exactly 0
            for cc in used:
                w = (1.0 if cc == c else 0.0) - 1.0 / len(used)
                W1[idx(r, cc, 0), c] += w      # +R
                W1[idx(r, cc, 1), c] -= w      # -G
        b1[c] = -THETA

    for c in range(GW):
        o = c - 19.5
        W2[c, 0] = o - TAU           # right
        W2[c, 1] = -o - TAU          # left
    W3[1, 0] = 1.0                   # MOVE_LEFT
    W3[0, 1] = 1.0                   # MOVE_RIGHT
    b3[2] = EPS                      # ATTACK unless a strafe unit fires
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    np.savez("weights.npz", **build())
