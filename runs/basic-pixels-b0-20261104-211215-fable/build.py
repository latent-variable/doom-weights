"""Hand-written weights for the ViZDoom `basic` scenario, `pixels` input.

No training, no fitting, no search: every number below is chosen by reasoning
about the scene (see explanation.md).  Running this file writes weights.npz.

The program the weights implement
---------------------------------
Layer 1 (40 units used): one "how much monster-red is in screen column j" unit
    per grid column, looking only at a band of rows that holds wall + monster
    and nothing else, with the local wall colour subtracted.
Layer 2 (43 units used): cleaned per-column redness (columns holding only a
    small share of all redness removed), plus three units that together say
    whether the monster covers the aim line (screen centre) with a margin on
    both sides.
Layer 3: LEFT   = cleaned redness left of the aim line,
         RIGHT  = cleaned redness right of the aim line,
         ATTACK = huge gain * (monster straddles the aim line with a margin).
"""
import os

import numpy as np

GRID_W, GRID_H = 40, 30
D = GRID_W * GRID_H * 3 + 1          # 3600 pixel values + AMMO2
H = 128
A = 3                                # 0 MOVE_LEFT, 1 MOVE_RIGHT, 2 ATTACK

# ---- design constants (all chosen by hand, see explanation.md) --------------
# Rows 11..15 are screen pixels 88..127.  With the status bar on, the horizon is
# at pixel row 104; the monster stands on the floor at the far wall and spans
# roughly rows 91..125, the far wall's floor line is at ~124 and the idle pistol
# starts below ~130.  So this band is "far wall + monster" only.
BAND_ROWS = (11, 12, 13, 14, 15)

# "Redness" of a cell: R - 1.25 G + 0.25 B.  The coefficients sum to zero, so
# any gray (R=G=B) of any brightness gives exactly 0; brown/tan/skin (G about
# halfway between R and B) gives little; a yellow-white muzzle flash gives ~0
# or less; a red body (R >> G ~ B) gives close to its full R - G.
COLOUR = (1.0, -1.25, 0.25)

# Centre-surround: from each column's band sum subtract the mean band sum of
# the columns 6..9 cells away on either side.  A wall tint that is the same
# (or changes smoothly) across the screen cancels; a blob a few cells wide
# (the monster is ~3 cells wide, and the offsets clear it up to ~6) does not.
SURROUND = (6, 7, 8, 9)

THETA1 = 0.05        # absolute floor on a column's redness (a full monster column is ~1)
THETA_EDGE = 0.10    # extra floor for columns whose surround is one-sided (screen edges)

PHI_CLEAN = 0.08     # layer 2: drop columns holding < 8% of all redness (texture speckle);
                     # a monster up to 6 columns wide has >= ~12% in each main column
PHI_AIM = 0.05       # shoot only if BOTH centre columns hold > 5% of all redness, i.e.
                     # the aim line cuts the monster with ~6% of its area on the small
                     # side: the aim is inside the sprite by ~23% of its half-width.
GAIN = 1000.0        # makes the shoot test a hard switch over the left/right scores
EPS = 1.0            # ATTACK needs strictly positive evidence

CL, CR = GRID_W // 2 - 1, GRID_W // 2      # columns 19 and 20: the aim line is between them


def build():
    W1 = np.zeros((D, H), np.float64)
    b1 = np.zeros(H, np.float64)
    W2 = np.zeros((H, H), np.float64)
    b2 = np.zeros(H, np.float64)
    W3 = np.zeros((H, A), np.float64)
    b3 = np.zeros(A, np.float64)

    def idx(r, c, k):
        return (r * GRID_W + c) * 3 + k

    # ---- layer 1: r_j = relu(band_red(j) - mean band_red(surround of j) - theta) ----
    # band_red(j) = sum over BAND_ROWS of (R - 1.25 G + 0.25 B) in column j
    for j in range(GRID_W):
        sur = [j + s * o for o in SURROUND for s in (-1, 1) if 0 <= j + s * o < GRID_W]
        one_sided = not (j - SURROUND[0] >= 0 and j + SURROUND[0] < GRID_W)
        for r in BAND_ROWS:
            for k in range(3):
                W1[idx(r, j, k), j] += COLOUR[k]
                for c in sur:
                    W1[idx(r, c, k), j] -= COLOUR[k] / len(sur)
        b1[j] = -(THETA1 + (THETA_EDGE if one_sided else 0.0))
    # (the AMMO2 input, index 3600, is deliberately unused)

    # ---- layer 2 ----
    # S = sum_k r_k  (all redness that passed layer 1)
    # units 0..39: s_j = relu(r_j - PHI_CLEAN * S)
    for j in range(GRID_W):
        W2[:GRID_W, j] -= PHI_CLEAN
        W2[j, j] += 1.0
    # units 40, 41, 42: pieces of  M = min(r_CL, r_CR) - PHI_AIM * S
    #   M = relu(r_CL - phi S) - relu(phi S - r_CL) - relu(r_CL - r_CR)
    P, N, Q = 40, 41, 42
    W2[:GRID_W, P] -= PHI_AIM
    W2[CL, P] += 1.0
    W2[:GRID_W, N] += PHI_AIM
    W2[CL, N] -= 1.0
    W2[CL, Q] += 1.0
    W2[CR, Q] -= 1.0

    # ---- layer 3 ----
    W3[0:CR, 0] = 1.0                 # MOVE_LEFT : cleaned redness left of the aim line
    W3[CR:GRID_W, 1] = 1.0            # MOVE_RIGHT: cleaned redness right of the aim line
    W3[P, 2] = GAIN                   # ATTACK    : GAIN * M - EPS
    W3[N, 2] = -GAIN
    W3[Q, 2] = -GAIN
    b3[2] = -EPS

    return {k: v.astype(np.float32) for k, v in
            dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3).items()}


def main():
    w = build()
    here = os.path.dirname(os.path.abspath(__file__))
    targets = {os.path.join(os.getcwd(), "weights.npz"), os.path.join(here, "weights.npz")}
    for path in targets:
        np.savez(path, **w)


if __name__ == "__main__":
    main()
