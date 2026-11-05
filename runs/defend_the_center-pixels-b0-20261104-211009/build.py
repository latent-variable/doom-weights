"""Hand-written weights for defend_the_center, pixels mode.  numpy only, no data.

Policy (see explanation.md):
  * default: TURN_LEFT  -> the player spins in place, never reversing (no oscillation).
  * ATTACK when the screen centre (mean of cols 19, 20 = the aim line x=160) differs in
    colour from the wall in rows 12 / 13, i.e. right at the horizon of the 3-D view
    (the view is 208 px tall above the 32-px status bar, so the horizon is y=104).
    "Differs from the wall" = differs from most of 10 one-column reference strips spread
    over the outer screen (a vote, so other monsters in a few strips do not matter).
    Row 12 (just above eye level) can only be covered by standing monsters, never by
    corpses lying on the floor, so it carries more weight than row 13.
  * never ATTACK while the yellow muzzle flash is visible above the pistol
    (the weapon is already busy: avoids refire on a dying monster and lock-in).
"""
import numpy as np

D, H, A = 30 * 40 * 3 + 2, 128, 3
GW = 40


def idx(r, c, k):
    return (r * GW + c) * 3 + k


# ---- constants, chosen by reasoning (see explanation.md) ----
T0 = 0.07            # colour difference (~ mean |channel diff|, 0..1) that means "not wall"
EPS = 0.02           # width of the soft step that turns a difference into a 0/1 vote
ROW_W = {12: 1.0, 13: 0.5}            # votes per reference strip, per row
CENTER = (19, 20)
REF_COLS = (0, 4, 8, 12, 27, 31, 35, 39)
ATTACK_THRESH = 5.5   # row 13 alone (max 4) never fires; row 12 alone needs >= 6 of 8
# very close monster (biting range): it towers far above the horizon, rows 9-10, and is
# compared only with the outermost columns, which even a monster at melee range leaves free
NEAR_ROWS = (9, 10)
NEAR_REFS = (0, 3, 36, 39)
NEAR_W = ATTACK_THRESH / 7.5   # all 8 near votes alone fire
FLASH_ROWS = range(16, 26)
FLASH_COLS = range(19, 21)
FLASH_COEF = np.array([1.0, 2.0, -3.0])   # "yellowness" R + 2G - 3B, relative to row mean
FLASH_THETA = 0.7
# 4 directions summing to zero: sum_i relu(u_i . d) / 3 is a norm-like size of d
TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)


def build():
    W1 = np.zeros((D, H)); b1 = np.zeros(H)
    W2 = np.zeros((H, H)); b2 = np.zeros(H)
    W3 = np.zeros((H, A)); b3 = np.zeros(A)

    # ---------- layer 1: colour differences centre - reference, and flash detectors ----------
    u = 0
    groups = {}                      # (row, ref col) -> 4 layer-1 units
    pairs = [(r, rc, ROW_W[r]) for r in ROW_W for rc in REF_COLS]
    pairs += [(r, rc, NEAR_W) for r in NEAR_ROWS for rc in NEAR_REFS]
    weight = {}
    for r, rc, wt in pairs:
        units = []
        for d in TETRA:
            for k in range(3):
                for c in CENTER:
                    W1[idx(r, c, k), u] += d[k] / len(CENTER)
                W1[idx(r, rc, k), u] -= d[k]
            units.append(u); u += 1
        groups[(r, rc)] = units
        weight[(r, rc)] = wt
    flash_units = []
    for r in FLASH_ROWS:
        for c in FLASH_COLS:
            for k in range(3):
                W1[idx(r, c, k), u] += FLASH_COEF[k]
                for c2 in range(GW):
                    W1[idx(r, c2, k), u] -= FLASH_COEF[k] / GW
            b1[u] = -FLASH_THETA
            flash_units.append(u); u += 1
    assert u <= H, u

    # ---------- layer 2: 0/1 votes "centre differs from strip", and a 0/1 flash flag ----------
    v = 0
    votes = {}                       # key -> (p, q); vote = (p - q) / EPS, clipped to [0, 1]
    for key, units in groups.items():
        p, q = v, v + 1
        for uu in units:
            W2[uu, p] = W2[uu, q] = 1.0 / 3.0
        b2[p] = -T0
        b2[q] = -(T0 + EPS)
        votes[key] = (p, q); v += 2
    K = 20.0
    fp, fq = v, v + 1
    for uu in flash_units:
        W2[uu, fp] = W2[uu, fq] = K
    b2[fq] = -1.0
    v += 2
    assert v <= H, v

    # ---------- layer 3: action scores ----------
    M = 10.0
    for (r, rc), (p, q) in votes.items():
        w = weight[(r, rc)] * M / EPS
        W3[p, 2] += w
        W3[q, 2] -= w
    b3[2] = -M * ATTACK_THRESH
    W3[fp, 2] -= 50.0 * M          # flash flag vetoes ATTACK
    W3[fq, 2] += 50.0 * M
    b3[0] = 0.0                    # TURN_LEFT: default (also wins ties)
    b3[1] = -1.0                   # TURN_RIGHT: never used

    out = dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    return {k: x.astype(np.float32) for k, x in out.items()}


if __name__ == "__main__":
    np.savez("weights.npz", **build())
