"""Hand-written weights for ViZDoom `basic`, pixels mode.  Run: python build.py -> weights.npz

Nothing here is learned or fitted: every number below is set by the reasoning in the comments
(and in explanation.md).
"""
import numpy as np

GW, GH, D, H, A = 40, 30, 3601, 128, 3

# Rows searched for the monster. The bottom rows hold the player's pistol and hand (flesh tones
# are reddish), so they are left out; the monster floats near eye level, i.e. mid-screen.
ROWS = list(range(2, 23))
# Three overlapping horizontal bands; a ~4-row-tall monster always lies mostly inside one.
BANDS = [list(range(2, 10)), list(range(8, 16)), list(range(14, 23))]
REDNESS = np.array([2.0, -1.0, -1.0])   # 2R - G - B: 0 for any gray, ~1 for saturated red
THETA = 0.35      # band redness contrast that still counts as background
FIRE_AT = 0.15   # crosshair coverage (in redness units above THETA) needed to shoot
FIRE_GAIN = 1000.0


def build():
    W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

    # ---- Layer 1: 120 units, one per (band b, column c) -------------------------------------
    # f(r,c) = 2R - G - B of a cell. S_cb = sum of f over band b's rows in column c,
    # T_c = sum of f over all ROWS in column c.
    #   E_cb = S_cb - (n_b / n) T_c              (vertical surround: a hue that tints a whole
    #                                             column evenly, e.g. a wall stripe, cancels)
    #   E_cb -= mean over columns of E_.b         (horizontal surround: floor / ceiling / wall
    #                                             tints that span the screen cancel)
    #   h1[b*40 + c] = relu(E_cb - THETA)         "red blob in band b of column c"
    n = len(ROWS)
    for b, band in enumerate(BANDS):
        rw = np.array([(1.0 if r in band else 0.0) - len(band) / n for r in ROWS])  # per row
        for c in range(GW):
            unit = b * GW + c
            cw = np.full(GW, -1.0 / GW); cw[c] += 1.0                               # per column
            for ri, r in enumerate(ROWS):
                for cc in range(GW):
                    base = (r * GW + cc) * 3
                    W1[base:base + 3, unit] = rw[ri] * cw[cc] * REDNESS
            b1[unit] = -THETA
    # (units 120..127 unused)

    # ---- Layer 2 ----------------------------------------------------------------------------
    # P_c = sum_b h1[b, c]  (red evidence in column c).  The crosshair is at x = 160, the
    # boundary between columns 19 and 20.
    #   m = sum_c P_c (19.5 - c)       signed moment, > 0 when the red mass is left of centre
    #   unit 0 = relu(m), unit 1 = relu(-m)
    #   unit 2 = relu(P_19), unit 3 = relu(P_19 - P_20)  ->  unit2 - unit3 = min(P_19, P_20):
    #   red on BOTH sides of the crosshair, i.e. the crosshair is on the monster.
    for b in range(len(BANDS)):
        for c in range(GW):
            j = b * GW + c
            W2[j, 0] = 19.5 - c
            W2[j, 1] = -(19.5 - c)
            if c == 19:
                W2[j, 2] = 1.0
                W2[j, 3] = 1.0
            if c == 20:
                W2[j, 3] = -1.0

    # ---- Output -----------------------------------------------------------------------------
    # q_left = relu(m), q_right = relu(-m): strafe toward the red mass (strafing left moves
    # the view so that things on the left come toward the centre).
    # q_attack = FIRE_GAIN * (min(P_19, P_20) - FIRE_AT): once the crosshair is solidly on red
    # this beats any move score (|m| stays below ~100); otherwise it is negative.
    # Nothing red anywhere: all moves 0, attack < 0 -> MOVE_LEFT; no ammo is wasted.
    W3[0, 0] = 1.0
    W3[1, 1] = 1.0
    W3[2, 2] = FIRE_GAIN
    W3[3, 2] = -FIRE_GAIN
    b3[2] = -FIRE_GAIN * FIRE_AT
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    np.savez("weights.npz", **build())
