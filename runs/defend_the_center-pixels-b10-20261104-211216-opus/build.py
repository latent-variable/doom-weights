"""Hand-written weights for defend_the_center, pixels mode.

Layer 1: 40 per-column "monster here" detectors, plus gun-recoil, muzzle-flash and out-of-ammo units.
Layer 2: left mass, right mass, crosshair-covered mass, near-miss mass (brake), recoil veto.
Output: turn toward the heavier side, shoot when the crosshair is covered, never during recoil.
See explanation.md.
"""
import numpy as np

D, H, A = 3602, 128, 3
GW, GH = 40, 30
ROWS = range(9, 15)            # wall band above eye level (y 72..119): no floor, no sky, no corpses
# Per-cell colour feature: zero-ish on the brown wall, positive on grey marines and pink demons.
# wall ~ (.43,.34,.27): b-g = -.075, r-g = +.09 ; monsters have b ~ g.
COEF = np.array([0.3, -1.3, 1.0])  # r, g, b
SCALE = 10.0                   # column feature units: 1.0 = 0.1 of summed colour feature
THRESH = 0.2                   # = 0.02 raw; wall noise is ~0.01
CL, CR = 19, 20                # the two columns either side of the crosshair (x = 160)
K_FIRE = 100.0
NEAR = [15, 16, 17, 18, 21, 22, 23, 24]
BRAKE_MASS = 3.0
K_BRAKE = 20.0
SCAN_BIAS = 1.0
BUSY_SCALE = 10.0
K_SUPPRESS = 5000.0


def idx(r, c, k):
    return (r * GW + c) * 3 + k


def build():
    W1 = np.zeros((D, H)); b1 = np.zeros(H)
    W2 = np.zeros((H, H)); b2 = np.zeros(H)
    W3 = np.zeros((H, A)); b3 = np.zeros(A)

    # ---- layer 1: s_c = relu(SCALE*(F_c - mean_c' F_c') - THRESH), F_c = sum over band rows of COEF . rgb
    for c in range(GW):
        for r in ROWS:
            for cc in range(GW):
                for k in range(3):
                    W1[idx(r, cc, k), c] = SCALE * COEF[k] * ((1.0 if cc == c else 0.0) - 1.0 / GW)
        b1[c] = -THRESH

    # ---- layer 2
    # weights favour columns near the centre, so the target being turned to stays the preferred one
    def w(c):
        return 1.0 + 1.0 * (1.0 - abs(c - 19.5) / 20.0)
    for c in range(GW):
        if c <= CL:
            W2[c, 0] = w(c)       # left mass
        else:
            W2[c, 1] = w(c)       # right mass
    W2[CL, 2] = 1.0; W2[CR, 2] = 1.0     # C: monster mass on the crosshair columns
    for c in NEAR:                         # N: big monster just beside the crosshair
        W2[c, 3] = 1.0
    b2[3] = -BRAKE_MASS

    # ---- layer 1, gun state (the pistol sprite is drawn over everything, so these cells only see the gun)
    # Only cells the hand/gun sprite always covers fully.  Z = Y - R + 1.2 D, all in luminance r+g+b:
    #   Y = cells the hand moves into when the gun kicks (29,21) (29,22) (25,19) (25,20)
    #   R = cells that look the same in every gun frame  (26,19) (26,20) (27,19) (27,20)
    #   D = lum(26,20) - lum(28,17): a bright-minus-dark gun contrast, sets the threshold
    # Every term is a difference of equal weights, so a red pain tint / flash light (c -> (1-a)c + a*tint)
    # only scales Z by (1-a) and never flips its sign.  ready ~ -0.38, recoil ~ +0.35, fired ~ +1.
    U_BUSY, U_FLASH, U_EMPTY = 40, 41, 42
    terms = [((29, 21), 1), ((29, 22), 1), ((25, 19), 1), ((25, 20), 1),
             ((26, 19), -1), ((26, 20), -1), ((27, 19), -1), ((27, 20), -1),
             ((26, 20), 1.2), ((28, 17), -1.2)]
    for (r, c), s in terms:
        for k in range(3):
            W1[idx(r, c, k), U_BUSY] += BUSY_SCALE * s
    b1[U_BUSY] = 0.0
    # muzzle flash: two cells above the barrel turn bright yellow (lum ~2.6 each, normally < 2)
    for k in range(3):
        W1[idx(17, 19, k), U_FLASH] = 1.0
        W1[idx(17, 20, k), U_FLASH] = 1.0
    b1[U_FLASH] = -3.5
    # out of ammo: x[3600] = ammo / 26  ->  relu(1 - ammo) is 1 only when empty
    W1[3600, U_EMPTY] = -26.0; b1[U_EMPTY] = 1.0
    # h2[4]: gun recoiling (a press now would trigger A_ReFire and a second, wasted shot); not while the flash
    # is up (pressing then does nothing, so attack is a free brake) and not with fists (ammo gone)
    W2[U_BUSY, 4] = 1.0; W2[U_FLASH, 4] = -30.0; W2[U_EMPTY, 4] = -1000.0

    # ---- output
    W3[0, 0] = 1.0                # TURN_LEFT  <- left mass
    W3[1, 1] = 1.0                # TURN_RIGHT <- right mass
    W3[2, 2] = K_FIRE             # ATTACK <- crosshair covered
    W3[3, 2] = K_BRAKE            # ATTACK as a brake: resets the turn acceleration
    W3[4, 2] = -K_SUPPRESS        # ...but not while the gun is recoiling
    b3[:] = [SCAN_BIAS, 0.0, -0.001]  # nothing (big) seen on the right: keep turning left (scan)
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    w = build()
    np.savez("weights.npz", **{k: v.astype(np.float32) for k, v in w.items()})
