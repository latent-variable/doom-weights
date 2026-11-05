"""Hand-written weights for the ViZDoom `basic` scenario, `objects` input mode.

No training of any kind: every number below is chosen by reasoning about the game
(see explanation.md). Running this file writes weights.npz.

Policy in words
---------------
  * Strafe toward the monster (toward the side of the screen that holds more of it).
  * Press ATTACK when the vertical centre line of the screen (where the bullet goes)
    is inside the monster's silhouette with a small margin.
  * While the player is clearly away from its spawn point (so it must be sliding
    toward the monster at speed), press ATTACK a little *before* the silhouette
    reaches the centre line: the pistol fires 4 tics after the press and the player
    keeps coasting in the meantime.
  * After a miss (ammo below 50) drop the early shots and use a plain, wide
    "centre line inside the silhouette" rule; after five misses, a tighter one.
"""
import numpy as np

ROWS, COLS, CH = 30, 40, 3
D_IN, HID, N_ACT = ROWS * COLS * CH + 1, 128, 3
AMMO = ROWS * COLS * CH                      # index of AMMO2 / 50
LEFT, RIGHT, ATTACK = 0, 1, 2

# ---- numbers chosen by hand -------------------------------------------------------
F0 = 0.05          # fresh rule: >= 5% of the monster's area on each side of the centre line
F1 = 0.03          # after 1..4 misses: a WIDE zone. A miss almost always means "arrived too
                   # fast"; a wide zone catches the player near a turning point, where it is slow.
F2 = 0.15          # after 5 or more misses (and always, as a subset): tighter, last resort
EPS = 0.02         # cells; keeps "no monster" from counting as aligned

# Lead levels. (rho threshold, lead in game units). rho is the left/right depth asymmetry
# of the walls, a stand-in for "how far has the player slid from its spawn point", which
# in turn bounds its speed from below. Thresholds correspond to roughly 12, 22, 42, 66, 100,
# 150 units of travel if the far wall is ~416 units away (later if the room is shallower,
# which is the safe direction). Each lead is the coasting distance during the pistol's
# 4-tic delay at ~1/1.6 of that travel, minus 4 units (the top levels shaved a little more:
# they are the ones with least slack if the sprite is boxier or wider than I assume).
LEADS = [(0.036, 4.0), (0.066, 7.0), (0.127, 10.0), (0.210, 12.0), (0.345, 13.0), (0.576, 14.0)]
MONSTER_W = 62.0   # the Cacodemon's hit box is 62 units wide = its sprite width
K_SHAPE = 1.06     # sprite width (px) ~= K_SHAPE * sqrt(sprite area in px)
KAPPA_FAR = 0.25   # lead rules stop applying once the centroid is this far past the centre

DEPTH_ROWS = (13, 14)                        # rows that show walls with or without a status bar
DEPTH_L, DEPTH_R = range(0, 13), range(27, 40)   # outer columns only: the monster is in the middle
Z_MIN = 0.25       # sum of the 64 depth cells below this = depth channel carries nothing

B_GATE = 1e5       # travel gates are effectively on/off, whatever the depth scale or monster size
B_SHOT = 1e5       # switches rules off by number of shots fired
B_Z = 1e6
K_FRAC = 1000.0    # gain of the area-fraction rules on the ATTACK score
K_LEAD = 100.0     # gain of the lead rules on the ATTACK score
# ------------------------------------------------------------------------------------


def idx(r, c, k):
    return (r * COLS + c) * CH + k


def build():
    W1 = np.zeros((D_IN, HID)); b1 = np.zeros(HID)
    W2 = np.zeros((HID, HID)); b2 = np.zeros(HID)
    W3 = np.zeros((HID, N_ACT)); b3 = np.zeros(N_ACT)

    n1 = [0]

    def unit1():
        n1[0] += 1
        return n1[0] - 1

    def monster(j, col_weight, bias=0.0):
        """hidden-1 unit j = relu(sum over cells of col_weight(c) * monster share + bias)"""
        for r in range(ROWS):
            for c in range(COLS):
                W1[idx(r, c, 0), j] = col_weight(c)
        b1[j] = bias

    left = lambda c: c < COLS // 2           # columns 0..19 are left of the bullet line

    # --- layer 1: features -----------------------------------------------------------
    # L, R = monster area (in cells) left / right of the centre line, T = L + R.
    dLR = unit1(); monster(dLR, lambda c: 1.0 if left(c) else -1.0)          # relu(L - R)
    dRL = unit1(); monster(dRL, lambda c: -1.0 if left(c) else 1.0)          # relu(R - L)
    aL = []
    for f in (F0, F1, F2):                                                   # relu(L - f*T - eps)
        j = unit1(); monster(j, lambda c, f=f: (1.0 - f) if left(c) else -f, -EPS)
        aL.append(j)
    # shots fired so far = 50 * (1 - ammo/50)
    shot1 = unit1(); W1[AMMO, shot1] = -50.0; b1[shot1] = 49.5               # >= 0.5 after 1 shot
    shot5 = unit1(); W1[AMMO, shot5] = -50.0; b1[shot5] = 45.5               # >= 0.5 after 5 shots
    fresh = unit1(); W1[AMMO, fresh] = 50.0; b1[fresh] = -49.5               # 0.5 before any shot, else 0
    # first moment: (centroid x - 160 px) * T
    mom = lambda c: 8.0 * c + 4.0 - 160.0
    mP = unit1(); monster(mP, lambda c: mom(c))                              # monster right of centre
    mN = unit1(); monster(mN, lambda c: -mom(c))                             # monster left of centre
    # width_px * T = 8 * K_SHAPE * T^1.5, as a piecewise-linear function of T
    knots = [0.0, 1.0]
    while knots[-1] < 100.0:
        knots.append(knots[-1] * 1.25)
    phi = lambda t: 8.0 * K_SHAPE * t ** 1.5
    slopes = [(phi(knots[i + 1]) - phi(knots[i])) / (knots[i + 1] - knots[i]) for i in range(len(knots) - 1)]
    phi_units = []                                                           # (unit, coefficient)
    for i, t0 in enumerate(knots[:-1]):
        j = unit1(); monster(j, lambda c: 1.0, -t0)                          # relu(T - t0)
        phi_units.append((j, slopes[i] - (slopes[i - 1] if i else 0.0)))
    # travel gates from wall depth: SL, SR = summed depth of the outer left / right cells
    def depth(j, wl, wr, bias=0.0):
        for r in DEPTH_ROWS:
            for c in DEPTH_L:
                W1[idx(r, c, 2), j] = wl
            for c in DEPTH_R:
                W1[idx(r, c, 2), j] = wr
        b1[j] = bias
    gate_r, gate_l = [], []
    for rho, _ in LEADS:
        # player has moved right  <=>  right wall nearer  <=>  SL - SR >= rho * (SL + SR)
        j = unit1(); depth(j, rho - 1.0, rho + 1.0); gate_r.append(j)        # > 0 when NOT far right
        j = unit1(); depth(j, rho + 1.0, rho - 1.0); gate_l.append(j)        # > 0 when NOT far left
    zlow = unit1(); depth(zlow, -1.0, -1.0, Z_MIN)                           # > 0 when depth is blank

    # --- layer 2: rules ----------------------------------------------------------------
    n2 = [0]

    def unit2():
        n2[0] += 1
        return n2[0] - 1

    goL = unit2(); W2[dLR, goL] = 1.0
    goR = unit2(); W2[dRL, goR] = 1.0
    # min(L, R) - f*T - eps > 0, i.e. the centre line is inside the silhouette with margin f
    rules = []
    for j, off_when in zip(aL, ((shot1,), (fresh, shot5), ())):
        g = unit2()
        W2[j, g] = 1.0
        W2[dLR, g] = -1.0
        for u in off_when:
            W2[u, g] = -B_SHOT
        rules.append((g, K_FRAC))
    # lead rules, one per level and side, only before the first shot
    for (rho, lead), gr, gl in zip(LEADS, gate_r, gate_l):
        kappa = 0.5 + lead / MONSTER_W
        for near, far, gate in ((mP, mN, gr), (mN, mP, gl)):
            g = unit2()
            for j, coef in phi_units:
                W2[j, g] = kappa * coef                    # kappa * width * T
            W2[near, g] = -1.0                             # - (centroid offset) * T on the approach side
            W2[far, g] = -kappa / KAPPA_FAR
            W2[gate, g] = -B_GATE
            W2[shot1, g] = -B_SHOT
            W2[zlow, g] = -B_Z
            rules.append((g, K_LEAD))

    # --- output -------------------------------------------------------------------------
    W3[goL, LEFT] = 1.0;  W3[goR, LEFT] = -1.0
    W3[goR, RIGHT] = 1.0; W3[goL, RIGHT] = -1.0
    for g, k in rules:
        W3[g, ATTACK] = k
    b3[ATTACK] = -0.01                                     # nothing visible: do not shoot

    assert n1[0] <= HID and n2[0] <= HID
    return {k: v.astype(np.float32) for k, v in
            dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3).items()}


if __name__ == "__main__":
    np.savez("weights.npz", **build())
