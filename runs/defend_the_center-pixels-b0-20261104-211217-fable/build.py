"""Hand-written weights for defend_the_center, pixels mode.

No learning: every number below is chosen by reasoning (see explanation.md).

The policy, in words
--------------------
The player stands in the exact centre of a round room, so in any screen row the
*background* looks the same in every column.  A monster is a place where a row's
colour departs from the colour of the same row a few columns away.

Only screen rows 12 and 13 are read (y = 96..111), averaged together.  With the
status bar on, the horizon is at y = 104, so these rows hold the head and chest
of every standing monster and nothing else that changes: no corpse on the floor,
no gun, no muzzle flash, no status bar.

  ATTACK  when both columns that touch the crosshair (19 and 20) are "monster".
  LEFT    when a monster sits just left of the centre (the coarse right turn
          stepped over it): step back onto it.
  RIGHT   otherwise: keep sweeping.

Layer 1: one unit per *condition*; it outputs how far the condition is violated.
Layer 2: one unit per *detector* = AND of its conditions.
Layer 3: OR of the detectors, and the priority ATTACK > LEFT > RIGHT.
"""
import os

import numpy as np

GRID_W, GRID_H = 40, 30
D = GRID_W * GRID_H * 3 + 2          # 3602: grid, then AMMO2/26, HEALTH/100
H = 128
I_AMMO, I_HEALTH = D - 2, D - 1

# ---- design constants (chosen, not fitted) --------------------------------
ROWS = (12, 13)                       # y 96..111: just above / just below the horizon
ZONE_ROWS = (11, 12)                  # y 88..103: only a monster nearer than ~300 units fills these
BIG_ROWS = (10, 11)                   # y 80..95: only a monster nearer than ~200 units fills these

# colour projections; a monster must differ from the wall in at least one of
# them, with one sign, at every place a detector looks
LUM, RG, GB = (1.0, 1.0, 1.0), (1.0, -1.0, 0.0), (0.0, 1.0, -1.0)

C19, C20 = [19], [20]                 # the two columns that touch the crosshair
NEAR_L, NEAR_R = [11, 12, 13, 14], [25, 26, 27, 28]     # background beside a small target
FAR_L, FAR_R = [0, 1, 2, 3, 4, 5], [34, 35, 36, 37, 38, 39]   # background beside a huge target
ZONE = [15, 16, 17, 18]               # "just stepped over" zone, left of the crosshair
ZREF_L, ZREF_R = [10, 11, 12, 13], [21, 22, 23, 24]

RHO = 0.2      # the less covered crosshair column must show > 20% of the other's contrast

# thresholds: (absolute, fraction of the reference's brightness R+G+B)
# hypothesis key: (projection, sign)
SMALL_T = {("lum", +1): (0.010, 0.07), ("lum", -1): (0.010, 0.12), "chroma": (0.012, 0.025)}
BIG_T = {("lum", +1): (0.020, 0.10), ("lum", -1): (0.020, 0.20), "chroma": (0.015, 0.035)}
ZONE_T = {("lum", +1): (0.015, 0.10), ("lum", -1): (0.015, 0.10), "chroma": (0.012, 0.025)}
AMMO_RATCHET = {"lum": 0.08, "chroma": 0.03}     # fire threshold grows as bullets are spent
HEALTH_RATCHET = {"lum": 0.08, "chroma": 0.03}   # step-back threshold grows as health is lost

K_AND = 200.0                         # sharpness of the AND
Q_ATT, Q_LEFT, Q_RIGHT = 100.0, 1.0, 0.5


def idx(r, c, k):
    return (r * GRID_W + c) * 3 + k


def build():
    W1 = np.zeros((D, H)); b1 = np.zeros(H)
    W2 = np.zeros((H, H)); b2 = np.zeros(H)
    W3 = np.zeros((H, 3)); b3 = np.zeros(3)

    def add_mean(u, cols, proj, scale, rows):
        """W1[:, u] += scale * proj . (mean colour of `cols` over `rows`)"""
        for r in rows:
            for c in cols:
                for k in range(3):
                    W1[idx(r, c, k), u] += scale * proj[k] / (len(cols) * len(rows))

    n1 = 0   # next free layer-1 unit
    n2 = 0   # next free layer-2 unit

    def condition(proj, sign, a, b, ref, thr, ammo=0.0, health=0.0, rows=ROWS):
        """Layer-1 unit: relu( threshold - sign*proj.[(a - ref) - RHO*(b - ref)] ).
        Zero when the condition holds, positive by the amount it fails.
        threshold = thr[0] + thr[1]*brightness(ref) + ammo*(1 - ammo/26) + health*(1 - health/100)
        """
        nonlocal n1
        u = n1; n1 += 1
        add_mean(u, a, proj, -sign, rows)
        rho = 0.0
        if b is not None:
            add_mean(u, b, proj, +sign * RHO, rows)
            rho = RHO
        add_mean(u, ref, proj, +sign * (1.0 - rho), rows)
        add_mean(u, ref, LUM, thr[1], rows)
        b1[u] = thr[0] + ammo + health
        W1[I_AMMO, u] = -ammo
        W1[I_HEALTH, u] = -health
        return u

    def detector(conds, out_action, out_weight):
        """Layer-2 unit: relu(1 - K * sum(violations)) = 1 iff every condition holds."""
        nonlocal n2
        v = n2; n2 += 1
        for u in conds:
            W2[u, v] = -K_AND
        b2[v] = 1.0
        W3[v, out_action] = out_weight

    for pname, proj in (("lum", LUM), ("rg", RG), ("gb", GB)):
        kind = "lum" if pname == "lum" else "chroma"
        for sign in (+1, -1):
            key = ("lum", sign) if kind == "lum" else "chroma"
            ar, hr = AMMO_RATCHET[kind], HEALTH_RATCHET[kind]
            # small target on the crosshair: both columns differ from the wall just to the
            # left AND just to the right ...
            conds = [condition(proj, sign, a, b, ref, SMALL_T[key], ammo=ar)
                     for (a, b) in ((C19, C20), (C20, C19)) for ref in (NEAR_L, NEAR_R)]
            # ... AND from the wall at one screen edge at least.  (Without this, two
            # monsters standing on the two near references make the empty wall between
            # them look like a target.)
            edge = [condition(proj, sign, C19 + C20, None, far, SMALL_T[key], ammo=ar)
                    for far in (FAR_L, FAR_R)]
            for e in edge:
                detector(conds + [e], 2, Q_ATT)
            # huge target on the crosshair (it hides the near references): compare with both
            # screen edges, two rows higher, where only a monster nearer than ~200 units
            # reaches.  The wall's top edge is curved on screen (higher at the screen edges),
            # so rows 10-11 alone could differ between centre and edge in an empty room with
            # a low wall; rows 12-13 cannot, so they must agree too (`edge`).
            conds = [condition(proj, sign, a, b, ref, BIG_T[key], ammo=ar, rows=BIG_ROWS)
                     for (a, b) in ((C19, C20), (C20, C19)) for ref in (FAR_L, FAR_R)]
            detector(conds + edge, 2, Q_ATT)
            # something just left of the crosshair
            conds = [condition(proj, sign, ZONE, None, ref, ZONE_T[key], health=hr, rows=ZONE_ROWS)
                     for ref in (ZREF_L, ZREF_R)]
            detector(conds, 0, Q_LEFT)

    assert n1 <= H and n2 <= H
    b3[1] = Q_RIGHT                  # default: sweep right
    return {k: v.astype(np.float32) for k, v in
            dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3).items()}


if __name__ == "__main__":
    w = build()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")
    np.savez(out, **w)
    print("wrote", out, {k: v.shape for k, v in w.items()})
