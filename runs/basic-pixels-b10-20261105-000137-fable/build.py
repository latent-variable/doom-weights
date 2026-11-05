"""Hand-written weights for the ViZDoom `basic` scenario, pixels mode.

Nothing here is trained or fitted: every number is derived from the game's geometry and
physics (see explanation.md).  Run: python build.py  ->  weights.npz
"""
import numpy as np

D, H, A = 3601, 128, 3
ROWS, COLS = 30, 40
CENTER = 19.5                 # crosshair (x = 160 px) in cell-centre column units
UNITS_PER_COL = 19.2          # 8 px at the monster's distance: 384 units away, focal length 160 px

# ---------------------------------------------------------------------------------------
# Physics used to place the shooting window (ZDoom walking strafe + pistol timing).
# Each tic: v += 0.75; x += v; v *= 0.90625.  ATTACK chosen at a decision fires the bullet
# 4 tics later, and during those 4 tics the player keeps sliding with no thrust.
# ---------------------------------------------------------------------------------------
FRICTION, ACCEL = 0.90625, 0.75
HIT_HALF = 31.0               # Cacodemon radius: a shot hits if it is within 31 units of its centre
MARGIN = 5.0                  # safety margin kept inside the hit box (about 2 px)
N_BINS = 7                    # decisions 0..6 of an uninterrupted strafe (later ones reuse the last)


def strafe_profile(n):
    """(position, velocity) at each decision while strafing from rest in one direction."""
    out, p, v = [], 0.0, 0.0
    for _ in range(n):
        out.append((p, v))
        for _ in range(4):
            v += ACCEL; p += v; v *= FRICTION
    return out


def slide4(v):
    return v * (1 + FRICTION + FRICTION ** 2 + FRICTION ** 3)


def wall_feature(p, eye=0.0):
    """Value of the layer-1 position cue when the player is p units right of the start.

    Above the far wall's top edge (rows 0..10) the screen shows gray ceiling (R = B) and
    the two brown side walls (R - B = 0.14).  Each side wall's top edge is a straight line
    through the screen centre whose slope depends on the distance to that wall, so the
    brown area right minus the brown area left grows steadily as the player moves right.
    `eye` is the view-bob offset of the eye (units); the wall top is 63 units above it.
    """
    ys = np.arange(240) + 0.5
    xs = np.arange(320) + 0.5
    e = (120.0 - ys)[:, None]                     # px above the horizon
    dx = (xs - 160.0)[None, :]
    top = 75.6 * (63.0 - eye) / 63.0              # wall-top height above the eye, in px-units
    left = (-dx) > e * (224.0 + p) / top          # room half-width 224
    right = dx > e * (224.0 - p) / top
    img = ((left | right) & (e > 0)).astype(np.float64)
    cells = img.reshape(30, 8, 40, 8).mean(axis=(1, 3)) * 0.14
    return cells[0:11, 20:].sum() - cells[0:11, :20].sum()


def view_bob(n, v):
    """Eye-height offset at decision n: ZDoom bobs the view by (speed^2 / 8) * sin(2 pi t / 20),
    and decision n is always at level time 14 + 4 n."""
    speed = v / FRICTION if n else 0.0
    return 0.5 * min(16.0, 0.25 * speed * speed) * np.sin(2 * np.pi * (14 + 4 * n) / 20.0)


profile = strafe_profile(N_BINS)
T_BIN = [(HIT_HALF - MARGIN + slide4(v)) / UNITS_PER_COL for _, v in profile]     # in columns
D_AT = [wall_feature(p, view_bob(n, v)) for n, (p, v) in enumerate(profile)]
# boundary between bin n-1 and bin n: halfway (geometrically, except the first) between the
# cue values the two decisions produce
D_EDGE = [0.5 * D_AT[1]] + [float(np.sqrt(D_AT[n - 1] * D_AT[n])) for n in range(2, N_BINS)]

W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)


def cell(r, c):
    return (r * COLS + c) * 3


# ---------------------------------------------------------------------------------------
# Layer 1a: monster-colour detectors.
# The monster is blue and red; walls and floor are brown (R : G : B about 1 : .73 : .5),
# the ceiling gray, the hand tan.  R + B - 2.1 G is <= 0 on all of those and > 0 on monster
# cells.  The monster always stands in rows 13..17 (it never changes distance or height).
# ---------------------------------------------------------------------------------------
THETA = 0.01
INNER = range(8, 32)          # columns the monster can occupy while we approach it


def add_monster(unit, r, c):
    i = cell(r, c)
    W1[i, unit] += 1.0; W1[i + 1, unit] += -2.1; W1[i + 2, unit] += 1.0


col_of = {}
u = 0
for c in INNER:                                       # one unit per cell (rows 13 + 17 share one)
    for rows in ([14], [15], [16], [13, 17]):
        for r in rows:
            add_monster(u, r, c)
        b1[u] = -THETA
        col_of[u] = c
        u += 1
for c in list(range(0, 8)) + list(range(32, 40)):     # far left / right: one unit per column
    for r in range(13, 18):
        add_monster(u, r, c)
    b1[u] = -THETA
    col_of[u] = c
    u += 1

# ---------------------------------------------------------------------------------------
# Layer 1b: "how far have I strafed" gates from the side walls.
# cue = sum(R - B) over rows 0..10, right half minus left half.
# gate_right[k] = relu(cue - D_EDGE[k]) is on once the player is past decision k going right,
# gate_left[k]  = relu(-cue - D_EDGE[k]) likewise going left.
# ---------------------------------------------------------------------------------------
gate_right, gate_left = [], []
for k in range(N_BINS - 1):
    for sign, store in ((1.0, gate_right), (-1.0, gate_left)):
        for r in range(0, 11):
            for c in range(COLS):
                s = sign * (1.0 if c >= 20 else -1.0)
                W1[cell(r, c), u] = s          # +R
                W1[cell(r, c) + 2, u] = -s     # -B
        b1[u] = -D_EDGE[k]
        store.append(u)
        u += 1
# Layer 1c: "I have not fired yet".  The last input is ammo / 50: 1.0 before the first shot,
# 0.98 or less afterwards.  The unit is 0.01 with full ammo and exactly 0 once a shot is gone.
FRESH = u
W1[D - 1, FRESH] = 1.0
b1[FRESH] = -0.99
u += 1
assert u <= H, u

# ---------------------------------------------------------------------------------------
# Layer 2: "the monster is outside the shooting window" units, one per speed bin and side.
# S = sum of detections, M = sum of detections * (column - centre)  ->  M / S = offset.
#   right[n] = relu( M - T_n * S - BIG * gate_right[n])   (on: monster right of window n)
#   left[n]  = relu(-M - T_n * S - BIG * gate_left[n])
# Window n is switched off once the player is past decision n, so the widest window still
# switched on is the one for the current speed.  T grows with speed because the player
# slides further toward the monster before the bullet leaves.
#   vis = S
# ---------------------------------------------------------------------------------------
BIG = 2000.0
right_units = list(range(0, N_BINS))
left_units = list(range(N_BINS, 2 * N_BINS))
VIS = 2 * N_BINS
for n in range(N_BINS):
    for k, c in col_of.items():
        W2[k, right_units[n]] = (c - CENTER) - T_BIN[n]
        W2[k, left_units[n]] = -(c - CENTER) - T_BIN[n]
    if n < N_BINS - 1:
        W2[gate_right[n], right_units[n]] = -BIG
        W2[gate_left[n], left_units[n]] = -BIG
# Fallback after a miss: if a shot has been fired and the game is still running, that shot
# missed.  From then on a narrow window (T_MISS) is also switched on, so the player first
# re-centres on the monster instead of repeating a shot from a spot that just failed.
T_MISS = 14.0 / UNITS_PER_COL
RIGHT_MISS, LEFT_MISS = VIS + 1, VIS + 2
for k, c in col_of.items():
    W2[k, RIGHT_MISS] = (c - CENTER) - T_MISS
    W2[k, LEFT_MISS] = -(c - CENTER) - T_MISS
W2[FRESH, RIGHT_MISS] = -1e5          # held off (by -1000) while ammo is full
W2[FRESH, LEFT_MISS] = -1e5
for k in col_of:
    W2[k, VIS] = 1.0

# ---------------------------------------------------------------------------------------
# Layer 3: strafe toward the monster while it is outside the window, otherwise shoot.
# ---------------------------------------------------------------------------------------
for n in range(N_BINS):
    W3[left_units[n], 0] = 1000.0      # MOVE_LEFT
    W3[right_units[n], 1] = 1000.0     # MOVE_RIGHT
W3[LEFT_MISS, 0] = 1000.0
W3[RIGHT_MISS, 1] = 1000.0
W3[VIS, 2] = 1.0                       # ATTACK: wins only when no move unit is on

np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

if __name__ == "__main__":
    print("T (units):", np.round(np.array(T_BIN) * UNITS_PER_COL, 1))
    print("cue at decisions:", np.round(D_AT, 2))
    print("bin edges:", np.round(D_EDGE, 2))
