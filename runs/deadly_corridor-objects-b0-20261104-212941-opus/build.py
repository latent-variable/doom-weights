"""Hand-written weights for deadly_corridor, objects mode.  See explanation.md.

Layer 1: pairs of ramps relu(k(f-t)) and relu(k(f-t)-1) for a handful of linear
         features f of the screen (monster mass in column bands, depth balance,
         pickup balance, health).
Layer 2: their difference = a clamped 0..1 step per feature, plus a few ANDs.
Layer 3: a fixed priority list: each action's score is a weighted sum of steps,
         with weights spaced so that the higher-priority rule always wins argmax.
"""
import numpy as np

GW, GH = 40, 30
D = GW * GH * 3 + 1          # 3601
HP = GW * GH * 3              # index of HEALTH/100
H = 128
FWD, MLEFT, MRIGHT, TLEFT, TRIGHT, ATK, FWDATK = range(7)


def idx(r, c, k):
    return (r * GW + c) * 3 + k


def cells(rows, cols, k, weight=1.0):
    """A weight vector over x summing channel k on the given rows x cols."""
    v = np.zeros(D, np.float64)
    for r in rows:
        for c in cols:
            v[idx(r, c, k)] += weight if np.isscalar(weight) else weight[c]
    return v


# ---- the features -----------------------------------------------------------
# Rows 0..12 lie above y = 104, the horizon when the Doom status bar is drawn
# (view 320 x 208).  A standing zombie (56 high, eye at 41) always pokes above
# the horizon; a corpse never does, so only "upper" monster pixels count as a
# live target.
UP = range(0, 13)
LOWBAND = range(13, 15)
NAVROWS = range(9, 18)
PICKROWS = range(11, 16)
MON, PICK, DEP = 0, 1, 2

feats = {}   # name -> (vector over x, bias-free), threshold t, ramp width
def feat(name, vec, t, width):
    feats[name] = (vec, t, width)

# Fire: a live monster under the crosshair (columns 19..20 = +-2.9 degrees).
feat('ON', cells(UP, range(19, 21), MON), 0.04, 0.04)
# Fine aim: monster just beside the crosshair -> strafe toward it.
feat('STR_L', cells(UP, range(15, 19), MON) - cells(UP, range(21, 25), MON), 0.04, 0.04)
feat('STR_R', cells(UP, range(21, 25), MON) - cells(UP, range(15, 19), MON), 0.04, 0.04)
# Finer aim, outranking the shot: the monster overlaps the crosshair but more
# of it lies in one neighbouring pair of columns than in the centre pair.
CEN = cells(UP, range(19, 21), MON)
feat('FINE_L', cells(UP, range(17, 19), MON) - cells(UP, range(21, 23), MON) - CEN, 0.03, 0.03)
feat('FINE_R', cells(UP, range(21, 23), MON) - cells(UP, range(17, 19), MON) - CEN, 0.03, 0.03)
# Coarse aim: monster mass farther out -> turn toward the heavier side.
feat('OUT_L', cells(UP, range(0, 15), MON) - cells(UP, range(25, 40), MON), 0.04, 0.04)
feat('OUT_R', cells(UP, range(25, 40), MON) - cells(UP, range(0, 15), MON), 0.04, 0.04)
# Something monster-like near the crosshair just below the horizon line
# (a far monster if there is no status bar, else maybe a corpse): walk and shoot.
feat('LO_C', cells(LOWBAND, range(18, 22), MON), 0.08, 0.08)
# Corridor heading from depth: left-right balance relative to total depth.
# sum_c w_c d_c > alpha * sum_c d_c  <=>  sum_c (w_c - alpha) d_c > 0
w = (19.5 - np.arange(GW)) / 19.5
ALPHA = 0.10
n = len(NAVROWS)
feat('NAV_L', cells(NAVROWS, range(GW), DEP, (w - ALPHA) / n), 0.0, 0.02)
feat('NAV_R', cells(NAVROWS, range(GW), DEP, (-w - ALPHA) / n), 0.0, 0.02)
# The armour vest (or any far pickup near the horizon) off to one side.
feat('PICK_L', cells(PICKROWS, range(4, 17), PICK) - cells(PICKROWS, range(23, 36), PICK), 0.03, 0.03)
feat('PICK_R', cells(PICKROWS, range(23, 36), PICK) - cells(PICKROWS, range(4, 17), PICK), 0.03, 0.03)
# Low health: stop fighting, run for the vest.
v = np.zeros(D); v[HP] = -1.0
feat('LOW', v, -0.30, 0.02)          # f = -health/100 > -0.30  <=> health < 30

names = list(feats)
F = len(names)

W1 = np.zeros((D, H)); b1 = np.zeros(H)
for i, nm in enumerate(names):
    vec, t, width = feats[nm]
    k = 1.0 / width
    W1[:, 2 * i] = k * vec;     b1[2 * i] = -k * t           # ramp a
    W1[:, 2 * i + 1] = k * vec; b1[2 * i + 1] = -k * t - 1.0  # ramp b = a - 1

W2 = np.zeros((H, H)); b2 = np.zeros(H)
step = {}
for i, nm in enumerate(names):
    W2[2 * i, i] = 1.0; W2[2 * i + 1, i] = -1.0           # clamp(k(f-t), 0, 1)
    step[nm] = i
ands = {}
j = F
for a, b in [('LOW', 'NAV_L'), ('LOW', 'NAV_R'), ('LOW', 'PICK_L'), ('LOW', 'PICK_R')]:
    for nm in (a, b):
        i = names.index(nm)
        W2[2 * i, j] += 1.0; W2[2 * i + 1, j] -= 1.0
    b2[j] = -1.0
    ands[(a, b)] = j
    j += 1

W3 = np.zeros((H, 7)); b3 = np.zeros(7)
b3[FWD] = 1.0                               # default: walk down the corridor
def put(unit, action, value):
    W3[unit, action] += value
put(step['LO_C'], FWDATK, 1.2)
put(step['NAV_L'], TLEFT, 1.5);  put(step['NAV_R'], TRIGHT, 1.5)
put(step['PICK_L'], TLEFT, 2.0); put(step['PICK_R'], TRIGHT, 2.0)
put(step['OUT_L'], TLEFT, 5.0);  put(step['OUT_R'], TRIGHT, 5.0)
put(step['STR_L'], MLEFT, 10.0); put(step['STR_R'], MRIGHT, 10.0)
put(step['ON'], ATK, 20.0)
put(step['FINE_L'], MLEFT, 25.0); put(step['FINE_R'], MRIGHT, 25.0)
# Low health beats every fighting rule (max fighting score is 25 + 10 = 35),
# but still yields to turning back toward the corridor or the vest.
put(step['LOW'], FWDATK, 50.0)
put(ands[('LOW', 'NAV_L')], TLEFT, 70.0);  put(ands[('LOW', 'NAV_R')], TRIGHT, 70.0)
put(ands[('LOW', 'PICK_L')], TLEFT, 70.0); put(ands[('LOW', 'PICK_R')], TRIGHT, 70.0)

out = {k: v.astype(np.float32) for k, v in
       dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3).items()}
np.savez('weights.npz', **out)
