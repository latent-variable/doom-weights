"""Hand-written weights for defend_the_center / pixels.  No learning: every number is chosen by reasoning
(see explanation.md).  Writes weights.npz; imports only numpy; reads no files.

Policy implemented (stateless: the pistol's animation frame on screen is the only "memory"):
  * ATTACK      when a monster is on the aim line (columns 19|20, eye-level rows 14-15) and the pistol
                is not in its recoil frames;
  * TURN_RIGHT  when a near monster's head, or a demon within ~400 units, stands just right of the line;
  * TURN_LEFT   otherwise (default scan), always while the pistol recoils (sweeps 33 degrees past the
                kill), and with priority when a very near monster is in the left half.
"""
import numpy as np

D, H, A = 3602, 128, 3
ROWS, COLS = 30, 40
R, G, B = 0, 1, 2
WALL = (0.4393, 0.3514, 0.2730)        # the wall's colour: identical in every frame seen (std 0.004 per cell)
GAIN = 100.0                           # 0.01 of colour over threshold -> unit output 1
REF_ROW, REF_COLS = 8, range(12, 28)   # always-wall cells: wall colour under the current lighting

W1 = np.zeros((D, H), np.float64); b1 = np.zeros(H, np.float64)
W2 = np.zeros((H, H), np.float64); b2 = np.zeros(H, np.float64)
W3 = np.zeros((H, A), np.float64); b3 = np.zeros(A, np.float64)
n1 = 0                                 # next free layer-1 unit
n2 = 0                                 # next free layer-2 unit


def idx(r, c, k):
    return (r * COLS + c) * 3 + k


def unit(terms, bias, gain=GAIN):
    """New layer-1 unit relu(gain * (sum(w * x[i]) + bias)). Returns its index."""
    global n1
    j = n1; n1 += 1
    for i, w in terms:
        W1[i, j] += gain * w
    b1[j] += gain * bias
    return j


def unit2(inputs, bias=0.0):
    """New layer-2 unit relu(sum(w * h1[j]) + bias). Returns its index."""
    global n2
    k = n2; n2 += 1
    for j, w in inputs:
        W2[j, k] += w
    b2[k] = bias
    return k


def below_ref(cells, ch, sign=1.0):
    """terms of sign * sum_cells(wall_ch - cell_ch), wall_ch measured on the reference cells (lighting-proof)."""
    t = [(idx(r, c, ch), -sign) for r, c in cells]
    t += [(idx(REF_ROW, c, ch), sign * len(cells) / len(REF_COLS)) for c in REF_COLS]
    return t


def below_const(cells, ch, sign=1.0):
    """(terms, bias) of sign * sum_cells(WALL_ch - cell_ch), with the wall colour as a constant (blood-proof)."""
    return [(idx(r, c, ch), -sign) for r, c in cells], sign * len(cells) * WALL[ch]


def rg(r, c, w=1.0):
    """terms of w * (R - G) of one cell (0 for gray gun metal / floor, ~.18 for skin, .20 for the red sight)."""
    return [(idx(r, c, R), w), (idx(r, c, G), -w)]


# ================================================================ layer 1
# ---- aim: monster body on the aim line (columns 19|20, rows 14-15) ---------------------------------------
# Per line column, "something of a monster is here".  Two groups: features blood squares cannot fake (always
# used) and features they can (dropped while blood flies).  The line must have a monster on BOTH columns.
solid, loose = {}, {}
for c in (19, 20):
    cells = [(14, c), (15, c)]
    t, bias = below_const(cells, R)
    tb, biasb = below_const(cells, B, -1.0)
    tr, biasr = below_const(cells, R, -1.0)
    solid[c] = [unit(t, bias - 0.10),                                  # less red than wall: marine (gray) body
                unit(tb, biasb - 0.06)]                                # bluer than wall: marine helmet, demon's pink head
    loose[c] = [unit(below_ref(cells, G), -0.07),                      # less green than wall: demon (dark red) body
                unit(below_ref(cells, R) + below_ref(cells, G), -0.05),   # darker in R+G: thin far bodies
                unit(tr, biasr - 0.06)]                                # redder than wall: marine's arm, demon's head
four = [(r, c) for r in (14, 15) for c in (19, 20)]
eight = [(r, c) for r in (14, 15) for c in (18, 19, 20, 21)]
# one strong column of dark body next to the line: near monster whose hit box (wider than the sprite) holds the line
uFM = unit(below_ref(four, G) + below_ref(four, R), -0.30)
# demon colour (less green, not less red) over >= ~4.5 px of columns 19-20: demon hit box is twice its torso
uQD = unit(below_ref(four, G) + below_ref(four, R, -1.0), -0.25)
# a big dark body filling most of columns 18-21: a near monster; more than stray blood squares can add up to
uBIG = unit(below_ref(eight, G) + below_ref(eight, R), -0.90)
# a demon within ~400 units standing right of the aim line, at most 29 degrees off: lock on to it
far_demon = [unit(below_ref([(r, c) for r in (14, 15) for c in cols], G) + below_ref([(r, c) for r in (14, 15) for c in cols], R, -1.0), -0.50)
             for cols in (range(21, 26), range(23, 29), range(26, 31))]
# most of one column of gray (marine) body right next to the line: its hit box is ~3 units wider than the sprite
t, bias = below_const(four, R)
uMS = unit(t, bias - 0.30)
# a NEAR marine with the line inside its sprite: both line columns are "not wall" in red (darker body OR
# brighter arm / chainsaw) and there is a good mass of gray body within 5 columns of the line.  Without
# the arm part a marine in melee whose arm side covers the line was never shot (and the turn rules
# then flipped left / right around it for ever).  Saturated units (0..1) so that layer 2 can AND them.
def sat_pair(terms, bias):
    return unit(terms, bias), unit(terms, bias - 0.01)        # first minus second = clip(100 * (x + bias), 0, 1)


occ = {}
for c in (19, 20):
    t, bias = below_const([(14, c), (15, c)], R)
    tn, biasn = below_const([(14, c), (15, c)], R, -1.0)
    occ[c] = sat_pair(t, bias - 0.10) + sat_pair(tn, biasn - 0.10)
mass = []                                   # two adjacent columns of gray body somewhere within 5 columns of the line
for c0 in (15, 16, 17, 21, 22, 23):         # (small blocks: a bright arm in a big block would cancel the body)
    t, bias = below_const([(r, c) for r in (14, 15) for c in (c0, c0 + 1)], R)
    mass.append(sat_pair(t, bias - 0.50))

# "blood": anything red in the sky rows = blood squares flying (we are being hit, or hit something close)
uBlood = unit([(idx(r, c, R), 1.0) for r in (0, 1, 2) for c in range(COLS)], -0.05, gain=1e5)


# ---- heads: bluer than wall = marine helmet or demon's pink head; blood (no blue) can never fake it ---------
def heads(cols, rows, th):
    t, bias = below_const([(r, c) for r in rows for c in cols], B, -1.0)
    return unit(t, bias - th)


RIGHT, LEFT = (range(21, 27), range(27, 33), range(33, 40)), (range(0, 7), range(7, 13), range(13, 19))
near_right = [heads(cols, rows, 0.05) for cols in RIGHT for rows in ((13,), (12,))]          # head top above y=112
close_right = [heads(cols, rows, th) for cols in RIGHT for rows, th in (((11,), 0.05), ((9, 10), 0.07))]
close_left = [heads(cols, rows, th) for cols in LEFT for rows, th in (((11,), 0.05), ((9, 10), 0.07))]

# ---- pistol busy (frames B and C) ---------------------------------------------------------------------------
# Differences of (R - G) between weapon cells plus G terms, no constant: brightness changes cannot flip the sign.
# frame B (hand raised): skin at (24,19),(24,20) where the idle frame and frame C show the dark slide.
uB = unit(rg(24, 19) + rg(24, 20) + rg(22, 20, -2.0) + [(idx(27, 19, G), -0.5)], 0.0)
# frame C (slide back): the red rear sight shows at (22,20). idle -.085, C +.066, B -.11, weapon still rising < 0.
uC = unit(rg(22, 20) + rg(26, 19, -1.0) + [(idx(26, 19, G), 0.389), (idx(22, 20, G), -0.5)], 0.0)

# ================================================================ layer 2
bl = (uBlood, -1.0)                                                # "- blood": switches a rule off while blood flies
a19, a20 = [(j, 1.0) for j in solid[19]], [(j, -1.0) for j in solid[20]]
t19, t20 = [(j, 1.0) for j in solid[19] + loose[19]], [(j, -1.0) for j in solid[20] + loose[20]]
MA0, MA1 = unit2(a19), unit2(a19 + a20)                            # min(solid19, solid20) = MA0 - MA1
MG0, MG1 = unit2(t19 + [bl]), unit2(t19 + t20 + [bl])              # min(all19, all20), off while blood flies
FM0 = unit2([(uFM, 1.0), bl])
QD0 = unit2([(uQD, 1.0), bl])
BG0 = unit2([(uBIG, 1.0)])
MS0 = unit2([(uMS, 1.0)])
sgn = (1.0, -1.0, 1.0, -1.0)
CB0 = unit2([(j, w) for c in (19, 20) for j, w in zip(occ[c], sgn)]
            + [(a, 1 / 6.0) for a, b in mass] + [(b, -1 / 6.0) for a, b in mass], -2.0)      # occ19 AND occ20 AND some mass
U0 = unit2([(uB, 1.0), (uC, 1.0)])                                 # busy, saturated to [0, 1] as U0 - U1
U1 = unit2([(uB, 1.0), (uC, 1.0)], -1.0)
# some head high in the right half: NR0 - NR1 (helmets / pink heads: blood cannot fake them)
allr = [(j, 1.0) for j in near_right + close_right]
NR0, NR1 = unit2(allr), unit2(allr, -1.0)
# a demon right of the line: FD0 - FD1.  Ignored while blood flies (blood has the demon's colour).
alld = [(j, 1.0) for j in far_demon] + [bl]
FD0, FD1 = unit2(alld), unit2(alld, -1.0)
CR0, CR1 = unit2([(j, 1.0) for j in close_right]), unit2([(j, 1.0) for j in close_right], -1.0)
CL0, CL1 = unit2([(j, 1.0) for j in close_left]), unit2([(j, 1.0) for j in close_left], -1.0)

# ================================================================ output
BUSY_R, BUSY_A = 1000.0, 1e6      # vetoes; attack evidence can reach a few thousand, so its veto must be far larger
b3[0] = 1.0                                                        # TURN_LEFT: 1 + 4 * close_left
W3[CL0, 0] = 4.0; W3[CL1, 0] = -4.0
W3[NR0, 1] = 2.0; W3[NR1, 1] = -2.0                                # TURN_RIGHT: 2 * near_right + 1.5 * demon_right + 4 * close_right
W3[FD0, 1] = 1.5; W3[FD1, 1] = -1.5
W3[CR0, 1] = 4.0; W3[CR1, 1] = -4.0
W3[U0, 1] = -BUSY_R; W3[U1, 1] = BUSY_R
for k, w in ((MA0, 1), (MA1, -1), (MG0, 1), (MG1, -1), (FM0, 1), (QD0, 1), (BG0, 1), (MS0, 1), (CB0, 100)):
    W3[k, 2] = 10.0 * w                                            # ATTACK: 10 * (any aim rule)
W3[U0, 2] = -BUSY_A; W3[U1, 2] = BUSY_A

assert n1 <= H and n2 <= H, (n1, n2)
np.savez("weights.npz", W1=W1.astype(np.float32), b1=b1.astype(np.float32), W2=W2.astype(np.float32),
         b2=b2.astype(np.float32), W3=W3.astype(np.float32), b3=b3.astype(np.float32))
