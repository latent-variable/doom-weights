"""Hand-written weights for deadly_corridor / objects mode.  No learning: every
number below is chosen by reasoning about what the unit should compute.

The network is used as a small logic machine:

  layer 1  thresholded linear measurements of the screen ("predicates"),
           each clipped to [0, 1]
  layer 2  ANDs of predicates (and their negations)
  layer 3  each AND votes for one action with a priority weight

See explanation.md for the behaviour this encodes.
"""
import numpy as np

ROWS, COLS, CH = 30, 40, 3
D = ROWS * COLS * CH + 1          # + health
H = 128
A = 7
FWD, MLEFT, MRIGHT, TLEFT, TRIGHT, ATT, FWD_ATT = range(7)
MON, PICK, DEPTH = 0, 1, 2
HEALTH = ROWS * COLS * CH


def idx(r, c, k):
    return (r * COLS + c) * CH + k


def form(rows, cols, k, w=1.0):
    """A linear form over the input: w * sum of channel k over rows x cols."""
    f = np.zeros(D, np.float64)
    for r in rows:
        for c in cols:
            f[idx(r, c, k)] += w
    return f


class Net:
    def __init__(self):
        self.W1 = np.zeros((D, H)); self.b1 = np.zeros(H)
        self.W2 = np.zeros((H, H)); self.b2 = np.zeros(H)
        self.W3 = np.zeros((H, A)); self.b3 = np.zeros(A)
        self.n1 = 0
        self.n2 = 0

    # ---- layer 1: predicates.  A predicate is returned as (const, {unit: coef}),
    # an affine function of layer-1 units whose value is clip(K*(f.x + off), 0, 1).
    def pred(self, f, off, K):
        """clip(K * (f.x + off), 0, 1) using two ReLUs: relu(z) - relu(z - 1)."""
        u = self.n1; self.n1 += 2
        self.W1[:, u] = K * f;     self.b1[u] = K * off
        self.W1[:, u + 1] = K * f; self.b1[u + 1] = K * off - 1.0
        return (0.0, {u: 1.0, u + 1: -1.0})

    def pred_pos(self, f, K):
        """clip(K * f.x, 0, 1) for a form that is never negative: 1 - relu(1 - K f.x)."""
        u = self.n1; self.n1 += 1
        self.W1[:, u] = -K * f; self.b1[u] = 1.0
        return (1.0, {u: -1.0})

    # ---- layer 2: AND of literals.  relu(1 - sum_pos (1 - P) - sum_neg P)
    def conj(self, pos=(), neg=()):
        v = self.n2; self.n2 += 1
        self.b2[v] = 1.0
        for c0, d in pos:
            self.b2[v] += c0 - 1.0
            for u, co in d.items():
                self.W2[u, v] += co
        for c0, d in neg:
            self.b2[v] -= c0
            for u, co in d.items():
                self.W2[u, v] -= co
        return v

    def vote(self, v, action, weight):
        self.W3[v, action] += weight

    def save(self, path):
        assert self.n1 <= H and self.n2 <= H, (self.n1, self.n2)
        np.savez(path, W1=self.W1.astype(np.float32), b1=self.b1.astype(np.float32),
                 W2=self.W2.astype(np.float32), b2=self.b2.astype(np.float32),
                 W3=self.W3.astype(np.float32), b3=self.b3.astype(np.float32))


def build():
    n = Net()
    PX = 1.0 / 64.0                       # one pixel, as a share of a cell

    # ------------------------------------------------------------------ monsters
    # Column sums of the monster channel over the whole screen height.
    # The crosshair sits on the line between columns 19 and 20.
    allr = range(0, ROWS)

    def mon(cols, rows=allr):
        return form(rows, cols, MON)

    FZ = mon([19, 20])                    # the two columns touching the crosshair
    R = mon(range(21, 40))                # everything right of them
    OUTL, DZL = mon(range(0, 16)), mon(range(16, 19))     # left: far out / within 3 cells
    OUTR = mon(range(24, 40))
    MID = mon(range(16, 24))              # within 4 cells of the crosshair

    on_x = n.pred_pos(FZ, 1.0 / (2 * PX))                 # a monster touches the crosshair
    # ... and those columns hold at least 12% of the monster pixels within 4 cells of
    # the crosshair.  A monster's hit box is about 40 units wide, wider than its body
    # but not than its outstretched gun or legs: when only such a fringe touches the
    # crosshair (measured on saved frames: a few % of the mass) the shot would miss.
    solid = n.pred(FZ - 0.12 * MID, 0.0, 64.0)
    r_any = n.pred_pos(R, 1.0 / (2 * PX))
    # Which side to deal with first: the heavier one, with each column's weight halving
    # every 6 columns away from the crosshair.  Turning towards a monster then makes
    # its side win by more on the next step, so the choice cannot flip back and forth
    # when two monsters are about the same size (as the first pair is).
    # Columns 16..18 count half: a monster there cannot be turned to (see 2. below),
    # so one equally far off on the right, which can, goes first.
    side_w = [2.0 ** (-abs(c - 19.5) / 6.0) * (0.5 if 16 <= c <= 18 else 1.0) for c in range(COLS)]
    LW = sum(side_w[c] * mon([c]) for c in range(0, 19))
    RW = sum(side_w[c] * mon([c]) for c in range(21, 40))
    l_gt_r = n.pred(1.2 * LW - RW, 0.0, 64.0)             # the left side wins (ties go left)
    outl_gt_dzl = n.pred(OUTL - DZL, 0.0, 64.0)           # ... and it is mostly far out
    # one monster lying across the line between columns 15 and 16 (at least 6 px of it
    # in column 15: a 5-cell turn then puts at least the 2 px `on_x` needs on the crosshair)
    o15 = n.pred(mon([15]), -4 * PX, 1.0 / (2 * PX))
    o16 = n.pred_pos(mon([16]), 1.0 / (2 * PX))
    dom_l = n.pred(OUTL - 4.0 * MID, 0.0, 64.0)           # a much bigger (closer) monster far left
    dom_r = n.pred(OUTR - 4.0 * MID, 0.0, 64.0)           # ... far right
    # Fine aim when turning is not allowed.  Practice showed the player can only move
    # about 16 units either side of the corridor's centre line (the wider "rooms" the
    # monsters stand in cannot be entered, and nothing in the depth picture marks the
    # limit), so a side-step is only worth it against a close monster (legs reach
    # rows 19+, under ~250 units), where a few units swing its bearing by degrees.
    # Further off, walk on instead: the bearing widens until a turn is safe.  Walking
    # can never get stuck; two practice games were lost side-stepping into that
    # invisible limit under fire.
    edge_rows, mid_rows = range(10, 20), range(12, 15)
    wall_l = n.pred(form(edge_rows, range(0, 2), DEPTH, -1.0 / 20.0), 0.027, 600.0)   # door frame
    wall_r = n.pred(form(edge_rows, range(38, 40), DEPTH, -1.0 / 20.0), 0.027, 600.0)  # beside us
    low = range(19, ROWS)
    close_l = n.pred_pos(mon(range(14, 20), low), 1.0 / (3 * PX))
    close_r = n.pred_pos(mon(range(20, 26), low), 1.0 / (3 * PX))

    # 1. Shoot a monster that is solidly on the crosshair, unless a much closer
    #    monster stands far off to one side: that one first.
    n.vote(n.conj(pos=[on_x, solid], neg=[dom_l, dom_r]), ATT, 128.0)
    # 2. Nothing touches the crosshair: turn the heavier side towards it.  A held turn
    #    speeds up to ~5 cells per step, so to rule out flipping left-right-left for
    #    ever, a left turn is only allowed when the monster reaches more than 4 cells
    #    out (it then cannot jump clean over the crosshair), and no turn at all once
    #    something touches the crosshair.
    n.vote(n.conj(pos=[r_any], neg=[l_gt_r, on_x]), TRIGHT, 32.0)
    n.vote(n.conj(pos=[r_any, dom_r], neg=[l_gt_r]), TRIGHT, 32.0)
    n.vote(n.conj(pos=[l_gt_r, outl_gt_dzl], neg=[on_x]), TLEFT, 32.0)
    n.vote(n.conj(pos=[l_gt_r, o15, o16], neg=[on_x]), TLEFT, 32.0)
    n.vote(n.conj(pos=[l_gt_r, dom_l]), TLEFT, 32.0)
    # 3. Turning not allowed: side-step towards a close monster, else walk on.
    n.vote(n.conj(pos=[l_gt_r, close_l], neg=[wall_l]), MLEFT, 8.0)
    n.vote(n.conj(pos=[l_gt_r, wall_l]), FWD, 8.0)
    n.vote(n.conj(pos=[l_gt_r], neg=[close_l]), FWD, 8.0)
    n.vote(n.conj(pos=[r_any, close_r], neg=[l_gt_r, wall_r]), MRIGHT, 8.0)
    n.vote(n.conj(pos=[r_any, wall_r], neg=[l_gt_r]), FWD, 8.0)
    n.vote(n.conj(pos=[r_any], neg=[l_gt_r, close_r]), FWD, 8.0)

    # ---------------------------------------------------------------- navigation
    # (votes of weight 1: only decide when no monster is in view)
    far, near = range(0, 20), range(20, ROWS)             # pickups beyond / within ~200 units
    lcols, ccols, rcols = range(0, 17), range(17, 23), range(23, 40)
    PC = form(allr, ccols, PICK)
    PSIDE = form(allr, lcols, PICK) + form(allr, rcols, PICK)
    PMID = form(far, range(12, 17), PICK) + form(far, range(23, 28), PICK)
    # which side: a distant pickup (the vest) outweighs any dropped item at our feet
    PDIFF = (1000.0 * (form(far, lcols, PICK) - form(far, rcols, PICK))
             + form(near, lcols, PICK) - form(near, rcols, PICK))
    p_ctr = n.pred_pos(PC, 1.0 / (1.5 * PX))              # a pickup within 3 cells of straight ahead
    p_side = n.pred_pos(PSIDE, 1.0 / (1.5 * PX))          # ... or further off to a side
    p_mid = n.pred_pos(PMID, 1.0 / (1.5 * PX))            # a distant one 3 to 8 cells off
    p_left = n.pred(PDIFF, 0.0, 64.0)
    # how far is the way ahead open?  mean depth just above the horizon, in the middle
    ahead = form(mid_rows, range(18, 22), DEPTH, 1.0 / 12.0)
    deep = n.pred(ahead, -0.30, 20.0)                     # ~550+ units
    open_ = n.pred(ahead, -0.10, 50.0)                    # ~200+ units

    n.vote(n.conj(pos=[p_ctr]), FWD, 1.0)                           # pickup ahead: walk to it
    # the vest is a little off to one side but the way ahead is open: keep walking.
    # (Turning back to it at once can undo a turn just made towards a monster that is
    # only a pixel wide and drops out of view; a practice game was lost to that loop.)
    n.vote(n.conj(pos=[p_mid, open_], neg=[p_ctr]), FWD, 1.0)
    for blocker in (p_mid, open_):
        n.vote(n.conj(pos=[p_side, p_left], neg=[p_ctr, blocker]), TLEFT, 1.0)   # pickup off to the left
        n.vote(n.conj(pos=[p_side], neg=[p_ctr, p_left, blocker]), TRIGHT, 1.0)  # ... to the right
    n.vote(n.conj(pos=[deep], neg=[p_ctr, p_side]), FWD, 1.0)       # nothing seen, open ahead: walk
    n.vote(n.conj(neg=[p_ctr, p_side, deep]), TRIGHT, 1.0)          # facing a wall: turn round
    return n


if __name__ == "__main__":
    net = build()
    net.save("weights.npz")
