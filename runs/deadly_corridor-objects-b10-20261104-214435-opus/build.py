"""Hand-written weights for deadly_corridor, objects mode.

Layer 1 extracts 12 linear image statistics (see features() below); the other
116 units are unused (all-zero).
Layer 2 turns them into near-binary gates: clip(k*(f-thr), 0, 1) is built from
two ReLUs, relu(k(f-thr)) - relu(k(f-thr)-1), the subtraction done in layer 3.
Layer 3 is a priority list: each gate adds a fixed score to one action, larger
scores for more urgent rules; MOVE_FORWARD has a base score of 1.
"""
import numpy as np

GW, GH = 40, 30
D = GW * GH * 3 + 1
H = 128
A = 7
FWD, LEFT, RIGHT, TL, TR, ATK, FATK = range(7)


def idx(r, c, k):
    return (r * GW + c) * 3 + k


def off(c):
    return c - 19.5          # column offset from the crosshair (between cols 19 and 20)


# ---- tunables (chosen by reasoning, adjusted after reading practice screens) ----
P = dict(
    ATTACK_MASS=0.01,    # monster share on the crosshair strip that triggers a shot
    SMALL_MASS=2.0,      # total monster share below which the target is "far": shoot while advancing
    FAR_S=1.0,           # full-screen moment that triggers a turn
    VEST_DEAD=3.5,       # vest within this many columns of centre counts as straight ahead
    VEST_S=0.005,        # vest moment that triggers a turn
    VEST_VIS=0.005,      # vest mass meaning "vest visible"
    VEST_ROWS=(13, 17),  # horizon band where the distant vest is drawn
    VEST_HALF=10.5,      # and only columns within this many of the centre
    NAV_ROWS=(10, 17),   # rows around the horizon used for depth heading
    NAV_T=0.4,           # depth asymmetry that triggers a heading turn
    BLOCK_T=6.0,         # centre-box depth sum below which we are facing a wall
)


def features():
    """Layer-1 weight columns: list of (name, weight vector over x, bias)."""
    F = []
    m = np.zeros((GH, GW)); s = np.zeros((GH, GW))
    for c in range(GW):
        o = off(c)
        m[:, c] = 1.0 if abs(o) < 2 else 0.0   # columns 18..21
        # nearest-to-centre weighting: largest just off centre, so the monster
        # closest to the crosshair dominates and turning toward it is stable
        s[:, c] = np.sign(o) * (22.0 - abs(o))
    def grid(k, w):
        v = np.zeros(D); v[k:GW * GH * 3:3] = w.reshape(-1); return v
    F.append(("xhair", grid(0, m), 0.0))
    wide = np.zeros((GH, GW)); wide[:, 15:25] = 1.0       # columns 15..24
    F.append(("xwide", grid(0, wide), 0.0))
    F.append(("mtotal", grid(0, np.ones((GH, GW))), 0.0))
    F.append(("far+", grid(0, s), 0.0)); F.append(("far-", grid(0, -s), 0.0))
    # The vest is far away at the horizon near the screen centre; items dropped
    # by dead zombies lie nearby on the floor (lower rows) or in the side
    # alcoves (screen edges), so only a central horizon window counts as "vest".
    v = np.zeros((GH, GW)); vm = np.zeros((GH, GW)); v0, v1 = P["VEST_ROWS"]
    for c in range(GW):
        o = off(c)
        if abs(o) <= P["VEST_HALF"]:
            v[v0:v1, c] = np.sign(o) if abs(o) > P["VEST_DEAD"] else 0.0
            vm[v0:v1, c] = 1.0
    F.append(("vest+", grid(1, v), 0.0)); F.append(("vest-", grid(1, -v), 0.0))
    F.append(("vestmass", grid(1, vm), 0.0))
    t = np.zeros((GH, GW)); r0, r1 = P["NAV_ROWS"]
    for c in range(GW):
        t[r0:r1, c] = off(c) / 19.5 / (r1 - r0)
    F.append(("depth+", grid(2, t), 0.0)); F.append(("depth-", grid(2, -t), 0.0))
    b = np.zeros((GH, GW)); b[8:22, 12:28] = 1.0
    F.append(("open", grid(2, b), 0.0))
    return F


def build():
    W1 = np.zeros((D, H), np.float32); b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, H), np.float32); b2 = np.zeros(H, np.float32)
    W3 = np.zeros((H, A), np.float32); b3 = np.zeros(A, np.float32)

    F = features()
    fi = {}
    for j, (name, w, bias) in enumerate(F):
        W1[:, j] = w; b1[j] = bias; fi[name] = j

    nxt = [0]
    def gate(terms, k, thr):
        """Two h2 units whose difference is clip(k*(sum w*h1[name] - thr), 0, 1)."""
        j = nxt[0]; nxt[0] += 2
        for name, w in terms:
            W2[fi[name], j] += k * w; W2[fi[name], j + 1] += k * w
        b2[j] = -k * thr; b2[j + 1] = -k * thr - 1.0
        return j

    def put(g, a, v):
        W3[g, a] += v; W3[g + 1, a] -= v

    b3[FWD] = 1.0
    gx = gate([("xhair", 1)], 100.0, P["ATTACK_MASS"])
    gs = gate([("mtotal", -1)], 2.0, -P["SMALL_MASS"])          # 1 when total mass small
    # shooting outranks every turn rule combined (12 + 8 + 4 + 3 < 40)
    put(gx, ATK, 40.0)
    # FATK = 40*gw + 40*gs - 39.5: fires only when a SMALL (far) target is roughly
    # ahead (columns 15..24); advancing on it makes it bigger, so aim converges
    # instead of the turn overshooting back and forth around a 1-column target.
    gw = gate([("xwide", 1)], 100.0, P["ATTACK_MASS"])
    put(gw, FATK, 40.0); put(gs, FATK, 40.0); b3[FATK] = -39.5
    put(gate([("far+", 1)], 1.0, P["FAR_S"]), TR, 12.0)
    put(gate([("far-", 1)], 1.0, P["FAR_S"]), TL, 12.0)
    put(gate([("vest+", 1)], 100.0, P["VEST_S"]), TR, 8.0)
    put(gate([("vest-", 1)], 100.0, P["VEST_S"]), TL, 8.0)
    vis = gate([("vestmass", 1)], 100.0, P["VEST_VIS"])
    put(vis, TR, -4.0); put(vis, TL, -4.0)
    put(gate([("depth+", 1)], 4.0, P["NAV_T"]), TR, 4.0)
    put(gate([("depth-", 1)], 4.0, P["NAV_T"]), TL, 4.0)
    # facing a wall: open < BLOCK_T  ->  turn right
    put(gate([("open", -1)], 1.0, -P["BLOCK_T"]), TR, 3.0)
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3), [f[0] for f in F]


if __name__ == "__main__":
    w, _ = build()
    np.savez("weights.npz", **w)
