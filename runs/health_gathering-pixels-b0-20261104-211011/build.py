"""Hand-written weights for health_gathering, pixels mode.

Layer 1 (128 relu units)
  * medkit blob detectors: per cell "non-greenness" s = R + B - G (the acid floor is
    green, medkits are white/red, so s jumps up on a medkit). Each detector takes a
    horizontal centre-surround (centre minus half of each equally wide neighbour) of s
    over a band of floor rows. A uniform floor, a straight wall/floor boundary
    (even a slanted one: it is linear, and a second difference cancels it) all give 0;
    a small object on the floor gives a positive response. Threshold in the bias.
  * floor-greenness units g = G - (R+B)/2 summed over regions, used for
      - Bf: "wall fairly close ahead" (little green floor in rows 19-21),
      - Bn: "wall very close ahead" (little green floor in rows 22-24),
      - gL, gR: amount of open floor on the left / right, to choose turn direction.
Layer 2
  * M0..M4: medkit evidence per steering sector (far left .. far right).
  * tL, tR: "turn left / right because blocked", biased toward the more open side.
  * Bn copy (veto on plain forward).
Output
  q0 TURN_LEFT  = M0 + 2 tL      q1 TURN_RIGHT = M4 + 2 tR
  q2 FORWARD    = 0.02 + M2 - 20 Bn
  q3 FWD+LEFT   = M1             q4 FWD+RIGHT  = M3
"""
import numpy as np

H, W = 30, 40
D = H * W * 3 + 1
NH = 128


def idx(r, c, k):
    return (r * W + c) * 3 + k


def main():
    W1 = np.zeros((D, NH), np.float32); b1 = np.zeros(NH, np.float32)
    W2 = np.zeros((NH, NH), np.float32); b2 = np.zeros(NH, np.float32)
    W3 = np.zeros((NH, 5), np.float32); b3 = np.zeros(5, np.float32)

    def add_s(u, r, c, w):            # w * (R + B - G) of cell (r, c)
        W1[idx(r, c, 0), u] += w; W1[idx(r, c, 2), u] += w; W1[idx(r, c, 1), u] -= w

    def add_g(u, r, c, w):            # w * (G - (R + B)/2) of cell (r, c)
        W1[idx(r, c, 1), u] += w; W1[idx(r, c, 0), u] -= w / 2; W1[idx(r, c, 2), u] -= w / 2

    u = 0
    K = 30.0                          # offset keeping a linear quantity positive through relu
    detectors = []                    # (unitP, unitQ, centre column, threshold, band weight)
    # band: rows, centre width, stride, threshold, priority weight, excluded columns
    bands = [
        (range(15, 17), 2, 2, 0.10, 1.0, set()),
        (range(17, 20), 2, 2, 0.15, 1.5, set()),
        (range(20, 25), 3, 2, 0.30, 2.0, set(range(15, 25))),   # clear of a weapon sprite
        (range(25, 28), 4, 1, 0.40, 2.5, set(range(13, 27))),
    ]
    for rows, w, stride, thr, pw, excl in bands:
        starts = list(range(0, W - 3 * w + 1, stride))
        if starts[-1] != W - 3 * w:
            starts.append(W - 3 * w)
        for start in starts:
            cols = list(range(start, start + 3 * w))
            if excl & set(cols):
                continue
            left, centre, right = cols[:w], cols[w:2 * w], cols[2 * w:]
            uP, uQ = u, u + 1
            for r in rows:
                for c in centre:
                    add_s(uP, r, c, 1.0)          # P = relu(K + centre - left)
                for c in left:
                    add_s(uP, r, c, -1.0)
                    add_s(uQ, r, c, -1.0)         # Q = relu(right - left)
                for c in right:
                    add_s(uQ, r, c, 1.0)
            b1[uP] = K
            detectors.append((uP, uQ, start + 1.5 * w - 0.5, thr, pw))
            u += 2

    flanks = list(range(6, 15)) + list(range(25, 34))
    # Bf = relu(1 - sum g / (0.06 N)) : 1 when no green floor there, 0 when floor is green
    def blocked_unit(rows):
        nonlocal u
        n = len(rows) * len(flanks)
        for r in rows:
            for c in flanks:
                add_g(u, r, c, -1.0 / (0.06 * n))
        b1[u] = 1.0
        u += 1
        return u - 1
    uBf = blocked_unit(range(19, 22))
    uBn = blocked_unit(range(22, 25))

    def green_unit(cols):
        nonlocal u
        for r in range(17, 22):
            for c in cols:
                add_g(u, r, c, 1.0)
        u += 1
        return u - 1
    ugL = green_unit(range(0, 12))
    ugR = green_unit(range(28, 40))
    assert u <= NH, u
    print('layer-1 units used:', u, 'detectors:', len(detectors))

    # ---- layer 2 ----
    # per detector: relu(P - Q - K - thr) = relu(min(centre-left, centre-right) - thr)
    sec_edges = [8, 16, 24, 32]
    v = 0
    det_h2 = []
    for (uP, uQ, cc, thr, pw) in detectors:
        W2[uP, v] = 1.0; W2[uQ, v] = -1.0; b2[v] = -K - thr
        det_h2.append((v, int(np.searchsorted(sec_edges, cc, side="right")), pw))
        v += 1
    vtL, vtR, vBn = v, v + 1, v + 2
    for vt, sign in ((vtL, 1.0), (vtR, -1.0)):
        W2[uBf, vt] = 1.0
        W2[uBn, vt] = 3.0
        W2[ugL, vt] = 0.02 * sign
        W2[ugR, vt] = -0.02 * sign
        b2[vt] = -0.2
    b2[vtR] += 0.01                   # break ties toward turning right
    W2[uBn, vBn] = 1.0
    assert vBn < NH

    # ---- output ----
    sector_action = [0, 3, 2, 4, 1]   # far left, left, centre, right, far right
    for (vd, s, pw) in det_h2:
        W3[vd, sector_action[s]] = pw
    W3[vtL, 0] = 2.0
    W3[vtR, 1] = 2.0
    W3[vBn, 2] = -20.0; b3[2] = 0.02

    np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


if __name__ == "__main__":
    main()
