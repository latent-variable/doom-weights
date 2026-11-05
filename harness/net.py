"""The network: fixed shape, plain numpy. Only its weights are written.

This file is given to the builder as source code.

    h1 = relu(x  @ W1 + b1)      x:  (D,)   W1: (D, 128)   b1: (128,)
    h2 = relu(h1 @ W2 + b2)             W2: (128, 128) b2: (128,)
    q  =      h2 @ W3 + b3              W3: (128, A)   b3: (A,)
    action = argmax(q)    (a tie goes to the lowest index)

D is env.input_size(scenario); A is the scenario's number of actions. Weights are
float32 and finite. Every weight starts at zero, so the untouched network always
picks action 0.
"""
import numpy as np

HIDDEN = 128
KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


def shapes(d, a):
    return {"W1": (d, HIDDEN), "b1": (HIDDEN,), "W2": (HIDDEN, HIDDEN), "b2": (HIDDEN,),
            "W3": (HIDDEN, a), "b3": (a,)}


def zeros(d, a):
    return {k: np.zeros(s, np.float32) for k, s in shapes(d, a).items()}


def load(path, d, a):
    """Read a weights .npz and check it exactly; raise ValueError on anything off."""
    with np.load(path) as f:
        got = {k: f[k] for k in f.files}
    want = shapes(d, a)
    if set(got) != set(want):
        raise ValueError(f"weights must hold exactly {sorted(want)}, got {sorted(got)}")
    out = {}
    for k, s in want.items():
        w = np.asarray(got[k])
        if w.shape != s:
            raise ValueError(f"{k} has shape {w.shape}, expected {s}")
        w = w.astype(np.float32)
        if not np.all(np.isfinite(w)):
            raise ValueError(f"{k} has a NaN or infinite value")
        out[k] = w
    return out


def forward(w, x):
    """Action scores and the two hidden layers, for one input vector."""
    h1 = np.maximum(x @ w["W1"] + w["b1"], 0.0)
    h2 = np.maximum(h1 @ w["W2"] + w["b2"], 0.0)
    return h2 @ w["W3"] + w["b3"], h1, h2


def act(w, x):
    return int(np.argmax(forward(w, x)[0]))
