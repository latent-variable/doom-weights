"""The encoding and the weight checks the builder is told about must hold exactly."""
import types

import numpy as np
import pytest

from harness import env as E
from harness import net as N


def test_object_grid_index_matches_the_spec():
    labels_buf = np.zeros((E.RENDER_H, E.RENDER_W), np.uint8)
    labels_buf[8:16, 96:104] = 7          # one full cell: row 1, column 12
    depth = np.full((E.RENDER_H, E.RENDER_W), 255, np.uint8)
    lab = types.SimpleNamespace(object_name="Cacodemon", value=7)
    x = E.encode("objects", None, labels_buf, [lab], depth, [50], "basic")
    assert x.shape == (E.input_size("basic"),)
    assert x[(1 * 40 + 12) * 3 + 0] == 1.0
    assert x[:3600:3].sum() == 1.0        # nothing else is a monster
    assert x[(0 * 40 + 0) * 3 + 2] == 1.0  # far depth
    assert x[-1] == 1.0                    # ammo 50 / 50


def test_pixels_are_cell_means():
    screen = np.zeros((E.RENDER_H, E.RENDER_W, 3), np.uint8)
    screen[:, :160, 0] = 255               # left half red
    x = E.encode("pixels", screen, None, [], None, [100, 26], "defend_the_center")
    assert x[(5 * 40 + 19) * 3 + 0] == 1.0 and x[(5 * 40 + 20) * 3 + 0] == 0.0


def test_zero_network_picks_action_zero_and_bad_weights_are_refused(tmp_path):
    d, a = E.input_size("basic"), 3
    w = N.zeros(d, a)
    assert N.act(w, np.ones(d, np.float32)) == 0
    bad = dict(w, W3=np.zeros((128, 4), np.float32))
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError):
        N.load(tmp_path / "bad.npz", d, a)
