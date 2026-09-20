"""Numpy twin of `train/exec.py` for deployment.

The submitted bot runs on one CPU core and cannot import JAX, so the executor
exists twice and the two must agree exactly.  A divergence here is silent in the
worst way: the net still runs, the move is still legal, and the bot simply plays
a different game than the one that was trained.

Parity with the JAX side is pinned by tests/test_np.py.
"""
from __future__ import annotations

import numpy as np

from .config import B, INT_BUILD, INT_DIR0, INT_HALF0, INT_PASS
from .np_obs import build_price

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PASS = (1, 0, 0, 0, 0)


def select_source(src_action, movable) -> int:
    """Flat index of the acting cell, or -1 when that cell cannot act."""
    cell = int(src_action)
    return cell if movable.reshape(-1)[cell] else -1


def execute(src_action: int, int_action: int, frame):
    """-> (5,) primitive tuple.  Never illegal."""
    src_action, int_action = int(src_action), int(int_action)
    passable = ~frame.mountains
    movable = frame.owned_cells & (frame.armies > 1)
    army = frame.armies.astype(np.int32)

    if int_action == INT_PASS:
        return PASS
    r, c = divmod(int(src_action), B)

    if int_action == INT_BUILD:
        # checked at the named cell; the price implies army > 1, so `movable`
        # is not required and gating on it would reject a legal build
        own_structures = (frame.castles | frame.generals) & frame.owned_cells
        plain = frame.owned_cells & ~frame.castles & ~frame.generals
        if plain[r, c] and army[r, c] >= build_price(own_structures)[r, c]:
            return (2, r, c, 0, 0)
        return PASS

    if select_source(src_action, movable) < 0:
        return PASS

    half = int_action >= INT_HALF0
    k = int_action - (INT_HALF0 if half else INT_DIR0)
    nr, nc = r + DIRS[k][0], c + DIRS[k][1]
    d = k if (0 <= nr < B and 0 <= nc < B and passable[nr, nc]) else -1
    if d < 0:
        return PASS
    return (0, r, c, d, 1 if half else 0)
