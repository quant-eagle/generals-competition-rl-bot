"""Turn a (source, intent) pair into one primitive action for the engine.

The action space is that of Straka et al. (arXiv:2606.23348), plus the one
thing their ruleset does not have:

    source = an explicit board cell            441   (H x W)
    intent = direction x {all, half}            8
             | build                            1    (castles are buildable here)
             | pass                             1

Higher-level actions -- "move toward target cell X" routed by BFS, or "act
from the largest stack" -- are deliberately absent.  They resolve to a concrete
move from board state at execution time, so a source and a target can be
jointly unexecutable, exact masking would cost one BFS per target, and the
policy cannot learn to avoid a failure whose cause it neither chooses nor
predicts.  The geodesic information such actions would carry lives in
observation planes instead (obs.bfs_from), where no mask problem can arise.

This function always emits a legal primitive or a pass.  50 faults in one game
is a forfeit, so it never gambles on a mask being exactly right.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (B, INT_BUILD, INT_DIR0, INT_HALF0, INT_PASS, N_CELLS,
                     N_DIRS)
from .obs import build_price

DIRS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)


def select_source(src_action, movable):
    """Flat index of the acting cell, or -1 when that cell cannot act.

    The cell is named by the policy itself, so an illegal (cell, intent) pair is
    something it can learn to stop choosing.
    """
    cell = jnp.clip(src_action, 0, N_CELLS - 1)
    return jnp.where(movable.reshape(-1)[cell], cell, -1).astype(jnp.int32)


def execute(src_action, int_action, obs):
    """-> (5,) primitive [kind, row, col, dir, split].

    kind: 0 move, 1 pass, 2 build.  `split` is 1 for a half move -- the engine
    moves `army // 2` on split_army == 1 and `army - 1` otherwise.
    """
    passable = ~obs.mountains
    movable = obs.owned_cells & (obs.armies > 1)
    army = obs.armies.astype(jnp.int32)

    is_build = int_action == INT_BUILD
    is_move = int_action < INT_BUILD          # 0..3 all, 4..7 half
    half = is_move & (int_action >= INT_HALF0)

    cell = jnp.clip(src_action, 0, N_CELLS - 1)
    r, c = cell // B, cell % B
    src = select_source(src_action, movable)

    k = jnp.clip(jnp.where(half, int_action - INT_HALF0, int_action - INT_DIR0),
                 0, N_DIRS - 1)
    nr, nc = r + DIRS[k, 0], c + DIRS[k, 1]
    inb = (nr >= 0) & (nr < B) & (nc >= 0) & (nc < B)
    raw_ok = inb & passable[jnp.clip(nr, 0, B - 1), jnp.clip(nc, 0, B - 1)]
    ok_move = is_move & (src >= 0) & raw_ok

    # Build is checked at the named cell and does not require `movable`: the
    # price is >= 35, which already implies army > 1, so gating on `movable`
    # could only reject a legal build.
    own_structures = (obs.castles | obs.generals) & obs.owned_cells
    plain = obs.owned_cells & ~obs.castles & ~obs.generals
    ok_build = is_build & plain[r, c] & (army[r, c] >= build_price(own_structures)[r, c])

    acts = ok_move | ok_build
    kind = jnp.where(ok_build, 2, jnp.where(ok_move, 0, 1))
    prim = jnp.stack([kind,
                      jnp.where(acts, r, 0),
                      jnp.where(acts, c, 0),
                      jnp.where(ok_move, k, 0),
                      jnp.where(ok_move & half, 1, 0)]).astype(jnp.int32)
    return prim


execute_v = jax.vmap(execute, in_axes=(0, 0, 0))
