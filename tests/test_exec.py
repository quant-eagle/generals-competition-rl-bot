"""Executor: every emitted primitive must be legal, and each branch of the
(cell, intent) action space must mean exactly what it says.

An illegal action costs a fault and 50 faults in one game is a forfeit, so
"never emits an illegal move" is a hard contract.  The action space is the
cell x direction grid of Straka et al. (arXiv:2606.23348) plus build.
"""
from collections import deque

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from generals import GeneralsEnv, get_observation
from generals.core.action import compute_valid_move_mask
from generals.modifiers.build_castles import build_cost_grid

from train import exec as X
from train import obs as O
from train.config import (B, INT_BUILD, INT_DIR0, INT_HALF0, INT_PASS, N_CELLS,
                          N_DIRS, N_INT, N_SRC)

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _seed_mask(flat_target):
    m = np.zeros((B, B), bool); m[int(flat_target) // B, int(flat_target) % B] = True
    return jnp.asarray(m)


def bfs_ref(passable, target):
    """Reference BFS with a queue, to check the min-plus relaxation."""
    tr, tc = target // B, target % B
    d = np.full((B, B), O.UNREACHABLE, np.int32)
    if not passable[tr, tc]:
        return d
    d[tr, tc] = 0
    q = deque([(tr, tc)])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < B and 0 <= nc < B and passable[nr, nc] \
                    and d[nr, nc] == O.UNREACHABLE:
                d[nr, nc] = d[r, c] + 1
                q.append((nr, nc))
    return d


@pytest.fixture(scope="module")
def states():
    env = GeneralsEnv(mode="competition")
    pool, st = env.reset(jr.PRNGKey(0))
    out, key = [st], jr.PRNGKey(5)
    for _ in range(5):
        for _ in range(14):
            acts = []
            for seat in (0, 1):
                key, k = jr.split(key)
                o = get_observation(st, seat)
                m = compute_valid_move_mask(o.armies, o.owned_cells, o.mountains)
                flat = jnp.argmax(jr.uniform(k, m.shape) * m)
                acts.append(jnp.array([0, (flat // 4) // B, (flat // 4) % B,
                                       flat % 4, 0], jnp.int32))
            _, st = env.step(st, jnp.stack(acts), pool)
        out.append(st)
    return out


def test_bfs_matches_a_reference_queue_search(states):
    """The min-plus relaxation must equal a real BFS."""
    for st in states[:3]:
        passable = np.asarray(~st.mountains)
        for target in (0, 123, 220, 440):
            got = np.asarray(O.bfs_from(jnp.asarray(passable), _seed_mask(target)))
            ref = bfs_ref(passable, target)
            # The relaxation propagates one step per iteration and runs
            # BFS_ITERS times, so anything further stays UNREACHABLE by design;
            # the invariant is agreement within the bound.
            near = ref <= O.BFS_ITERS
            assert np.array_equal(got[near], ref[near])
            assert (got[~near] >= O.BFS_ITERS).all(), \
                "cells beyond the bound must not report a short distance"


@pytest.fixture(scope="module")
def rich(states):
    """A state with enough army to make builds affordable.

    Random play never accumulates a build price on one cell, so build legality
    would go untested on the natural states.  Only `armies` is touched, so the
    board topology and ownership stay real.
    """
    st = states[-1]
    own = st.ownership[0] & ~st.castles & ~st.generals
    return st._replace(armies=jnp.where(own, 200, st.armies))


def _sweep(o):
    """Every branch of the action space, vmapped: cells x every intent.

    Covers movable cells, non-movable cells (which must produce a pass, never
    an illegal move) and every intent including both split modes.
    """
    flat = np.asarray(o.owned_cells & (o.armies > 1)).reshape(-1)
    movable = np.flatnonzero(flat)
    still = np.flatnonzero(~flat)
    srcs = [int(c) for c in movable[:32]] + [int(c) for c in still[:8]]
    ints = list(range(N_INT))
    sa = jnp.asarray(np.repeat(srcs, len(ints)), jnp.int32)
    ia = jnp.asarray(np.tile(ints, len(srcs)), jnp.int32)
    prim = jax.vmap(X.execute, in_axes=(0, 0, None))(sa, ia, o)
    return np.asarray(sa), np.asarray(ia), np.asarray(prim)


def test_every_emitted_action_is_legal(states, rich):
    """Sweep every branch of the joint space: nothing emitted may be rejected."""
    checked = {0: 0, 1: 0, 2: 0}
    splits = set()
    for st in (states[1], states[3], rich):
        for seat in (0, 1):
            o = get_observation(st, seat)
            valid = np.asarray(compute_valid_move_mask(
                o.armies, o.owned_cells, o.mountains))
            cost = np.asarray(build_cost_grid(st, seat))
            army, own = np.asarray(o.armies), np.asarray(o.owned_cells)
            plain = own & ~np.asarray(o.castles) & ~np.asarray(o.generals)
            sa, ia, prim = _sweep(o)
            for j in range(len(sa)):
                kind, r, c, d, split = [int(v) for v in prim[j]]
                # split is 1 exactly for a half move, never for build/pass
                assert split == (1 if (kind == 0 and ia[j] >= INT_HALF0)
                                 else 0), \
                    f"({sa[j]},{ia[j]}) emitted split={split} for kind={kind}"
                if kind == 2:
                    assert plain[r, c] and army[r, c] >= cost[r, c], \
                        f"({sa[j]},{ia[j]}) emitted an unaffordable build"
                elif kind == 0:
                    assert valid[r, c, d], \
                        f"({sa[j]},{ia[j]}) emitted an illegal move"
                checked[kind] += 1
                if kind == 0:
                    splits.add(split)
    # every branch must have fired, including both split modes
    assert checked[0] > 150, f"too few moves exercised: {checked}"
    assert checked[1] > 0, f"pass branch never fired: {checked}"
    assert checked[2] > 0, f"build branch never fired: {checked}"
    assert splits == {0, 1}, f"both split modes must be exercised, got {splits}"


def test_the_geodesic_planes_route_around_obstacles():
    """The distance planes must bend around mountains.  A 7-layer trunk cannot
    compute paths that reach 55 steps, and would otherwise approximate the
    distance as Euclidean."""
    passable = np.ones((B, B), bool)
    passable[10, :] = False
    passable[10, 20] = True                 # one gap in the wall
    seed = np.zeros((B, B), bool); seed[0, 0] = True
    d = np.asarray(O.bfs_from(jnp.asarray(passable), jnp.asarray(seed)))
    assert d[11, 0] == 51, f"must route through the gap, got {d[11, 0]}"
    assert d[10, 20] == 30, "the gap itself is 10 down + 20 across"
    assert d[10, 5] >= O.UNREACHABLE, "walled-off cells stay unreachable"
    assert d[0, 0] == 0, "the seed is distance zero"


def test_cell_plus_direction_moves_that_cell_that_way(states):
    """Naming a cell and a direction must move that cell that way; a blocked
    or off-board step must become a pass."""
    checked = 0
    for st in states[1:4]:
        o = get_observation(st, 0)
        movable = np.asarray(o.owned_cells & (o.armies > 1)).reshape(-1)
        passable = np.asarray(~o.mountains)
        for cell in np.flatnonzero(movable)[:6]:
            for k in range(N_DIRS):
                prim = X.execute(jnp.int32(int(cell)),
                                 jnp.int32(INT_DIR0 + k), o)
                kind, r, c, d, _ = [int(v) for v in np.asarray(prim)]
                nr, nc = cell // B + DIRS[k][0], cell % B + DIRS[k][1]
                legal = (0 <= nr < B and 0 <= nc < B and passable[nr, nc])
                if not legal:
                    assert kind == 1, "an off-board or blocked step must pass"
                    continue
                assert (kind, r * B + c, d) == (0, int(cell), k)
                checked += 1
    assert checked > 20, f"only {checked} raw-direction moves exercised"


def test_build_acts_at_the_source_cell(rich):
    """Build applies at the named source cell, so every legal build site is
    reachable."""
    found = False
    for st in (rich,):
        for seat in (0, 1):
            o = get_observation(st, seat)
            cost = np.asarray(build_cost_grid(st, seat))
            army, own = np.asarray(o.armies), np.asarray(o.owned_cells)
            plain = own & ~np.asarray(o.castles) & ~np.asarray(o.generals)
            sites = np.flatnonzero((plain & (army >= cost)).reshape(-1))
            for cell in sites[:3]:
                prim = X.execute(jnp.int32(int(cell)), jnp.int32(INT_BUILD), o)
                kind, r, c, _, _ = [int(v) for v in np.asarray(prim)]
                assert (kind, r * B + c) == (2, int(cell))
                found = True
    assert found, "the rich fixture must expose an affordable build site"


def test_pass_intent_passes_whatever_the_source_says(states):
    o = get_observation(states[0], 0)
    for sa in (0, 3, 10, 200, 440):
        prim = X.execute(jnp.int32(sa), jnp.int32(INT_PASS), o)
        assert int(prim[0]) == 1


def test_masks_are_never_empty_and_agree_with_the_executor(states):
    """The legality grid is exact: a (cell, intent) pair is offered iff the
    executor can carry it out."""
    for st in states:
        for seat in (0, 1):
            o = get_observation(st, seat)
            grid = O.legal_mask(o)
            assert grid.shape == (N_SRC, N_INT)
            assert bool(grid[:, INT_PASS].any()), "pass must always be legal"
            assert bool(grid.any())

            # an offered cell can act, and a cell that can act is offered
            movable = o.owned_cells & (o.armies > 1)
            army = np.asarray(o.armies)
            own = np.asarray(o.owned_cells)
            plain = own & ~np.asarray(o.castles) & ~np.asarray(o.generals)
            cost = np.asarray(O.build_price(
                (o.castles | o.generals) & o.owned_cells))
            can_act = ((np.asarray(movable)) | (plain & (army >= cost))).reshape(-1)
            g = np.asarray(grid)
            if can_act.any():
                assert (g[:, :INT_PASS].any(-1) == can_act).all(), \
                    "source marginal must be exactly the cells that can act"
                payable = (plain & (army >= cost)).reshape(-1)
                assert (g[:, INT_BUILD] == payable).all(), \
                    "BUILD must be legal at exactly the cells that can pay"

            # (cell, dir) is offered iff that cell can actually take that step
            passable = np.asarray(~o.mountains)
            mv = np.asarray(movable).reshape(-1)
            for k in range(N_DIRS):
                rr, cc = np.divmod(np.arange(N_SRC), B)
                nr, nc = rr + DIRS[k][0], cc + DIRS[k][1]
                inb = (nr >= 0) & (nr < B) & (nc >= 0) & (nc < B)
                want = mv & inb & passable[np.clip(nr, 0, B - 1),
                                           np.clip(nc, 0, B - 1)]
                assert (g[:, INT_DIR0 + k] == want).all(), \
                    f"direction {k} is not exactly legal"
                # half intents exist in the action space but are masked out:
                # strong play uses them on under 2% of moves
                assert not g[:, INT_HALF0 + k].any(), \
                    "half-moves must never be offered"


def test_bfs_iterations_cover_the_board_eccentricity(states):
    """The relaxation propagates distance one cell per iteration, so BFS_ITERS
    below a board's eccentricity leaves far cells at UNREACHABLE.  Competition
    boards reach an eccentricity of 55, well past 2 * 21."""
    worst = 0
    for st in states:
        passable = np.asarray(~st.mountains).reshape(-1)
        for s in np.flatnonzero(passable)[::23]:
            d = np.full(B * B, -1, np.int32)
            d[s] = 0
            q = deque([int(s)])
            while q:
                c = q.popleft()
                r, cc = divmod(c, B)
                for dr, dc in DIRS:
                    nr, nc = r + dr, cc + dc
                    if 0 <= nr < B and 0 <= nc < B:
                        n = nr * B + nc
                        if passable[n] and d[n] < 0:
                            d[n] = d[c] + 1
                            q.append(n)
            worst = max(worst, int(d.max()))
    assert O.BFS_ITERS > worst, (
        f"BFS_ITERS={O.BFS_ITERS} does not cover an eccentricity of {worst}; "
        f"far cells would read as unreachable")
