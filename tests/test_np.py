"""Numpy deployment twins vs the JAX originals: planes, masks, executor.

The submitted bot cannot import JAX, so the encoder and the executor exist
twice.  If they drift, the deployed bot is not the trained one, and unlike a
network mismatch the failure is silent: the net still runs, the move is still
legal, and it simply plays a different game.  Hence a test per twin.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from generals import GeneralsEnv, get_observation
from generals.core.action import compute_valid_move_mask

from train import exec as X
from train import np_exec, np_obs
from train import obs as O
from train.config import (B, INT_BUILD, INT_DIR0, INT_PASS, N_CELLS, N_DIRS,
                          N_INT, N_PLANES, N_SRC)

FIELDS = ("armies", "generals", "castles", "mountains", "neutral_cells",
          "owned_cells", "opponent_cells", "fog_cells", "structures_in_fog",
          "owned_land_count", "owned_army_count", "opponent_land_count",
          "opponent_army_count", "timestep")


def to_numpy(o):
    return SimpleNamespace(**{f: np.asarray(getattr(o, f)) for f in FIELDS})


@pytest.fixture(scope="module")
def states():
    """A fogged sequence, so memory has actually accumulated something."""
    env = GeneralsEnv(mode="competition")
    pool, st = env.reset(jr.PRNGKey(0))
    out, key = [st], jr.PRNGKey(7)
    for _ in range(5):
        for _ in range(12):
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


@pytest.fixture(scope="module")
def rich(states):
    """Enough army that the build branch is reachable."""
    st = states[-1]
    own = st.ownership[0] & ~st.castles & ~st.generals
    return st._replace(armies=jnp.where(own, 200, st.armies))


def test_build_price_matches_jax(states):
    for st in states:
        o = get_observation(st, 0)
        own = (o.castles | o.generals) & o.owned_cells
        assert np.allclose(np.asarray(O.build_price(own)),
                           np_obs.build_price(np.asarray(own)))


def test_encoded_planes_match_plane_for_plane(states):
    """Includes the memory substrate, which only diverges after several ticks."""
    jm, nm = O.init_memory(), np_obs.Memory()
    for st in states:
        for seat in (0, 1):
            o = get_observation(st, seat)
            if seat == 0:
                jm = O.update(jm, o)
                nm.update(to_numpy(o))
                a = np.asarray(O.encode(jm, o))
                b = np_obs.encode(nm, to_numpy(o))
                bad = [i for i in range(N_PLANES)
                       if not np.allclose(a[i], b[i], atol=1e-5)]
                assert not bad, f"planes diverged: {bad}"


def test_legal_masks_match_exactly(states, rich):
    for st in list(states) + [rich]:
        for seat in (0, 1):
            o = get_observation(st, seat)
            n = to_numpy(o)
            jg, ng = O.legal_mask(o), np_obs.legal_mask(n)
            assert np.array_equal(np.asarray(jg), ng), "legality grid drift"


def test_executor_twins_emit_the_same_primitive(states, rich):
    """Every branch: movable and still cells x every intent, both splits."""
    seen = {0: 0, 1: 0, 2: 0}
    for st in list(states[1:]) + [rich]:
        for seat in (0, 1):
            o = get_observation(st, seat)
            n = to_numpy(o)
            flat = np.asarray(o.owned_cells & (o.armies > 1)).reshape(-1)
            srcs = ([int(c) for c in np.flatnonzero(flat)[:24]]
                    + [int(c) for c in np.flatnonzero(~flat)[:8]])
            for sa in srcs:
                for ia in range(N_INT):          # both split modes, build, pass
                    jp = X.execute(jnp.int32(sa), jnp.int32(ia), o)
                    npp = np_exec.execute(sa, ia, n)
                    jp = tuple(int(v) for v in np.asarray(jp))
                    assert jp == tuple(npp), \
                        f"({sa},{ia}): jax {jp} vs numpy {npp}"
                    seen[jp[0]] += 1
    assert seen[0] > 150 and seen[2] > 0, seen


def test_bfs_twins_agree():
    """The geodesic planes must match, including the BFS_ITERS cap, or the
    deployed bot sees a field the trained policy never saw."""
    rng = np.random.default_rng(0)
    for _ in range(4):
        passable = rng.random((B, B)) > 0.25
        seed = np.zeros((B, B), bool)
        seed[rng.integers(B), rng.integers(B)] = True
        a = np.asarray(O.bfs_from(jnp.asarray(passable), jnp.asarray(seed)))
        b = np_obs.bfs_from(passable, seed)
        assert np.array_equal(a, b), "jax and numpy geodesics diverge"


def test_wire_frame_reproduces_the_padded_training_view(states):
    """The evaluator sends an H x W frame; training uses 21x21 padded with
    mountains.  Getting the padding wrong shifts every plane."""
    st = states[2]
    o = get_observation(st, 0)
    ty = np.zeros((B, B), np.int32)
    ty[np.asarray(o.mountains)] = np_obs.TYPE_MOUNTAIN
    ty[np.asarray(o.fog_cells)] = np_obs.TYPE_FOG
    ty[np.asarray(o.structures_in_fog)] = np_obs.TYPE_SIF
    ty[np.asarray(o.castles)] = np_obs.TYPE_CASTLE
    ty[np.asarray(o.generals)] = np_obs.TYPE_GENERAL
    ty[np.asarray(o.neutral_cells) & (ty == 0)] = np_obs.TYPE_PLAIN
    ow = np.where(np.asarray(o.owned_cells), 1,
                  np.where(np.asarray(o.opponent_cells), 2, 0))
    f = np_obs.frame_to_obs(int(o.timestep), int(o.owned_land_count),
                            int(o.owned_army_count), int(o.opponent_land_count),
                            int(o.opponent_army_count), ty, ow,
                            np.asarray(o.armies))
    assert np.array_equal(f.owned_cells, np.asarray(o.owned_cells))
    assert np.array_equal(f.opponent_cells, np.asarray(o.opponent_cells))
    assert f.timestep == int(o.timestep)
