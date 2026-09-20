"""Observation encoding -- on device, one seat, fully static-shaped for vmap.

Under fog the current frame is not a sufficient statistic, so the encoder
carries `Memory` across ticks: last-seen enemy ownership and army, staleness,
a sticky enemy-general sighting, and multi-timescale EMAs of army motion.  No
hand-built threat, strike or gather scores -- the network learns those.  The
enemy general is almost always one of the largest stacks in enemy land and sits
among the oldest untouched cells, so last-seen army and staleness are exactly
the two channels that carry the target-selection signal.

The one family of engineered features is geodesic distance (`bfs_from`): paths
around mountains run far longer than a 7-layer trunk can propagate, so the
network could only approximate them.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from generals.modifiers.build_castles import (BASE_COST, PROXIMITY_DECAY,
                                              PROXIMITY_PENALTY, _RADIUS)

from .config import (AGE_SCALE, ARMY_SCALE, B, BONUS_PERIOD, BOOL_IDX,
                     CASTLE_SCALE, EMA_ALPHAS, FLOAT_IDX, INT_BUILD,
                     N_BOOL, N_CELLS, N_DIRS, N_EMA,
                     N_FLOAT, N_INT, N_PACKED, N_PLANES, N_SCALAR, N_SERIES,
                     N_SRC, N_TSAMP, SCALAR_IDX,
                     STRUCT_PERIOD, TOTAL_SCALE, TRING, TSCALES)

# plain Python, not a jnp array: these index a static shift, and under a
# scan trace `int(jnp_array[k])` raises rather than folding to a constant.
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _price_kernel() -> jnp.ndarray:
    """(2R+1, 2R+1) manhattan surcharge kernel: max(0, 14 - 2d)."""
    r = jnp.arange(-_RADIUS, _RADIUS + 1)
    d = jnp.abs(r)[:, None] + jnp.abs(r)[None, :]
    return jnp.maximum(0, PROXIMITY_PENALTY - PROXIMITY_DECAY * d).astype(jnp.float32)


_KERNEL = _price_kernel()


def build_price(own_structures: jnp.ndarray) -> jnp.ndarray:
    """(B, B) bool own structures -> (B, B) f32 castle price per cell.

    Mirrors build_castles.build_cost_grid exactly: 35 everywhere plus the
    surcharge kernel summed over the builder's own general and castles.  A build
    7+ cells from all your structures costs exactly 35, which is why strong
    players space their castles 7 apart.
    """
    x = own_structures.astype(jnp.float32)[None, None]
    k = _KERNEL[None, None]
    surcharge = lax.conv_general_dilated(
        x, k, window_strides=(1, 1), padding=((_RADIUS, _RADIUS),) * 2)
    return BASE_COST + surcharge[0, 0]


def neighbor(x: jnp.ndarray, dr: int, dc: int, fill) -> jnp.ndarray:
    """result[r, c] = x[r+dr, c+dc], `fill` outside the board."""
    pad = jnp.pad(x, 1, constant_values=fill)
    return lax.dynamic_slice(pad, (1 + dr, 1 + dc), (B, B))


class Memory(NamedTuple):
    """Per-seat, per-env persistent state.  Reset with `init_memory()`."""
    ghost_owner: jnp.ndarray      # (B, B) bool  last-seen enemy ownership
    ghost_army: jnp.ndarray       # (B, B) f32   last-seen army on those cells
    last_seen_t: jnp.ndarray      # (B, B) i32   -1 == never observed
    ever_seen: jnp.ndarray        # (B, B) bool
    enemy_general: jnp.ndarray    # (B, B) bool  sticky sighting
    ema: jnp.ndarray              # (N_EMA, 2, B, B) f32, multi-timescale motion


class Rings(NamedTuple):
    """Per-seat multi-scale history of the five aggregate series.

    (N_SERIES, n_scales, TRING) f32, already normalised.  Scale k holds every
    TSCALES[k]-th sample, so the four rings together span 16 / 64 / 256 / 1024
    ticks at 1 / 4 / 16 / 64 tick resolution off 320 numbers.

    Chosen over a raw history buffer because it is O(1) per tick and storable:
    the seed bank injects positions mid-game, so the representation must be
    something a seed can carry.  An exponential resample of a raw buffer is
    neither -- reconstructing it at injection would need the buffer itself."""
    buf: jnp.ndarray              # (N_SERIES, len(TSCALES), TRING) f32


def init_rings() -> Rings:
    return Rings(jnp.zeros((N_SERIES, len(TSCALES), TRING), jnp.float32))


def series_values(obs, own_castles: jnp.ndarray) -> jnp.ndarray:
    """(N_SERIES,) f32, normalised on the same scales as the matching planes so
    the encoder and the board see the quantities in the same units."""
    return jnp.stack([
        jnp.log1p(obs.owned_army_count.astype(jnp.float32)) / TOTAL_SCALE,
        obs.owned_land_count.astype(jnp.float32) / (B * B),
        own_castles.sum().astype(jnp.float32) / CASTLE_SCALE,
        jnp.log1p(obs.opponent_army_count.astype(jnp.float32)) / TOTAL_SCALE,
        obs.opponent_land_count.astype(jnp.float32) / (B * B),
    ])


def update_rings(rings: Rings, t, vals: jnp.ndarray) -> Rings:
    """Push `vals` into every ring whose stride divides t.  Exact, not a
    resample: ring k is the series sampled every TSCALES[k] ticks."""
    out = []
    for k, stride in enumerate(TSCALES):
        cur = rings.buf[:, k]
        rolled = jnp.roll(cur, -1, axis=-1).at[:, -1].set(vals)
        out.append(jnp.where(t % stride == 0, rolled, cur))
    return Rings(jnp.stack(out, axis=1))


def push(rings: Rings, obs) -> Rings:
    """One tick of the five series into the rings.  Mirrors `update` for
    `Memory`: called once per seat per tick, before `series_of`."""
    return update_rings(rings, obs.timestep,
                        series_values(obs, obs.castles & obs.owned_cells))


def series_of(rings: Rings) -> jnp.ndarray:
    """(N_SERIES, N_TSAMP) -- what the TemporalEncoder consumes."""
    return rings.buf.reshape(N_SERIES, N_TSAMP)


def init_memory() -> Memory:
    z = jnp.zeros((B, B), jnp.float32)
    f = jnp.zeros((B, B), bool)
    return Memory(f, z, jnp.full((B, B), -1, jnp.int32), f, f,
                  jnp.zeros((N_EMA, 2, B, B), jnp.float32))


def _visible(obs) -> jnp.ndarray:
    """Cells this seat can actually see this tick.

    The engine splits the unseen board into `fog_cells` (nothing known) and
    `structures_in_fog` (an obstacle is known to be there, nothing else), so
    visibility is the complement of their union.
    """
    return ~obs.fog_cells & ~obs.structures_in_fog


def update(mem: Memory, obs) -> Memory:
    vis = _visible(obs)
    enemy_army_now = jnp.where(obs.opponent_cells, obs.armies, 0).astype(jnp.float32)
    ghost_army = jnp.where(vis, enemy_army_now, mem.ghost_army)

    # Multi-timescale motion.  EMA'd in the same log1p/ARMY_SCALE units the
    # matching planes use, so `x - E_k` is a difference the trunk can take
    # directly rather than a comparison across two scalings.
    #
    # The two fields do not measure the same thing.  `own_army` is always
    # visible, so its EMAs are true motion.  `ghost_army` is frozen while a
    # cell sits in fog, so its EMA converges to that frozen value and
    # `ghost - E_k(ghost)` is a belief-revision signal -- "what I just learned
    # differs from what I have believed for k ticks" -- not enemy motion.
    own_army = jnp.where(obs.owned_cells, obs.armies, 0).astype(jnp.float32)
    cur = jnp.stack([jnp.log1p(own_army), jnp.log1p(ghost_army)]) / ARMY_SCALE
    a = jnp.asarray(EMA_ALPHAS, jnp.float32)[:, None, None, None]
    return Memory(
        ghost_owner=jnp.where(vis, obs.opponent_cells, mem.ghost_owner),
        ghost_army=ghost_army,
        last_seen_t=jnp.where(vis, obs.timestep.astype(jnp.int32), mem.last_seen_t),
        ever_seen=mem.ever_seen | vis,
        enemy_general=mem.enemy_general | (obs.generals & obs.opponent_cells),
        ema=mem.ema + a * (cur[None] - mem.ema),
    )


BFS_ITERS = 64        # longest geodesic seen on competition boards is 55
UNREACHABLE = 1 << 15


def bfs_from(passable: jnp.ndarray, seeds: jnp.ndarray) -> jnp.ndarray:
    """Geodesic distance to the nearest seeded cell, for K seed masks at once.

    `seeds` is (B, B) or (K, B, B); the result matches.  The K fields are
    relaxed in one scan rather than K scans: the relaxation is 64 sequential
    steps and the rollout is latency-bound, so three separate calls put 192
    dependent steps on the critical path where one batched call puts 64.  Same
    FLOPs, a third of the depth, ~9% of rollout time.

    `passable` must be the agent's known terrain (`~obs.mountains`, which the
    engine fog-masks), never ground truth, or the field encodes passability the
    agent has not discovered.
    """
    flat = seeds.ndim == 2
    sd = seeds[None] if flat else seeds
    d = jnp.where(sd & passable[None], 0, UNREACHABLE).astype(jnp.int32)

    def relax(d, _):
        best = d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = jnp.roll(d, (dr, dc), (1, 2))
            if dr == 1:
                shifted = shifted.at[:, 0].set(UNREACHABLE)
            elif dr == -1:
                shifted = shifted.at[:, B - 1].set(UNREACHABLE)
            elif dc == 1:
                shifted = shifted.at[:, :, 0].set(UNREACHABLE)
            else:
                shifted = shifted.at[:, :, B - 1].set(UNREACHABLE)
            best = jnp.minimum(best, shifted + 1)
        return jnp.where(passable[None], jnp.minimum(d, best), UNREACHABLE), None

    d, _ = jax.lax.scan(relax, d, None, length=BFS_ITERS)
    return d[0] if flat else d


def _dists(passable, seeds) -> jnp.ndarray:
    """K normalised geodesic planes in one scan: 0 on a seeded cell, 1 far or
    unreachable.  An empty seed yields an all-1 plane, which is the honest
    reading -- "no such cell is known" -- rather than a spurious zero."""
    return jnp.minimum(bfs_from(passable, seeds), BFS_ITERS) / float(BFS_ITERS)


def encode(mem: Memory, obs) -> jnp.ndarray:
    """(Memory already updated for this tick, Observation) -> (N_PLANES, B, B).

    Plane order is `config.PLANES` and is load-bearing: the numpy deployment
    twin indexes the same order, pinned by the parity test.
    """
    t = obs.timestep.astype(jnp.float32)
    own_army = jnp.where(obs.owned_cells, obs.armies, 0).astype(jnp.float32)
    enemy_army = jnp.where(obs.opponent_cells, obs.armies, 0).astype(jnp.float32)
    age = jnp.where(mem.ever_seen,
                    t - mem.last_seen_t.astype(jnp.float32),
                    jnp.inf)
    own_structures = (obs.castles | obs.generals) & obs.owned_cells
    passable = ~obs.mountains

    def sc(x):  # broadcast a scalar to a plane
        return jnp.full((B, B), x, jnp.float32)

    planes = [
        obs.owned_cells,                                   # own_cells
        obs.opponent_cells,                                # enemy_cells
        obs.neutral_cells,                                 # neutral
        obs.mountains,                                     # mountains
        obs.castles & obs.owned_cells,                     # own_castles
        obs.castles & ~obs.owned_cells,                    # other_castles
        obs.generals & obs.owned_cells,                    # own_general
        obs.fog_cells,                                     # fog
        obs.structures_in_fog,                             # structures_in_fog
        mem.ever_seen,                                     # ever_seen
        jnp.log1p(own_army) / ARMY_SCALE,                  # own_army
        jnp.log1p(enemy_army) / ARMY_SCALE,                # enemy_army
        mem.ghost_owner,                                   # ghost_owner
        jnp.log1p(mem.ghost_army) / ARMY_SCALE,            # ghost_army
        jnp.tanh(age / AGE_SCALE),                         # staleness (1 if never)
        mem.enemy_general,                                 # enemy_general_seen
        # field-major, matching PLANES: all own scales, then all enemy scales
        *mem.ema[:, 0], *mem.ema[:, 1],                    # ema_own_*, ema_enemy_*
        # Geodesic fields.  `passable` is the agent's known terrain: the
        # engine fog-masks observed mountains, so nothing here encodes ground
        # the agent has not seen.
        *_dists(passable, jnp.stack([                      # one batched scan
            mem.enemy_general,                             # dist_enemy_general
            obs.generals & obs.owned_cells,                # dist_own_general
            obs.neutral_cells,                             # dist_frontier
        ])),
        (build_price(own_structures) - BASE_COST) / BASE_COST,
        sc((BONUS_PERIOD - t % BONUS_PERIOD) / BONUS_PERIOD),   # bonus_in
        sc((STRUCT_PERIOD - t % STRUCT_PERIOD) / STRUCT_PERIOD),  # struct_in
        sc(obs.owned_land_count.astype(jnp.float32) / (B * B)),
        sc(jnp.log1p(obs.owned_army_count.astype(jnp.float32)) / TOTAL_SCALE),
        sc(jnp.log1p(obs.opponent_army_count.astype(jnp.float32)) / TOTAL_SCALE),
    ]
    out = jnp.stack([p.astype(jnp.float32) for p in planes])
    assert out.shape == (N_PLANES, B, B), (out.shape, N_PLANES)
    return out


# --- trajectory-buffer codec ------------------------------------------------
# `planes` dominates device memory and is what caps `envs`, not FLOPs.  These
# two functions are the only place plane storage differs from plane semantics:
# the network's input contract is unchanged, and `unpack(pack(x)) == x` exactly
# for the bool and scalar groups (the float group is fp16 either way).

_BOOL_I = jnp.asarray(BOOL_IDX)
_FLOAT_I = jnp.asarray(FLOAT_IDX)
_SCALAR_I = jnp.asarray(SCALAR_IDX)


def pack(x: jnp.ndarray):
    """(N_PLANES, B, B) -> (bits (N_PACKED, B, B) u8, f (N_FLOAT, B, B) f16,
    s (N_SCALAR,) f16).  Lossless for the bool and scalar groups."""
    bits = jnp.packbits(x[_BOOL_I] > 0.5, axis=0)
    return (bits, x[_FLOAT_I].astype(jnp.float16),
            x[_SCALAR_I, 0, 0].astype(jnp.float16))


def unpack(bits, f, sc) -> jnp.ndarray:
    """Inverse of `pack`.  The scalars are broadcast back to full planes here
    rather than stored as 441 identical copies per sample."""
    b = jnp.unpackbits(bits, axis=0, count=N_BOOL).astype(f.dtype)
    out = jnp.zeros((N_PLANES, B, B), f.dtype)
    out = out.at[_BOOL_I].set(b)
    out = out.at[_FLOAT_I].set(f)
    return out.at[_SCALAR_I].set(sc[:, None, None])


def legal_mask(obs):
    """-> (N_CELLS, N_INT) bool: exactly which (cell, intent) pairs act.

    One grid, not two marginals.  Separate source and intent masks can only say
    "this cell can act" and "some cell can serve this intent", so a pair drawn
    from the two can still be illegal and silently become a pass.  Masking the
    joint grid removes that by construction: there is no legal action the
    engine will refuse.  This is affordable because the source is an explicit
    cell, so legality is a pure function of the observation.
    """
    movable = obs.owned_cells & (obs.armies > 1)
    passable = ~obs.mountains

    # a direction is legal for this cell, not for some cell
    step = jnp.stack([movable & neighbor(passable, dr, dc, False)
                      for dr, dc in DIRS], -1).reshape(N_CELLS, N_DIRS)

    own_structures = (obs.castles | obs.generals) & obs.owned_cells
    plain = obs.owned_cells & ~obs.castles & ~obs.generals
    can_build = (plain & (obs.armies >= build_price(own_structures))
                 ).reshape(N_CELLS, 1)

    # Half-moves are masked out: top ladder players use them for 0-2% of
    # moves, and an unconstrained policy over-uses them.  The head keeps its
    # (441, 10) shape; the mask simply never offers intents 4-7, in training
    # and deployment identically.
    no_half = jnp.zeros_like(step)
    real = jnp.concatenate([step, no_half, can_build], -1)
    # Pass is offered wherever the cell can do something real, so the marginal
    # over cells stays meaningful; on a board where nothing can act it is
    # offered everywhere so the distribution is still defined.
    any_real = real.any(-1)
    pass_ok = (any_real | ~any_real.any())[:, None]
    grid = jnp.concatenate([real, pass_ok], -1)
    assert grid.shape == (N_CELLS, N_INT)
    return grid


def priv_planes(state, seat: int) -> jnp.ndarray:
    """(N_PRIV, B, B) full-state planes for the centralized critic only.

    The value of a strike or gather decision depends on the hidden general and
    hidden enemy stacks.  These never reach the actor's input, so nothing
    privileged can leak into the deployed policy.
    """
    opp = 1 - seat
    enemy_owner = state.ownership[opp]
    return jnp.stack([
        (state.generals & enemy_owner).astype(jnp.float32),
        jnp.log1p(jnp.where(enemy_owner, state.armies, 0).astype(jnp.float32))
        / ARMY_SCALE,
        enemy_owner.astype(jnp.float32),
    ])


# vmapped forms used by the rollout: (n_env, ...) over the leading axis
pack_v = jax.vmap(pack)
unpack_v = jax.vmap(unpack)
update_v = jax.vmap(update)
push_v = jax.vmap(push)
series_of_v = jax.vmap(series_of)
encode_v = jax.vmap(encode)
legal_mask_v = jax.vmap(legal_mask)
priv_planes_v = jax.vmap(priv_planes, in_axes=(0, None))
