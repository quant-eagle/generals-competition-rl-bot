"""Board curriculum: identical competition rules, varying only difficulty.

Stages widen two axes together -- board size and the walking distance between
generals -- from a cramped board where the enemy is a few steps away to the true
competition distribution.  The point is credit assignment: on a close board a
game resolves in tens of ticks, so the terminal reward is dense; at competition
distance the median ladder game lasts ~320 ticks.

Every other rule is pinned to the `competition` preset (mountain density, castle
generation, build-castles, deathtouch, truncation, fog, pad_to=21) so a stage
differs from the real game in difficulty only.  `pad_to` is 21 everywhere, so
all pools share a shape and switching stages never retraces the rollout.

The last stage is the competition distribution.  Earlier stages are scaffolding
and training must finish on S4, or the shipped policy is tuned to boards the
evaluator never generates.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
from generals import GeneralsEnv

from .config import DEATHTOUCH_TURN as DEATHTOUCH
from .config import DRAW_TURN as TRUNCATION

PAD = 21


class Stage(NamedTuple):
    name: str
    min_size: int
    max_size: int
    min_dist: int
    max_dist: int | None


# min_generals_distance is a walking (BFS) distance in the generator, not a
# straight line.
STAGES = (
    Stage("S0-cramped",   10, 12,  4,  7),
    Stage("S1-near",      12, 15,  6, 10),
    Stage("S2-mid",       15, 18, 10, 14),
    Stage("S3-far",       18, 21, 14, 17),
    Stage("S4-competition", 18, 21, 17, None),   # the real distribution
)


_ENV_CACHE: dict[tuple[str, int], GeneralsEnv] = {}


def make_env(stage: Stage, pool_size: int = 2048) -> GeneralsEnv:
    """A competition env with only the difficulty axis changed.

    `mode="competition"` is deliberately not used: a named mode is authoritative
    and would override the distance we are varying.  Every other field is copied
    from that preset verbatim.
    """
    # Cached per (stage, size).  `reset` is jitted per env instance, so a fresh
    # env for every pool refresh recompiles it (~1 min).  Reusing the instance
    # makes a refresh ~60 ms, which makes board rotation effectively free.
    hit = _ENV_CACHE.get((stage.name, pool_size))
    if hit is not None:
        return hit
    env = GeneralsEnv(
        pool_size=pool_size,
        min_grid_size=stage.min_size,
        max_grid_size=stage.max_size,
        pad_to=PAD,
        truncation=TRUNCATION,
        perfect_info=False,
        mountain_density_range=(0.24, 0.26),
        num_castles_range=(9, 11),
        castle_val_range=(20, 26),
        min_generals_distance=stage.min_dist,
        max_generals_distance=stage.max_dist,
        build_castles=True,
        deathtouch_turn=DEATHTOUCH,
    )
    _ENV_CACHE[(stage.name, pool_size)] = env
    return env


def n_combos(stage: Stage) -> int:
    """How many (h, w) size combinations this stage spans."""
    n = stage.max_size - stage.min_size + 1
    return n * n


def make_pool(key, stage: Stage, size: int):
    """A pool of starting states for `stage`, padded to 21 like every other.

    `size` must be at least `n_combos(stage)`: the env splits its pool across
    every (h, w) combination, so a size below the combo count floors to zero
    boards per combo and yields an empty pool -- which would fail far away from
    here, as an out-of-range gather during the first reset.
    """
    combos = n_combos(stage)
    if size < combos:
        raise ValueError(
            f"pool size {size} < {combos} size-combos for {stage.name}; "
            f"the env would generate 0 boards per combo")
    pool, _ = make_env(stage, pool_size=size).reset(key)
    got = pool.armies.shape[0]
    if got == 0:
        raise RuntimeError(f"{stage.name}: generated an empty pool")
    return jax.tree.map(lambda x: x[:size], pool)


def stage_for(frac: float) -> int:
    """Which stage index at training fraction `frac` in [0, 1].

    Weighted so the competition stage gets the largest share: scaffolding is
    cheap to learn and the real distribution is what we are scored on.
    """
    cuts = (0.08, 0.20, 0.34, 0.50)     # S0..S3 boundaries; S4 takes the rest
    return sum(frac >= c for c in cuts)
