"""Numpy twin of `train/obs.py` for deployment -- single frame, no JAX.

The submitted bot runs on one CPU core with a 150 ms budget and cannot import
JAX, so the observation encoder exists twice.  Parity with the JAX original is
pinned by tests/test_np.py: identical planes, identical legal mask, identical
action decoding.  If they drift, the bot we upload is not the bot we trained.

Plane order comes from `config.PLANES` and is load-bearing.
"""
from __future__ import annotations

import numpy as np

from .config import (AGE_SCALE, ARMY_SCALE, B, BONUS_PERIOD, CASTLE_SCALE,
                     EMA_ALPHAS, INT_BUILD, INT_DIR0, INT_PASS,
                     N_CELLS, N_DIRS, N_EMA, N_INT, N_PLANES, N_SERIES,
                     N_SRC, N_TSAMP, STRUCT_PERIOD,
                     TOTAL_SCALE, TRING, TSCALES)

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

BASE_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2
_RADIUS = (PROXIMITY_PENALTY - 1) // PROXIMITY_DECAY      # 6


def _price_kernel() -> np.ndarray:
    r = np.arange(-_RADIUS, _RADIUS + 1)
    d = np.abs(r)[:, None] + np.abs(r)[None, :]
    return np.maximum(0, PROXIMITY_PENALTY - PROXIMITY_DECAY * d).astype(np.float32)


_KERNEL = _price_kernel()


def build_price(own_structures: np.ndarray) -> np.ndarray:
    """(B, B) bool -> (B, B) f32 castle price; mirrors build_castles."""
    x = own_structures.astype(np.float32)
    pad = np.pad(x, _RADIUS)
    out = np.full((B, B), float(BASE_COST), np.float32)
    for di in range(-_RADIUS, _RADIUS + 1):
        for dj in range(-_RADIUS, _RADIUS + 1):
            w = _KERNEL[di + _RADIUS, dj + _RADIUS]
            if w > 0:
                out += w * pad[_RADIUS + di:_RADIUS + di + B,
                               _RADIUS + dj:_RADIUS + dj + B]
    return out


class Memory:
    """Persistent per-game memory; the numpy mirror of obs.Memory."""

    __slots__ = ("ghost_owner", "ghost_army", "last_seen_t", "ever_seen",
                 "enemy_general", "ema")

    def __init__(self):
        z = lambda: np.zeros((B, B), np.float32)
        f = lambda: np.zeros((B, B), bool)
        self.ghost_owner = f()
        self.ghost_army = z()
        self.last_seen_t = np.full((B, B), -1, np.int32)
        self.ever_seen = f()
        self.enemy_general = f()
        self.ema = np.zeros((N_EMA, 2, B, B), np.float32)

    def update(self, o) -> None:
        vis = ~o.fog_cells & ~o.structures_in_fog
        enemy_army_now = np.where(o.opponent_cells, o.armies, 0).astype(np.float32)
        self.ghost_owner = np.where(vis, o.opponent_cells, self.ghost_owner)
        self.ghost_army = np.where(vis, enemy_army_now, self.ghost_army)
        self.last_seen_t = np.where(vis, int(o.timestep), self.last_seen_t)
        self.ever_seen = self.ever_seen | vis
        self.enemy_general = self.enemy_general | (o.generals & o.opponent_cells)
        # mirrors obs.update exactly -- see the note there on why the two
        # fields do not measure the same thing (own is motion, ghost is
        # belief revision).
        own_army = np.where(o.owned_cells, o.armies, 0).astype(np.float32)
        cur = np.stack([np.log1p(own_army),
                        np.log1p(self.ghost_army)]) / ARMY_SCALE
        a = np.asarray(EMA_ALPHAS, np.float32)[:, None, None, None]
        self.ema = self.ema + a * (cur[None] - self.ema)


class Rings:
    """numpy twin of `obs.Rings` -- the deployed bot must maintain the same
    multi-scale history or it feeds the network a distribution it never saw."""
    __slots__ = ("buf",)

    def __init__(self):
        self.buf = np.zeros((N_SERIES, len(TSCALES), TRING), np.float32)

    def push(self, o) -> None:
        vals = np.array([
            np.log1p(float(o.owned_army_count)) / TOTAL_SCALE,
            float(o.owned_land_count) / (B * B),
            float((o.castles & o.owned_cells).sum()) / CASTLE_SCALE,
            np.log1p(float(o.opponent_army_count)) / TOTAL_SCALE,
            float(o.opponent_land_count) / (B * B),
        ], np.float32)
        t = int(o.timestep)
        for k, stride in enumerate(TSCALES):
            if t % stride == 0:
                self.buf[:, k, :-1] = self.buf[:, k, 1:]
                self.buf[:, k, -1] = vals

    def series(self) -> np.ndarray:
        return self.buf.reshape(N_SERIES, N_TSAMP)


BFS_ITERS = 64
UNREACHABLE = 1 << 15


def bfs_from(passable: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """numpy twin of obs.bfs_from -- multi-source geodesic distance.

    The result must match the JAX min-plus relaxation exactly, including its
    BFS_ITERS cap: a cell further than 64 steps stays UNREACHABLE on the JAX
    side, so it must here too, or the shipped bot sees a field the trained
    policy never saw."""
    d = np.where(seed & passable, 0, UNREACHABLE).astype(np.int32)
    for _ in range(BFS_ITERS):
        best = d.copy()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(d, (dr, dc), (0, 1))
            if dr == 1:    sh[0] = UNREACHABLE
            elif dr == -1: sh[B - 1] = UNREACHABLE
            elif dc == 1:  sh[:, 0] = UNREACHABLE
            else:          sh[:, B - 1] = UNREACHABLE
            best = np.minimum(best, sh + 1)
        d = np.where(passable, np.minimum(d, best), UNREACHABLE)
    return d


def _dist(passable, seed) -> np.ndarray:
    return np.minimum(bfs_from(passable, seed), BFS_ITERS) / float(BFS_ITERS)


def _dists(passable, seeds) -> np.ndarray:
    return np.stack([_dist(passable, s) for s in seeds])


def encode(mem: Memory, o) -> np.ndarray:
    """(Memory already updated for this tick, obs) -> (N_PLANES, B, B) f32."""
    t = float(o.timestep)
    own_army = np.where(o.owned_cells, o.armies, 0).astype(np.float32)
    enemy_army = np.where(o.opponent_cells, o.armies, 0).astype(np.float32)
    age = np.where(mem.ever_seen, t - mem.last_seen_t.astype(np.float32), np.inf)
    own_structures = (o.castles | o.generals) & o.owned_cells
    sc = lambda x: np.full((B, B), x, np.float32)

    planes = [
        o.owned_cells, o.opponent_cells, o.neutral_cells, o.mountains,
        o.castles & o.owned_cells, o.castles & ~o.owned_cells,
        o.generals & o.owned_cells, o.fog_cells, o.structures_in_fog,
        mem.ever_seen,
        np.log1p(own_army) / ARMY_SCALE,
        np.log1p(enemy_army) / ARMY_SCALE,
        mem.ghost_owner,
        np.log1p(mem.ghost_army) / ARMY_SCALE,
        np.tanh(age / AGE_SCALE),
        mem.enemy_general,
        *mem.ema[:, 0], *mem.ema[:, 1],           # ema_own_*, ema_enemy_*
        *_dists(~o.mountains, np.stack([                  # one batched pass
            mem.enemy_general,                            # dist_enemy_general
            o.generals & o.owned_cells,                   # dist_own_general
            o.neutral_cells,                              # dist_frontier
        ])),
        (build_price(own_structures) - BASE_COST) / BASE_COST,
        sc((BONUS_PERIOD - t % BONUS_PERIOD) / BONUS_PERIOD),     # bonus_in
        sc((STRUCT_PERIOD - t % STRUCT_PERIOD) / STRUCT_PERIOD),  # struct_in
        sc(float(o.owned_land_count) / (B * B)),
        sc(np.log1p(float(o.owned_army_count)) / TOTAL_SCALE),
        sc(np.log1p(float(o.opponent_army_count)) / TOTAL_SCALE),
    ]
    out = np.stack([np.asarray(p, np.float32) for p in planes])
    assert out.shape == (N_PLANES, B, B)
    return out


def neighbor(x: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """result[r, c] = x[r+dr, c+dc], `fill` outside the board."""
    pad = np.pad(x, 1, constant_values=fill)
    return pad[1 + dr:1 + dr + B, 1 + dc:1 + dc + B]


def legal_mask(frame):
    """-> (N_CELLS, N_INT) bool.  Mirrors `train/obs.legal_mask` exactly."""
    movable = frame.owned_cells & (frame.armies > 1)
    passable = ~frame.mountains

    step = np.stack([movable & neighbor(passable, dr, dc, False)
                     for dr, dc in DIRS], -1).reshape(N_CELLS, N_DIRS)

    own_structures = (frame.castles | frame.generals) & frame.owned_cells
    plain = frame.owned_cells & ~frame.castles & ~frame.generals
    can_build = (plain & (frame.armies >= build_price(own_structures))
                 ).reshape(N_CELLS, 1)

    # half-moves are masked out, as in train/obs.legal_mask
    real = np.concatenate([step, np.zeros_like(step), can_build], -1)
    any_real = real.any(-1)
    pass_ok = (any_real | (not any_real.any()))[:, None]
    grid = np.concatenate([real, pass_ok], -1)
    assert grid.shape == (N_CELLS, N_INT)
    return grid


# --- wire frame -> observation ----------------------------------------------
#
# The evaluator sends an H x W frame (18-21 per side) but the policy is trained
# on 21x21 boards that the env pads with mountains.  The agent must reproduce
# that padding or every plane is shifted relative to training.  Out-of-board
# cells are mountains in the true state, so they appear as `mountains` where the
# seat has vision and as `structures_in_fog` otherwise -- exactly what the
# engine's observation would emit for them.

TYPE_FOG, TYPE_PLAIN, TYPE_MOUNTAIN = 0, 1, 2
TYPE_CASTLE, TYPE_GENERAL, TYPE_SIF = 3, 4, 5
OWNER_ME, OWNER_OPP = 1, 2


def _dilate(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for a in range(3):
        for b_ in range(3):
            out |= p[a:a + B, b_:b_ + B]
    return out


class Frame:
    """Observation fields the encoder needs, in the engine's own semantics."""

    __slots__ = ("armies", "generals", "castles", "mountains", "neutral_cells",
                 "owned_cells", "opponent_cells", "fog_cells",
                 "structures_in_fog", "owned_land_count", "owned_army_count",
                 "opponent_land_count", "opponent_army_count", "timestep")


def frame_to_obs(turn, my_land, my_army, opp_land, opp_army,
                 type_grid, owner_grid, army_grid) -> Frame:
    H, W = type_grid.shape
    pad = np.ones((B, B), bool)
    pad[:H, :W] = False                      # True where the board does not exist

    ty = np.zeros((B, B), np.int32); ty[:H, :W] = type_grid
    ow = np.zeros((B, B), np.int32); ow[:H, :W] = owner_grid
    am = np.zeros((B, B), np.int32); am[:H, :W] = army_grid

    f = Frame()
    f.owned_cells = ow == OWNER_ME
    f.opponent_cells = ow == OWNER_OPP
    visible = _dilate(f.owned_cells)

    f.mountains = (ty == TYPE_MOUNTAIN) | (pad & visible)
    f.structures_in_fog = (ty == TYPE_SIF) | (pad & ~visible)
    f.fog_cells = (ty == TYPE_FOG) & ~pad
    f.castles = (ty == TYPE_CASTLE) & ~pad
    f.generals = (ty == TYPE_GENERAL) & ~pad
    # The engine's neutral ownership is `passable & ~general`, so mountains
    # and generals are not neutral -- and padded cells are mountains, so they
    # never are either.  A neutral cell on the wire is visible, unowned and
    # passable.
    f.neutral_cells = (ow == 0) & ((ty == TYPE_PLAIN) | (ty == TYPE_CASTLE)) \
        & ~pad
    f.armies = am
    f.owned_land_count = int(my_land)
    f.owned_army_count = int(my_army)
    f.opponent_land_count = int(opp_land)
    f.opponent_army_count = int(opp_army)
    f.timestep = int(turn)
    return f
