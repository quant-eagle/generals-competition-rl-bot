"""Exact observation-memory reconstruction from full-state replays.

A seeded RL episode cannot just drop the policy into a mid-game board: the
fogged observation is only Markov together with its memory (last-seen ownership
and army, staleness, the sticky general sighting).  Starting from a blank memory
would present a state no real game ever produces -- every cell never seen while
the seat owns sixty of them.

Replays store per-tick `armies`/`owners` and a cumulative `seen` mask, so
memory is recoverable exactly rather than approximated:

  * per-tick visibility is the 3x3 dilation of that seat's owned cells;
  * `last_seen[t]` is then a running maximum over visible ticks;
  * ghosts are a gather of the board at each cell's own last-seen tick.

Used by tools/build_rl_seeds.py to build the bank the rollout injects from.
"""
from __future__ import annotations

import numpy as np

from .config import (ARMY_SCALE, B, CASTLE_SCALE, EMA_ALPHAS, N_EMA,
                     N_SERIES, TOTAL_SCALE, TRING, TSCALES)

DIRS3 = [(a, b) for a in range(3) for b in range(3)]


def dilate(mask: np.ndarray) -> np.ndarray:
    """3x3 max-pool over the trailing two axes — the engine's vision rule."""
    p = np.pad(mask, [(0, 0)] * (mask.ndim - 2) + [(1, 1), (1, 1)])
    out = np.zeros_like(mask)
    h, w = mask.shape[-2:]
    for a, b in DIRS3:
        out |= p[..., a:a + h, b:b + w]
    return out


class GameMemory:
    """Reconstructed per-tick memory for one seat of one replay."""

    __slots__ = ("last_seen", "ever_seen", "ghost_owner", "ghost_army",
                 "egen_seen", "vis", "T", "ema", "rings")

    def __init__(self, armies, owners, generals, seat, castles=None):
        T = armies.shape[0]
        opp = 1 - seat
        self.T = T
        self.vis = dilate(owners == seat)

        tt = np.where(self.vis, np.arange(T, dtype=np.int32)[:, None, None], -1)
        self.last_seen = np.maximum.accumulate(tt, axis=0)
        self.ever_seen = self.last_seen >= 0

        rr, cc = np.meshgrid(np.arange(B), np.arange(B), indexing="ij")
        idx = np.clip(self.last_seen, 0, T - 1)
        self.ghost_owner = (owners[idx, rr, cc] == opp) & self.ever_seen
        self.ghost_army = np.where(self.ghost_owner,
                                   armies[idx, rr, cc], 0).astype(np.int32)

        # generals never move, so a sighting is just "was that cell ever seen"
        gr, gc = general_pos(generals, owners, opp)
        self.egen_seen = self.ever_seen[:, gr, gc]

        self.ema = self.rings = None
        if castles is not None:
            self._temporal(armies, owners, castles, seat, opp)

    def _temporal(self, armies, owners, castles, seat, opp):
        """Replay the two temporal states forward: the motion EMAs and the
        economy rings.  Both are recursions, so they cannot be gathered like
        the ghosts -- they have to be stepped tick by tick, exactly as
        obs.update and obs.push do it, or an injected seed continues a history
        the environment would never have produced."""
        T = self.T
        own = owners == seat
        own_army = np.where(own, armies, 0).astype(np.float32)
        cur = np.stack([np.log1p(own_army),
                        np.log1p(self.ghost_army.astype(np.float32))],
                       axis=1) / ARMY_SCALE              # (T, 2, B, B)

        a = np.asarray(EMA_ALPHAS, np.float32)[:, None, None, None]
        self.ema = np.zeros((T, N_EMA, 2, B, B), np.float32)
        e = np.zeros((N_EMA, 2, B, B), np.float32)
        for t in range(T):
            e = e + a * (cur[t][None] - e)
            self.ema[t] = e

        # the five series, in obs.series_values order and units
        opp_m = owners == opp
        vals = np.stack([
            np.log1p((armies * own).sum((1, 2)).astype(np.float32)) / TOTAL_SCALE,
            own.sum((1, 2)).astype(np.float32) / (B * B),
            (castles & own).sum((1, 2)).astype(np.float32) / CASTLE_SCALE,
            np.log1p((armies * opp_m).sum((1, 2)).astype(np.float32)) / TOTAL_SCALE,
            opp_m.sum((1, 2)).astype(np.float32) / (B * B),
        ], axis=1)                                        # (T, N_SERIES)

        self.rings = np.zeros((T, N_SERIES, len(TSCALES), TRING), np.float32)
        r = np.zeros((N_SERIES, len(TSCALES), TRING), np.float32)
        for t in range(T):
            for k, stride in enumerate(TSCALES):
                if t % stride == 0:
                    r[:, k, :-1] = r[:, k, 1:]
                    r[:, k, -1] = vals[t]
            self.rings[t] = r

    def at(self, t: int) -> dict[str, np.ndarray]:
        return {
            "last_seen_t": self.last_seen[t],
            "ever_seen": self.ever_seen[t],
            "ghost_owner": self.ghost_owner[t],
            "ghost_army": self.ghost_army[t].astype(np.float32),
            "egen_seen": bool(self.egen_seen[t]),
            "ema": None if self.ema is None else self.ema[t],
            "rings": None if self.rings is None else self.rings[t],
        }


def general_pos(generals: np.ndarray, owners: np.ndarray, seat: int
                ) -> tuple[int, int]:
    """(row, col) of `seat`'s general.  Ownership is read at t=0, before any
    capture could have changed it."""
    m = generals & (owners[0] == seat)
    assert m.sum() == 1, f"expected exactly one general for seat {seat}, got {m.sum()}"
    flat = int(np.argmax(m))
    return flat // B, flat % B


def validate(mem: GameMemory, stored_seen: np.ndarray) -> None:
    """Cross-check the reconstruction against the replay's own cumulative mask.

    They must agree exactly: `ever_seen` is the accumulation of the same
    per-tick visibility the replay recorded.
    """
    if not np.array_equal(mem.ever_seen, stored_seen):
        bad = int((mem.ever_seen != stored_seen).sum())
        raise AssertionError(
            f"reconstructed ever_seen disagrees with replay in {bad} cells")
