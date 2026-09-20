"""Replay decoder: omniscient state-sequence replays -> actions, fog views, shards.

Replays store full board states per tick and no actions.  This infers the action
pair behind every transition.  Pure NumPy plus engine imports: game logic is
never reimplemented -- every inferred pair is verified by stepping the real
engine transition (the build-castles and deathtouch modifiers around
`game.step`, the same composition the competition ruleset uses).

Replay schema: ticks[i] is the full state at engine time i (tick 0 = initial
state). Transition t -> t+1: builds resolve, both moves resolve (chase >
reinforce > smaller-army order), time becomes t+1, then growth runs at the new
time (structures +1 on even ticks, all owned tiles +1 on ticks % 50 == 0; both
stack) -- unless a winner was set, in which case the loser's cells transfer and
no growth happens.

Public API:
    load_replay(path)                       -> replay dict (.json or .json.zst)
    infer_actions(replay)                   -> list[(a0, a1)] 5-int wire tuples
    decode_replay(replay)                   -> DecodedReplay (actions + castle masks + diagnostics)
    fog_view(replay, tick_idx, player)      -> FogView (exact stdio frame contents)
    validate(replay)                        -> dict report (bit-exact engine replay gate)
    emit_dataset(manifest_filter=None, ...) -> dict summary; shards under data/shards/

Action wire format of the competition protocol: (kind, row, col, dir, split);
kind 0=move 1=pass 2=build; dir 0=up 1=down 2=left 3=right; split 0=all-but-one
1=half(floor). Identical to the engine's internal 5-int action vector.
"""
from __future__ import annotations

import argparse
import json
import time as _time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import zstandard

import jax
import jax.numpy as jnp

from generals.core import game
from generals.modifiers import build_castles as _bc
from generals.modifiers import deathtouch as _dt

# competition ruleset constants, as in the engine's competition preset
DEATHTOUCH_TURN = 800
TRUNCATION = 1200

PASS = (1, 0, 0, 0, 0)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right == game.DIRECTIONS

_DECODED_KEY = "__decoded__"

# build_cost_grid is engine code but not jitted at source; jit it here (player static).
_cost_grid_jit = jax.jit(_bc.build_cost_grid, static_argnums=1)


# --------------------------------------------------------------------------- io

def load_replay(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    if str(path).endswith(".zst"):
        raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 28)
    return json.loads(raw)


# ------------------------------------------------------------------ engine step

def _step_engine(state, a0, a1):
    """One competition-ruleset transition: builds resolve first (rewritten to
    passes), then the deathtouch-wrapped base step."""
    actions = jnp.asarray(np.array([a0, a1], dtype=np.int32))
    state, actions = _bc.apply_build_actions(state, actions)
    return _dt.step(state, actions, DEATHTOUCH_TURN)


# --------------------------------------------------------------------- results

@dataclass
class DecodedReplay:
    actions: list                 # list[(a0, a1)]; ambiguous slots hold (PASS, PASS)
    ambiguous: list               # tick indices t whose transition t->t+1 was unresolved
    castles_by_tick: np.ndarray   # (n_ticks, H, W) bool -- castle mask of state at tick i
    n_transitions: int
    engine_winner: int            # winner per engine replay (-1 draw/none)
    replay_winner: int            # winner field from the replay json (-1 if draw/absent)
    multimatch: list = field(default_factory=list)   # ticks where >1 distinct state matched
    recovered_castles: list = field(default_factory=list)  # (tick, [(r, c), ...])
    elapsed_s: float = 0.0

    @property
    def ambiguity_rate(self) -> float:
        return len(self.ambiguous) / max(1, self.n_transitions)


@dataclass
class FogView:
    """Exactly the stdio frame a player sees at one tick of the competition protocol."""
    type_grid: np.ndarray   # (H, W) int8: 0 fog 1 plain 2 mountain 3 castle 4 general 5 structure-in-fog
    owner_grid: np.ndarray  # (H, W) int8: 0 neutral/unknown 1 me 2 opponent
    army_grid: np.ndarray   # (H, W) int32: 0 outside vision
    scalars: tuple          # (turn, my_land, my_army, opp_land, opp_army)

    def to_frame(self) -> str:
        """Render the wire frame text, byte-identical to the engine's encoder."""
        lines = [" ".join(str(int(x)) for x in self.scalars)]
        for grid in (self.type_grid, self.owner_grid, self.army_grid):
            for row in grid:
                lines.append(" ".join(str(int(x)) for x in row))
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- decoder

class _Decoder:
    def __init__(self, replay: dict):
        self.replay = replay
        self.H = replay["dims"]["rows"]
        self.W = replay["dims"]["cols"]
        ticks = replay["ticks"]
        self.n_ticks = len(ticks)
        self.armies = np.array([t["armies"] for t in ticks], dtype=np.int32)
        self.owners = np.array([t["owners"] for t in ticks], dtype=np.int8)
        if self.armies.shape != (self.n_ticks, self.H, self.W):
            raise ValueError(f"tick grid shape mismatch: {self.armies.shape}")

        self.mountains = np.zeros((self.H, self.W), dtype=bool)
        for r, c in replay["mountains"]:
            self.mountains[r, c] = True
        if replay.get("castles"):
            raise ValueError("replay has neutral castles -- not a competition-mode replay")
        self.generals = np.zeros((self.H, self.W), dtype=bool)
        self.gen_pos = [tuple(replay["generals"][0]), tuple(replay["generals"][1])]
        for r, c in self.gen_pos:
            self.generals[r, c] = True
        self.passable = ~self.mountains

        rw = replay.get("winner")
        self.replay_winner = rw if rw in (0, 1) else -1

    # -- state helpers --------------------------------------------------------

    def initial_state(self):
        grid = np.zeros((self.H, self.W), dtype=np.int32)
        grid[self.mountains] = -2
        grid[self.gen_pos[0]] = 1
        grid[self.gen_pos[1]] = 2
        return game.create_initial_state(jnp.asarray(grid))

    def matches_tick(self, state, t: int) -> bool:
        if not np.array_equal(np.asarray(state.armies), self.armies[t]):
            return False
        own0 = np.asarray(state.ownership[0])
        own1 = np.asarray(state.ownership[1])
        og = np.full((self.H, self.W), -1, dtype=np.int8)
        og[own0] = 0
        og[own1] = 1
        return np.array_equal(og, self.owners[t])

    def state_from_tick(self, base_state, t: int):
        """Force-build an engine state from replay tick t (resync after an
        unresolved transition). Castle mask carried over from base_state."""
        own0 = jnp.asarray(self.owners[t] == 0)
        own1 = jnp.asarray(self.owners[t] == 1)
        neutral = jnp.asarray(self.passable) & ~own0 & ~own1
        return base_state._replace(
            armies=jnp.asarray(self.armies[t]),
            ownership=jnp.stack([own0, own1]),
            ownership_neutral=neutral,
            time=jnp.int32(t),
        )

    # -- diff analysis ---------------------------------------------------------

    def growth_grid(self, T: int, castles: np.ndarray) -> np.ndarray:
        """Army growth the engine applies at post-increment time T, using
        post-move ownership (= replay tick T)."""
        owned = self.owners[T] >= 0
        g = np.zeros((self.H, self.W), dtype=np.int32)
        if T % 50 == 0:
            g += owned
        if T % 2 == 0:
            g += (owned & (castles | self.generals))
        return g

    def touched_cells(self, t: int, castles: np.ndarray, growth: bool) -> np.ndarray:
        """Cells whose army (net of growth) or owner changed across t -> t+1.
        Executed moves always touch their source (army strictly drops or owner
        flips) so candidate enumeration from this mask is complete."""
        T = t + 1
        g = self.growth_grid(T, castles) if growth else 0
        resid = self.armies[T].astype(np.int64) - self.armies[t] - g
        return (resid != 0) | (self.owners[T] != self.owners[t])

    # -- candidate actions -----------------------------------------------------

    def move_cands(self, p: int, t: int, touched: np.ndarray,
                   resid: np.ndarray | None = None) -> list:
        """All candidate moves for player p, best-guess-first.

        Ordering is a pure heuristic (completeness untouched): moves whose
        destination is also touched and whose moved-army size matches the
        source's residual drop are tried first, which makes the engine-verified
        first match cheap on ordinary turns.
        """
        prev_a = self.armies[t]
        own = self.owners[t] == p
        scored = []
        for r, c in np.argwhere(touched & own & (prev_a >= 2)):
            a = prev_a[r, c]
            # split 0 moves a-1, split 1 moves a//2; identical when a == 2.
            splits = ((0, a - 1),) if a == 2 else ((0, a - 1), (1, a // 2))
            drop = -int(resid[r, c]) if resid is not None else None
            for d, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and self.passable[nr, nc]:
                    for s, k in splits:
                        score = 0
                        if touched[nr, nc]:
                            score += 2
                        if drop is not None and k == drop:
                            score += 1
                        scored.append((-score, len(scored), (0, int(r), int(c), d, s)))
        scored.sort()
        return [cand for _, _, cand in scored]

    def build_cands(self, p: int, t: int, touched: np.ndarray, state) -> list:
        prev_a = self.armies[t]
        own = self.owners[t] == p
        castles_now = np.asarray(state.castles)
        rough = touched & own & (prev_a >= _bc.BASE_COST) & ~self.generals & ~castles_now
        if not rough.any():
            return []
        cost = np.asarray(_cost_grid_jit(state, p))
        return [(2, int(r), int(c), 0, 0) for r, c in np.argwhere(rough & (prev_a >= cost))]

    def capture_cands(self, attacker: int, t: int) -> list:
        """Moves by `attacker` landing on the opponent's general (capture or
        deathtouch). Small closed set: at most 4 sources x 2 splits."""
        gr, gc = self.gen_pos[1 - attacker]
        prev_a = self.armies[t]
        own = self.owners[t] == attacker
        out = []
        for d, (dr, dc) in enumerate(DIRS):
            # source such that source + DIRS[d] == general
            r, c = gr - dr, gc - dc
            if 0 <= r < self.H and 0 <= c < self.W and own[r, c] and prev_a[r, c] >= 2:
                splits = (0,) if prev_a[r, c] == 2 else (0, 1)
                for s in splits:
                    out.append((0, r, c, d, s))
        return out

    # -- transition inference ---------------------------------------------------

    def try_pairs(self, state, t: int, c0: list, c1: list, check_winner: bool):
        """Step the engine over candidate pairs; return (pair, new_state) or None.

        Candidates are ordered pass-first, so the first match is the minimal
        explanation. When builds are among the candidates the resulting hidden
        castle set could differ between matches, so all pairs are scanned and
        distinct resulting states are disambiguated by one-step lookahead.
        """
        any_build = any(a[0] == 2 for a in c0) or any(a[0] == 2 for a in c1)
        matches = []
        n_pairs = 0
        for a0, a1 in product(c0, c1):
            n_pairs += 1
            if n_pairs > 5000:
                break
            new_state, info = _step_engine(state, a0, a1)
            if not self.matches_tick(new_state, t + 1):
                continue
            if check_winner and int(new_state.winner) != self.replay_winner:
                continue
            if not any_build:
                return (a0, a1), new_state
            matches.append(((a0, a1), new_state))
        if not matches:
            return None
        # group by full resulting state (armies/owners already pinned by the
        # replay -- only the castle mask and winner can differ)
        groups = []
        for pair, st in matches:
            key = (np.asarray(st.castles).tobytes(), int(st.winner))
            for gkey, _, _ in groups:
                if gkey == key:
                    break
            else:
                groups.append((key, pair, st))
        if len(groups) == 1:
            return matches[0]
        # >1 distinct hidden states: disambiguate by whether the next transition
        # is explainable (castle growth on the next even tick separates them)
        self._multimatch_ticks.append(t)
        if t + 2 < self.n_ticks:
            for _, pair, st in groups:
                if self.infer_transition(st, t + 1, record=False) is not None:
                    return pair, st
        return matches[0]

    def infer_transition(self, state, t: int, record: bool = True):
        """Infer the action pair for transition t -> t+1. Returns
        ((a0, a1), new_state) or None if no candidate pair reproduces the tick."""
        final = t == self.n_ticks - 2
        strategies = []

        if final and self.replay_winner in (0, 1):
            # a decisive final tick: the winner's action is a move onto the
            # loser's general; the loser's action shows in the army diff (owner
            # diff is swamped by the loser-cell transfer). Growth may or may not
            # have run (base capture: no; deathtouch override: yes) -- union both.
            w = self.replay_winner
            l = 1 - w

            def _capture():
                castles = np.asarray(state.castles)
                touched = (self.touched_cells(t, castles, growth=False)
                           | self.touched_cells(t, castles, growth=True))
                cw = self.capture_cands(w, t)
                cl = (self.move_cands(l, t, touched) + [PASS]
                      + self.build_cands(l, t, touched, state))
                return (cw, cl) if w == 0 else (cl, cw)

            strategies.append((_capture, True))

        def _normal():
            castles = np.asarray(state.castles)
            T = t + 1
            resid = (self.armies[T].astype(np.int64) - self.armies[t]
                     - self.growth_grid(T, castles))
            touched = (resid != 0) | (self.owners[T] != self.owners[t])
            cands = []
            for p in (0, 1):
                # moves best-first, then pass, then builds; an executed candidate
                # move always changes the state, so it can never collide with pass.
                cands.append(self.move_cands(p, t, touched, resid) + [PASS]
                             + self.build_cands(p, t, touched, state))
            return cands[0], cands[1]

        strategies.append((_normal, final))

        if final and self.replay_winner == -1 and t + 1 < TRUNCATION:
            # mutual capture / mutual deathtouch draw: both moved onto each
            # other's general (the engine reports winner -1).
            def _mutual():
                return ([PASS] + self.capture_cands(0, t),
                        [PASS] + self.capture_cands(1, t))

            strategies.append((_mutual, True))

        for build_cands, check_winner in strategies:
            c0, c1 = build_cands()
            got = self.try_pairs(state, t, c0, c1, check_winner)
            if got is not None:
                return got

        # Missed-castle recovery: an earlier unresolved transition may have
        # hidden a build; the phantom castle shows up as an unexplained +1 on
        # even ticks. Hypothesize castles there and retry once.
        T = t + 1
        if record and T % 2 == 0:
            castles = np.asarray(state.castles)
            resid = (self.armies[T].astype(np.int64) - self.armies[t]
                     - self.growth_grid(T, castles))
            stable = (self.owners[T] == self.owners[t]) & (self.owners[t] >= 0)
            cand = (resid == 1) & stable & ~castles & ~self.generals & self.passable
            if cand.any():
                patched = state._replace(castles=state.castles | jnp.asarray(cand))
                got = self.infer_transition(patched, t, record=False)
                if got is not None:
                    self._recovered.append((t, [tuple(map(int, rc)) for rc in np.argwhere(cand)]))
                    return got
        return None

    # -- full decode -------------------------------------------------------------

    def decode(self) -> DecodedReplay:
        t0 = _time.perf_counter()
        state = self.initial_state()
        if not self.matches_tick(state, 0):
            raise ValueError("replay tick 0 does not match engine initial state")

        n_trans = self.n_ticks - 1
        castles_by_tick = np.zeros((self.n_ticks, self.H, self.W), dtype=bool)
        actions: list = []
        ambiguous: list = []
        self._multimatch_ticks: list = []
        self._recovered: list = []

        for t in range(n_trans):
            got = self.infer_transition(state, t)
            if got is None:
                ambiguous.append(t)
                actions.append((PASS, PASS))
                state = self.state_from_tick(state, t + 1)  # resync, keep castles
            else:
                pair, state = got
                actions.append(pair)
            castles_by_tick[t + 1] = np.asarray(state.castles)

        return DecodedReplay(
            actions=actions,
            ambiguous=ambiguous,
            castles_by_tick=castles_by_tick,
            n_transitions=n_trans,
            engine_winner=int(state.winner),
            replay_winner=self.replay_winner,
            multimatch=self._multimatch_ticks,
            recovered_castles=self._recovered,
            elapsed_s=_time.perf_counter() - t0,
        )


# ------------------------------------------------------------------- public API

def decode_replay(replay: dict) -> DecodedReplay:
    """Decode (with per-replay-dict caching)."""
    cached = replay.get(_DECODED_KEY)
    if cached is not None:
        return cached
    result = _Decoder(replay).decode()
    replay[_DECODED_KEY] = result
    return result


def infer_actions(replay: dict) -> list:
    """List of (a0, a1) 5-int action tuples, one per transition.
    Ambiguous transitions hold (PASS, PASS); indices in decode_replay(replay).ambiguous."""
    return decode_replay(replay).actions


def validate(replay: dict) -> dict:
    """The acceptance gate: replay the inferred actions through the real engine
    from the initial state and check every tick reproduces bit-exactly."""
    t0 = _time.perf_counter()
    dec = decode_replay(replay)
    d = _Decoder(replay)
    state = d.initial_state()
    first_mismatch = None
    if not d.matches_tick(state, 0):
        first_mismatch = 0
    else:
        for t, (a0, a1) in enumerate(dec.actions):
            state, _ = _step_engine(state, a0, a1)
            if not d.matches_tick(state, t + 1):
                first_mismatch = t + 1
                break
    winner_ok = int(state.winner) == d.replay_winner if first_mismatch is None else False
    bit_exact = first_mismatch is None
    return {
        "ok": bit_exact and winner_ok and not dec.ambiguous,
        "bit_exact": bit_exact,
        "winner_ok": winner_ok,
        "first_mismatch_tick": first_mismatch,
        "n_transitions": dec.n_transitions,
        "n_ambiguous": len(dec.ambiguous),
        "ambiguous_ticks": list(dec.ambiguous),
        "ambiguity_rate": dec.ambiguity_rate,
        "multimatch_ticks": list(dec.multimatch),
        "recovered_castles": list(dec.recovered_castles),
        "decode_s": dec.elapsed_s,
        "validate_s": _time.perf_counter() - t0,
    }


# ------------------------------------------------------------------- fog views

def _visible_mask(own: np.ndarray) -> np.ndarray:
    """8-neighborhood visibility (3x3 max-pool), as the engine computes it. Works
    on (H, W) or batched (T, H, W) ownership masks."""
    H, W = own.shape[-2:]
    pad = [(0, 0)] * (own.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(own, pad)
    v = np.zeros_like(own)
    for di in range(3):
        for dj in range(3):
            v |= p[..., di:di + H, dj:dj + W]
    return v


def _fog_arrays(d: _Decoder, castles_by_tick: np.ndarray, player: int, ticks: slice):
    """Vectorized fog reconstruction for a range of ticks. Mirrors the engine's
    observation and wire encoding exactly."""
    armies = d.armies[ticks]
    owners = d.owners[ticks]
    castles = castles_by_tick[ticks]
    own = owners == player
    opp = owners == (1 - player)
    vis = _visible_mask(own)

    struct = d.mountains | castles                       # (T, H, W) via broadcast
    type_grid = np.ones(armies.shape, dtype=np.int8)     # plain
    type_grid[~vis & ~struct] = 0                        # fog
    type_grid[~vis & struct] = 5                         # structure-in-fog
    type_grid[vis & d.mountains] = 2
    type_grid[vis & castles] = 3
    type_grid[vis & d.generals] = 4

    owner_grid = np.zeros(armies.shape, dtype=np.int8)
    owner_grid[own & vis] = 1
    owner_grid[opp & vis] = 2

    army_grid = (armies * vis).astype(np.int32)

    n = armies.shape[0]
    turn0 = ticks.start or 0
    scalars = np.stack([
        np.arange(turn0, turn0 + n, dtype=np.int64),
        own.sum((1, 2)),
        (armies * own).sum((1, 2)),
        opp.sum((1, 2)),
        (armies * opp).sum((1, 2)),
    ], axis=1)
    return type_grid, owner_grid, army_grid, scalars


def fog_view(replay: dict, tick_idx: int, player: int) -> FogView:
    """Exactly the stdio observation `player` sees at `tick_idx`.

    Castle positions are hidden state (ticks carry only armies/owners), so this
    runs (cached) action inference to track builds.
    """
    dec = decode_replay(replay)
    d = _Decoder(replay)
    if not 0 <= tick_idx < d.n_ticks:
        raise IndexError(f"tick_idx {tick_idx} out of range [0, {d.n_ticks})")
    tg, og, ag, sc = _fog_arrays(d, dec.castles_by_tick, player,
                                 slice(tick_idx, tick_idx + 1))
    return FogView(type_grid=tg[0], owner_grid=og[0], army_grid=ag[0],
                   scalars=tuple(int(x) for x in sc[0]))


# --------------------------------------------------------------------- dataset

def emit_dataset(manifest_filter=None,
                 manifest_path: str | Path = "data/manifest.parquet",
                 replays_dir: str | Path = "data/replays",
                 out_dir: str | Path = "data/shards",
                 shard_size: int = 32,
                 require_valid: bool = True) -> dict:
    """Emit sharded npz per-player sequences under out_dir.

    One sequence per (replay, player): fogged raw grids (type int8 / owner int8 /
    army int32), scalar lines, the player's wire action per tick and an
    action_ok mask (False on ambiguous transitions). Raw grids only -- feature
    encoding is the consumer's job.

    manifest_filter: None (all rows) or callable(row_dict) -> bool.
    require_valid: skip replays that fail the bit-exact validation gate.
    """
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(manifest_path).to_pylist()
    if manifest_filter is not None:
        rows = [r for r in rows if manifest_filter(r)]

    shard_seqs: list = []
    shard_meta: list = []
    shard_paths: list = []
    summary = {"replays": 0, "skipped_missing": 0, "skipped_invalid": 0,
               "sequences": 0, "transitions": 0, "ambiguous": 0, "shards": []}

    def flush():
        if not shard_seqs:
            return
        idx = len(shard_paths)
        path = out_dir / f"shard_{idx:04d}.npz"
        payload = {}
        for i, seq in enumerate(shard_seqs):
            for k, v in seq.items():
                payload[f"s{i}_{k}"] = v
        payload["meta"] = np.frombuffer(
            json.dumps(shard_meta).encode(), dtype=np.uint8)
        np.savez_compressed(path, **payload)
        shard_paths.append(str(path))
        summary["shards"].append({"path": str(path), "sequences": len(shard_seqs)})
        shard_seqs.clear()
        shard_meta.clear()

    for row in rows:
        rid = row["replay_id"]
        path = Path(replays_dir) / f"{rid}.json.zst"
        if not path.exists():
            summary["skipped_missing"] += 1
            continue
        replay = load_replay(path)
        report = validate(replay)
        dec = decode_replay(replay)
        if require_valid and not report["bit_exact"]:
            summary["skipped_invalid"] += 1
            continue
        d = _Decoder(replay)
        n_trans = dec.n_transitions
        ok_mask = np.ones(n_trans, dtype=bool)
        ok_mask[dec.ambiguous] = False

        # players[a_side] is side A; side->name mapping for metadata
        names = replay.get("players", ["?", "?"])
        for p in (0, 1):
            tg, og, ag, sc = _fog_arrays(d, dec.castles_by_tick, p, slice(0, n_trans))
            actions_p = np.array([pair[p] for pair in dec.actions], dtype=np.int16)
            shard_seqs.append({
                "type": tg, "owner": og, "army": ag,
                "scalars": sc.astype(np.int64),
                "action": actions_p, "action_ok": ok_mask.copy(),
            })
            outcome = 1 if dec.replay_winner == p else (-1 if dec.replay_winner == 1 - p else 0)
            shard_meta.append({
                "replay_id": int(rid), "player_idx": p,
                "player_name": names[p] if p < len(names) else "?",
                "opponent_name": names[1 - p] if 1 - p < len(names) else "?",
                "outcome": outcome, "seed": replay.get("seed"),
                "rows": d.H, "cols": d.W,
                "n_transitions": n_trans,
                "n_ambiguous": len(dec.ambiguous),
            })
            summary["sequences"] += 1
            if len(shard_seqs) >= shard_size:
                flush()
        summary["replays"] += 1
        summary["transitions"] += n_trans
        summary["ambiguous"] += len(dec.ambiguous)

    flush()
    summary["shard_paths"] = shard_paths
    return summary


def read_shard(path: str | Path) -> tuple:
    """Load one shard -> (list of sequence dicts, list of meta dicts)."""
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        seqs = []
        for i in range(len(meta)):
            seqs.append({k: z[f"s{i}_{k}"]
                         for k in ("type", "owner", "army", "scalars", "action", "action_ok")})
    return seqs, meta


# -------------------------------------------------------------------------- cli

def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate-all", help="run the bit-exact gate on cached replays")
    v.add_argument("--replays-dir", default="data/replays")
    v.add_argument("--limit", type=int, default=None)

    e = sub.add_parser("emit", help="emit per-player sequence shards")
    e.add_argument("--manifest", default="data/manifest.parquet")
    e.add_argument("--replays-dir", default="data/replays")
    e.add_argument("--out-dir", default="data/shards")
    e.add_argument("--shard-size", type=int, default=32)

    args = ap.parse_args()
    if args.cmd == "validate-all":
        paths = sorted(Path(args.replays_dir).glob("*.json.zst"))
        if args.limit:
            paths = paths[: args.limit]
        n_ok = n_bit = tot_trans = tot_amb = 0
        t0 = _time.perf_counter()
        for p in paths:
            rep = validate(load_replay(p))
            n_ok += rep["ok"]
            n_bit += rep["bit_exact"]
            tot_trans += rep["n_transitions"]
            tot_amb += rep["n_ambiguous"]
            flag = "OK " if rep["ok"] else "FAIL"
            extra = ""
            if not rep["bit_exact"]:
                extra = f" first_mismatch={rep['first_mismatch_tick']}"
            if rep["n_ambiguous"]:
                extra += f" ambiguous={rep['ambiguous_ticks'][:5]}"
            print(f"{flag} {p.name:22s} trans={rep['n_transitions']:5d} "
                  f"amb={rep['n_ambiguous']:3d} decode={rep['decode_s']:6.2f}s "
                  f"validate={rep['validate_s'] - rep['decode_s']:6.2f}s{extra}")
        dt = _time.perf_counter() - t0
        print(f"\n{n_ok}/{len(paths)} fully ok, {n_bit}/{len(paths)} bit-exact; "
              f"ambiguous {tot_amb}/{tot_trans} transitions "
              f"({100.0 * tot_amb / max(1, tot_trans):.3f}%); {dt:.1f}s total "
              f"({dt / max(1, len(paths)):.2f}s/replay)")
    elif args.cmd == "emit":
        summary = emit_dataset(manifest_path=args.manifest,
                               replays_dir=args.replays_dir,
                               out_dir=args.out_dir,
                               shard_size=args.shard_size)
        summary.pop("shards", None)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
