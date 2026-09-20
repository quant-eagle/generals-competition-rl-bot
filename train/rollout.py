"""On-device vectorized self-play rollout with seed-bank injection.

Everything stays on the GPU: the engine is JAX, so states, observations, the
policy forward, the executor and the reward all live on device and nothing
crosses PCIe inside the loop.

Two departures from the engine's gym-style `step`, both deliberate:

  * the competition modifiers are called directly (build-castles, then
    deathtouch) so no auto-reset fires -- the rollout needs the pre-reset
    successor state;
  * resets draw from a mixture of fresh boards and mid-game seed states taken
    from top-ladder replays, which is the backward curriculum (Salimans & Chen,
    arXiv:1812.03381).  A fresh policy finds a committed 15-cell march by
    chance with probability ~4^-15 and the terminal reward is one bit per ~300
    decisions; starting episodes at `blind_commit` / `strike_conversion` states
    with a bounded horizon turns that into a 40-60 step problem.

The reward is sparse: terminal win/loss/draw only, gamma = 1, no potential
shaping (see learn.py for why).

The rollout also stores Q(s,a) and Vbar(s) under the behaviour (tempered)
distribution, which is what Q-boosting's Expected-SARSA backup needs.

The learner always occupies seat 0.  Seeds are canonicalised to match: the
tagged seat is relabelled to 0, which is why the seed bank stores memory for
both seats.  Seeded episodes truncate into the value bootstrap rather than
terminating, so a horizon cut is not mistaken for a loss.

One decision per tick: there is no action repeat or decision skipping.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from generals.core.game import GameState, get_observation
from generals.modifiers import build_castles as bc
from generals.modifiers import deathtouch as dt

from . import exec as X
from . import obs as O
from .config import (B, DEATHTOUCH_TURN, DRAW_TURN, INT_BUILD, N_INT,
                     N_DEPTH, N_PLANES, N_PRIV,
                     N_EMA, N_SERIES, N_TSAMP, TRING, TSCALES,
                     N_SRC, NetCfg)
from .learn import q_vbar, q_taken
from .net import act_logits_v, forward_v

NEG_HALF = -1e29   # masked logits are -1e30; anything above this is legal


def cstep(state, actions):
    """One competition tick: builds resolve first, then deathtouch-aware step."""
    state, actions = bc.apply_build_actions(state, actions)
    return dt.step(state, actions, DEATHTOUCH_TURN)


cstep_v = jax.vmap(cstep)
get_obs_v = jax.vmap(get_observation, in_axes=(0, None))


def _sel(cond, a, b):
    """Per-env select over a pytree: cond is (N,), leaves are (N, ...)."""
    return jax.tree.map(
        lambda x, y: jnp.where(cond.reshape((-1,) + (1,) * (x.ndim - 1)), x, y),
        a, b)


# Scenario tags of the seed bank, in `--scen-weights` order.
SCENARIOS = ("strike_conversion", "blind_commit", "defense_warning",
             "castle_decision")


class SeedBank(NamedTuple):
    """Device-resident mid-game seeds, from tools/build_rl_seeds.py.

    `scen_idx` / `scen_seat` / `scen_n` support stratified sampling.  Sampling
    uniformly over the bank would inherit the corpus's own mix, in which
    castle decisions -- among the most outcome-relevant -- are the least
    represented scenario.  The tag is per-seat, so the seat carrying it is
    stored alongside and becomes the learner.
    """
    armies: jnp.ndarray        # (S, B, B) int32
    owners: jnp.ndarray        # (S, B, B) int8, -1 neutral
    castles: jnp.ndarray
    generals: jnp.ndarray
    mountains: jnp.ndarray
    t: jnp.ndarray             # (S,)
    mem: tuple                 # per seat: (last_seen, ever, gowner, garmy, ema)
    rings: tuple               # per seat: (S, N_SERIES, n_scales, TRING) f16
    scen_idx: jnp.ndarray      # (n_scen, n_depth, max_n) seed index, padded
    scen_seat: jnp.ndarray     # (n_scen, n_depth, max_n) seat carrying the tag
    scen_n: jnp.ndarray        # (n_scen, n_depth) real length of each row
    scen_horizon: jnp.ndarray  # (n_scen,) ticks granted per scenario
    depth_ticks: jnp.ndarray   # (S,) ticks from this row's start to its tag

    @staticmethod
    def load(path: str, index: str | None = None, manifest: str | None = None,
             min_elo: float = 3000.0, winners_only: bool = True,
             balance_players: bool = True):
        """-> (SeedBank, scenario_names).

        Names are returned separately, not stored on the tuple: a NamedTuple
        passed through jit has every field traced, and a tuple of strings is not
        a traceable array.

        Selection.  The bank exists so episodes start from positions where a
        win is available, which is what teaches conversion.  Unfiltered it does
        not do that: about half the (seed, seat) pairs are a strong player's
        opponent losing, which under a sparse terminal reward on a 40-tick
        horizon is -1 whatever the policy does -- noise, not gradient.  So a
        pair is kept only if that seat won and its player was rated at least
        `min_elo`.

        Balance.  Few players clear the Elo bar and they survive in uneven
        numbers, so the curriculum would teach one player's style much harder
        than another's.  `balance_players` tiles each player's rows up to the
        largest player's count, per scenario, so all are drawn equally.
        Tiling repeats rows rather than discarding any.

        Requires `index` (winner) and `manifest` (Elo, names).  Without them
        everything is kept and a line says so -- a missing file must not
        silently turn the curriculum back into a coin flip.
        """
        d = np.load(path)
        n_seed = d["t"].shape[0]
        has_ema = "ema0" in d.files
        ema_z = jnp.zeros((n_seed, N_EMA, 2, B, B), jnp.float16)
        mem = tuple(
            (jnp.asarray(d[f"last_seen{s}"], jnp.int32),
             jnp.asarray(d[f"ever_seen{s}"]),
             jnp.asarray(d[f"ghost_owner{s}"]),
             jnp.asarray(d[f"ghost_army{s}"], jnp.float32),
             jnp.asarray(d[f"ema{s}"], jnp.float16) if has_ema else ema_z)
            for s in (0, 1))
        # The aggregate-series rings.  A bank without them falls back to zeros
        # and says so: a seeded episode then starts as if the game had no
        # economic past, which must not happen silently.
        n_scale = len(TSCALES)
        if "rings0" in d.files:
            rings = tuple(jnp.asarray(d[f"rings{s_}"], jnp.float16)
                          for s_ in (0, 1))
        else:
            print("seed bank: no rings -- seeded episodes start with a blank "
                  "economy history; rebuild with tools/build_rl_seeds.py",
                  flush=True)
        if not has_ema:
            print("seed bank: no EMA planes -- seeded episodes start with zero "
                  "motion history, which reads as 'nothing has moved for 128 "
                  "ticks'; rebuild with tools/build_rl_seeds.py", flush=True)
            z = jnp.zeros((d["t"].shape[0], N_SERIES, n_scale, TRING),
                          jnp.float16)
            rings = (z, z)

        # Backward-curriculum depth.  A bank without it is a single-depth bank:
        # every row starts at its tagged moment.
        n_row = d["t"].shape[0]
        depth_idx = (d["depth_idx"] if "depth_idx" in d.files
                     else np.zeros(n_row, np.int8))
        depth_ticks = (d["depth_ticks"] if "depth_ticks" in d.files
                       else np.zeros(n_row, np.int16))
        n_depth = int(depth_idx.max()) + 1
        if n_depth == 1:
            print("seed bank: single depth -- every seeded episode starts at "
                  "the tagged moment, so the bank can only teach conversion, "
                  "never creation", flush=True)

        tags = d["tags"]                       # (n, 2, n_scen)
        names = tuple(str(x) for x in d["scenarios"])
        n = tags.shape[0]

        eligible = np.ones((n, 2), bool)
        who = np.full((n, 2), "", object)
        if (index and manifest and Path(index).exists()
                and Path(manifest).exists() and "replay_id" in d.files):
            import pandas as pd
            rid = d["replay_id"]
            winner = pd.read_parquet(index).set_index(
                "replay_id")["winner"].reindex(rid).to_numpy()
            mf = pd.read_parquet(manifest).set_index("replay_id").reindex(rid)
            a_is_seat0 = mf["a_side"].to_numpy() == 0
            for seat in (0, 1):
                is_a = (seat == 0) == a_is_seat0
                elo = np.where(is_a, mf["elo_a"].to_numpy(),
                               mf["elo_b"].to_numpy())
                who[:, seat] = np.where(is_a, mf["player_a"].to_numpy(),
                                        mf["player_b"].to_numpy())
                ok = np.ones(n, bool)
                if winners_only:
                    ok &= (winner == seat)
                if min_elo:
                    ok &= np.nan_to_num(elo, nan=-1.0) >= min_elo
                eligible[:, seat] = ok
        else:
            why = ("bank has no replay_id" if "replay_id" not in d.files
                   else "no index/manifest")
            print(f"seed bank: {why} -- keeping all seats, including losing "
                  f"and low-Elo ones", flush=True)

        rows_i, rows_s = [], []
        for j in range(len(names)):
            for dep in range(n_depth):
                keep = tags[:, :, j] & eligible & (depth_idx == dep)[:, None]
                if not keep.any():      # never empty a (scenario, depth)
                    print(f"seed bank: {names[j]} depth {dep} had no eligible "
                          f"seat; falling back to unfiltered", flush=True)
                    keep = tags[:, :, j] & (depth_idx == dep)[:, None]
                i_idx, seat = np.where(keep)
                if balance_players:
                    pl = who[i_idx, seat]
                    uniq = np.unique(pl)
                    if len(uniq) > 1:   # one group (or no metadata) is a no-op
                        grp = [np.where(pl == u)[0] for u in uniq]
                        # rows with no manifest player yield NaN "players"
                        # that match nothing; drop their empty groups, and
                        # skip balancing if fewer than two survive
                        grp = [g for g in grp if len(g)]
                        if len(grp) > 1:
                            m = max(len(g) for g in grp)
                            take = np.concatenate(
                                [g[np.arange(m) % len(g)] for g in grp])
                            i_idx, seat = i_idx[take], seat[take]
                rows_i.append(i_idx.astype(np.int32))
                rows_s.append(seat.astype(np.int32))

        width = max(len(r) for r in rows_i)
        pad = lambda r: np.pad(r, (0, width - len(r)), mode="wrap")
        shape = (len(names), n_depth, width)
        return SeedBank(
            jnp.asarray(d["armies"], jnp.int32), jnp.asarray(d["owners"]),
            jnp.asarray(d["castles"]), jnp.asarray(d["generals"]),
            jnp.asarray(d["mountains"]), jnp.asarray(d["t"], jnp.int32), mem,
            rings,
            jnp.asarray(np.stack([pad(r) for r in rows_i]).reshape(shape)),
            jnp.asarray(np.stack([pad(r) for r in rows_s]).reshape(shape)),
            jnp.asarray(np.array([len(r) for r in rows_i], np.int32)
                        .reshape(len(names), n_depth)),
            jnp.asarray(d["scen_horizon"], jnp.int32),
            jnp.asarray(depth_ticks, jnp.int32)), names

    @staticmethod
    def placeholder(names: tuple = SCENARIOS):
        """-> (SeedBank, scenario_names) holding one blank row.

        For training without replay data.  The rollout is compiled against a
        bank's shapes, so pure self-play still needs one; with `p_seed = 0` no
        reset ever draws from it.  Every (scenario, depth) slot points at row 0
        and reports length 1, so the sampling arithmetic stays well defined.
        """
        n_scen, z = len(names), jnp.zeros((1, B, B), bool)
        mem = tuple((jnp.zeros((1, B, B), jnp.int32), z, z,
                     jnp.zeros((1, B, B), jnp.float32),
                     jnp.zeros((1, N_EMA, 2, B, B), jnp.float16))
                    for _ in (0, 1))
        ring = jnp.zeros((1, N_SERIES, len(TSCALES), TRING), jnp.float16)
        slot = jnp.zeros((n_scen, N_DEPTH, 1), jnp.int32)
        return SeedBank(
            jnp.zeros((1, B, B), jnp.int32), jnp.full((1, B, B), -1, jnp.int8),
            z, z, z, jnp.zeros((1,), jnp.int32), mem, (ring, ring),
            slot, slot, jnp.ones((n_scen, N_DEPTH), jnp.int32),
            jnp.full((n_scen,), 40, jnp.int32),
            jnp.zeros((1,), jnp.int32)), tuple(names)

    @property
    def size(self) -> int:
        return self.armies.shape[0]


class Carry(NamedTuple):
    state: GameState
    mem0: O.Memory             # learner (seat 0)
    mem1: O.Memory             # opponent (seat 1)
    ring0: O.Rings             # aggregate-series history, per seat
    ring1: O.Rings
    horizon: jnp.ndarray       # (N,) tick at which this episode truncates
    scen: jnp.ndarray          # (N,) scenario of a seeded episode, -1 if fresh
    opp: jnp.ndarray           # (N,) seat-1 policy: 0 mirror-net, 1 expander,
                               # 2 hunter (the league; seeds are always 0)
    key: jnp.ndarray


class Batch(NamedTuple):
    """(T, N, ...) trajectory slice.  Planes are fp16 to keep the buffer small."""
    # Planes are stored packed: booleans bitpacked, broadcast scalars kept as
    # scalars.  Less than half the size of (T, N, N_PLANES, B, B) fp16,
    # losslessly -- and this buffer, not FLOPs, is what caps `envs`.
    pbits: jnp.ndarray         # (T, N, N_PACKED, B, B) uint8
    pfloat: jnp.ndarray        # (T, N, N_FLOAT, B, B) fp16
    pscalar: jnp.ndarray       # (T, N, N_SCALAR) fp16
    series: jnp.ndarray        # (T, N, N_SERIES, N_TSAMP), the temporal tokens
    priv: jnp.ndarray          # (T, N, N_PRIV, B, B)
    legal: jnp.ndarray         # (T, N, N_SRC, N_INT) exact joint mask
    a_src: jnp.ndarray         # (T, N)
    a_int: jnp.ndarray         # (T, N)
    logp: jnp.ndarray          # log-prob of the joint (cell, intent) action
    q_a: jnp.ndarray           # Q(s, a) under the behaviour policy
    vbar: jnp.ndarray          # sum_a pi(a|s) Q(s, a), same
    reward: jnp.ndarray        # sparse: nonzero only at a terminal
    terminal: jnp.ndarray      # true game end (win, loss or draw)
    done: jnp.ndarray          # terminal or seed-horizon truncation
    gen_target: jnp.ndarray    # (T, N) enemy-general cell, aux label
    died: jnp.ndarray          # (T, N) learner lost here, for the danger head
    ep_scen: jnp.ndarray       # (T, N) scenario of an episode ending here, -1
                               # otherwise -- the backward-curriculum mastery
                               # signal
    valid: jnp.ndarray         # (T, N) trainable sample: seat-1 lanes of
                               # league (scripted-opponent) envs are False --
                               # their actions came from a script, so PPO's
                               # ratio is undefined on them


def _general_of(generals, owner_mask):
    flat = jnp.argmax(generals & owner_mask)
    return jnp.stack([flat // B, flat % B])


def _seed_state(bank: SeedBank, i, seat, hz):
    """Build a GameState + both memories from seed `i`, learner as seat 0."""
    own = bank.owners[i] == seat
    opp = bank.owners[i] == (1 - seat)
    gen, mnt = bank.generals[i], bank.mountains[i]
    state = GameState(
        armies=bank.armies[i],
        ownership=jnp.stack([own, opp]),
        ownership_neutral=~own & ~opp & ~mnt,
        generals=gen, castles=bank.castles[i], mountains=mnt, passable=~mnt,
        general_positions=jnp.stack([_general_of(gen, own),
                                     _general_of(gen, opp)]),
        time=bank.t[i], winner=jnp.int32(-1), pool_idx=jnp.int32(0))

    # `seat` is traced, so both seats' memories are built and then selected
    # elementwise -- a tracer cannot index the Python tuple bank.mem.
    def mem_for(s: int):
        ls, ev, go, ga, em = jax.tree.map(lambda x: x[i], bank.mem[s])
        egen = gen & (bank.owners[i] == (1 - s))
        return O.Memory(ghost_owner=go, ghost_army=ga, last_seen_t=ls,
                        ever_seen=ev,
                        enemy_general=egen & ev,
                        ema=em.astype(jnp.float32))

    m0, m1 = mem_for(0), mem_for(1)
    r0 = O.Rings(bank.rings[0][i].astype(jnp.float32))
    r1 = O.Rings(bank.rings[1][i].astype(jnp.float32))
    is0 = seat == 0
    sel = lambda a, b: jax.tree.map(lambda x, y: jnp.where(is0, x, y), a, b)
    # The episode ends at (tagged tick + scenario horizon) at every depth:
    # t is the actual start, depth_ticks the distance from it to the tag.  A
    # deeper start is therefore a longer task on the same target, not a
    # different one -- which is what makes depths comparable and lets a
    # mastery criterion advance between them.
    return (state, sel(m0, m1), sel(m1, m0), sel(r0, r1), sel(r1, r0),
            bank.t[i] + bank.depth_ticks[i] + hz)


def reset(key, n, pool, bank: SeedBank, p_seed: float, scen_w, scen_depth,
          league_frac: float = 0.0):
    """Fresh boards mixed with seeds; returns states and matching memories.

    Seeds are drawn scenario-first with weights `scen_w`, then uniformly within
    the scenario, and the learner takes the seat that carries the tag.

    `league_frac` of fresh envs draw a scripted seat-1 opponent (half
    expander, half hunter) instead of the mirror -- the league.  Seeded envs
    always stay mirror: the curriculum measures conversion against the live
    defender.
    """
    k_pick, k_pool, k_scen, k_pos, k_lg = jax.random.split(key, 5)
    use_seed = jax.random.uniform(k_pick, (n,)) < p_seed
    pool_i = jax.random.randint(k_pool, (n,), 0, pool.armies.shape[0])

    scen = jax.random.categorical(k_scen, jnp.log(scen_w), shape=(n,))
    dep = scen_depth[scen]              # current backward-curriculum depth
    pos = (jax.random.uniform(k_pos, (n,))
           * bank.scen_n[scen, dep]).astype(jnp.int32)
    seed_i = bank.scen_idx[scen, dep, pos]
    seat = bank.scen_seat[scen, dep, pos]

    fresh = jax.tree.map(lambda x: x[pool_i], pool)
    blank = jax.vmap(lambda _: O.init_memory())(jnp.arange(n))
    blank_r = jax.vmap(lambda _: O.init_rings())(jnp.arange(n))
    s_state, s_m0, s_m1, s_r0, s_r1, s_h = jax.vmap(
        _seed_state, in_axes=(None, 0, 0, 0))(
            bank, seed_i, seat, bank.scen_horizon[scen])

    state = _sel(use_seed, s_state, fresh)
    horizon = jnp.where(use_seed, s_h, DRAW_TURN)
    u = jax.random.uniform(k_lg, (n,))
    opp = jnp.where(u < league_frac / 2, 1,
                    jnp.where(u < league_frac, 2, 0)).astype(jnp.int32)
    opp = jnp.where(use_seed, 0, opp)
    return (state, _sel(use_seed, s_m0, blank), _sel(use_seed, s_m1, blank),
            _sel(use_seed, s_r0, blank_r), _sel(use_seed, s_r1, blank_r),
            horizon, jnp.where(use_seed, scen, -1), opp)


def init_carry(key, n, pool, bank, p_seed, scen_w, scen_depth,
               league_frac: float = 0.0) -> Carry:
    k, k2 = jax.random.split(key)
    state, m0, m1, r0, r1, h, sc, op = reset(k, n, pool, bank, p_seed, scen_w,
                                             scen_depth, league_frac)
    return Carry(state, m0, m1, r0, r1, h, sc, op, k2)


def sample_joint(key, logits, temp, eps_build: float = 0.0):
    """Sample one action from the (cell, intent) grid; return its log-prob.

    `logits` is (n, N_SRC, N_INT), already masked to the exact legality grid, so
    every action with non-zero probability is one the engine will execute.

    Eps-build exploration.  A castle costs army now and repays only much later,
    so a young policy drives BUILD to ~0 almost immediately; fresh games then
    contain no post-build states, the critic never prices castle income, and
    the build advantage stays negative -- a self-sealing equilibrium.  The
    behaviour policy therefore mixes in eps mass over the legal builds only:

        b = (1 - eps') * pi + eps' * uniform(legal builds),  eps' = 0 when none

    This is exploration, not shaping: the recorded logp is the mixture's, PPO's
    ratio pi/b corrects for it exactly, and the reward is untouched.  If castles
    are genuinely bad the gradient still says so -- it just gets to see the
    evidence.  Vbar keeps using pi (the target policy), never b, and evaluation
    stays greedy on pi.
    """
    n_lane = logits.shape[0]
    lp = jax.nn.log_softmax((logits / temp).reshape(n_lane, -1))
    pi = jnp.exp(lp)
    # `eps_build` may be a traced scalar (the anneal threads it through the
    # jit), and a Python comparison on a tracer is an error.  Skip the mixture
    # only for a concrete zero; a traced value always takes the mixture path,
    # whose math degrades exactly to pi at eps=0.
    if not (isinstance(eps_build, (int, float)) and eps_build <= 0.0):
        legal_b = (logits[:, :, INT_BUILD] > NEG_HALF).reshape(n_lane, N_SRC)
        nb = legal_b.sum(-1)
        ub = jnp.zeros((n_lane, N_SRC, N_INT))
        ub = ub.at[:, :, INT_BUILD].set(
            legal_b / jnp.maximum(nb, 1)[:, None]).reshape(n_lane, -1)
        e = jnp.where(nb > 0, eps_build, 0.0)[:, None]
        b = (1.0 - e) * pi + e * ub
        a = jax.random.categorical(key, jnp.log(b + 1e-30))
        n = jnp.arange(n_lane)
        return (a // N_INT, a % N_INT, jnp.log(b[n, a] + 1e-30),
                pi.reshape(logits.shape))
    a = jax.random.categorical(key, lp)
    n = jnp.arange(n_lane)
    return (a // N_INT, a % N_INT, lp[n, a],
            pi.reshape(logits.shape))


def make_step(cfg: NetCfg, selfplay: str = "mirror",
              eps_build: float = 0.0, draw_reward: float = 0.0,
              train_cap: int = DRAW_TURN, win_speed: float = 0.0,
              league_frac: float = 0.0):
    """Build the single-tick rollout step for a fixed config.

    `selfplay="mirror"` plays the current policy in both seats and trains on
    both, as Straka et al. (arXiv:2606.23348) do, so no half of a rollout is
    discarded.  It only works mirrored: PPO's ratio needs `logp_old` from the
    policy that actually chose the action, and under the snapshot ladder seat 1
    is played by a lagging snapshot, so its data is off-policy by an unbounded
    amount.  `selfplay="ladder"` plays FIFO opponents and trains on seat 0 only.

    Mirroring also makes the two seats one forward over 2N samples rather than
    two over N, which is a better GEMM shape than either alone.

    `pool` and `bank` are arguments, not closure constants: closed over, XLA
    bakes gigabytes of them into the executable, which blows up both compile
    time and VRAM.  `p_seed` is traced for the same family of reasons -- the
    curriculum changes it between iterations, and a baked-in value would
    retrace the whole scan.
    """
    if league_frac > 0:
        from . import arena
        _exp_v = jax.vmap(arena.scripted("expander"))
        _hun_v = jax.vmap(arena.scripted("hunter"))

    def step(carry: Carry, params, opp_params, p_seed, temp, scen_w,
             scen_depth, pool, bank, eps=None):
        # `eps` overrides the closure eps_build when threaded (the anneal);
        # None keeps the constant.
        e_bld = eps_build if eps is None else eps
        st = carry.state
        key, ka, kb, kr = jax.random.split(carry.key, 4)
        n = carry.horizon.shape[0]

        o0, o1 = get_obs_v(st, 0), get_obs_v(st, 1)
        m0, m1 = O.update_v(carry.mem0, o0), O.update_v(carry.mem1, o1)
        ring0, ring1 = O.push_v(carry.ring0, o0), O.push_v(carry.ring1, o1)
        x0, x1 = O.encode_v(m0, o0), O.encode_v(m1, o1)
        z0, z1 = O.series_of_v(ring0), O.series_of_v(ring1)
        lg0, lg1 = O.legal_mask_v(o0), O.legal_mask_v(o1)
        priv0 = O.priv_planes_v(st, 0)

        if selfplay == "mirror":
            # one forward over both seats
            cat = lambda a, b: jnp.concatenate([a, b])
            priv1 = O.priv_planes_v(st, 1)
            logits, qp, _, _ = forward_v(
                params, cat(x0, x1), cat(z0, z1), cat(priv0, priv1),
                cat(lg0, lg1), cfg)
            a_s, a_i, logp, pi = sample_joint(ka, logits, temp, e_bld)
            vbar, q_a = q_vbar(qp, pi), q_taken(qp, a_s, a_i)
            b_s, b_i = a_s[n:], a_i[n:]
            a_s, a_i = a_s[:n], a_i[:n]
        else:
            logits, qp, _, _ = forward_v(params, x0, z0, priv0, lg0, cfg)
            a_s, a_i, logp, pi = sample_joint(ka, logits, temp, e_bld)
            vbar, q_a = q_vbar(qp, pi), q_taken(qp, a_s, a_i)
            olog = act_logits_v(opp_params, x1, z1, lg1, cfg)
            b_s, b_i, _, _ = sample_joint(kb, olog, temp)

        prim0 = X.execute_v(a_s, a_i, o0)
        prim1 = X.execute_v(b_s, b_i, o1)
        if league_frac > 0:
            # The league: a scripted seat 1 for the drawn fraction of fresh
            # envs.  Scripts act on the raw observation; the mirror forward
            # for those envs still runs (its seat-1 samples are masked out of
            # training via `valid`, not skipped -- one traced path).
            sel = lambda c, a, b: jnp.where(
                c.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
            prim1 = sel(carry.opp == 1, _exp_v(o1), prim1)
            prim1 = sel(carry.opp == 2, _hun_v(o1), prim1)
        nxt, info = cstep_v(st, jnp.stack([prim0, prim1], 1))

        # Reaching the turn cap is a draw: terminal, so it bootstraps 0 and is
        # folded into `terminal` rather than treated as a horizon truncation.
        # `draw_reward` prices it for both seats.  The competition scores a
        # clock-out as a draw, but under a free draw mirror play reaches the
        # cap so often that a dominating position learns value ~0 and the
        # policy hoards instead of closing.  A negative draw_reward makes
        # running out the clock cost something.  Draws are then not zero-sum,
        # so seat 1 gets its own branch instead of -r0.
        #
        # `train_cap` tightens the clock-out in training only.  Top-ladder
        # games are mostly decided well before it (median ~320 ticks), so a
        # cap near 600 with a priced draw concentrates experience where games
        # are decided and roughly doubles the terminals per iteration.
        won = info.winner >= 0
        # The cap applies to fresh games only.  Seeded episodes carry their own
        # horizon (tag + scenario grant) and must truncate into the value
        # bootstrap there; pricing one as a draw mid-conversion would inject
        # noise into exactly the signal the seeds exist to produce.
        # carry.scen is the exact discriminator (-1 fresh, >=0 seeded).
        fresh = carry.scen < 0
        drawn = (((nxt.time >= train_cap) & fresh)
                 | (nxt.time >= DRAW_TURN)) & ~won
        terminal = won | drawn
        truncated = (nxt.time >= carry.horizon) & ~terminal
        done = terminal | truncated

        draw_r = jnp.where(drawn, jnp.float32(draw_reward), 0.0)
        # win_speed scales the win by earliness: a kill at tick 0 pays
        # (1 + win_speed), at the cap 1.0.  With gamma=1 a sparse reward
        # otherwise carries no gradient toward killing sooner.  Terminal-only,
        # so the sparse pipeline is untouched, and symmetric: the loser pays
        # what the winner gains, so decided games stay zero-sum.
        wmag = 1.0 + win_speed * jnp.clip(
            (train_cap - nxt.time) / train_cap, 0.0, 1.0).astype(jnp.float32)
        r0 = jnp.where(info.winner == 0, wmag,
                       jnp.where(info.winner == 1, -wmag, draw_r))
        gp = st.general_positions
        gt0 = gp[:, 1, 0] * B + gp[:, 1, 1]      # seat 0 hunts seat 1's general
        d0 = (info.winner == 1).astype(jnp.float32)

        if selfplay == "mirror":
            # decided games are zero-sum (seat 1 = -r0); a draw pays draw_r to
            # both seats; every per-seat label mirrors the same way
            cat = lambda a, b: jnp.concatenate([a, b])
            gt1 = gp[:, 0, 0] * B + gp[:, 0, 1]
            r1 = jnp.where(drawn, draw_r, -r0)
            reward = cat(r0, r1)
            d1 = (info.winner == 0).astype(jnp.float32)   # seat 1 died
            gen_target, died = cat(gt0, gt1), cat(d0, d1)
            terminal_o, done_o = cat(terminal, terminal), cat(done, done)
            planes, priv = cat(x0, x1), cat(priv0, priv1)
            lgl = cat(lg0, lg1)
            act_s, act_i = cat(a_s, b_s), cat(a_i, b_i)
            series = cat(z0, z1)
            valid = cat(jnp.ones(n, bool), carry.opp == 0)
        else:
            reward, gen_target, died = r0, gt0, d0
            terminal_o, done_o = terminal, done
            planes, priv, lgl = x0, priv0, lg0
            act_s, act_i = a_s, a_i
            series = z0
            valid = jnp.ones(n, bool)

        rs, rm0, rm1, rr0, rr1, rh, rsc, rop = reset(kr, n, pool, bank,
                                                     p_seed, scen_w,
                                                     scen_depth, league_frac)
        state = _sel(done, rs, nxt)
        m0 = _sel(done, rm0, m0)
        m1 = _sel(done, rm1, m1)
        # the rings are per-episode history: a new episode must not inherit the
        # previous game's economy, the same reason the memories reset here
        ring0 = _sel(done, rr0, ring0)
        ring1 = _sel(done, rr1, ring1)
        # The scenario that just ended, for the backward-curriculum mastery
        # signal; -1 for fresh episodes and for ticks where nothing ended.
        #
        # `done`, not `terminal`.  A seeded episode that runs out its horizon
        # without converting is truncated, not terminal; attributing on
        # `terminal` would drop those, and the metric would become "of the
        # seeded episodes that ended in a kill, how many did seat 0 win" --
        # nearly tautological, because seat 0 is the advantaged seat.  Failing
        # to convert in time is exactly what the criterion exists to detect,
        # and a truncated episode carries reward 0, so counting it makes the
        # rate mean "converted within the horizon".
        ep_scen = jnp.where(done, carry.scen, -1)
        scen_next = jnp.where(done, rsc, carry.scen)
        opp_next = jnp.where(done, rop, carry.opp)
        horizon = jnp.where(done, rh, carry.horizon)

        pb, pf, ps = O.pack_v(planes)
        out = (pb, pf, ps, series.astype(jnp.float16),
               priv.astype(jnp.float16), lgl,
               act_s, act_i, logp, q_a, vbar, reward, terminal_o, done_o,
               gen_target, died, ep_scen, valid)
        return Carry(state, m0, m1, ring0, ring1, horizon,
                     scen_next, opp_next, key), out

    def rollout(carry: Carry, params, opp_params, p_seed, temp, scen_w,
                scen_depth, pool, bank, T: int, eps=None):
        def body(c, _):
            return step(c, params, opp_params, p_seed, temp, scen_w,
                        scen_depth, pool, bank, eps)
        carry, out = jax.lax.scan(body, carry, None, length=T)
        return carry, Batch(*out)

    return step, rollout
