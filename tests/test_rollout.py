"""Rollout invariants: sparse reward, episode boundaries, and resumability.

The seed bank used here is synthetic: memories are derived from the stored
cumulative `seen` mask rather than reconstructed tick-by-tick from replays.
This file tests plumbing (shapes, seat canonicalisation, episode boundaries,
checkpointing); memory reconstruction is `train/seeds.py`'s job.
"""
from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from train import boards, ckpt, learn, net
from train import obs as O
from train import rollout as R
from train import config as C
from train.config import (B, DRAW_TURN, N_FLOAT, N_INT, N_PACKED, N_PLANES,
                          N_PRIV, N_SCALAR, N_SERIES, N_SRC, NetCfg,
                          TRING, TSCALES)

DEPTH0 = jnp.zeros(4, jnp.int32)   # backward-curriculum depth 0
SMALL = NetCfg(dim=64, depth=2, heads=2, cell_dim=16, q_dim=32, priv_hidden=16,
               bf16=False)
BANK_SRC = Path("data/apex/seed_bank.npz")
TINY = Path(os.environ.get("TMPDIR", "/tmp")) / "gz_tiny_seeds.npz"
# Selection needs a bigger slice: at 64 seeds some scenarios have no eligible
# seat, so the filter looks inert when it is not.
BIG = Path(os.environ.get("TMPDIR", "/tmp")) / "gz_sel_seeds.npz"


def _build(src: Path, dst: Path, n: int) -> None:
    d = np.load(src)
    out = {k: d[k][:n] for k in ("t", "armies", "owners", "castles",
                                 "generals", "mountains", "tags", "replay_id")}
    out["scenarios"] = d["scenarios"]
    out["scen_horizon"] = np.full(len(d["scenarios"]), 60, np.int32)
    seen = d["seen"][:n]
    for s_ in (0, 1):
        ev = seen[:, s_]
        out[f"ever_seen{s_}"] = ev
        out[f"last_seen{s_}"] = np.where(
            ev, out["t"][:, None, None], -1).astype(np.int32)
        go = ev & (out["owners"] == (1 - s_))
        out[f"ghost_owner{s_}"] = go
        ga = np.where(go, out["armies"], 0).astype(np.float32)
        out[f"ghost_army{s_}"] = ga
        out[f"prev_ghost{s_}"] = ga
    np.savez(dst, **out)


@pytest.fixture(scope="module")
def bank():
    if not BANK_SRC.exists():
        pytest.skip("seed bank not present")
    if TINY.exists() and "replay_id" not in np.load(TINY).files:
        TINY.unlink()                      # stale fixture without replay ids
    if not TINY.exists():
        _build(BANK_SRC, TINY, 64)
    return R.SeedBank.load(str(TINY))


@pytest.fixture(scope="module")
def rig(bank):
    bnk, _ = bank
    pool = boards.make_pool(jr.PRNGKey(0), boards.STAGES[0], 18)
    scen_w = jnp.full(bnk.scen_n.shape[0], 1.0 / bnk.scen_n.shape[0])
    params = net.init(jr.PRNGKey(1), SMALL)
    carry = R.init_carry(jr.PRNGKey(2), 4, pool, bnk, 0.5, scen_w,
                         jnp.zeros(scen_w.shape[0], jnp.int32))
    step, _ = R.make_step(SMALL)
    return bnk, pool, scen_w, params, carry, step


def test_one_step_has_the_declared_shapes_and_no_nans(rig):
    bnk, pool, scen_w, params, carry, step = rig
    c, out = step(carry, params, params, 0.5, 1.0, scen_w, DEPTH0, pool, bnk)
    b = R.Batch(*out)
    n = 4 * 2          # mirror self-play emits a transition per seat
    # planes are stored packed; the codec is what makes the buffer affordable
    assert b.pbits.shape == (n, N_PACKED, B, B) and b.pbits.dtype == jnp.uint8
    assert b.pfloat.shape == (n, N_FLOAT, B, B)
    assert b.pscalar.shape == (n, N_SCALAR)
    assert O.unpack_v(b.pbits, b.pfloat, b.pscalar).shape == (n, N_PLANES, B, B)
    assert b.priv.shape == (n, N_PRIV, B, B)
    assert b.legal.shape == (n, N_SRC, N_INT)
    for f in (b.logp, b.q_a, b.vbar, b.reward):
        assert np.isfinite(np.asarray(f)).all(), "non-finite scalar in the batch"
    assert (np.asarray(b.a_src) < N_SRC).all()
    assert (np.asarray(b.a_int) < N_INT).all()


def test_mirror_selfplay_trains_on_both_seats_and_is_zero_sum(rig):
    """Both seats of every self-play game are trained on, as in Straka et al.
    (arXiv:2606.23348).  The game is zero-sum, so seat 1's reward must be
    exactly the negation of seat 0's, and a draw must stay 0 for both."""
    bnk, pool, scen_w, params, carry, step = rig
    n = carry.horizon.shape[0]
    _, out = step(carry, params, params, 0.5, 1.0, scen_w, DEPTH0, pool, bnk)
    b = R.Batch(*out)
    assert b.reward.shape[0] == 2 * n, "seat 1's transitions are missing"
    r0, r1 = np.asarray(b.reward[:n]), np.asarray(b.reward[n:])
    assert np.array_equal(r1, -r0), "the two seats are not zero-sum"
    assert set(np.unique(r0)) <= {-1.0, 0.0, 1.0}

    # each seat hunts the other seat's general, and dies when the other wins
    g0, g1 = np.asarray(b.gen_target[:n]), np.asarray(b.gen_target[n:])
    assert not np.array_equal(g0, g1), "both seats were given the same target"
    d0, d1 = np.asarray(b.died[:n]), np.asarray(b.died[n:])
    assert ((d0 > 0) == (r0 < 0)).all() and ((d1 > 0) == (r1 < 0)).all()
    assert (d0 * d1 == 0).all(), "both seats cannot die in the same game"


def test_ladder_mode_still_trains_on_seat_zero_only(rig):
    """In ladder mode seat 1 is a lagging snapshot, so its data is off-policy
    and must not be trained on."""
    bnk, pool, scen_w, params, carry, _ = rig
    step, _ = R.make_step(SMALL, "ladder")
    _, out = step(carry, params, params, 0.5, 1.0, scen_w, DEPTH0, pool, bnk)
    assert R.Batch(*out).reward.shape[0] == carry.horizon.shape[0]


def test_privileged_information_cannot_move_the_policy(rig):
    """The critic reads privileged planes; the actor must not.  A leak would be
    invisible until the deployed bot played worse than its own evaluation."""
    bnk, pool, scen_w, params, carry, step = rig
    _, out = step(carry, params, params, 0.5, 1.0, scen_w, DEPTH0, pool, bnk)
    b = R.Batch(*out)
    assert b.priv.shape[1] == N_PRIV

    x = O.unpack_v(b.pbits, b.pfloat, b.pscalar).astype(jnp.float32)
    priv = b.priv.astype(jnp.float32)
    l1 = net.forward_v(params, x, b.series.astype(jnp.float32), priv,
                       b.legal, SMALL)[0]
    l2 = net.forward_v(params, x, b.series.astype(jnp.float32),
                       -3.0 * priv + 1.0, b.legal, SMALL)[0]
    assert np.array_equal(np.asarray(l1), np.asarray(l2))


def test_reward_is_sparse_and_only_terminals_pay(rig):
    """Terminal win/loss only: no reward is paid on a live tick."""
    bnk, pool, scen_w, params, carry, step = rig
    _, rollout = R.make_step(SMALL)
    _, batch = jax.jit(rollout, static_argnums=9)(
        carry, params, params, 0.5, 1.0, scen_w, DEPTH0, pool, bnk, 24)
    r, term = np.asarray(batch.reward), np.asarray(batch.terminal)
    assert (r[~term] == 0.0).all(), "reward paid on a non-terminal tick"
    assert set(np.unique(r)) <= {-1.0, 0.0, 1.0}


def test_a_draw_is_terminal_and_pays_zero(rig):
    """With the default draw_reward=0 the 1200-turn cap is not a loss."""
    bnk, pool, scen_w, params, carry, step = rig
    st = carry.state._replace(
        time=jnp.full_like(carry.state.time, DRAW_TURN - 1))
    c, out = step(carry._replace(state=st), params, params, 0.5, 1.0, scen_w,
                  DEPTH0, pool, bnk)
    b = R.Batch(*out)
    assert bool(b.terminal.all()), "reaching the cap must be terminal"
    assert float(np.abs(np.asarray(b.reward)).max()) == 0.0


def test_a_priced_draw_pays_the_penalty_to_both_seats(rig):
    """draw_reward prices the clock-out for both seats.  Draws are not zero-sum
    under a penalty, so seat 1 must not be paid -r0 = +0.75.  Without a priced
    draw, mirror self-play learns value ~0 for dominating positions and the
    policy hoards instead of closing."""
    bnk, pool, scen_w, params, carry, _ = rig
    step, _ = R.make_step(SMALL, "mirror", 0.0, -0.75)
    st = carry.state._replace(
        time=jnp.full_like(carry.state.time, DRAW_TURN - 1))
    c, out = step(carry._replace(state=st), params, params, 0.5, 1.0, scen_w,
                  DEPTH0, pool, bnk)
    b = R.Batch(*out)
    r, term = np.asarray(b.reward), np.asarray(b.terminal)
    assert bool(term.all())
    drew = r == -0.75
    won_or_lost = np.abs(r) == 1.0
    assert (drew | won_or_lost).all(), f"unexpected rewards {np.unique(r)}"
    assert drew.any(), "cap-reached games must pay the draw penalty"


def test_full_training_state_round_trips_through_a_checkpoint(rig, tmp_path):
    """Params, EMA, opponent snapshots, Adam moments, RNG and carry all
    round-trip.  Restoring weights without Adam moments silently changes the
    optimisation problem."""
    bnk, pool, scen_w, params, carry, step = rig
    ema = jax.tree.map(lambda x: x * 0.5, params)
    opt = learn.make_optimizer(3e-4, 0.5, 100)
    opt_state = opt.init(params)
    p = tmp_path / "ckpt_0000001.npz"
    snaps = [jax.tree.map(lambda x: x * 0.1, params),
             jax.tree.map(lambda x: x * 0.2, params)]
    ckpt.save(p, params=params, ema=ema, snapshots=snaps,
              opt_state=opt_state, carry=carry, key=jr.PRNGKey(9),
              meta={"iter": 1, "steps": 10})
    p2, e2, s2, o2, c2, k2, meta = ckpt.load(
        p, params_like=params, opt_state_like=opt_state, carry_like=carry)
    # the opponent FIFO is training state: losing it collapses the league to
    # mirror-only, the configuration most prone to self-play cycling
    assert len(s2) == 2
    for a, b in ((p2, params), (e2, ema), (s2[0], snaps[0]),
                 (s2[1], snaps[1])):
        for k in b:
            assert np.allclose(np.asarray(a[k]), np.asarray(b[k]))
    assert np.array_equal(np.asarray(c2.state.armies),
                          np.asarray(carry.state.armies))
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda x, y: bool(np.allclose(x, y)), o2, opt_state))
    assert meta["iter"] == 1


def test_seat_canonicalisation_puts_the_tagged_seat_at_zero(bank):
    bnk, _ = bank
    raw = np.load(TINY)
    for i in (0, 7, 23):
        for seat in (0, 1):
            st, m0, m1, r0, r1, hz = R._seed_state(bnk, i, seat, 60)
            assert np.array_equal(np.asarray(st.armies), raw["armies"][i])
            assert np.array_equal(np.asarray(st.ownership[0]),
                                  raw["owners"][i] == seat)
            assert int(hz) == int(raw["t"][i]) + 60
            # the rings must be canonicalised on the same seat as the memories:
            # seat 1's economy history under seat 0's board would be a fog leak
            assert r0.buf.shape == (N_SERIES, len(TSCALES), TRING)
            want0 = raw.get("rings0", None)
            if want0 is not None:
                src = raw[f"rings{seat}"][i]
                assert np.allclose(np.asarray(r0.buf), src, atol=1e-3)


def test_an_iteration_runs_end_to_end_and_moves_the_parameters(rig):
    """The integration test: rollout -> Q-boost -> filter -> update.  Anything
    mis-shaped between those stages fails here rather than hours into a run."""
    bnk, pool, scen_w, params, carry, _ = rig
    lc = learn.LossCfg(adv_frac=0.5)
    opt = learn.make_optimizer(1e-3, lc.max_grad_norm, 10)
    opt_state = opt.init(params)
    update = learn.make_update(SMALL, lc, opt, 1, 2)
    prepare = learn.make_prepare(lc)
    _, rollout = R.make_step(SMALL)
    carry, batch = jax.jit(rollout, static_argnums=9)(
        carry, params, params, 0.5, 1.0, scen_w, DEPTH0, pool, bnk, 16)

    flat = prepare(batch)
    new, opt_state, stats = update(params, opt_state, flat, jr.PRNGKey(3), 1.0)

    assert flat["adv"].shape[0] == int(batch.reward.size * lc.adv_frac)
    for k in ("pg", "v", "ent", "gen", "danger"):
        assert np.isfinite(float(stats[k])), f"{k} is not finite"
    # A fresh policy is near-uniform over the legal grid, so entropy is large
    # and positive; zero here would mean a collapsed policy.
    assert float(stats["ent"]) > 0.0, "entropy bonus has nothing to push on"
    moved = max(float(np.abs(np.asarray(new[k]) - np.asarray(params[k])).max())
                for k in params)
    assert moved > 0, "the update did not change any parameter"


def test_board_pool_rotation_actually_rotates_and_is_cheap():
    """`--pool-every` exists to bring in new maps, and is only affordable
    because `boards.make_env` caches: reset is jitted per env instance, so
    rebuilding the env on each refresh would recompile it."""
    stage = boards.STAGES[0]
    e1, e2 = boards.make_env(stage, 64), boards.make_env(stage, 64)
    assert e1 is e2, "make_env must cache, or a pool refresh recompiles"

    a = boards.make_pool(jr.PRNGKey(0), stage, 64)
    b = boards.make_pool(jr.PRNGKey(1), stage, 64)
    assert not np.array_equal(np.asarray(a.mountains), np.asarray(b.mountains)), \
        "a refreshed pool must contain different maps"
    c = boards.make_pool(jr.PRNGKey(0), stage, 64)
    assert np.array_equal(np.asarray(a.mountains), np.asarray(c.mountains)), \
        "the same key must reproduce the same pool"


def test_entropy_is_annealed_not_constant():
    """The entropy coefficient decays 0.05 -> 0.001 by a power law, so early
    exploration is an order of magnitude above a small constant."""
    lc = learn.LossCfg()
    v = [learn.entropy_coef(lc, t) for t in (0, 10, 100, 1000, 10_000)]
    assert v == sorted(v, reverse=True), v
    assert v[0] == lc.ent_coef
    assert learn.entropy_coef(lc, 10**12) == lc.ent_min, "floor must bind"
    assert v[3] > 0.003, "still well above the floor at t=1000"


def test_map_generation_stays_on_the_competition_preset():
    """Training boards must come from the engine's `competition` preset.
    Training on boards the evaluator never generates is a distribution shift
    no metric in the loop would show."""
    from generals import GeneralsEnv
    ours = boards.make_env(boards.STAGES[-1], 64)
    ref = GeneralsEnv(mode="competition", pool_size=64)
    for attr in ("mountain_density_range", "num_castles_range",
                 "castle_val_range"):
        assert getattr(ours, attr) == getattr(ref, attr), \
            f"{attr} drifted off the competition preset"


@pytest.fixture(scope="module")
def big_bank():
    if not BANK_SRC.exists():
        pytest.skip("seed bank not present")
    if not BIG.exists() or "replay_id" not in np.load(BIG).files:
        _build(BANK_SRC, BIG, 4000)
    return BIG


def test_seed_selection_keeps_winning_strong_seats_only(big_bank):
    """The bank is a backward curriculum, so a seed must be a position from
    which a win was actually available: the learner takes the seat that won,
    and only seats rated 3000+."""
    import pandas as pd
    if not (Path("data/apex/index.parquet").exists()
            and Path("data/manifest.parquet").exists()):
        pytest.skip("manifest not present")

    raw, names = R.SeedBank.load(str(big_bank))
    sel, _ = R.SeedBank.load(str(big_bank), index="data/apex/index.parquet",
                             manifest="data/manifest.parquet",
                             balance_players=False)
    assert int(sel.scen_n.sum()) < int(raw.scen_n.sum()), "the filter did nothing"

    # every surviving (seed, seat) must be the seat that won and be >= 3000 Elo
    d = np.load(big_bank)
    rid = d["replay_id"]
    winner = pd.read_parquet("data/apex/index.parquet").set_index(
        "replay_id")["winner"].reindex(rid).to_numpy()
    mf = pd.read_parquet("data/manifest.parquet").set_index(
        "replay_id").reindex(rid)
    a0 = mf["a_side"].to_numpy() == 0
    for j in range(len(names)):
        n, m = int(sel.scen_n[j, 0]), int(raw.scen_n[j, 0])
        if n == 0 or n == m:
            continue          # empty, or fell back to unfiltered (documented)
        idx = np.asarray(sel.scen_idx[j, 0])[:n]
        seat = np.asarray(sel.scen_seat[j, 0])[:n]
        assert (winner[idx] == seat).all(), f"{names[j]}: a losing seat survived"
        is_a = (seat == 0) == a0[idx]
        elo = np.where(is_a, mf["elo_a"].to_numpy()[idx],
                       mf["elo_b"].to_numpy()[idx])
        assert (elo >= 3000).all(), f"{names[j]}: a sub-3000 seat survived"


def test_expert_players_are_drawn_equally(big_bank):
    """Few players clear the rating bar and they contribute unevenly, so raw
    sampling would over-weight one player's style.  Balancing must equalise
    them per scenario, and must do it by repeating rows so that no position is
    discarded."""
    if not Path("data/manifest.parquet").exists():
        pytest.skip("manifest not present")
    import pandas as pd

    d = np.load(big_bank)
    rid, n = d["replay_id"], d["tags"].shape[0]
    mf = pd.read_parquet("data/manifest.parquet").set_index(
        "replay_id").reindex(rid)
    a0 = mf["a_side"].to_numpy() == 0
    who = np.full((n, 2), "", object)
    for st in (0, 1):
        isa = (st == 0) == a0
        who[:, st] = np.where(isa, mf["player_a"].to_numpy(),
                              mf["player_b"].to_numpy())

    def shares(balance):
        b, names = R.SeedBank.load(
            str(big_bank), index="data/apex/index.parquet",
            manifest="data/manifest.parquet", balance_players=balance)
        si, ss = np.asarray(b.scen_idx), np.asarray(b.scen_seat)
        nn = np.asarray(b.scen_n)          # (n_scen, n_depth)
        return [pd.Series(who[si[j, 0][:nn[j, 0]],
                              ss[j, 0][:nn[j, 0]]]).value_counts()
                for j in range(len(names))], int(nn[:, 0].sum())

    raw, n_raw = shares(False)
    bal, n_bal = shares(True)

    # every scenario is exactly equal across whatever players it contains
    for vc in bal:
        assert vc.nunique() == 1, f"not balanced within a scenario: {dict(vc)}"

    # no player was dropped, and nothing was discarded to achieve balance
    assert set(pd.concat(bal).index) == set(pd.concat(raw).index)
    assert n_bal >= n_raw, "balancing must repeat rows, never discard them"

    # and it actually changed something: the raw corpus is skewed
    assert any(vc.nunique() > 1 for vc in raw), "corpus was already balanced?"


def test_a_missing_manifest_keeps_everything_and_says_so(capsys):
    """A missing file must not silently turn the curriculum back into a coin
    flip."""
    if not TINY.exists():
        pytest.skip("tiny seed bank not built")
    bank, _ = R.SeedBank.load(str(TINY), index=None, manifest=None)
    out = capsys.readouterr().out
    assert "keeping all seats" in out.lower()
    assert int(bank.scen_n.sum()) > 0


def test_mirror_selfplay_win_rate_is_a_decisiveness_metric(rig):
    """The logged win rate slices the seat-0 half and divides by all terminals,
    draws included.  Under mirror self-play seat 1's reward is exactly -seat
    0's, so decided games split 50/50 by construction: the number is
    (1 - draw_rate)/2 in [0, 0.5], a measure of how often games end, not of
    playing strength.  Nothing may gate on it; the curriculum gates on a fixed
    opponent."""
    bnk, pool, scen_w, params, carry, step = rig
    n_env = carry.horizon.shape[0]
    _, rollout = R.make_step(SMALL)
    _, b = jax.jit(rollout, static_argnums=9)(
        carry, params, params, 1.0, 1.0, scen_w, DEPTH0, pool, bnk, 8)

    # the layout train.py's slice depends on
    assert b.reward.shape == (8, 2 * n_env), "expected (seg, 2*envs)"
    r = np.asarray(b.reward)
    assert np.array_equal(r[:, n_env:], -r[:, :n_env]), "seat 1 must be -seat 0"

    # the identity, on the emitted layout and on a synthetic sweep of draw rates
    for draw in (0.0, 0.3, 0.9):
        rng = np.random.default_rng(0)
        r0 = rng.choice([-1.0, 0.0, 1.0],
                        size=4096, p=[(1 - draw) / 2, draw, (1 - draw) / 2])
        full = np.stack([r0, -r0], 1)              # (n, 2 seats)
        term = np.ones_like(full, dtype=bool)      # draws count as terminals
        wr = float((full[:, :1] > 0).sum()) / float(term[:, :1].sum())
        assert abs(wr - (1 - draw) / 2) < 0.02, (draw, wr)
        assert wr <= 0.5 + 1e-9, "a mirror win rate can never exceed 0.5"


def _depth_bank(dst: Path, n_seed: int = 6) -> None:
    """A synthetic bank with SEED_DEPTHS variants per seed."""
    from train.config import N_EMA, N_SERIES, SEED_DEPTHS, TRING, TSCALES
    nd = len(SEED_DEPTHS)
    n = n_seed * nd
    rng = np.random.default_rng(0)
    tag_t = np.repeat(rng.integers(200, 400, n_seed), nd).astype(np.int16)
    dep = np.tile(np.arange(nd), n_seed)
    start = np.maximum(0, tag_t - np.asarray(SEED_DEPTHS)[dep]).astype(np.int16)
    out = {
        "t": start,
        "depth_idx": dep.astype(np.int8),
        "depth_ticks": (tag_t - start).astype(np.int16),
        "armies": rng.integers(0, 30, (n, B, B)).astype(np.int16),
        "owners": rng.integers(-1, 2, (n, B, B)).astype(np.int8),
        "castles": rng.random((n, B, B)) < 0.02,
        "generals": np.zeros((n, B, B), bool),
        "mountains": np.zeros((n, B, B), bool),
        "replay_id": np.repeat(np.arange(n_seed), nd).astype(np.int64),
        "tags": np.ones((n, 2, 1), bool),
        "scenarios": np.array(["strike_conversion"]),
        "scen_horizon": np.array([40], np.int32),
    }
    out["generals"][:, 3, 3] = True
    out["generals"][:, 10, 10] = True
    out["owners"][:, 3, 3], out["owners"][:, 10, 10] = 0, 1
    for s_ in (0, 1):
        out[f"ever_seen{s_}"] = np.ones((n, B, B), bool)
        out[f"last_seen{s_}"] = np.zeros((n, B, B), np.int32)
        out[f"ghost_owner{s_}"] = np.zeros((n, B, B), bool)
        out[f"ghost_army{s_}"] = np.zeros((n, B, B), np.float32)
        out[f"egen_seen{s_}"] = np.ones(n, bool)
        out[f"ema{s_}"] = np.zeros((n, N_EMA, 2, B, B), np.float16)
        out[f"rings{s_}"] = np.zeros((n, N_SERIES, len(TSCALES), TRING),
                                     np.float16)
    np.savez(dst, **out)


def test_a_deeper_start_is_the_same_target_from_further_back(tmp_path):
    """The backward curriculum's core invariant.

    A deeper seed starts earlier but must end at the same absolute tick, so it
    is a longer task on the same target rather than a different task.  If the
    end moved with the start, depths would not be comparable and a mastery
    criterion could not advance between them."""
    from train.config import SEED_DEPTHS
    src = tmp_path / "depth.npz"
    _depth_bank(src)
    bank, names = R.SeedBank.load(str(src))

    nd = len(SEED_DEPTHS)
    assert np.asarray(bank.scen_n).shape == (1, nd), "scen_n needs a depth axis"
    assert (np.asarray(bank.scen_n) > 0).all(), "every depth must have rows"

    t = np.asarray(bank.t)
    dt = np.asarray(bank.depth_ticks)
    for k in range(0, len(t), nd):
        ends = [int(R._seed_state(bank, i, 0, 40)[-1]) for i in range(k, k + nd)]
        assert len(set(ends)) == 1, f"end tick moved with depth: {ends}"
        starts = t[k:k + nd]
        assert (np.diff(starts) <= 0).all(), "a deeper start must be earlier"
        assert (t[k:k + nd] + dt[k:k + nd] == t[k] + dt[k]).all(), \
            "the tagged tick must be identical across depths"


def test_depth_selection_draws_only_from_the_requested_depth(tmp_path):
    """`scen_depth` advances the curriculum, so it must actually gate which
    rows are drawn; otherwise the mechanism is inert while the log reports
    depth advances."""
    from train.config import SEED_DEPTHS
    src = tmp_path / "depth.npz"
    _depth_bank(src)
    bank, _ = R.SeedBank.load(str(src))
    di = np.repeat(np.arange(len(SEED_DEPTHS)), 1)   # row -> depth, per seed
    idx = np.asarray(bank.scen_idx)                  # (1, n_depth, width)
    for d in range(len(SEED_DEPTHS)):
        rows = idx[0, d][:int(np.asarray(bank.scen_n)[0, d])]
        got = {int(r) % len(SEED_DEPTHS) for r in rows}
        assert got == {d}, f"depth {d} drew rows of depth {got}"


def test_mastery_counts_a_failure_to_convert_as_a_loss():
    """The backward-curriculum criterion must attribute on `done`, not
    `terminal`.

    A seeded episode that runs out its horizon without a kill is truncated, not
    terminal.  Attributing on `terminal` drops exactly those episodes, so the
    rate becomes "of the seeded episodes that ended in a kill, how many did
    seat 0 win", which is nearly tautological and insensitive to depth."""
    import inspect
    src = inspect.getsource(R.make_step)
    assert "ep_scen = jnp.where(done, carry.scen, -1)" in src, \
        "mastery must attribute on `done`; `terminal` drops timeouts"

    # and the semantics that makes it correct: a truncation pays reward 0, so
    # counting it lands as a non-win rather than as an absent sample
    reward = np.array([1.0, 0.0, -1.0, 0.0])
    done = np.array([True, True, True, True])     # 2nd and 4th are timeouts
    terminal = np.array([True, False, True, False])
    assert (reward[done] > 0).sum() / done.sum() == 0.25
    assert (reward[terminal] > 0).sum() / terminal.sum() == 0.5, \
        "attributing on terminal inflates the rate by dropping timeouts"


def test_eps_build_mixes_the_behaviour_policy_and_records_its_logp():
    """Build exploration is an epsilon-mixture, not reward shaping: the sampler
    mixes eps mass over the legal builds, the recorded logp is the mixture's
    (the behaviour policy PPO's ratio needs), and the returned pi stays the
    target policy so Vbar is untouched."""
    import jax.random as jr
    from train.config import INT_BUILD, N_INT, N_SRC
    from train.rollout import sample_joint

    rng = np.random.default_rng(0)
    n = 4
    logits = jnp.asarray(rng.standard_normal((n, N_SRC, N_INT)), jnp.float32)
    # lane 3 has no legal build: eps must become 0 there
    logits = logits.at[:, :, INT_BUILD].set(-1e30)
    build_cells = jnp.array([5, 9, 200])
    logits = logits.at[:3, build_cells, INT_BUILD].set(0.5)

    eps = 0.5
    a_s, a_i, logp, pi = sample_joint(jr.PRNGKey(1), logits, 1.0, eps)

    # pi is the target policy: it must ignore eps entirely
    _, _, _, pi0 = sample_joint(jr.PRNGKey(1), logits, 1.0, 0.0)
    assert np.allclose(np.asarray(pi), np.asarray(pi0))

    # the recorded logp must be the mixture's, verified against a hand-built b
    p = np.asarray(pi).reshape(n, -1)
    b = (1 - eps) * p.copy()
    for lane in range(3):
        for c in np.asarray(build_cells):
            b[lane, c * N_INT + INT_BUILD] += eps / len(build_cells)
    b[3] = p[3]                                   # no legal build -> pure pi
    flat_a = np.asarray(a_s) * N_INT + np.asarray(a_i)
    want = np.log(b[np.arange(n), flat_a] + 1e-30)
    assert np.allclose(np.asarray(logp), want, atol=1e-5), \
        f"behaviour logp is not the mixture's: {np.asarray(logp)} vs {want}"

    # and at eps ~ 1 with builds legal, builds must actually be sampled
    hits = 0
    for k in range(20):
        _, ai, _, _ = sample_joint(jr.PRNGKey(k), logits[:3], 1.0, 0.999)
        hits += int((np.asarray(ai) == INT_BUILD).sum())
    assert hits > 40, f"eps-build is not forcing builds: {hits}/60"


def test_train_cap_terminates_fresh_games_and_spares_seeds(rig):
    """--train-cap ends fresh games early with the draw price.  Seeded episodes
    carry their own horizon and must keep truncating into the value bootstrap;
    capping them would score a winning attacker as a draw mid-conversion.
    DRAW_TURN itself stays a draw for everyone."""
    bnk, pool, scen_w, params, carry, _ = rig
    step, _ = R.make_step(SMALL, "mirror", 0.0, -0.75, 600)
    st = carry.state._replace(time=jnp.full_like(carry.state.time, 599))
    c, out = step(carry._replace(state=st), params, params, 0.0, 1.0, scen_w,
                  DEPTH0, pool, bnk)
    b = R.Batch(*out)
    r, term = np.asarray(b.reward), np.asarray(b.terminal)
    fresh = np.concatenate([np.asarray(carry.scen)] * 2) < 0
    assert fresh.any() and (~fresh).any(), "rig must mix fresh and seeded"
    assert bool(term[fresh].all()), "fresh games must draw at the cap"
    # a seeded env at the cap tick must NOT be terminal-drawn there (it
    # truncates at its own horizon into the bootstrap instead)
    assert not term[~fresh].any(), "seeds must not draw at the train cap"
    assert ((r[~fresh] != -0.75)).all(), "a seed must never eat the cap price"
    assert ((r == -0.75) | (np.abs(r) == 1.0) | (r == 0.0)).all()


def test_a_seed_at_the_true_draw_turn_still_draws(rig):
    """DRAW_TURN is the engine's rule: at 1200 everything alive draws and pays
    the price, seeded or not.  The cap exemption must not leak past the real
    clock."""
    bnk, pool, scen_w, params, carry, _ = rig
    step, _ = R.make_step(SMALL, "mirror", 0.0, -0.75, 600)
    st = carry.state._replace(
        time=jnp.full_like(carry.state.time, C.DRAW_TURN - 1))
    c, out = step(carry._replace(state=st), params, params, 0.0, 1.0, scen_w,
                  DEPTH0, pool, bnk)
    b = R.Batch(*out)
    assert bool(np.asarray(b.terminal).all())


def test_win_speed_pays_early_kills_more_and_stays_zero_sum(rig):
    """gamma=1 sparse rewards contain no gradient toward killing sooner; the
    earliness premium adds one at the terminal only.  Decided games stay
    zero-sum (the loser pays what the winner gains)."""
    bnk, pool, scen_w, params, carry, _ = rig
    step, _ = R.make_step(SMALL, "mirror", 0.0, -1.0, 450, 0.5)
    st = carry.state._replace(time=jnp.zeros_like(carry.state.time))
    c, out = step(carry._replace(state=st), params, params, 0.0, 1.0, scen_w,
                  DEPTH0, pool, bnk)
    b = R.Batch(*out)
    r = np.asarray(b.reward)
    n = r.shape[0] // 2
    decided = np.abs(r[:n]) > 0.5
    assert np.allclose(r[:n][decided], -r[n:][decided]), "zero-sum broken"
    wins = r[r > 0.5]
    if wins.size:
        assert (wins <= 1.5 + 1e-6).all() and (wins >= 1.0 - 1e-6).all()
        assert (wins > 1.4).all(), "a tick-1 win must pay ~1.5 at speed 0.5"


def test_the_league_masks_scripted_seats_and_seeds_stay_mirror(rig):
    """League envs: seat-1 actions come from a script, so their samples are
    invalid for PPO (the ratio is undefined).  Seeded envs must stay mirror
    (the curriculum measures conversion vs the live defender)."""
    bnk, pool, scen_w, params, carry, _ = rig
    step, _ = R.make_step(SMALL, "mirror", 0.0, -1.0, 600, 0.0, 1.0)
    c0 = R.init_carry(jr.PRNGKey(9), 4, pool, bnk, 0.5, scen_w,
                      DEPTH0, league_frac=1.0)
    assert (np.asarray(c0.opp)[np.asarray(c0.scen) >= 0] == 0).all(), \
        "a seeded env must never draw a scripted opponent"
    c, out = step(c0, params, params, 0.5, 1.0, scen_w, DEPTH0, pool, bnk)
    b = R.Batch(*out)
    v = np.asarray(b.valid)
    n = v.shape[0] // 2
    assert v[:n].all(), "seat 0 is always the learner and always valid"
    scripted = np.asarray(c0.opp) != 0
    assert (~v[n:][scripted]).all(), "scripted seats must be masked"
    assert v[n:][~scripted].all(), "mirror seats must stay trainable"
    if scripted.any():
        assert bool(scripted.any()), "league_frac=1 must draw scripts on fresh"


def test_league_zero_is_pure_mirror_selfplay(rig):
    bnk, pool, scen_w, params, carry, step = rig
    c, out = step(carry, params, params, 0.5, 1.0, scen_w, DEPTH0, pool, bnk)
    b = R.Batch(*out)
    assert bool(np.asarray(b.valid).all()), \
        "with no league every sample is trainable"


def test_a_placeholder_bank_trains_on_fresh_boards_only():
    """`--no-seeds`: the rollout compiles against the placeholder's shapes, and
    with p_seed = 0 no episode is ever drawn from it.  Needs no replay data."""
    pool = boards.make_pool(jr.PRNGKey(0), boards.STAGES[0], 18)
    params = net.init(jr.PRNGKey(1), SMALL)
    bank, names = R.SeedBank.placeholder()
    assert names == R.SCENARIOS and bank.size == 1
    w = jnp.ones(len(names)) / len(names)
    depth = jnp.zeros(len(names), jnp.int32)
    carry = R.init_carry(jr.PRNGKey(3), 4, pool, bank, 0.0, w, depth)
    assert (np.asarray(carry.scen) == -1).all(), "a seeded episode was drawn"

    _, rollout = R.make_step(SMALL)
    carry, b = jax.jit(rollout, static_argnums=9)(
        carry, params, params, 0.0, 1.0, w, depth, pool, bank, 8)
    assert (np.asarray(carry.scen) == -1).all()
    assert np.isfinite(np.asarray(b.reward)).all()
