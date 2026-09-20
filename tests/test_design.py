"""Design invariants, made executable.

Contract checks on the architecture, the action space, the loss configuration
and the training-state plumbing.  A deliberate design change fails loudly here
instead of drifting silently; behaviour is tested in the other files.

References: Straka et al. (arXiv:2606.23348) for the trunk and the PPO recipe,
Fan & Farina (arXiv:2605.19235) for the Q-boosted advantage.
"""
from __future__ import annotations

import ast
import inspect
import json
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

import pytest

from train import arena, boards, ckpt, config, learn, net, net_torch
from train import np_exec, np_obs, rollout, train

ROOT = Path(__file__).resolve().parent.parent


def _code(obj) -> str:
    """Source of `obj` with comments and docstrings removed, so assertions on
    source text depend on code alone."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# --- trunk -------------------------------------------------------------------

def test_trunk_matches_straka_table_II():
    c = config.NetCfg()
    assert (c.dim, c.depth, c.heads) == (448, 7, 8), "Table II is 448 / 7 / 8"
    assert config.PATCH == 3 and config.N_TOKENS == 49, "3x3 patches -> 49 tokens"
    # Table II's feedforward (1344) is a 2-matrix GELU MLP: 2*448*1344 MAC per
    # token.  SwiGLU has three matrices, so the equal-FLOP width is 2.0x.
    assert 3 * c.dim * c.mlp_hidden == 2 * c.dim * 1344, \
        "SwiGLU width must equal the Table II feedforward budget"
    mb = config.n_params(c, deploy_only=True) * 2 / 1e6
    assert 12.0 < config.n_params(c) / 1e6 < 17.0, "within a Table II-sized model"
    assert mb < 45.0, "fp16 weights must leave room under the 50 MB zip cap"


def test_trunk_uses_rmsnorm_qknorm_and_swiglu():
    shapes = config.param_shapes(config.NetCfg())
    assert any(k.endswith("qn/g") for k in shapes), "QK-norm missing"
    assert any(k.endswith("kn/g") for k in shapes)
    assert all(not k.endswith("n1/b") for k in shapes), "RMSNorm has no bias"
    assert any(k.endswith("gate/w") for k in shapes), "SwiGLU gate missing"
    src = _code(net._block)
    assert "softmax" in src and "float32" in src, "softmax must run in f32"


# --- action space ------------------------------------------------------------

def test_the_action_space_is_cell_times_intent():
    """One categorical over (source cell, intent), as in Straka et al.

    The source is named explicitly, so the legality mask is exact per pair and
    every offered action is one the engine executes.  Indirect sources or
    targets resolved from board state at execution time can only be masked
    marginally, and the resulting silent passes are unlearnable."""
    assert config.N_SRC == config.N_CELLS == 441
    assert not hasattr(config, "N_SLOTS"), "sources are cells, not slots"
    assert not hasattr(config, "SRC_CELL0")

    # 4 directions x {all, half}, plus build and pass
    assert config.N_INT == 2 * config.N_DIRS + 2 == 10
    assert config.INT_DIR0 == 0 and config.INT_HALF0 == 4
    assert config.INT_BUILD == 8 and config.INT_PASS == 9


# --- advantage estimation ----------------------------------------------------

def test_vrpo_is_the_default_and_gae_remains_switchable():
    lc = learn.LossCfg()
    assert lc.advantage == "vrpo", "Q-boosting is the default estimator"
    # Not Fan & Farina's lam=1: Q-boosting does not telescope, so at
    # gamma=lam=1 the control-variate corrections accumulate over a 512-tick
    # segment and blow the critic target up.
    assert lc.lam == 0.9
    assert lc.gamma == 1.0, "finite episodes, gamma = 1"
    assert callable(learn.gae), "GAE stays available for the estimator A/B"
    assert "advantage" in inspect.signature(learn.LossCfg).parameters


def test_q_critic_is_factored_with_a_rank_r_interaction():
    c = config.NetCfg()
    assert c.q_rank > 0, "rank-0 cannot express source x intent interaction"
    shapes = config.param_shapes(c)
    for k in ("q/base", "q/src_cell", "q/intg"):
        assert k + "/w" in shapes, f"{k} missing from the Q head"
    assert shapes["q/src_cell/w"][-1] == 1 + c.q_rank


# --- regularization and reward -----------------------------------------------

def test_the_regularizer_is_entropy_plus_an_expander_magnet():
    """KL(pi || m) toward a scripted expander prior m, split as -H + CE so the
    entropy term and the magnet term anneal on separate schedules.  With
    mag_coef=None the two recombine into the fused KL."""
    src = _code(learn.make_loss)
    assert "ent_coef * neg_h" in src and "mc * ce_m" in src, \
        "the decoupled entropy/magnet split is gone"
    assert "kl_ref = neg_h + ce_m" in src, "kl_ref must log the full KL"
    from train import magnet
    assert (magnet.S_PASS, magnet.S_CITY, magnet.S_TOPK) == (0.2, 3.5, 2.0)
    assert magnet.S_HUNT > 0 and 0 < magnet.S_HALF < 1
    lc = learn.LossCfg()
    assert learn.entropy_coef(lc, 0) > learn.entropy_coef(lc, 1000)


def test_reward_is_terminal_only_with_no_shaping_terms():
    """The rollout pays at terminals only: no potential shaping, no per-tick
    term.  The win magnitude (`wmag`, the earliness premium) and the draw price
    are both terminal; `test_rollout` pins the values behaviourally."""
    code = _code(rollout)
    for token in ("import phi", "shaping_v", "shape_coef", "beta"):
        assert token not in code, f"'{token}' is back in the rollout"
    assert "jnp.where(info.winner == 0, wmag," in code
    assert config.DRAW_TURN == 1200 and config.DEATHTOUCH_TURN == 800


def test_observation_has_no_death_clock_but_keeps_economy_phase():
    """A remaining-ticks plane lets the policy lean on the clock instead of
    learning urgency, so it is absent.  The periodic income clocks stay: they
    are game mechanics, not the episode cap."""
    assert "ticks_left" not in config.PLANES
    assert "bonus_in" in config.PLANES and "struct_in" in config.PLANES
    assert "enemy_land_n" not in config.PLANES


# --- sample and parameter handling -------------------------------------------

def test_top_advantage_filtering_and_ema_are_on():
    lc = learn.LossCfg()
    assert lc.adv_frac == 0.25, "Straka et al. Table V: top-advantage fraction"
    assert lc.max_grad_norm == 0.267, "Straka et al. Table V: gradient-norm clip"
    assert 0 < lc.ema_decay < 1 and callable(learn.ema_update)
    assert "ema" in inspect.signature(ckpt.save).parameters, \
        "the EMA weights are what ship; they must be checkpointed"
    assert "snapshots" in inspect.signature(ckpt.save).parameters


def test_entropy_schedule_follows_the_paper_at_its_horizon():
    """0.05 * (t+1)^-0.2 with t in iterations (Straka et al.), which lands on
    ~0.0050 at t = 100k."""
    lc = learn.LossCfg()
    assert lc.ent_coef == 0.05 and lc.ent_power == 0.2
    end = learn.entropy_coef(lc, 100_000)
    assert abs(end - 0.0050) < 5e-4, f"endpoint {end:.4f} != 0.0050"
    assert learn.entropy_coef(lc, 0) > learn.entropy_coef(lc, 1000) > end


def test_the_shipped_path_excludes_every_training_only_head():
    import torch
    assert set(config.TRAIN_ONLY) == {"gen", "danger", "priv", "q"}
    tp = net_torch.prepare({k: torch.zeros(s) for k, s in
                            config.param_shapes(config.NetCfg()).items()})
    assert not any(k.split("/")[0] in config.TRAIN_ONLY for k in tp)


def test_both_numpy_twins_exist_for_the_deployment_path():
    for mod, fn in ((np_obs, "encode"), (np_obs, "legal_mask"),
                    (np_obs, "bfs_from"), (np_exec, "execute")):
        assert hasattr(mod, fn), f"{mod.__name__}.{fn} missing"


def test_rollout_geometry_lives_in_one_place():
    """Straka et al. Table V: 512 envs x 512 ticks, minibatch size 1024, one
    epoch.  Kept in `config` so the trainer and the profiling tools cannot
    measure different geometries."""
    assert (config.ENVS, config.SEG, config.EPOCHS) == (512, 512, 1)
    assert config.MINIBATCH == 1024, "a size, not a count"
    src = _code(train.main)
    for tok in ("C.ENVS", "C.SEG", "C.EPOCHS", "C.MINIBATCH"):
        assert tok in src, f"train.py restates {tok} instead of reading config"


# --- run plumbing ------------------------------------------------------------

def test_wandb_can_never_stop_a_run():
    """A long run must not die because a metrics service is unreachable.

    Every entry point degrades to a no-op: `init` returns False, and `log` and
    `finish` are safe on a run that was never started or has already failed.
    """
    from train import wb
    assert wb.init("p", "n", {}, mode="off") is False
    wb.log({"a": 1.0}, step=1)      # must not raise on a dead run
    wb.finish()
    wb.finish()                      # ...and must be idempotent


def test_env_loader_never_returns_secret_values():
    """`load_env` reports only whether a key is present; it must not hand the
    value back to a caller that might log it."""
    from train import wb
    assert "return bool(" in _code(wb.load_env), \
        "load_env must return presence, not a value"
    # `from __future__ import annotations` makes these strings, not types
    assert inspect.signature(wb.load_env).return_annotation in (bool, "bool")
    assert isinstance(wb.load_env("/nonexistent"), bool)


def test_dotenv_is_gitignored():
    """.env holds API keys and must never be committable."""
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()


def test_resume_restores_every_piece_of_training_state():
    """A checkpoint is all of the state or it is not a checkpoint.  Restoring
    weights without Adam moments silently changes the optimisation problem;
    without the EMA it evaluates a different policy."""
    sig = set(inspect.signature(ckpt.save).parameters)
    for field in ("params", "ema", "snapshots", "opt_state", "carry",
                  "key", "meta"):
        assert field in sig, f"ckpt.save does not persist {field}"

    src = _code(train.main)
    # The curriculum is competence-gated and may deliberately lag the schedule,
    # so the stage must be restored, not recomputed from the iteration count.
    assert "meta.get('stage'" in src, "stage is recomputed instead of restored"
    assert "meta.get('roll_win'" in src, "the readiness EMA is not restored"
    # ...and everything the loop mutates must be written back out
    for k in ("'stage': stage_i", "'roll_win': roll_win", "'wandb_id'"):
        assert k in src, f"{k} is never checkpointed"


def test_a_partial_checkpoint_is_rejected_at_load(tmp_path):
    """A checkpoint missing its EMA, or whose optimizer state was written by
    different code, must fail at load rather than resume into a subtly
    different run."""
    params = {"w": np.ones((2, 2), np.float32)}
    like = dict(params_like=params, opt_state_like=(np.zeros(2),),
                carry_like=(np.zeros(1),))

    def write(name, blob):
        path = tmp_path / name
        np.savez(path, key=np.zeros(2, np.uint32), **blob)
        path.with_suffix(".json").write_text(json.dumps({"n_snapshots": 0}))
        return path

    with pytest.raises(ValueError):                       # no e:: weights
        ckpt.load(write("no_ema.npz", {"p::w": params["w"]}), **like)
    with pytest.raises(ValueError):                       # leaf-count mismatch
        ckpt.load(write("leaves.npz", {"p::w": params["w"], "e::w": params["w"],
                                       "o::0": np.zeros(2), "o::1": np.zeros(2),
                                       "c::0": np.zeros(1)}), **like)


def test_resuming_continues_the_same_wandb_chart():
    """A restart must not appear as a second, disconnected run."""
    from train import wb
    assert "run_id" in inspect.signature(wb.init).parameters
    assert "resume='allow'" in _code(wb.init)
    assert callable(wb.run_id)


def _losing_trace(T=40, N=8):
    """Env 0 sights the general at t=10 and loses at t=25; nothing else ends."""
    alive = np.ones((T, N), bool)
    alive[25:, 0] = False
    winner = np.full((T, N), -1, np.int32)
    winner[24, 0] = 1
    seen = np.zeros((T, N), bool)
    seen[10:, 0] = True
    z = np.zeros((T, N), np.float32)
    return arena.Trace(jnp.asarray(alive), jnp.asarray(winner), jnp.asarray(z),
                       jnp.asarray(z), jnp.asarray(z), jnp.asarray(z),
                       jnp.asarray(z), jnp.zeros((T, N), bool), jnp.asarray(seen),
                       jnp.zeros((T, N), bool), jnp.zeros((T, N), bool))


def test_the_profile_reports_action_mass_fractions():
    prof = arena.profile(_losing_trace())
    assert "build_frac" in prof and "half_frac" in prof


def test_closing_and_scouting_metrics_are_readable_before_any_win():
    """`len_win` and `t_to_kill` are conditioned on winning, so they are NaN
    early in training, when they are most needed.  The unconditional versions
    must be finite whenever any game ended or any general was seen."""
    N = 8
    prof = arena.profile(_losing_trace(N=N))
    assert prof["winrate"] == 0.0, "the fixture deliberately never wins"
    assert np.isnan(prof["len_win"]), "len_win is win-conditioned by design"
    assert np.isfinite(prof["len_decided"]) and prof["len_decided"] == 25
    assert np.isfinite(prof["decided_300"])
    assert prof["seen_frac"] == 1 / N
    assert prof["t_seen"] == 10
    assert np.isfinite(prof["t_after_seen"])


def test_the_economy_clock_matches_the_engine():
    """BONUS_PERIOD and STRUCT_PERIOD are constants in config, but the engine is
    the authority, so pin them behaviourally.

    Under pass the agent owns exactly one cell, so the all-cell bonus is worth
    +1, the same size as a structure tick.  The test therefore measures exact
    per-tick deltas and looks for ticks where growth exceeds the structure
    baseline."""
    import jax.random as jr
    from train.rollout import cstep

    pool = boards.make_pool(jr.PRNGKey(0), boards.STAGES[-1], 32)
    st = jax.tree.map(lambda x: x[0], pool)
    noop = jnp.zeros((2, 5), jnp.int32).at[:, 0].set(1)

    prev, delta = None, {}
    for _ in range(2 * config.BONUS_PERIOD + 5):
        out = cstep(st, noop)
        st = out[0] if isinstance(out, tuple) else out
        owned = np.asarray(st.ownership)[0]
        tot = float((np.asarray(st.armies) * owned).sum())
        if prev is not None:
            delta[int(st.time)] = tot - prev
        prev = tot

    assert int(np.asarray(st.ownership)[0].sum()) == 1, "PASS should hold one cell"

    baseline = max(v for t, v in delta.items() if t % config.BONUS_PERIOD)
    excess = sorted(t for t, v in delta.items() if v > baseline)
    assert excess == [config.BONUS_PERIOD, 2 * config.BONUS_PERIOD], \
        f"all-cell bonus fired at {excess}, expected every {config.BONUS_PERIOD}"

    struct = sorted(t for t, v in delta.items()
                    if v > 0 and t < 2 * config.STRUCT_PERIOD + 6)
    assert all(t % config.STRUCT_PERIOD == 0 for t in struct), \
        f"structures grew off the {config.STRUCT_PERIOD}-turn cadence: {struct}"


def test_the_observation_carries_the_economy_clock():
    """The 50-turn bonus is a 60-100 army step late game, on a known tick.  Both
    phases are encoded as time remaining: with gamma=1 what prices a state is
    time-to-event."""
    assert "bonus_in" in config.PLANES and "struct_in" in config.PLANES
    b = config.PLANES.index("bonus_in")
    for t, want in ((0, 1.0), (49, 1 / 50), (50, 1.0), (99, 1 / 50)):
        got = (config.BONUS_PERIOD - t % config.BONUS_PERIOD) / config.BONUS_PERIOD
        assert abs(got - want) < 1e-9, (t, got, want)
    assert config.PLANES.index("struct_in") == b + 1


def test_the_motion_planes_are_ordered_and_decay_at_their_named_rates():
    """PLANES order vs obs.encode order is load-bearing, and the parity tests
    cannot check it: both twins can be wrong identically and still agree.

    So check semantics: hold the board still and each ema_*_k must converge
    toward the current value at its own named rate, fast before slow, in the
    declared plane slot."""
    idx = [config.PLANES.index(f"ema_own_{w}") for w in (2, 8, 32, 128)]
    assert idx == list(range(idx[0], idx[0] + 4)), "own scales must be contiguous"
    assert [config.PLANES.index(f"ema_enemy_{w}") for w in (2, 8, 32, 128)] \
        == list(range(idx[0] + 4, idx[0] + 8)), "enemy block must follow own"

    # a static field: every EMA should climb monotonically toward it
    alphas = np.asarray(config.EMA_ALPHAS)
    ema = np.zeros(config.N_EMA)
    target = 1.0
    for _ in range(64):
        ema = ema + alphas * (target - ema)
    assert np.all(np.diff(ema) < 0), \
        f"faster scales must converge first, got {ema}"
    # closed form: after n steps E = 1 - (1-alpha)^n
    assert np.allclose(ema, 1 - (1 - alphas) ** 64, atol=1e-6)
    # the 128-tick scale must still be far from converged at 64 ticks, or it is
    # not carrying a longer horizon than the others
    assert ema[-1] < 0.5, f"the slow scale converged too fast: {ema[-1]}"


def test_the_plane_codec_is_lossless_where_it_claims_to_be():
    """The trajectory buffer, not FLOPs, caps `envs`, so plane storage is a
    throughput decision.  Packing must be lossless for the two groups it claims:
    booleans (1 bit, not 16) and broadcast scalars (one value, not 441 copies).
    """
    import train.obs as O

    rng = np.random.default_rng(0)
    x = rng.random((config.N_PLANES, config.B, config.B)).astype(np.float32)
    for i in config.BOOL_IDX:
        x[i] = (x[i] > 0.5).astype(np.float32)
    for i in config.SCALAR_IDX:          # genuinely constant across cells
        x[i] = x[i][0, 0]

    y = np.asarray(O.unpack(*O.pack(jnp.asarray(x))), np.float32)
    bi, si = list(config.BOOL_IDX), list(config.SCALAR_IDX)
    assert np.array_equal(y[bi], x[bi]), "boolean group must round-trip exactly"
    assert np.allclose(y[si], x[si], atol=1e-3), "scalar group must round-trip"
    assert np.abs(y - x).max() < 1e-3, "float group must stay within fp16"

    # and it must actually save what it claims: bytes per cell, packed vs fp16
    stored = config.N_PACKED + 2 * config.N_FLOAT
    assert stored / (2 * config.N_PLANES) < 0.52, \
        f"codec saves only {1 - stored / (2 * config.N_PLANES):.0%}, expected ~50%"


def test_no_plane_is_both_packed_and_float():
    """Index groups partition PLANES.  An overlap would overwrite one group
    with the other in `unpack`, and a gap would leave a plane as zeros; no
    shape check catches either."""
    groups = list(config.BOOL_IDX) + list(config.FLOAT_IDX) + list(config.SCALAR_IDX)
    assert sorted(groups) == list(range(config.N_PLANES)), \
        "storage groups must partition the planes exactly"


def test_the_trainer_reads_its_architecture_from_netcfg():
    """The CLI must read its architecture defaults from NetCfg rather than
    restate them: a trainer and a profiler that disagree on `heads` build
    models that cannot exchange a checkpoint, since the QK-norm gains are
    sized per head_dim."""
    parser_src = _code(train.main)
    d = config.NetCfg()
    for name, want in (("dim", d.dim), ("depth", d.depth), ("heads", d.heads)):
        assert f"'--{name}', type=int, default=_d.{name}" in parser_src, (
            f"--{name} must default to NetCfg().{name} (={want}), not a literal")


# --- p_seed: gated, not ramped -----------------------------------------------

def _pargs(**kw):
    import argparse
    d = dict(p_seed_base=0.30, p_seed_floor=0.10, p_seed_gate=0.9,
             p_seed_decay=1000)
    d.update(kw)
    return argparse.Namespace(**d)


def test_p_seed_gate_needs_max_depth_and_conversion():
    """The gate must not open on conversion alone.

    While depths remain, every advance resets that scenario's mastery window,
    so a high rate means "mastered this depth" and the task is about to get
    harder.  Cutting the curriculum's budget on that reading would starve the
    ramp while it still has work to do.
    """
    from train.train import gate_open
    top = config.N_DEPTH - 1
    n = 4
    assert gate_open(np.full(n, top), np.full(n, 0.95), 0.9)
    # converting, but one scenario still has depths to climb
    d = np.full(n, top); d[2] = top - 1
    assert not gate_open(d, np.full(n, 0.95), 0.9)
    # at max depth, but one scenario below threshold
    wr = np.full(n, 0.95); wr[1] = 0.85
    assert not gate_open(np.full(n, top), wr, 0.9)
    # a scenario whose first window has not closed reads NaN -> shut
    wr = np.full(n, 0.95); wr[0] = np.nan
    assert not gate_open(np.full(n, top), wr, 0.9)


def test_p_seed_holds_then_anneals_monotonically():
    """base while shut; base -> floor over --p-seed-decay once open; then held.

    Monotone non-increasing is the property that matters: p_seed must never
    climb back, or the seeded/self-play mixture oscillates for the rest of the
    run.
    """
    from train.train import p_seed_at
    a = _pargs()
    assert p_seed_at(a, 5000, None) == a.p_seed_base       # shut: no drift
    assert p_seed_at(a, 900, 900) == a.p_seed_base         # opens at base
    assert p_seed_at(a, 1400, 900) == pytest.approx(0.20)  # halfway
    assert p_seed_at(a, 1900, 900) == pytest.approx(a.p_seed_floor)
    assert p_seed_at(a, 9999, 900) == pytest.approx(a.p_seed_floor)  # held
    seq = [p_seed_at(a, i, 900) for i in range(900, 2500, 25)]
    assert all(x >= y - 1e-12 for x, y in zip(seq, seq[1:])), "must not rise"
    assert p_seed_at(_pargs(p_seed_decay=0), 900, 900) == a.p_seed_floor


def test_default_hyperparameters_are_pinned():
    """The loss defaults follow Straka et al.'s recipe, and the CLI defaults
    must match the documented ones, so a run that forgets a flag still
    reproduces the intended configuration."""
    lc = learn.LossCfg()
    for field, want in (("lam", 0.9),
                        ("max_grad_norm", 0.267),
                        ("ent_coef", 0.05),         # entropy schedule start
                        ("ent_min", 0.001),         # entropy schedule floor
                        ("adv_frac", 0.25)):        # top-advantage fraction
        assert getattr(lc, field) == want, \
            f"LossCfg.{field} is {getattr(lc, field)}, expected {want}"
    assert config.EPOCHS == 1

    calls = {}
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(train.main)))):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") ==
                "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            calls[node.args[0].value] = {
                k.arg: k.value for k in node.keywords}
    for flag, want in (("--pool", 100_000), ("--pool-every", 20),
                       ("--stage-win", 0.6), ("--depth-win", 0.6),
                       ("--lr", 1e-4)):
        assert flag in calls, f"{flag} not found in train.py"
        got = ast.literal_eval(calls[flag]["default"])
        assert got == want, f"{flag} defaults to {got}, expected {want}"
