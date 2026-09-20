"""The training loop: rollout -> Q-boost -> filtered PPO update, on device.

One iteration is a single compiled step -- jitted on one device, `shard_map`ped
across several -- so the engine, both policy forwards, the advantage scan and
the optimizer never leave the accelerator.  The host only schedules: curriculum
gates, board-pool refreshes, evaluation and checkpoints.

Progress is printed every few iterations (env-steps/s, losses, episode
outcomes, where the action mass sits, curriculum mix, VRAM) so a run is
watchable from the log rather than inferred from a checkpoint hours later.

State is fully resumable: `ckpt.py` stores params, the EMA weights, Adam
moments, RNG and the rollout carry atomically, and the curriculum state rides
in the metadata.

The opponent is the current policy (mirror self-play, both seats trained on).
`--league` swaps a fraction of fresh games to scripted opponents, and
`--selfplay ladder` plays a FIFO of frozen snapshots instead.

Usage:
    python -u -m train.train --iters 20000            # fresh
    python -u -m train.train --resume                 # continue latest
    python -u -m train.train --devices 4              # data parallel
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time
from pathlib import Path

# XLA preallocates 75% of the card by default, leaving a quarter of it unused.
# Must be set before importing jax.  0.92 leaves room for the CUDA context and
# the driver's own allocations.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import jax
import jax.numpy as jnp
import numpy as np

from jax.sharding import PartitionSpec as P

from . import arena, boards, ckpt, learn, net, parallel, wb
from . import config as C
from .config import INT_BUILD, INT_HALF0, NetCfg, gflops, n_params
from .rollout import SeedBank, init_carry, make_step

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def vram_gb() -> float:
    try:
        s = jax.local_devices()[0].memory_stats() or {}
        return s.get("bytes_in_use", 0) / 1e9
    except Exception:
        return float("nan")


def gate_open(scen_depth, seed_wr, thresh: float) -> bool:
    """Has the backward curriculum finished its job?

    Two conditions, both necessary.  Every scenario must be at the deepest
    start -- while depths remain, each advance resets that scenario's mastery
    window, so a high conversion means "mastered this depth", not "mastered the
    task", and acting on it would cut the curriculum's budget mid-ramp.  And
    every scenario's most recent window must convert at `thresh`.

    `seed_wr` is NaN for a scenario whose first window has not closed; NaN fails
    the comparison, so the gate stays shut until all four have been measured.
    """
    return bool((np.asarray(scen_depth) == C.N_DEPTH - 1).all()
                and np.all(np.asarray(seed_wr) >= thresh))


def p_seed_at(a, it: int, seed_gate_it):
    """Seeded-reset probability: base until the gate opens, then -> floor.

    Monotone non-increasing by construction, which is the property that matters:
    `seed_gate_it` is latched (and checkpointed), so a scenario dipping back
    below the threshold after the gate opens cannot push p_seed back up and
    restart the mixture oscillating.
    """
    if seed_gate_it is None:
        return a.p_seed_base
    # decay <= 0 means "step straight to the floor", not "ramp over one
    # iteration" -- the guard is against div-by-zero, not a hidden schedule.
    g = (1.0 if a.p_seed_decay <= 0
         else min(1.0, (it - seed_gate_it) / a.p_seed_decay))
    return a.p_seed_base + (a.p_seed_floor - a.p_seed_base) * g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/apex/seed_bank_with_memory.npz")
    ap.add_argument("--no-seeds", action="store_true",
                    help="pure self-play from fresh boards: no seed bank, no "
                         "backward curriculum")
    # Seed selection.  Unfiltered, half the bank is a strong player's opponent
    # losing -- a position worth -1 whatever the policy does.  Keep only
    # winning seats at or above --seed-min-elo.
    ap.add_argument("--seed-index", default="data/apex/index.parquet")
    ap.add_argument("--seed-manifest", default="data/manifest.parquet")
    ap.add_argument("--seed-min-elo", type=float, default=3000.0)
    ap.add_argument("--seed-losers", action="store_true",
                    help="also seed from losing seats (default: winners only)")
    ap.add_argument("--seed-unbalanced", action="store_true",
                    help="draw seeds in raw survivor proportions; the default "
                         "equalises the contributing players")
    ap.add_argument("--out", default=str(CKPT_DIR))
    ap.add_argument("--resume", action="store_true")
    # Architecture defaults are read from NetCfg, never restated: a literal
    # here that drifts from the dataclass trains a model no test or profiler
    # measures, and whose checkpoints they cannot load.
    _d = NetCfg()
    ap.add_argument("--dim", type=int, default=_d.dim)
    ap.add_argument("--depth", type=int, default=_d.depth)
    ap.add_argument("--heads", type=int, default=_d.heads)
    # Rollout geometry of Straka et al. (arXiv:2606.23348), Table V: 512 envs x
    # 512 ticks x 2 seats, of which the 0.25 advantage filter keeps a quarter,
    # trained in minibatches of 1024 for one epoch.
    ap.add_argument("--envs", type=int, default=C.ENVS)
    ap.add_argument("--seg", type=int, default=C.SEG,
                    help="rollout ticks/iter")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="peak of the power-law schedule")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS,
                    help="passes over the filtered batch")
    ap.add_argument("--devices", type=int, default=1,
                    help="data-parallel devices; --envs is per device")
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per gradient step; exact, cuts backward "
                         "activation memory by this factor")
    ap.add_argument("--minibatch", type=int, default=C.MINIBATCH,
                    help="minibatch size; the count is derived from the kept "
                         "batch, so this means the same thing at any --envs")
    ap.add_argument("--pool", type=int, default=100_000, help="boards per stage")
    # The board pool rotates so the policy cannot memorise map layouts over a
    # curriculum stage.  A refresh costs ~60 ms (boards.make_env is cached).
    ap.add_argument("--pool-every", type=int, default=20,
                    help="regenerate the board pool every N iterations; 0 = never")
    # The board curriculum advances on competence, not on a fixed fraction of
    # the run -- and competence is measured against a fixed scripted opponent,
    # because the self-play win rate sits near 0.5 by construction and says
    # nothing about this board difficulty.  The schedule remains as a floor: a
    # stage may lag it by at most one, since a policy that never sees the
    # competition distribution is worse than one advanced early.
    ap.add_argument("--stage-win", type=float, default=0.6,
                    help="win rate vs --stage-opponent required to advance")
    ap.add_argument("--stage-opponent", default="expander",
                    choices=("expander", "rusher", "hunter", "harvester"))
    ap.add_argument("--stage-every", type=int, default=100,
                    help="evaluate the curriculum gate every N iterations; 0 "
                         "reverts to schedule-only")
    ap.add_argument("--stage-games", type=int, default=96)
    ap.add_argument("--stage-ticks", type=int, default=300)
    ap.add_argument("--scen-weights", default="0.25,0.30,0.15,0.30",
                    help="strike_conversion,blind_commit,defense_warning,"
                         "castle_decision")
    # Backward curriculum (after Salimans & Chen, arXiv:1812.03381): the depth
    # ramp below moves each scenario's start earlier once it converts reliably.
    # Note the defender is the same improving policy, so conversion is an arms
    # race; a bar well below --stage-win can already mean mastery.
    #
    # p_seed is the orthogonal mixture knob, and it is gated rather than
    # ramped: hold at --p-seed-base until every scenario is at max depth and
    # converting at >= --p-seed-gate, then anneal to --p-seed-floor.  Ramping
    # it up instead would spend a growing share of the budget on the
    # curriculum just as it stops teaching, and move gradient out of the full
    # games -- the only episodes long enough to show a castle repaying.
    ap.add_argument("--depth-win", type=float, default=0.6,
                    help="seeded win rate that advances a scenario's start "
                         "point earlier")
    # Seed horizons are sized for the source games' tempo; a policy that
    # closes more slowly would read real conversions as timeouts.
    ap.add_argument("--seed-horizon-scale", default="1.0",
                    help="multiply seed horizons at load: one float for all "
                         "scenarios, or comma list in bank scenario order")
    ap.add_argument("--depth-games", type=int, default=400,
                    help="seeded episodes required before a depth advance")
    ap.add_argument("--p-seed-base", type=float, default=0.30,
                    help="seeded-reset probability while the backward "
                         "curriculum is still advancing")
    ap.add_argument("--p-seed-floor", type=float, default=0.10,
                    help="seeded-reset probability after the gate opens")
    ap.add_argument("--p-seed-gate", type=float, default=0.9,
                    help="per-scenario conversion rate, at max depth, that "
                         "opens the anneal to --p-seed-floor")
    ap.add_argument("--p-seed-decay", type=int, default=1000,
                    help="iterations over which p_seed falls base -> floor")
    ap.add_argument("--temp-start", type=float, default=1.0)
    ap.add_argument("--temp-end", type=float, default=0.7)
    ap.add_argument("--ent-coef", type=float, default=None)
    ap.add_argument("--adv-frac", type=float, default=None,
                    help="top-advantage fraction; 1.0 disables the filter")
    ap.add_argument("--advantage", default=None, choices=("vrpo", "gae"),
                    help="estimator; vrpo is Q-boosting, gae the baseline it "
                         "is A/B-ed against")
    ap.add_argument("--lam", type=float, default=None, help="trace decay")
    ap.add_argument("--snapshot-every", type=int, default=250)
    # The default is pure mirror self-play with no opponent pool, as in Straka
    # et al.; only --selfplay ladder needs a snapshot FIFO.
    ap.add_argument("--n-snapshots", type=int, default=0)
    # eps-build exploration (see rollout.sample_joint): the behaviour policy
    # mixes eps mass over the legal builds and PPO's ratio corrects for it.
    # Annealed linearly over --eps-build-anneal iterations (0 = constant) down
    # to --eps-build-floor.  Scaffold, then wean: the mixture guarantees early
    # castle experience, and a policy that profits from builds keeps them on
    # its own; a small floor keeps that experience from dying out entirely.
    ap.add_argument("--eps-build", type=float, default=0.0)
    ap.add_argument("--eps-build-anneal", type=int, default=0)
    ap.add_argument("--eps-build-floor", type=float, default=0.0)
    # Iteration by which the magnet (prior anchor) fades to zero.  The entropy
    # bonus is decoupled from it (learn.py) and keeps its own schedule.
    ap.add_argument("--magnet-out", type=int, default=30_000)
    ap.add_argument("--gen-coef", type=float, default=None,
                    help="weight of the enemy-general localisation auxiliary")
    # Draw pricing (see rollout.py): the competition clock-out is a draw, and
    # under a free draw mirror play learns that hoarding beats closing.  A
    # negative value prices the clock-out.
    ap.add_argument("--draw-reward", type=float, default=0.0)
    # Training-only clock: fresh games end (as a draw, priced by
    # --draw-reward) at this tick instead of DRAW_TURN.
    ap.add_argument("--train-cap", type=int, default=C.DRAW_TURN)
    # Earliness premium on wins (see rollout.py); 0 is a flat +1.
    ap.add_argument("--win-speed", type=float, default=0.0)
    # Multiplies the whole lr schedule, floor included.
    ap.add_argument("--lr-scale", type=float, default=1.0)
    # The league: this fraction of fresh envs face a scripted seat 1 (half
    # expander, half hunter) instead of the mirror; their seat-1 samples are
    # masked out of training.  Seeded envs always stay mirror.
    ap.add_argument("--league", type=float, default=0.0)
    # "mirror" plays the current policy in both seats and trains on both,
    # doubling the data per game.  It only works mirrored: PPO's ratio needs
    # logp_old from the policy that chose the action, and a lagging snapshot in
    # seat 1 is off-policy by an unbounded amount.  "ladder" plays the snapshot
    # FIFO and trains on seat 0 only.
    ap.add_argument("--selfplay", default="mirror",
                    choices=("mirror", "ladder"))
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--keep-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-games", type=int, default=192)
    ap.add_argument("--eval-ticks", type=int, default=420)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default="auto", choices=("auto", "off"),
                    help="'auto' logs when WANDB_API_KEY is present (.env or "
                         "environment) and is silent when it is not")
    ap.add_argument("--wandb-project", default="generals-zero")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--cache", default=".jax_cache",
                    help="XLA persistent cache dir; '' to disable")
    a = ap.parse_args()

    if a.cache:
        ckpt.enable_compilation_cache(a.cache)
    cfg = NetCfg(dim=a.dim, depth=a.depth, heads=a.heads)
    lc = learn.LossCfg()
    for name, val in (("ent_coef", a.ent_coef), ("gen_coef", a.gen_coef),
                      ("adv_frac", a.adv_frac), ("advantage", a.advantage),
                      ("lam", a.lam)):
        if val is not None:
            lc = lc._replace(**{name: val})
    out = Path(a.out)
    key = jax.random.PRNGKey(a.seed)

    prior = ckpt.latest(out) if a.resume else None
    prior_meta = ckpt.meta_of(prior) if prior is not None else {}
    run_name = a.run_name or prior_meta.get("run_name") or (
        f"gz-d{cfg.dim}L{cfg.depth}-{lc.advantage}-"
        f"{a.envs}x{a.seg}-{time.strftime('%m%d-%H%M')}")
    wb.init(a.wandb_project, run_name,
            {**{k: v for k, v in vars(a).items() if k != "run_name"},
             **cfg.__dict__, **lc._asdict(),
             "params": n_params(cfg), "gflops": gflops(cfg),
             "transitions_per_iter": a.envs * a.seg},
            mode=a.wandb, run_id=prior_meta.get("wandb_id"))
    print(f"devices {jax.devices()} | {cfg}", flush=True)
    print(f"loss {lc}", flush=True)
    if a.no_seeds:
        # p_seed is pinned at 0, so no reset draws from the placeholder and
        # the depth and p_seed gates, which count seeded episodes, never fire.
        bank, scen_names = SeedBank.placeholder()
        a.p_seed_base = a.p_seed_floor = 0.0
        print("no seed bank: pure self-play from fresh boards", flush=True)
    elif not Path(a.seeds).exists():
        raise SystemExit(
            f"seed bank {a.seeds} not found.  Build one with "
            f"tools/build_rl_seeds.py, or pass --no-seeds to train on pure "
            f"self-play.")
    else:
        bank, scen_names = SeedBank.load(
            a.seeds, index=a.seed_index, manifest=a.seed_manifest,
            min_elo=a.seed_min_elo, winners_only=not a.seed_losers,
            balance_players=not a.seed_unbalanced)
    hs = [float(x) for x in str(a.seed_horizon_scale).split(",")]
    if len(hs) == 1:
        hs = hs * len(scen_names)
    if len(hs) != len(scen_names):
        raise SystemExit(f"--seed-horizon-scale needs 1 or {len(scen_names)} "
                         f"values ({scen_names}), got {hs}")
    if any(x != 1.0 for x in hs):
        bank = bank._replace(scen_horizon=jnp.asarray(
            np.asarray(bank.scen_horizon) * np.asarray(hs), jnp.int32))
        print(f"seed horizons x{hs}: "
              f"{dict(zip(scen_names, np.asarray(bank.scen_horizon).tolist()))}",
              flush=True)
    n_scen = len(scen_names)
    scen_w = jnp.asarray([float(x) for x in a.scen_weights.split(",")])
    scen_w = scen_w / scen_w.sum()
    key, k = jax.random.split(key)
    stage_i = 0
    pool = boards.make_pool(k, boards.STAGES[stage_i], a.pool)
    sn = np.asarray(bank.scen_n)                   # (n_scen, n_depth)
    print(f"seed bank {bank.size:,} | eligible per scenario x depth "
          f"{dict(zip(scen_names, sn.tolist()))} "
          f"| depths {C.SEED_DEPTHS[:sn.shape[1]]} ticks before the tag "
          f"| weights {dict(zip(scen_names, np.asarray(scen_w).round(2)))}",
          flush=True)
    print(f"stage {boards.STAGES[stage_i].name} | {a.pool} boards", flush=True)

    key, k = jax.random.split(key)
    params = net.init(k, cfg)
    ema = jax.tree.map(jnp.copy, params)
    # steps_per_iter converts optax's optimizer-step count to iterations for
    # the lr schedule (see learn.power_law_schedule).
    kept0 = int(a.envs * a.devices * a.seg * 2 * lc.adv_frac)
    opt = learn.make_optimizer(a.lr, lc.max_grad_norm, a.iters,
                               steps_per_iter=max(1, kept0 // a.minibatch),
                               lr_scale=a.lr_scale)
    opt_state = opt.init(params)
    n_dev = a.devices
    axis = parallel.AXIS if n_dev > 1 else None
    # --minibatch is a size; the per-device count follows from the kept batch
    kept = int(a.envs * a.seg * 2 * lc.adv_frac)      # both seats
    n_mb = max(1, kept // a.minibatch)
    update = learn.make_update(cfg, lc, opt, a.epochs, n_mb, a.accum,
                               axis_name=axis)
    prepare = learn.make_prepare(lc)
    if a.selfplay == "ladder" and not a.n_snapshots:
        raise SystemExit("--selfplay ladder needs --n-snapshots > 0; with 0 "
                         "there is no opponent pool to draw from.")
    _, rollout_fn = make_step(cfg, a.selfplay, a.eps_build, a.draw_reward,
                              a.train_cap, a.win_speed, a.league)

    def train_step(carry, key, params, opp, opt_state, pool, bank, scen_w,
                   scen_depth, p_seed, temp, mag, eps, mag_c):
        """One iteration.  Identical for 1 and N devices -- jitted below for
        one, `shard_map`ped for many -- so the two cannot drift apart.

        Everything large is an argument; nothing is captured.  `pool` is
        reassigned on every stage advance and refresh, and a closed-over array
        is baked in at trace time, which would pin the boards to iteration 0.
        `bank` would become a multi-GB compile-time constant embedded in the
        executable on every device; as arguments both are ordinary replicated
        buffers.
        """
        # `shard_map` hands each device its shard without squeezing the sharded
        # axis, so a (n_dev, 2) key array arrives as (1, 2) and
        # jax.random.split rejects a batched key.  `ndim` is static, so this is
        # free, and a no-op on the single-device path.
        if key.ndim > 1:
            key = key[0]
        if carry.key.ndim > 1:
            carry = carry._replace(key=carry.key[0])
        carry, batch = rollout_fn(carry, params, opp, p_seed, temp, scen_w,
                                  scen_depth, pool, bank, a.seg, eps)
        flat = prepare(batch)
        # metrics are pulled off `batch` before it is released (seat 0 only)
        n_t = batch.terminal[:, :a.envs].sum()
        wins = (batch.reward[:, :a.envs] > 0).sum()
        # where the action mass sits: half-moves and builds
        half_f = ((batch.a_int >= INT_HALF0) & (batch.a_int < INT_BUILD)).mean()
        build_f = (batch.a_int == INT_BUILD).mean()
        # backward-curriculum mastery: per-scenario outcomes of seeded episodes
        # only.  n_scen is static, so the loop unrolls.
        sc, rw = batch.ep_scen[:, :a.envs], batch.reward[:, :a.envs]
        seed_n = jnp.stack([(sc == j).sum() for j in range(n_scen)])
        seed_w = jnp.stack([((sc == j) & (rw > 0)).sum() for j in range(n_scen)])
        del batch
        params, opt_state, stats = update(params, opt_state, flat, key, temp,
                                          mag, mag_coef=mag_c)
        met = (n_t, wins, half_f, build_f, seed_n, seed_w)
        if axis is not None:
            # counts sum across devices, rates mean -- otherwise the logged win
            # rate would depend on the device count
            met = (jax.lax.psum(n_t, axis), jax.lax.psum(wins, axis),
                   jax.lax.pmean(half_f, axis), jax.lax.pmean(build_f, axis),
                   jax.lax.psum(seed_n, axis), jax.lax.psum(seed_w, axis))
        # restore the leading axis so the returned carry matches out_specs
        if n_dev > 1:
            carry = carry._replace(key=carry.key[None])
        return carry, params, opt_state, stats, met

    if n_dev > 1:
        mesh = parallel.make_mesh(n_dev)
        D, R = P(parallel.AXIS), P()
        step_fn = parallel.make_step(
            mesh, train_step,
            in_specs=(D, D, R, R, R, R, R, R, R, R, R, R, R, R),
            out_specs=(D, R, R, R, (R, R, R, R, R, R)))
        print(f"data parallel: {n_dev} devices x {a.envs} envs = "
              f"{n_dev * a.envs} total", flush=True)
    else:
        mesh, step_fn = None, jax.jit(train_step)

    key, k = jax.random.split(key)
    # built for the total env count then split, so every device draws from one
    # generator rather than n_dev independently seeded ones
    carry = init_carry(k, a.envs * n_dev, pool, bank, a.p_seed_base, scen_w,
                       jnp.zeros(n_scen, jnp.int32), league_frac=a.league)
    if n_dev > 1:
        # Shard the flat (n_dev*envs, ...) carry on axis 0, so each device
        # receives (envs, ...) directly.  The PRNG key is not env-batched, so
        # it is split to (n_dev, 2) first to shard evenly.
        carry = carry._replace(key=jax.random.split(carry.key, n_dev))
        carry = parallel.shard(mesh, carry)

    start, seen = 1, 0
    roll_win = 0.5          # running seat-0 wins / terminals
    gate_wr = float("nan")  # last curriculum-gate result vs the fixed opponent
    # p_seed gate state.  `seed_wr` is the last measured conversion per
    # scenario (NaN until a first window closes); `seed_gate_it` is the
    # iteration the gate opened, or None.  Both persist across a resume --
    # re-opening a gate that already opened would re-raise p_seed.
    seed_wr = np.full(n_scen, np.nan)
    seed_gate_it = None
    # Backward curriculum.  Every scenario starts at its tagged moment (depth
    # 0), where the opportunity already exists, and moves its start earlier
    # once it converts reliably -- so the task grows from "convert what you are
    # handed" to "create it, then convert it".  Advanced per scenario on a
    # mastery criterion, never on a schedule.
    scen_depth = np.zeros(n_scen, np.int32)
    seed_n_c = np.zeros(n_scen)     # seeded episodes since the last advance
    seed_w_c = np.zeros(n_scen)     # ...of which won
    snapshots = [jax.tree.map(jnp.copy, params)] if a.n_snapshots else []
    latest = prior
    if latest is not None:
        fresh_carry = carry
        params, ema, snapshots, opt_state, carry, key, meta = ckpt.load(
            latest, params_like=params, opt_state_like=opt_state,
            carry_like=carry)
        # The carry is sized by total envs (devices x --envs), and ckpt.load
        # validates leaf count, not shapes.  It is only in-flight game state,
        # so on a device-count change it is rebuilt rather than failing;
        # everything that matters (params, EMA, snapshots, Adam state, stage,
        # depths) is device-count independent and restores unchanged.
        want, got = fresh_carry.horizon.shape[0], carry.horizon.shape[0]
        if want != got:
            print(f"--- carry rebuilt: checkpoint has {got:,} envs, this run "
                  f"has {want:,} ({n_dev} devices x {a.envs}).  In-flight "
                  f"episodes are discarded; params, Adam state, EMA, "
                  f"snapshots, stage and depths all carried over.", flush=True)
            carry = fresh_carry
        elif n_dev > 1:
            carry = carry._replace(key=jax.random.split(key, n_dev))
            carry = parallel.shard(mesh, carry)
        start, seen = meta["iter"] + 1, meta["steps"]
        # The curriculum is competence-gated and may deliberately lag the
        # schedule, so the stage is restored, not recomputed from `frac` --
        # recomputing would silently undo a "not ready yet" decision.  Same for
        # the readiness EMA.
        stage_i = meta.get("stage", boards.stage_for(start / a.iters))
        roll_win = meta.get("roll_win", 0.5)
        seed_gate_it = meta.get("seed_gate_it")  # or a resume re-raises it
        if meta.get("seed_wr"):
            seed_wr = np.asarray(meta["seed_wr"], np.float64)
            # the bank can grow scenarios between resumes; new ones start
            # unmastered at depth 0 with no window
            if len(seed_wr) < n_scen:
                seed_wr = np.pad(seed_wr, (0, n_scen - len(seed_wr)),
                                 constant_values=np.nan)
        if meta.get("scen_depth"):            # ...or restart the curriculum
            scen_depth = np.asarray(meta["scen_depth"], np.int32)
            if len(scen_depth) < n_scen:
                scen_depth = np.pad(scen_depth,
                                    (0, n_scen - len(scen_depth)))
        key, k = jax.random.split(key)
        pool = boards.make_pool(k, boards.STAGES[stage_i], a.pool)
        print(f"resumed {latest.name} at iter {meta['iter']} "
              f"({seen / 1e6:.1f}M env-steps) stage {boards.STAGES[stage_i].name}"
              f" win~{roll_win:.2f}", flush=True)
    elif a.resume:
        print("--resume given but no checkpoint found; starting fresh",
              flush=True)

    # mirror self-play yields a transition per seat per tick
    n_tr = a.envs * n_dev * a.seg * (2 if a.selfplay == "mirror" else 1)
    print(f"params {n_params(cfg):,} ({n_params(cfg, True):,} shipped, "
          f"{gflops(cfg):.2f} GFLOP) | {a.envs} envs x {a.seg} ticks = "
          f"{n_tr:,} transitions/iter, training on {int(n_tr * lc.adv_frac):,} "
          f"| disk free {ckpt.disk_free_gb(out):.1f} GB", flush=True)

    # Evaluation runs greedy at deployment precision on the EMA weights:
    # training uses bf16 for throughput, but the shipped bot is an f32 argmax
    # on one CPU core, and the EMA is what ships.
    eval_cfg = dataclasses.replace(cfg, bf16=False)
    eval_run = jax.jit(arena.make_eval(eval_cfg), static_argnums=3)
    # Only compiled once: every stage pads to the same 21x21, so the same
    # executable serves all of them.
    gate_run = jax.jit(arena.make_eval(eval_cfg,
                                       arena.scripted(a.stage_opponent)),
                       static_argnums=3)
    key, k = jax.random.split(key)
    eval_pool = boards.make_pool(k, boards.STAGES[-1], a.eval_games)

    # Pipelined metric drain.  The host tail -- ~20 device->host scalar
    # transfers, wandb, mastery windows -- is a fixed ~0.6 s that would
    # otherwise sit between the device finishing iteration N and the host
    # launching N+1.  Instead N+1 is launched first and N's metrics drain while
    # the device works.  Control state (mastery, depth advances, p_seed gate,
    # roll_win) therefore lags the launch by exactly one iteration, which none
    # of those mechanisms can perceive.  At most two iterations are in flight:
    # draining N syncs on N's outputs before N+2 launches.
    pending = None

    def drain(rec, dt):
        nonlocal roll_win, seed_gate_it, seed_n_c, seed_w_c
        it_r, stats, met = rec["it"], rec["stats"], rec["met"]
        n_t, wins_b, half_b, build_b = (float(x) for x in met[:4])
        seed_n_c += np.asarray(met[4]); seed_w_c += np.asarray(met[5])
        if n_t > 0:
            roll_win += 0.04 * (wins_b / n_t - roll_win)

        # advance any scenario that has mastered its current depth; the window
        # resets on every evaluation
        for j in range(n_scen):
            if seed_n_c[j] < a.depth_games:
                continue
            wr = seed_w_c[j] / seed_n_c[j]
            if wr >= a.depth_win and scen_depth[j] + 1 < C.N_DEPTH:
                scen_depth[j] += 1
                print(f"--- DEPTH {scen_names[j]} -> "
                      f"{C.SEED_DEPTHS[scen_depth[j]]} ticks before the tag "
                      f"at iter {it_r} (converted {wr:.2f} of "
                      f"{int(seed_n_c[j])} within the horizon at depth "
                      f"{C.SEED_DEPTHS[scen_depth[j] - 1]})", flush=True)
                wb.log({f"depth/{scen_names[j]}": int(scen_depth[j])}, step=it_r)
            seed_wr[j] = wr
            seed_n_c[j] = seed_w_c[j] = 0.0

        # p_seed gate (see the comment at the argparse flag)
        if seed_gate_it is None and gate_open(scen_depth, seed_wr,
                                              a.p_seed_gate):
            seed_gate_it = it_r
            print(f"--- p_seed GATE OPEN at iter {it_r}: all scenarios at max "
                  f"depth converting {np.array2string(seed_wr, precision=2)} "
                  f">= {a.p_seed_gate}; annealing p_seed {a.p_seed_base:.2f} -> "
                  f"{a.p_seed_floor:.2f} over {a.p_seed_decay} iters",
                  flush=True)

        wb.log({"perf/steps_per_s": n_tr / dt, "perf/env_steps": rec["seen"],
                "perf/vram_gb": vram_gb(),
                "loss/pg": float(stats["pg"]), "loss/q": float(stats["v"]),
                "loss/q_explained_var": float(stats["q_ev"]),
                "loss/entropy": float(stats["ent"]),
                "loss/kl_ref": float(stats["kl_ref"]),
                "loss/kl_behaviour": float(stats["kl"]),
                "loss/clipfrac": float(stats["clipfrac"]),
                "loss/adv_std": float(stats["adv_std"]),
                "aux/general": float(stats["gen"]),
                "aux/danger": float(stats["danger"]),
                "curriculum/stage": stage_i,
                "curriculum/selfplay_win_rate": roll_win,
                "episode/decisive_rate": min(1.0, 2.0 * roll_win)
                                         if a.selfplay == "mirror" else float("nan"),
                "curriculum/p_seed": rec["p_seed"], "curriculum/temp": rec["temp"],
                **{f"depth/{scen_names[j]}_depth": int(scen_depth[j])
                   for j in range(n_scen)},
                **{f"depth/{scen_names[j]}_seen": float(seed_n_c[j])
                   for j in range(n_scen)},
                **{f"depth/{scen_names[j]}_wr":
                   float(seed_w_c[j] / max(seed_n_c[j], 1.0))
                   for j in range(n_scen)},
                "curriculum/ent_coef": learn.entropy_coef(lc, it_r),
                "action/half_frac": half_b, "action/build_frac": build_b,
                "episode/terminals": n_t,
                "episode/win_rate": wins_b / max(n_t, 1.0)}, step=it_r)

        if it_r % a.log_every == 0 or it_r == start:
            print(f"[{it_r:6d}/{a.iters}] {n_tr / dt:7.0f} steps/s "
                  f"| {rec['seen'] / 1e6:7.2f}M "
                  f"| pg {float(stats['pg']):+.4f} q {float(stats['v']):.4f} "
                  f"qev {float(stats['q_ev']):+.2f} "
                  f"ent {float(stats['ent']):.3f} mkl {float(stats['kl_ref']):.3f} "
                  f"kl {float(stats['kl']):+.4f} clip {float(stats['clipfrac']):.3f} "
                  f"| gen {float(stats['gen']):.3f} dgr {float(stats['danger']):.3f} "
                  f"| half {half_b:.3f} bld {build_b:.3f} "
                  f"| term {n_t:4.0f} win {wins_b / max(n_t, 1):.2f} "
                  f"| {boards.STAGES[stage_i].name} "
                  f"gate{gate_wr:.2f} dec{min(1.0, 2 * roll_win):.2f} "
                  f"dep{''.join(str(int(x)) for x in scen_depth)}"
                  f"/{seed_w_c.sum() / max(seed_n_c.sum(), 1.0):.2f}"
                  f"x{int(seed_n_c.sum())} "
                  f"ent_c {learn.entropy_coef(lc, it_r):.3f} "
                  f"seed {rec['p_seed']:.2f} T {rec['temp']:.2f} "
                  f"| vram {vram_gb():.1f}G "
                  f"| eta {(a.iters - it_r) * dt / 3600:.1f}h", flush=True)

    t_start = time.time()
    for it in range(start, a.iters + 1):
        frac = it / a.iters
        # p_seed: hold at base until the gate opens, then anneal to the floor.
        # It is a probability per reset, not per transition -- seeded episodes
        # truncate at their scenario horizon (~94 ticks on average) against a
        # fresh episode's ~300, so a base of 0.30 is only ~12% of transitions.
        # Once the curriculum is mastered the budget goes back to self-play: a
        # seeded fragment never teaches the opening, the seed set is finite and
        # can be memorised, and its positions drift away from the ones the
        # policy's own play reaches.
        p_seed = p_seed_at(a, it, seed_gate_it)
        temp = a.temp_start + (a.temp_end - a.temp_start) * frac

        want = boards.stage_for(frac)
        # The gate is evaluated on a cadence, not only when the schedule
        # proposes: the schedule is a fraction of --iters, and a policy that
        # masters the small boards early should not keep training on them.
        # Advancement is mastery-driven, with the schedule as a catch-up floor.
        due = (a.stage_every and it % a.stage_every == 0) or want > stage_i
        if due and stage_i + 1 < len(boards.STAGES):
            # Measure competence on this stage's boards against a fixed
            # opponent, on the EMA weights (what would ship).
            gate_st = jax.tree.map(lambda x: x[:a.stage_games], pool)
            gate_wr = float(arena.profile(
                gate_run(ema, ema, gate_st, a.stage_ticks)[0])["winrate"])
            # ...and never lag the schedule by more than one stage
            if gate_wr >= a.stage_win or want > stage_i + 1:
                stage_i += 1
                key, k = jax.random.split(key)
                pool = boards.make_pool(k, boards.STAGES[stage_i], a.pool)
                print(f"--- stage -> {boards.STAGES[stage_i].name} at iter {it}"
                      f" (vs {a.stage_opponent} {gate_wr:.2f}, self {roll_win:.2f}"
                      f", scheduled {want})", flush=True)
            else:
                print(f"--- stage held at {boards.STAGES[stage_i].name} iter {it}"
                      f" (vs {a.stage_opponent} {gate_wr:.2f} < {a.stage_win})",
                      flush=True)
            wb.log({"curriculum/gate_winrate": gate_wr}, step=it)
        elif a.pool_every and it % a.pool_every == 0:
            key, k = jax.random.split(key)
            pool = boards.make_pool(k, boards.STAGES[stage_i], a.pool)

        # opponent: the current policy, or one of the FIFO snapshots.  Keeping
        # the mirror in the mix matters: a short FIFO alone is prone to
        # self-play cycling, since it never averages over older history the way
        # fictitious play does.
        key, ko, ku = jax.random.split(key, 3)
        if n_dev > 1:                      # one key per device, sharded on `d`
            ku = parallel.shard(mesh, jax.random.split(ku, n_dev))
        if a.selfplay == "mirror":
            opp = params            # unused by the mirror path, kept for shape
        else:
            j = int(jax.random.randint(ko, (), 0, len(snapshots) + 1))
            opp = params if j == len(snapshots) else snapshots[j]

        t_launch = time.time()
        # One step: rollout -> advantages/filter -> update.  Metrics are not
        # read here -- the next iteration launches first and `drain` processes
        # this one's while the device works (see the comment at `pending`).
        carry, params, opt_state, stats, met = step_fn(
            carry, ku, params, opp, opt_state, pool, bank, scen_w,
            jnp.asarray(scen_depth),
            jnp.float32(p_seed), jnp.float32(temp),
            jnp.float32(learn.entropy_coef(lc, it)),
            jnp.float32(a.eps_build if not a.eps_build_anneal else
                        max(a.eps_build_floor, a.eps_build *
                            (1.0 - it / a.eps_build_anneal))),
            # magnet coefficient: entropy schedule x linear fade to zero by
            # --magnet-out
            jnp.float32(learn.entropy_coef(lc, it)
                        * max(0.0, 1.0 - it / a.magnet_out)))
        ema = learn.ema_update(ema, params, lc.ema_decay)
        seen += n_tr
        rec = {"it": it, "stats": stats, "met": met, "p_seed": p_seed,
               "temp": temp, "seen": seen, "t": t_launch}
        if pending is not None:
            drain(pending, t_launch - pending["t"])
        pending = rec

        if a.eval_every and (it % a.eval_every == 0 or it == a.iters):
            tr, final = eval_run(ema, ema, eval_pool, a.eval_ticks)
            prof = arena.profile(tr, final)
            sdist, naxes = arena.style_distance(prof)
            wb.log({f"eval/{k}": v for k, v in prof.items()
                    if isinstance(v, float) and v == v}
                   | {"eval/style_dist": sdist}, step=it)
            print(f"EVAL it={it} steps={seen/1e6:.1f}M "
                  f"style_dist={sdist:.3f}/{naxes}ax "
                  f"win={prof['winrate']:.3f} loss={prof['lossrate']:.3f} "
                  f"unfin={prof['unfinished']:.3f} len={prof['len_win']:.0f} "
                  f"lenDec={prof['len_decided']:.0f} "
                  f"by300={prof['decided_300']:.2f} "
                  f"seen={prof['seen_frac']:.2f}@{prof['t_seen']:.0f} "
                  f"ttk={prof['t_after_seen']:.0f} "
                  f"top1peak={prof['top1_peak']:.3f} "
                  # top1_peak is a share, so it is printed with its
                  # denominators: one stack is a large fraction of a small army
                  f"land100={prof['land100']:.0f} army100={prof['army100']:.0f} "
                  f"spear_commit={prof['spear_commit']:.1f} "
                  f"idle={prof['idle']:.3f} "
                  f"bld={prof['build_frac']:.4f} half={prof['half_frac']:.3f} "
                  f"march={prof['g3_marches']:.3f}", flush=True)

        if (it % a.ckpt_every == 0 or it == a.iters) and pending is not None:
            # metrics for this iteration must be drained before its checkpoint
            # meta is written, or seed_wr/scen_depth lag the saved weights.
            drain(pending, time.time() - pending["t"])
            pending = None
        if it % a.ckpt_every == 0 or it == a.iters:
            ckpt.save(out / f"{ckpt.STEM}{it:07d}.npz", params=params, ema=ema,
                      snapshots=snapshots, opt_state=opt_state,
                      carry=carry, key=key,
                      meta={"iter": it, "steps": seen, "cfg": cfg.__dict__,
                            "lr": a.lr, "envs": a.envs, "seg": a.seg,
                            # loop state the curriculum and the chart depend on
                            "stage": stage_i, "roll_win": roll_win,
                            "seed_gate_it": seed_gate_it,
                            "seed_wr": seed_wr.tolist(),
                            "scen_depth": scen_depth.tolist(),
                            "run_name": run_name, "wandb_id": wb.run_id()})
            ckpt.prune(out, a.keep_last, a.keep_every)
            if ckpt.disk_free_gb(out) < 2.0:
                print("ABORT: disk below 2 GB free", flush=True)
                return 1

    print(f"done: {seen / 1e6:.1f}M env-steps in "
          f"{(time.time() - t_start) / 3600:.2f}h", flush=True)
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
