"""Where a training iteration spends its time.

Throughput is the binding constraint: the simulator runs far faster than the
network, so every lever is about network forwards per game, not env steps.  This
splits one iteration into its stages and reports the share of each, so
optimisation targets the measured bottleneck rather than the assumed one.

Each stage is timed with an explicit `block_until_ready` on its entire output,
because JAX is asynchronous: blocking on one leaf of a multi-output scan lets
the rest stay in flight and charges its cost to the next stage.

    python -u tools/profile_iter.py --seeds data/apex/seed_bank_with_memory.npz
    python -u tools/profile_iter.py --envs 512,1024,2048 --reps 6
"""
from __future__ import annotations

import os
# must precede `import jax`; matches train/train.py, so the memory ceiling this
# profiler measures is the one training gets
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import boards, ckpt, learn, net                  # noqa: E402
from train import config as NetCfgDefaults                  # noqa: E402
from train.config import NetCfg, gflops                     # noqa: E402
from train.rollout import SeedBank, init_carry, make_step   # noqa: E402


def block(x):
    jax.block_until_ready(x)
    return x


def profile(cfg, lc, bank, scen_w, envs, seg, epochs, n_mb, reps, warmup,
            accum=1):
    key = jax.random.PRNGKey(0)
    key, k = jax.random.split(key)
    pool = boards.make_pool(k, boards.STAGES[-1], 2048)
    key, k = jax.random.split(key)
    params = net.init(k, cfg)
    ref = jax.tree.map(jnp.copy, params)
    opt = learn.make_optimizer(3e-4, lc.max_grad_norm, 1000)
    opt_state = opt.init(params)
    update = learn.make_update(cfg, lc, opt, epochs, n_mb, accum)
    prepare = learn.make_prepare(lc)
    _, roll = make_step(cfg)
    roll = jax.jit(roll, static_argnums=9)
    key, k = jax.random.split(key)
    depth0 = jnp.zeros(scen_w.shape[0], jnp.int32)
    carry = init_carry(k, envs, pool, bank, 0.3, scen_w, depth0)

    acc = {"rollout": 0.0, "advantage": 0.0, "reference": 0.0, "update": 0.0}
    n_tr = 0        # captured in-loop: `flat` is released before the loop ends
    for i in range(warmup + reps):
        key, ku = jax.random.split(key)
        t0 = time.perf_counter()
        carry, batch = roll(carry, params, params, 0.3, 1.0, scen_w, depth0,
                            pool, bank, seg)
        block((carry, batch))
        t1 = time.perf_counter()

        flat = block(prepare(batch))
        n_tr = int(flat["adv"].shape[0] / lc.adv_frac)
        # Release the trajectory buffer before the update, exactly as
        # train/train.py does.  Held through the update it adds several GB that
        # training never carries, and the profiler would report an OOM ceiling
        # stricter than the real one.
        del batch
        t2 = time.perf_counter()

        t3 = time.perf_counter()

        params, opt_state, stats = update(params, opt_state, flat, ku, 1.0)
        block(params)
        t4 = time.perf_counter()
        del flat

        if i >= warmup:
            acc["rollout"] += t1 - t0
            acc["advantage"] += t2 - t1
            acc["reference"] += t3 - t2
            acc["update"] += t4 - t3
    total = sum(acc.values())
    # env-steps (game ticks) and transitions (training samples) diverge under
    # mirror self-play: both seats of a tick are two samples from one tick.
    return {k: v / reps for k, v in acc.items()}, total / reps, envs * seg, n_tr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/apex/seed_bank_with_memory.npz")
    ap.add_argument("--envs", default=str(NetCfgDefaults.ENVS))
    ap.add_argument("--seg", type=int, default=NetCfgDefaults.SEG)
    ap.add_argument("--epochs", type=int, default=NetCfgDefaults.EPOCHS)
    ap.add_argument("--minibatch", type=int, default=NetCfgDefaults.MINIBATCH)
    # Defaults come from NetCfg, never restated here, so a config change cannot
    # leave this tool profiling a different architecture than the trainer uses.
    _d = NetCfg()
    ap.add_argument("--dim", type=int, default=_d.dim)
    ap.add_argument("--depth", type=int, default=_d.depth)
    ap.add_argument("--heads", type=int, default=_d.heads)
    ap.add_argument("--mlp-mult", type=float, default=_d.mlp_mult)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--adv-frac", type=float, default=0.25)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cache", default=".jax_cache",
                    help="persistent XLA compilation cache ('' disables)")
    a = ap.parse_args()

    if a.cache:
        ckpt.enable_compilation_cache(a.cache)
    cfg = NetCfg(dim=a.dim, depth=a.depth, heads=a.heads,
                 mlp_mult=a.mlp_mult)
    lc = learn.LossCfg(adv_frac=a.adv_frac)
    bank, _ = SeedBank.load(a.seeds)
    scen_w = jnp.full(bank.scen_n.shape[0], 1.0 / bank.scen_n.shape[0])
    print(f"{cfg}\n{gflops(cfg):.2f} GFLOP/forward | adv_frac {lc.adv_frac} | "
          f"{a.epochs} epochs x {a.minibatch} minibatches", flush=True)

    for envs in [int(x) for x in a.envs.split(",")]:
        acc, total, n, tr = profile(cfg, lc, bank, scen_w, envs, a.seg,
                                    a.epochs, a.minibatch, a.reps, a.warmup, a.accum)
        print(f"\n--- {envs} envs x {a.seg} ticks = {n:,} env-steps, "
              f"{tr:,} transitions/iter", flush=True)
        for k, v in acc.items():
            print(f"  {k:>10} {v * 1e3:8.1f} ms {100 * v / total:5.1f}%",
                  flush=True)
        print(f"  {'TOTAL':>10} {total * 1e3:8.1f} ms  -> "
              f"{n / total:,.0f} env-steps/s, {tr / total:,.0f} transitions/s",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
