"""Deployment latency of the network forward alone, on one CPU core.

The full turn (parse -> encode -> forward -> decode) is measured off the
packaged zip by `tools/latency_full.py`.  The forward dominates that budget, so
this is the quick check that decides width and depth before anything is built
on top of them.

    taskset -c 0 python -u tools/latency_trunk.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.config import (N_PLANES, N_SERIES, N_TSAMP, B,   # noqa: E402
                          NetCfg, gflops, n_params)
from train import net_torch                                # noqa: E402

BUDGET_MS = 150.0

# Capacity vs throughput at a similar FLOP budget: the smaller rows trade depth
# for width, and width is what decides how well the matmuls fill the hardware.
VARIANTS = {
    "default D448 L7": NetCfg(),
    "D448 L5":      NetCfg(dim=448, depth=5, heads=7),
    "D512 L4":      NetCfg(dim=512, depth=4, heads=8),
    "D384 L7":      NetCfg(dim=384, depth=7, heads=6),
    "D320 L10":    NetCfg(dim=320, depth=10, heads=5),
}


def bench(cfg: NetCfg, reps: int, warmup: int) -> tuple[float, float]:
    p = net_torch.prepare(net_torch.random_params(cfg))
    x = torch.randn(1, N_PLANES, B, B)
    z = torch.zeros(1, N_SERIES, N_TSAMP)
    with torch.inference_mode():
        for _ in range(warmup):
            net_torch.forward(p, x, z, cfg)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            net_torch.forward(p, x, z, cfg)
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(1)

    print(f"{'variant':>15} {'params':>9} {'fp16 MB':>8} {'GFLOP':>7} "
          f"{'med ms':>8} {'p95 ms':>8} {'budget':>8}")
    for name, cfg in VARIANTS.items():
        med, p95 = bench(cfg, a.reps, a.warmup)
        mb = n_params(cfg, deploy_only=True) * 2 / 1e6
        print(f"{name:>15} {n_params(cfg)/1e6:8.2f}M {mb:8.1f} "
              f"{gflops(cfg):7.2f} {med:8.1f} {p95:8.1f} "
              f"{100*p95/BUDGET_MS:7.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
