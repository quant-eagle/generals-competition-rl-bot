"""Full per-turn latency: parse -> encode -> forward -> decode, on one core.

The 150 ms move budget covers the whole path, not the network alone: the numpy
observation encode and the legal mask ship too.  This unpacks a packaged
submission zip and times exactly what the bot runs per turn.

    taskset -c 0 python -u tools/latency_full.py --zip dist/submission.zip
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

BUDGET_MS = 150.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="dist/submission.zip")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--size", type=int, default=21)
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(a.zip) as z:
        z.extractall(tmp)
    sys.path.insert(0, str(tmp))

    import torch
    torch.set_num_threads(1)
    import config, net_torch, np_exec, np_obs   # noqa: E401

    d = np.load(tmp / "weights.npz")
    cfg = config.NetCfg(bf16=False, **{k[5:]: int(d[k]) for k in d.files
                                       if k.startswith("cfg::")})
    p = net_torch.prepare({k: torch.from_numpy(d[k].astype(np.float32))
                           for k in d.files if not k.startswith("cfg::")})

    n = a.size
    rng = np.random.default_rng(0)
    ty = rng.integers(0, 6, (n, n)).astype(np.int32)
    ow = rng.integers(0, 3, (n, n)).astype(np.int32)
    am = rng.integers(0, 40, (n, n)).astype(np.int32)
    mem, rings = np_obs.Memory(), np_obs.Rings()

    ts = []
    with torch.inference_mode():
        for i in range(a.warmup + a.reps):
            t0 = time.perf_counter()
            f = np_obs.frame_to_obs(i, 60, 200, 55, 190, ty, ow, am)
            mem.update(f)
            rings.push(f)
            x = torch.from_numpy(np_obs.encode(mem, f))[None]
            z = torch.from_numpy(rings.series())[None]
            lg = np_obs.legal_mask(f)
            a_s, a_i = net_torch.act(p, x, z, torch.from_numpy(lg), cfg)
            np_exec.execute(a_s, a_i, f)
            dt = (time.perf_counter() - t0) * 1e3
            if i >= a.warmup:
                ts.append(dt)
    ts.sort()
    med = statistics.median(ts)
    p95 = ts[int(0.95 * (len(ts) - 1))]
    print(f"board {n}x{n} | threads={torch.get_num_threads()} | "
          f"{a.reps} turns", flush=True)
    print(f"full turn: median {med:.1f} ms | p95 {p95:.1f} ms | "
          f"max {ts[-1]:.1f} ms  (budget {BUDGET_MS:.0f} ms)", flush=True)
    ok = p95 <= BUDGET_MS * 0.75
    print(f"{'PASS' if ok else 'TIGHT/FAIL'} -- p95 is "
          f"{p95 / BUDGET_MS:.0%} of the hard budget", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
