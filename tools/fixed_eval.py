"""CPU sidecar: absolute skill curve against fixed scripted opponents.

Once the curriculum reaches its final stage every live training metric is
self-referential (EMA vs EMA).  Strategy cycling looks like "mirror metrics
fine, fixed-opponent strength falling", which those metrics cannot show.  This
watches the checkpoint directory and evaluates each new checkpoint's EMA
against a scripted opponent on competition boards, on CPU, without touching the
GPU or the training process.  The resulting curve detects cycling and is the
metric for choosing which checkpoint ships.

    JAX_PLATFORMS=cpu python -u tools/fixed_eval.py --dir checkpoints --games 64
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import numpy as np

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train import arena, boards, ckpt  # noqa: E402
from train.config import NetCfg        # noqa: E402


def ema_of(path: Path):
    d = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    cfg = NetCfg(**{k: v for k, v in meta["cfg"].items()
                    if k in NetCfg.__dataclass_fields__})
    import jax.numpy as jnp
    p = {k[3:]: jnp.asarray(d[k]) for k in d.files if k.startswith("e::")}
    return p, dataclasses.replace(cfg, bf16=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="checkpoints")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=900)
    ap.add_argument("--once", default=None, help="evaluate one checkpoint, exit")
    ap.add_argument("--opponents", default="expander",
                    help="comma list: expander,hunter,rusher,harvester")
    a = ap.parse_args()

    pool = boards.make_pool(jax.random.PRNGKey(7), boards.STAGES[-1], a.games)
    opponents = a.opponents.split(",")
    runs: dict = {}
    done_cfg = None
    seen: set[str] = set()

    def evaluate(path: Path):
        nonlocal done_cfg
        p, cfg = ema_of(path)
        if not runs or cfg != done_cfg:
            runs.clear()
            for name in opponents:
                runs[name] = jax.jit(
                    arena.make_eval(cfg, arena.scripted(name)),
                    static_argnums=3)
            done_cfg = cfg
        for name in opponents:
            t0 = time.time()
            tr, final = runs[name](p, p, pool, a.ticks)
            prof = arena.profile(tr, final)
            print(f"FIXED {path.stem} vs {name}: win {prof['winrate']:.3f} "
                  f"loss {prof['lossrate']:.3f} "
                  f"unfin {prof['unfinished']:.3f} "
                  f"bld {prof['build_frac']:.4f} idle {prof['idle']:.3f} "
                  f"({time.time()-t0:.0f}s, {int(pool.armies.shape[0])} "
                  f"games)", flush=True)

    if a.once:
        evaluate(Path(a.once))
        return 0
    while True:
        cks = sorted(Path(a.dir).glob("ckpt_*.npz"))
        for c in cks:
            if c.stem not in seen:
                seen.add(c.stem)
                try:
                    evaluate(c)
                except Exception as e:               # keep watching regardless
                    print(f"FIXED {c.stem} FAILED: {type(e).__name__} {e}",
                          flush=True)
        time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
