"""Evaluate a checkpoint: play-style profile in the mirror, plus a scripted gauntlet.

Greedy throughout, on the final curriculum stage (the competition board
distribution), on the EMA weights.  The EMA is the policy that ships (Straka et
al., arXiv:2606.23348, report ~+30 Elo for it), so evaluating the raw params
would grade a policy that is never submitted.

Pass/fail lines:
  rusher  win rate > 70% vs the scripted Rusher
  top1    top1_peak above the untrained control: does the policy gather army?

    python -u tools/evaluate.py --ckpt checkpoints/ckpt_0001000.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import arena, boards, net                      # noqa: E402
from train.config import NetCfg                            # noqa: E402


UNTRAINED_TOP1 = 0.225      # top1_peak of a randomly initialised policy


def load_params(path: Path, which: str = "ema"):
    """`which` selects the EMA weights (what ships) or the raw params."""
    d = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    cfg = NetCfg(**{k: v for k, v in meta["cfg"].items()
                    if k in NetCfg.__dataclass_fields__})
    tag = {"ema": "e::", "raw": "p::"}[which]
    params = {k[3:]: jnp.asarray(d[k]) for k in d.files if k.startswith(tag)}
    if not params:
        raise SystemExit(f"{path.name} carries no '{tag}' weights")
    return params, cfg, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=520)
    ap.add_argument("--opponents",
                    default="self,rusher,expander,hunter,harvester")
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    params, cfg, meta = load_params(Path(a.ckpt), a.weights)
    # evaluate at deployment precision, not training precision
    cfg = dataclasses.replace(cfg, bf16=False)
    print(f"{a.ckpt} | iter {meta.get('iter')} | "
          f"{meta.get('steps', 0)/1e6:.1f}M env-steps | {a.weights} weights | "
          f"{cfg}", flush=True)

    pool = boards.make_pool(jax.random.PRNGKey(a.seed), boards.STAGES[-1],
                            a.games)
    state = jax.tree.map(lambda x: x[:a.games], pool)

    for opp in a.opponents.split(","):
        run = jax.jit(arena.make_eval(cfg, None if opp == "self"
                                      else arena.scripted(opp)),
                      static_argnums=3)
        tr, final = run(params, params, state, a.ticks)
        prof = arena.profile(tr, final)
        print(f"\n===== vs {opp} ({a.games} games, greedy) =====", flush=True)
        if opp == "self":
            print(arena.report(prof), flush=True)
            t1 = prof["top1_peak"]
            print(f"  top1 {'PASS' if t1 > UNTRAINED_TOP1 else 'FAIL'} "
                  f"({t1:.3f} vs untrained control {UNTRAINED_TOP1}, "
                  f"top ladder bots 0.333)", flush=True)
        else:
            print(f"  win {prof['winrate']:.3f}  draw {prof['drawrate']:.3f}  "
                  f"median len {prof['len_win']:.0f}", flush=True)
            if opp == "rusher":
                print(f"  rusher {'PASS' if prof['winrate'] > 0.70 else 'FAIL'} "
                      f"(bar: >70% vs Rusher)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
