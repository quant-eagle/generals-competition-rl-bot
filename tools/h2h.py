"""Head-to-head arena between two checkpoints (EMA vs EMA).

Mirror training's live win rate sits at 0.5 by construction, and scripted
opponents saturate once every game is a win or a draw, so neither can say
whether iteration X -> Y improved the policy against itself.  This can: greedy
net-vs-net on competition boards, both seat orders on the same boards so map
luck cancels.

    JAX_PLATFORMS=cpu python -u tools/h2h.py A.npz B.npz --games 64 --ticks 900
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train import arena, boards                       # noqa: E402
from tools.fixed_eval import ema_of                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=900)
    a = ap.parse_args()

    pa, cfg = ema_of(Path(a.a))
    pb, cfg_b = ema_of(Path(a.b))
    if cfg != cfg_b:
        raise SystemExit(f"config mismatch: {cfg} vs {cfg_b}")
    pool = boards.make_pool(jax.random.PRNGKey(7), boards.STAGES[-1], a.games)
    run = jax.jit(arena.make_eval(cfg), static_argnums=3)

    t0 = time.time()
    w = l = 0.0
    for p0, p1, sgn in ((pa, pb, +1), (pb, pa, -1)):   # both seat orders
        prof = arena.profile(*run(p0, p1, pool, a.ticks))
        if sgn > 0:
            w += prof["winrate"]; l += prof["lossrate"]
        else:
            w += prof["lossrate"]; l += prof["winrate"]
    n = 2 * a.games
    aw, al = w / 2, l / 2
    print(f"H2H {Path(a.a).stem} vs {Path(a.b).stem}: "
          f"win {aw:.3f} loss {al:.3f} draw {1 - aw - al:.3f} "
          f"({n} games both orders, {a.ticks} ticks, {time.time()-t0:.0f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
