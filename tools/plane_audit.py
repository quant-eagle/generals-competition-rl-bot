"""Which observation planes does a trained policy actually use?

Two independent probes on a checkpoint's EMA, over states from its own greedy
play:

  1. Zero-ablation: zero one plane everywhere and count how often the greedy
     (cell, intent) argmax changes.  Zeroing is off-manifold, so a high number
     proves use while a low number only suggests the plane is removable.
  2. Saliency: mean |d logit_chosen / d plane * plane| -- how hard the trunk
     reads each plane on-manifold.

The temporal series (the two ring-buffer tokens) is ablated as a final row.

    JAX_PLATFORMS=cpu python -u tools/plane_audit.py --ckpt <ckpt.npz>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp
import numpy as np

from train import boards, net, obs as O
from train.config import N_INT, N_PLANES, PLANES, NetCfg
from train.rollout import cstep_v, get_obs_v
from train import exec as X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--every", type=int, default=12)
    a = ap.parse_args()

    d = np.load(a.ckpt)
    meta = json.loads(Path(a.ckpt).with_suffix(".json").read_text())
    cfg = dataclasses.replace(
        NetCfg(**{k: v for k, v in meta["cfg"].items()
                  if k in NetCfg.__dataclass_fields__}), bf16=False)
    p = {k[3:]: jnp.asarray(d[k]) for k in d.files if k.startswith("e::")}
    print(f"{a.ckpt} iter {meta['iter']} | collecting states from greedy play",
          flush=True)

    st = boards.make_pool(jax.random.PRNGKey(3), boards.STAGES[-1], a.envs)
    n = int(st.armies.shape[0])
    mem = jax.vmap(lambda _: O.init_memory())(jnp.arange(n))
    rings = jax.vmap(lambda _: O.init_rings())(jnp.arange(n))
    logits_fn = jax.jit(lambda *args: net.act_logits_v(*args, cfg))

    xs, zs, lgs = [], [], []
    for t in range(a.ticks):
        o0 = get_obs_v(st, 0)
        mem = O.update_v(mem, o0)
        rings = O.push_v(rings, o0)
        x = O.encode_v(mem, o0)
        z = O.series_of_v(rings)
        lg = O.legal_mask_v(o0)
        if t % a.every == 0 and t > 60:       # mid/late game: enemies met,
                                              # castles buildable
            xs.append(np.asarray(x)); zs.append(np.asarray(z))
            lgs.append(np.asarray(lg))
        fl = logits_fn(p, x, z, lg).reshape(n, -1)
        aj = jnp.argmax(fl, -1)
        prim = X.execute_v(aj // N_INT, aj % N_INT, o0)
        st, _ = cstep_v(st, jnp.stack([prim, prim], 1))
    x = jnp.asarray(np.concatenate(xs)); z = jnp.asarray(np.concatenate(zs))
    lg = jnp.asarray(np.concatenate(lgs))
    m = x.shape[0]
    print(f"{m} states collected", flush=True)

    base = jnp.argmax(logits_fn(p, x, z, lg).reshape(m, -1), -1)

    # saliency: d(chosen logit)/d(planes), one backward over the batch
    def chosen_logit(xx, zz, ll, idx):
        return net.act_logits(p, xx, zz, ll, cfg).reshape(-1)[idx]
    g = jax.vmap(jax.grad(chosen_logit), in_axes=(0, 0, 0, 0))(x, z, lg, base)
    sal = np.asarray(jnp.abs(g * x).sum((2, 3)).mean(0))       # (N_PLANES,)

    # A plane that is (near-)zero in every collected state cannot be audited:
    # zeroing it is a no-op and grad*input vanishes by construction, so flag it
    # instead of printing a misleading 0.0%.
    live = np.asarray(jnp.abs(x).mean((0, 2, 3)))
    rows = []
    for i in range(N_PLANES):
        agree = float((jnp.argmax(
            logits_fn(p, x.at[:, i].set(0.0), z, lg).reshape(m, -1), -1)
            == base).mean())
        dead = " [plane ~zero in data -- unmeasurable]" if live[i] < 1e-4 else ""
        rows.append((PLANES[i], 1 - agree, sal[i], live[i]))
        print(f"  zeroed {PLANES[i]:22s} argmax changed {100*(1-agree):5.1f}% "
              f"| saliency {sal[i]:9.4f} | mean|x| {live[i]:.4f}{dead}",
              flush=True)
    agree_z = float((jnp.argmax(
        logits_fn(p, x, jnp.zeros_like(z), lg).reshape(m, -1), -1)
        == base).mean())
    print(f"  zeroed temporal series     argmax changed "
          f"{100*(1-agree_z):5.1f}%", flush=True)

    rows.sort(key=lambda r: -r[1])
    print("\nleast-used planes (by ablation):", flush=True)
    for name, chg, s, lv in rows[-8:]:
        tag = " (unmeasurable here)" if lv < 1e-4 else ""
        print(f"  {name:22s} {100*chg:5.1f}% changed | sal {s:8.4f}{tag}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
