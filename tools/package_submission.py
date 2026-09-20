"""Package a checkpoint into a competition submission zip.

Takes the deploy-only parameters (the critic, auxiliary and privileged heads
exist at training time only and are dropped), vendors the modules the bot
imports, and rewrites their relative imports since the zip has no package.

fp16 is the default, because the zip is capped at 50 MB and fp32 weights for
the default model would not fit.  `--check` verifies that halving precision
does not change the greedy action before anything ships.

    python -u tools/package_submission.py --ckpt checkpoints/ckpt_0001000.npz
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.config import TRAIN_ONLY, NetCfg          # noqa: E402

# Fields the deployment twin needs to rebuild the architecture.  Training-only
# widths (q_dim, q_rank, priv_hidden) are absent: the shipped path does not
# contain those heads.  Config is written under a `cfg::` prefix so the bot
# separates it from weights by namespace, not by guessing at key shape (weight
# names such as `pos` carry no '/').
CFG_PREFIX = "cfg::"
CFG_KEYS = ("dim", "depth", "heads", "cell_dim", "dec_convs")

VENDOR = ("config.py", "np_obs.py", "np_exec.py", "net_torch.py")


def deploy_params(ckpt: Path) -> tuple[dict, NetCfg]:
    """The EMA (`e::`) weights of the policy path, not the raw params.

    The EMA is what every evaluation scores, and Straka et al.
    (arXiv:2606.23348) play it too, reporting ~+30 Elo over the raw weights.
    """
    d = np.load(ckpt)
    meta = json.loads(ckpt.with_suffix(".json").read_text())
    cfg = NetCfg(**{k: v for k, v in meta["cfg"].items()
                    if k in NetCfg.__dataclass_fields__})
    out = {}
    for k in d.files:
        if not k.startswith("e::"):
            continue
        name = k[3:]
        head = name.split("/")[0]
        if head in TRAIN_ONLY:
            continue
        out[name] = np.asarray(d[k])
    if not out:
        raise SystemExit(f"{ckpt} carries no e:: (EMA) weights")
    return out, cfg


def check_fp16(params: dict, cfg: NetCfg, n: int = 64) -> float:
    """Fraction of states where fp16 weights change the greedy action."""
    import torch

    from train import net_torch
    from train.config import B, N_PLANES

    p32 = net_torch.prepare({k: torch.from_numpy(v.astype(np.float32))
                             for k, v in params.items()})
    p16 = net_torch.prepare({k: torch.from_numpy(v.astype(np.float16)
                                                 .astype(np.float32))
                             for k, v in params.items()})
    from train.config import N_SERIES, N_TSAMP
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, N_PLANES, B, B, generator=g)
    z = torch.randn(n, N_SERIES, N_TSAMP, generator=g) * 0.1
    with torch.inference_mode():
        l32 = net_torch.forward(p32, x, z, cfg)
        l16 = net_torch.forward(p16, x, z, cfg)
    # one flat head: the action changes iff the joint argmax moves
    a32 = l32.reshape(n, -1).argmax(-1)
    a16 = l16.reshape(n, -1).argmax(-1)
    return float((a32 != a16).float().mean())


def smoke_test(stage: Path) -> None:
    """Load the staged bot exactly as the competition harness will and play a
    few turns.  main.py swallows every exception and falls back to a heuristic,
    so a broken import would not crash -- it would silently ship a much weaker
    bot.  The only reliable check is to run it.
    """
    import subprocess
    probe = r"""
import sys, numpy as np
sys.path.insert(0, %r)
import main
m = main.Model()
assert m.ok, "model failed to load -- the bot would fall back to the heuristic"
rng = np.random.default_rng(0)
n = 21
ty = rng.integers(0, 6, (n, n)).astype(np.int32)
ow = rng.integers(0, 3, (n, n)).astype(np.int32)
am = rng.integers(1, 40, (n, n)).astype(np.int32)
last = None
for t in range(3):                       # 3 turns: exercises memory state too
    out = m.move([t, 60, 200, 55, 190], ty, ow, am, n, n)
    assert out is not None, f"move() returned None on turn {t}"
    last = out
print("smoke ok:", last)
""" % str(stage)
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise SystemExit("staged bot failed to run:\n" + r.stdout + r.stderr)
    print(r.stdout.strip(), flush=True)
    # running the bot leaves bytecode behind; it must not reach the zip
    shutil.rmtree(stage / "__pycache__", ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="dist/submission.zip")
    ap.add_argument("--fp32", action="store_true", help="ship f32 weights")
    ap.add_argument("--check", action="store_true", default=True)
    a = ap.parse_args()

    ckpt = Path(a.ckpt)
    params, cfg = deploy_params(ckpt)
    n_par = sum(v.size for v in params.values())
    print(f"{ckpt.name}: {n_par:,} deploy parameters | {cfg}", flush=True)

    if a.check and not a.fp32:
        drift = check_fp16(params, cfg)
        print(f"fp16 argmax drift: {drift:.3%}", flush=True)
        if drift > 0.02:
            print("REFUSING fp16: >2% of states change action. Use --fp32.",
                  flush=True)
            return 1

    stage = Path("dist/_stage")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    dtype = np.float32 if a.fp32 else np.float16
    np.savez_compressed(stage / "weights.npz",
                        **{CFG_PREFIX + k: np.int32(getattr(cfg, k))
                           for k in CFG_KEYS},
                        **{k: v.astype(dtype) for k, v in params.items()})

    for name in VENDOR:
        src = (REPO / "train" / name).read_text()
        # The zip has no package, so every relative import must become flat.
        # A missed one raises ImportError inside the bot, which main.py turns
        # into silent heuristic play -- so rewrite generally, then assert
        # nothing is left.
        src = re.sub(r"from \.([A-Za-z_][A-Za-z0-9_]*) import",
                     r"from \1 import", src)
        if re.search(r"from \.\w", src) or "import ." in src:
            raise SystemExit(f"{name}: unrewritten relative import would ship")
        (stage / name).write_text(src)
    for name in ("main.py", "run.sh"):
        shutil.copy(REPO / "train" / "submission" / name, stage / name)

    smoke_test(stage)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.iterdir()):
            z.write(f, f.name)
    mb = out.stat().st_size / 1e6
    raw = sum(f.stat().st_size for f in stage.iterdir()) / 1e6
    print(f"wrote {out}: {mb:.1f} MB zipped / {raw:.1f} MB unpacked "
          f"({len(list(stage.iterdir()))} files)", flush=True)
    if mb > 50:
        print("OVER the 50 MB zip limit.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
