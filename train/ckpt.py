"""Resumable training state -- params, EMA, Adam moments, RNG, rollout carry.

The failure this exists to prevent: a crash mid-run whose checkpoint restores
weights but re-initialises the optimizer.  Adam's moments carry the effective
learning rate; restarting them mid-run silently changes the optimisation
problem.  So a checkpoint here is all of the state or it is not a checkpoint.

Pytrees (opt_state, carry) are stored as flat leaf lists.  Their structure is
recovered at load time by rebuilding an empty instance from the same code path
and reusing its treedef -- which also means a checkpoint written by different
code fails loudly at load rather than silently mis-binding leaves.

Writes are atomic: temp file then rename, so a kill during a write cannot leave
a truncated checkpoint as the newest one.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import jax
import numpy as np

STEM = "ckpt_"


def enable_compilation_cache(path: str | Path) -> None:
    """Persist XLA executables across processes.

    The training graph -- a rollout scan holding the engine and the ViT
    forwards, plus a two-level update scan with backward -- takes minutes to
    compile and autotune.  That is fine once inside a multi-hour run and
    intolerable when relaunching, so the cache is on by default.  It is keyed
    on the HLO, so a code change invalidates only what actually changed.
    """
    import jax
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(p))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def _leaves(tree):
    lv, treedef = jax.tree_util.tree_flatten(tree)
    return [np.asarray(x) for x in lv], treedef


def save(path: Path, *, params, ema, snapshots, opt_state, carry, key,
         meta: dict) -> None:
    """Atomically write the complete training state.

    `ema` (the evaluated and shipped weights) and the `snapshots` FIFO (the
    ladder's opponents) are part of the state: neither is derivable from
    `params`, and resuming without them would evaluate a different policy and
    silently collapse the ladder to mirror-only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, np.ndarray] = {}
    for tag, d in (("p", params), ("e", ema)):
        for k, v in d.items():
            blob[f"{tag}::{k}"] = np.asarray(v)
    for i, snap in enumerate(snapshots):
        for k, v in snap.items():
            blob[f"s{i}::{k}"] = np.asarray(v)
    meta = dict(meta, n_snapshots=len(snapshots))
    for tag, tree in (("o", opt_state), ("c", carry)):
        lv, _ = _leaves(tree)
        for i, x in enumerate(lv):
            blob[f"{tag}::{i}"] = x
    blob["key"] = np.asarray(key)

    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **blob)
    os.replace(tmp, path)                       # atomic within a filesystem
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))


def load(path: Path, *, params_like, opt_state_like, carry_like):
    """Restore state.  `*_like` supply the pytree structure only."""
    d = np.load(path)
    def grab(tag):
        pre = f"{tag}::"
        return {k[len(pre):]: jax.numpy.asarray(d[k])
                for k in d.files if k.startswith(pre)}
    params, ema = grab("p"), grab("e")
    meta_path = path.with_suffix(".json")
    n_snap = json.loads(meta_path.read_text()).get("n_snapshots", 0)
    snapshots = [grab(f"s{i}") for i in range(n_snap)]
    missing = set(params_like) ^ set(params)
    if missing:
        raise ValueError(f"checkpoint params do not match the model: {sorted(missing)[:6]}")
    if set(ema) != set(params):
        raise ValueError("checkpoint is missing its EMA weights")

    def rebuild(tag, like):
        _, treedef = _leaves(like)
        n = sum(1 for k in d.files if k.startswith(f"{tag}::"))
        lv = [jax.numpy.asarray(d[f"{tag}::{i}"]) for i in range(n)]
        if n != treedef.num_leaves:
            raise ValueError(
                f"checkpoint '{tag}' has {n} leaves, code expects "
                f"{treedef.num_leaves} -- written by different code?")
        return jax.tree_util.tree_unflatten(treedef, lv)

    opt_state = rebuild("o", opt_state_like)
    carry = rebuild("c", carry_like)
    meta = json.loads(meta_path.read_text())
    return (params, ema, snapshots, opt_state, carry,
            jax.numpy.asarray(d["key"]), meta)


def meta_of(path: Path) -> dict:
    """The sidecar metadata alone -- no array loading.

    Needed before the model exists: the W&B run id and the run name have to be
    known at init so a resume continues the same chart.
    """
    j = path.with_suffix(".json")
    return json.loads(j.read_text()) if j.exists() else {}


def iter_of(p: Path) -> int:
    m = re.search(rf"{STEM}(\d+)", p.name)
    return int(m.group(1)) if m else -1


def latest(dirpath: Path) -> Path | None:
    cks = sorted(dirpath.glob(f"{STEM}*.npz"), key=iter_of)
    return cks[-1] if cks else None


def prune(dirpath: Path, keep_last: int = 3, keep_every: int = 2000) -> list[Path]:
    """Keep the most recent `keep_last` plus milestone multiples of `keep_every`.

    The newest checkpoint is never removed, so a prune can never leave the run
    without a resume point.  Returns what was deleted.
    """
    cks = sorted(dirpath.glob(f"{STEM}*.npz"), key=iter_of)
    if not cks:
        return []
    keep = set(cks[-keep_last:])
    keep.add(cks[-1])
    keep |= {p for p in cks if keep_every and iter_of(p) % keep_every == 0}
    gone = []
    for p in cks:
        if p not in keep:
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)
            gone.append(p)
    return gone


def disk_free_gb(path: Path) -> float:
    st = os.statvfs(path if path.exists() else path.parent)
    return st.f_bavail * st.f_frsize / 1e9
