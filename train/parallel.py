"""Data parallelism for the trainer, on `jax.shard_map`.

Why more devices rather than a bigger one.  The rollout is a strictly
sequential scan over the segment's ticks and holds more than half of an
iteration, so a fatter per-tick batch cannot shorten it: doubling envs on one
device buys under 10% throughput.  Adding devices multiplies independent scans
instead of fattening one, which is the axis that scales.  The gradient
all-reduce is ~60 MB per optimizer step -- milliseconds against a
multi-second iteration, even over PCIe without NVLink -- so the step is
nowhere near communication-bound on consumer cards.

Why `shard_map` and not `pmap`.  `pmap` forces every replicated argument to
carry an explicit leading device axis, which then has to be stripped before
checkpointing and re-added on resume -- exactly the kind of bookkeeping that
silently corrupts a restart.  Under `shard_map` replicated arrays stay ordinary
arrays, so params, optimizer state, EMA and snapshots checkpoint and resume
unchanged from the single-device path, and a run can resume on a different
device count.

What is sharded.  Only the environments and their keys, on one axis named "d".
Everything else -- params, optimizer state, board pool, seed bank -- is
replicated.  `--envs` is per device.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

AXIS = "d"


def make_mesh(n_devices: int):
    """A 1-D mesh over the first `n_devices` local devices."""
    devs = jax.devices()
    if n_devices > len(devs):
        raise SystemExit(f"asked for {n_devices} devices, {len(devs)} present: "
                         f"{devs}")
    return jax.make_mesh((n_devices,), (AXIS,), devices=devs[:n_devices])


def shard(mesh, tree, spec=P(AXIS)):
    """Place a tree on the mesh with `spec` (default: split on the env axis)."""
    s = NamedSharding(mesh, spec)
    return jax.tree.map(lambda x: jax.device_put(x, s), tree)


def replicate(mesh, tree):
    """Place a tree on every device, unsharded.  The result is ordinary
    arrays -- there is no leading device axis to strip before checkpointing."""
    return shard(mesh, tree, P())


# There is deliberately no `split_envs` helper.  Reshaping (n_dev*envs, ...)
# into (n_dev, envs, ...) is the `pmap` idiom; `shard_map` shards the existing
# leading axis and hands each device its shard without removing that axis, so
# a pre-split array would arrive as (1, envs, ...).  Shard the flat carry on
# axis 0 and each device sees exactly (envs, ...).
#
# The one exception is the PRNG key, which is not env-batched: it is split to
# (n_dev, 2) so axis 0 shards evenly, and therefore does arrive as (1, 2).
# `train_step` indexes it back to a single key.


def make_step(mesh, inner, in_specs, out_specs):
    """`shard_map` + `jit` `inner` with explicit specs.

    Specs are passed in rather than inferred: which argument is sharded and
    which is replicated is the whole contract of a data-parallel step, and an
    explicit list makes it reviewable.

    `check_vma=False` because the replicated outputs (params, optimizer state)
    are only provably replicated to a human: they are identical across devices
    because `learn.make_update` calls `lax.pmean` on the gradient before the
    optimizer step, which the tracer cannot verify.  The equivalence test in
    tests/test_learn.py is what holds that invariant.
    """
    return jax.jit(jax.shard_map(inner, mesh=mesh, in_specs=in_specs,
                                 out_specs=out_specs, check_vma=False))
