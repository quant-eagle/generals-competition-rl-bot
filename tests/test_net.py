"""Parity and shape contract for the ViT trunk.

The network exists twice: JAX for training, torch for deployment.  A silent
divergence ships a different policy from the one that was trained, so every
component has a parity test.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from train import config as C
from train import net, net_torch

CFG = C.NetCfg(bf16=False)
SMALL = C.NetCfg(dim=64, depth=2, heads=2, cell_dim=16, q_dim=32,
                 priv_hidden=16, bf16=False)


def _obs(rng, n=1):
    return rng.standard_normal((n, C.N_PLANES, C.B, C.B), dtype=np.float32)


def _series(rng, n=1):
    """The five aggregate series, already normalised as `obs.series_values`
    emits them (all O(1)), so parity is tested on realistic magnitudes."""
    return rng.standard_normal((n, C.N_SERIES, C.N_TSAMP),
                               dtype=np.float32) * 0.3


def _to_torch(params):
    return net_torch.prepare({k: torch.from_numpy(np.asarray(v))
                              for k, v in params.items()})


@pytest.mark.parametrize("cfg", [SMALL, CFG])
def test_jax_and_torch_agree_on_logits(cfg):
    p = net.init(jax.random.PRNGKey(0), cfg)
    tp = _to_torch(p)
    x = _obs(np.random.default_rng(1), 3)
    z = _series(np.random.default_rng(11), 3)

    jl = jax.jit(net.act_logits_v, static_argnums=4)(
        p, jnp.asarray(x), jnp.asarray(z),
        jnp.ones((3, C.N_SRC, C.N_INT), bool), cfg)
    tl = net_torch.forward(tp, torch.from_numpy(x), torch.from_numpy(z), cfg)

    d = np.abs(np.asarray(jl) - tl.numpy()).max()
    assert d < 2e-4, f"policy head diverged by {d:.2e}"


def test_patchify_round_trips_cell_for_cell():
    """A cell's features must come back to the cell they came from."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((C.N_PLANES, C.B, C.B), dtype=np.float32))
    t = net._patchify(x)                       # (49, 9*23)
    back = net._unshuffle(t, C.N_PLANES)       # (21, 21, 23)
    assert np.allclose(back, np.asarray(x).transpose(1, 2, 0))

    tt = net_torch._patchify(torch.from_numpy(np.asarray(x))[None])
    assert np.allclose(np.asarray(t), tt[0].numpy())


def test_masked_logits_are_neg_inf_on_exactly_the_illegal_pairs():
    """One head over the grid, so masking is per pair: a legal source with an
    illegal intent cannot be expressed."""
    p = net.init(jax.random.PRNGKey(0), SMALL)
    x = jnp.asarray(_obs(np.random.default_rng(2))[0])
    z = jnp.asarray(_series(np.random.default_rng(12))[0])
    g = np.zeros((C.N_SRC, C.N_INT), bool)
    g[0, 3] = g[7, C.INT_PASS] = g[300, C.INT_BUILD] = True
    lg = jnp.asarray(g)
    out = net.act_logits(p, x, z, lg, SMALL)
    assert out.shape == (C.N_SRC, C.N_INT)
    assert bool(jnp.all(out[~lg] == net.NEG_INF))
    assert bool(jnp.all(jnp.isfinite(out[lg])))


def test_q_is_factored_and_vbar_matches_the_explicit_sum():
    """Vbar = sum_a pi(a) Q(s,a) must equal the closed form used in learn.py.

    The Q-boosting trace rests on this identity, so it is checked against a
    brute-force joint expectation.
    """
    p = net.init(jax.random.PRNGKey(3), SMALL)
    x = jnp.asarray(_obs(np.random.default_rng(4))[0])
    priv = jnp.asarray(np.random.default_rng(5).standard_normal(
        (C.N_PRIV, C.B, C.B), dtype=np.float32))
    z = jnp.asarray(_series(np.random.default_rng(13))[0])
    g, f = net.trunk(p, x, z, SMALL)
    q0, qs, qi, u, v = net.q_parts(p, f, g, priv, SMALL)

    ps = jax.random.uniform(jax.random.PRNGKey(6), (C.N_SRC,)); ps /= ps.sum()
    pi = jax.random.uniform(jax.random.PRNGKey(7), (C.N_INT,)); pi /= pi.sum()

    joint = q0 + qs[:, None] + qi[None, :] + u @ v.T           # (N_SRC, N_INT)
    brute = float(ps @ joint @ pi)
    closed = float(q0 + ps @ qs + pi @ qi + (ps @ u) @ (pi @ v))
    assert abs(brute - closed) < 1e-3 * max(1.0, abs(brute))


def test_privileged_planes_never_reach_the_actor():
    """The deployed twin cannot even see the critic's parameters."""
    tp = net_torch.prepare({k: torch.zeros(s)
                            for k, s in C.param_shapes(CFG).items()})
    assert not [k for k in tp if k.split("/")[0] in C.TRAIN_ONLY]
    assert C.n_params(CFG, deploy_only=True) < C.n_params(CFG)


def test_shipped_size_fits_the_zip_budget():
    """The submission zip is capped at 50 MB, and fp16 weights are what ship."""
    mb = C.n_params(CFG, deploy_only=True) * 2 / 1e6
    assert mb < 45.0, f"{mb:.1f} MB of fp16 weights leaves no room for code"


def test_every_deploy_parameter_survives_the_packaging_filter():
    """The deployed parameter set is exactly the policy path.  `pos` is the
    one key with no '/' in its name, so it is checked explicitly: a filter
    keyed on name shape would drop it."""
    shapes = C.param_shapes(CFG)
    deploy = {k for k in shapes if k.split("/")[0] not in C.TRAIN_ONLY}
    assert "pos" in deploy and "patch/w" in deploy
    assert not any(k.startswith(("q/", "priv/", "gen/", "danger/"))
                   for k in deploy)

    tp = net_torch.prepare({k: torch.zeros(s) for k, s in shapes.items()})
    assert set(tp) == deploy, sorted(deploy ^ set(tp))[:6]
