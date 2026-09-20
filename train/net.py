"""The network in JAX: a 3x3-patch ViT trunk plus policy and critic heads.

    trunk(p, x, series, cfg) -> (g, f)   g: (dim,)  f: (B, B, cell_dim)

`g` is the pooled token vector the danger head reads; `f` carries per-cell
features for the policy, the critic and the general auxiliary.

Heads:
    policy   (441, 10) one masked categorical over (cell, intent), the shape
             of Straka et al. (arXiv:2606.23348) plus build and pass
    Q        centralized action-value, factored + rank-r; training only
    general  (441,)  auxiliary: where is the enemy general
    danger   scalar  auxiliary: do we die within 32 ticks

The auxiliaries are predicted from the fog trunk and supervised by privileged
truth, so they sharpen representation without entering the reward and therefore
cannot distort the optimum.  The critic takes privileged planes as its own
input; the actor's input never does, so hidden state cannot reach the deployed
policy.  That asymmetry is a tested invariant.

Shapes come from `config.param_shapes`; `net_torch` mirrors this file primitive
for primitive.  Softmax and every norm run in f32 on both sides -- normalising
or exponentiating in bf16 is where mixed precision and JAX/torch parity both go
wrong.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (B, GRID, N_CELLS, N_INT, N_PLANES, N_SEQ, N_SERIES,
                     N_SRC, N_TEMPORAL, N_TOKENS, N_TSAMP,
                     PATCH, NetCfg, param_shapes)

NEG_INF = -1e30
EPS = 1e-6


def _cast(x, cfg: NetCfg):
    """Compute dtype for matmul-heavy paths.

    Always an explicit cast, so a caller may hand in the fp16 planes the rollout
    stored: f16 -> bf16 is bit-identical to f16 -> f32 -> bf16 (every f16 value
    is exact in f32), and it avoids materialising an f32 copy of a 1.3 GB
    buffer.  With bf16 off this upcasts instead, so the f32 eval path is safe.
    """
    return x.astype(jnp.bfloat16 if cfg.bf16 else jnp.float32)


def init(key, cfg: NetCfg) -> dict[str, jnp.ndarray]:
    """Fan-in scaled normal; zero biases; unit RMSNorm gains.

    Residual output projections are scaled by (2L)^-1/2 so the residual stream
    variance does not grow with depth.  Policy and Q output projections start
    near zero, so the first iterations are close to uniform-over-legal with
    Q ~ 0 -- which keeps the early KL and the Expected-SARSA trace well behaved.
    """
    shapes = param_shapes(cfg)
    names = sorted(shapes)
    resid = (2.0 * cfg.depth) ** -0.5
    near_zero = ("cell_int/w",
                 "q/src_cell/w", "q/intg/w",
                 "q/base/w", "gen/w", "danger/w")
    params = {}
    for rk, name in zip(jax.random.split(key, len(names)), names):
        s = shapes[name]
        if name.endswith("/b"):
            params[name] = jnp.zeros(s)
        elif name.endswith("/g"):
            params[name] = jnp.ones(s)
        elif name == "pos":
            params[name] = 0.02 * jax.random.normal(rk, s)
        else:
            fan_in = 1
            for x in s[:-1]:
                fan_in *= x
            scale = fan_in ** -0.5
            if name.endswith("proj/w") or name.endswith("down/w"):
                scale *= resid
            if name in near_zero:
                scale *= 0.01
            params[name] = scale * jax.random.normal(rk, s)
    return params


# --- primitives (mirrored one-for-one in net_torch) -------------------------

def _lin(p, name, x):
    y = x @ p[name + "/w"].astype(x.dtype)
    b = p.get(name + "/b")
    return y if b is None else y + b.astype(x.dtype)


def _cv(p, name, x):
    """(H, W, Cin) -> (H, W, Cout), 3x3 SAME."""
    w = p[name + "/w"].astype(x.dtype)
    y = jax.lax.conv_general_dilated(
        x[None], w, (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]
    return y + p[name + "/b"].astype(x.dtype)


def _rms(p, name, x):
    """RMSNorm in f32 regardless of the activation dtype."""
    dt = x.dtype
    f = x.astype(jnp.float32)
    y = f * jax.lax.rsqrt((f * f).mean(-1, keepdims=True) + EPS) * p[name + "/g"]
    return y.astype(dt)


def _silu(x):
    return x * jax.nn.sigmoid(x)


def _patchify(x):
    """(N_PLANES, B, B) -> (N_TOKENS, PATCH*PATCH*N_PLANES).

    Token (gr, gc) holds its 3x3 block in row-major sub-order; `_unshuffle` is
    the exact inverse, so a cell's features come back to the cell it came from.
    """
    h = x.transpose(1, 2, 0).reshape(GRID, PATCH, GRID, PATCH, N_PLANES)
    return h.transpose(0, 2, 1, 3, 4).reshape(N_TOKENS, PATCH * PATCH * N_PLANES)


def _unshuffle(t, dc):
    """(N_TOKENS, PATCH*PATCH*Dc) -> (B, B, Dc).  Inverse of `_patchify`."""
    h = t.reshape(GRID, GRID, PATCH, PATCH, dc).transpose(0, 2, 1, 3, 4)
    return h.reshape(B, B, dc)


def _block(p, name, h, cfg: NetCfg):
    hd, nh = cfg.head_dim, cfg.heads
    a = _rms(p, name + "n1", h)
    # One transpose to (3, heads, tokens, head_dim), so both attention einsums
    # are plain batched matmuls over a leading head axis.  Holding q/k/v as
    # (tokens, heads, dim) instead makes XLA shuffle a (batch, 51, 8, 56)
    # tensor around each einsum -- ~5% of the forward for identical arithmetic.
    qkv = _lin(p, name + "qkv", a).reshape(N_SEQ, 3, nh, hd) \
        .transpose(1, 2, 0, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]
    # QK-norm: RMSNorm over the head dimension, gains shared across heads.  The
    # cheapest guard against attention-logit blowup.
    q = _rms(p, name + "qn", q)
    k = _rms(p, name + "kn", k)
    # q and k come out of a bf16 matmul, so upcasting them before the score
    # matmul cannot recover precision; it only forces a slower f32 matmul.  The
    # softmax runs in f32, which is where the precision actually matters.
    s = jnp.einsum("hqd,hkd->hqk", q, k).astype(jnp.float32) * (hd ** -0.5)
    w = jax.nn.softmax(s, axis=-1).astype(v.dtype)
    o = jnp.einsum("hqk,hkd->hqd", w, v).transpose(1, 0, 2) \
        .reshape(N_SEQ, nh * hd)
    h = h + _lin(p, name + "proj", o)

    m = _rms(p, name + "n2", h)
    return h + _lin(p, name + "down", _silu(_lin(p, name + "gate", m))
                    * _lin(p, name + "up", m))


def temporal_tokens(p, series, cfg: NetCfg):
    """(N_SERIES, N_TSAMP) -> (N_TEMPORAL, dim).  After Straka et al.

    A plain MLP over the flattened multi-scale rings.  It carries no spatial
    information at all -- that is the memory planes' job -- and exists because
    the board frame cannot express "our army has been flat for 200 ticks while
    theirs doubled"."""
    z = _silu(_lin(p, "temp/in", _cast(series.reshape(-1), cfg)))
    return _lin(p, "temp/out", z).reshape(N_TEMPORAL, cfg.dim) + \
        p["ttype"].astype(z.dtype)


def trunk(p, x, series, cfg: NetCfg):
    """(N_PLANES, B, B), (N_SERIES, N_TSAMP)
       -> (pooled (dim,), cells (B, B, cell_dim))."""
    h = _lin(p, "patch", _cast(_patchify(x), cfg)) + p["pos"].astype(
        jnp.bfloat16 if cfg.bf16 else jnp.float32)
    h = jnp.concatenate([h, temporal_tokens(p, series, cfg)])   # (N_SEQ, D)
    for i in range(cfg.depth):
        h = _block(p, f"blk{i}/", h, cfg)
    h = _rms(p, "ln_f", h)
    g = h.mean(0).astype(jnp.float32)          # pools patches and history

    # the decoder un-patchifies, so it sees the patch tokens only -- the two
    # temporal tokens have no cell to belong to.
    f = _unshuffle(_lin(p, "dec", h[:N_TOKENS]), cfg.cell_dim)
    for j in range(cfg.dec_convs):
        f = _silu(_cv(p, f"dcv{j}", f))
    return g, _rms(p, "dec_n", f).astype(jnp.float32)


# --- heads ------------------------------------------------------------------

def policy_logits(p, f):
    """(N_SRC, N_INT) raw logits: one head over the (cell, intent) grid.

    Deliberately not factored into a per-cell source head and a global intent
    head.  A shared "north" logit would pool gradient across cells, but which
    direction is right depends on which cell, so that sharing aliases a genuine
    distinction.  Factoring also makes the legality mask marginal -- an intent
    mask can only say "some cell can step north", so a sampled (cell, intent)
    pair can still be illegal.  One head over the joint grid is masked exactly.
    """
    return _lin(p, "cell_int", f).reshape(N_SRC, N_INT)


def _mask(logits, legal):
    """`legal` is the exact (N_SRC, N_INT) grid: every False pair is unreachable."""
    return jnp.where(legal, logits, NEG_INF)


def q_parts(p, f, g, priv, cfg: NetCfg):
    """Centralized action-value, factored so Vbar stays exact and cheap.

        Q(s, i, j) = q0 + qs[i] + qi[j] + <u[i], v[j]>

    Returns (q0, qs (N_SRC,), qi (N_INT,), u (N_SRC, r), v (N_INT, r)).
    Never materializes the 441 x 10 joint table.
    """
    # bf16 like the trunk: q/mix is a 441 x (Dc+Cp) x q_dim matmul per sample,
    # the one part of the critic large enough to matter.  Outputs go back to
    # f32 -- Vbar is an expectation over 451 terms and the trace sums it over a
    # whole episode.
    e = _silu(_cv(p, "priv/c", _cast(priv, cfg).transpose(1, 2, 0)))
    z = _silu(_lin(p, "q/mix", jnp.concatenate([_cast(f, cfg), e], -1)))
    z = z.astype(jnp.float32)                                    # (B, B, q_dim)
    zg = z.mean((0, 1))
    r = cfg.q_rank

    src = _lin(p, "q/src_cell", z).reshape(N_SRC, 1 + r)
    intent = _lin(p, "q/intg", zg).reshape(N_INT, 1 + r)
    q0 = _lin(p, "q/base", zg).squeeze(-1)
    return q0, src[:, 0], intent[:, 0], src[:, 1:], intent[:, 1:]


def forward(p, x, series, priv, legal, cfg: NetCfg):
    """Everything the trainer needs from one observation."""
    g, f = trunk(p, x, series, cfg)
    return (_mask(policy_logits(p, f), legal),
            q_parts(p, f, g, priv, cfg),
            _lin(p, "gen", f).reshape(N_CELLS),
            _lin(p, "danger", g).squeeze(-1))


def act_logits(p, x, series, legal, cfg: NetCfg):
    """Policy-only forward -- the path the deployed twin mirrors."""
    _, f = trunk(p, x, series, cfg)
    return _mask(policy_logits(p, f), legal)


forward_v = jax.vmap(forward, in_axes=(None, 0, 0, 0, 0, None))
act_logits_v = jax.vmap(act_logits, in_axes=(None, 0, 0, 0, None))
