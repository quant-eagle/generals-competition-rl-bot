"""Torch twin of the ViT trunk -- the code that actually ships.

Training runs in JAX on GPU; the competition runs one CPU core with torch.
This module is pure-functional over the flat parameter dict from
`config.param_shapes`, which is stored in JAX convention; `prepare()`
transposes 4-D conv kernels to OIHW once at load so the per-turn path does no
reshaping.  Only the policy path lives here: Q, danger, general and privileged
heads exist at training time only.

Every reshape, norm and softmax mirrors `net.py` exactly.  The two places a
twin silently diverges are (a) normalising in low precision and (b) a
transposed patch/shuffle order, so both are written out longhand here rather
than delegated to a torch built-in whose memory order differs (F.pixel_shuffle
orders channels (C, r, r); our decoder emits (r, r, C)).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import (B, GRID, N_CELLS, N_INT, N_PLANES, N_SEQ, N_SERIES, N_SRC,
                     N_TEMPORAL, N_TOKENS, N_TSAMP, PATCH,
                     TRAIN_ONLY, NetCfg)

NEG_INF = -1e30
EPS = 1e-6


def prepare(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """JAX-convention weights -> torch-ready (conv HWIO -> OIHW), f32."""
    out = {}
    for k, v in params.items():
        if k.split("/")[0] in TRAIN_ONLY:
            continue
        v = v.float()
        out[k] = v.permute(3, 2, 0, 1).contiguous() if v.ndim == 4 else v
    return out


def _lin(p, name, x):
    y = x @ p[name + "/w"]
    b = p.get(name + "/b")
    return y if b is None else y + b


def _cv(p, name, x):
    return F.conv2d(x, p[name + "/w"], p[name + "/b"], padding=1)


def _rms(p, name, x):
    f = x.float()
    y = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS) * p[name + "/g"]
    return y.to(x.dtype)


def _silu(x):
    return x * torch.sigmoid(x)


def _patchify(x):
    """(n, N_PLANES, B, B) -> (n, N_TOKENS, PATCH*PATCH*N_PLANES)."""
    n = x.shape[0]
    h = x.permute(0, 2, 3, 1).reshape(n, GRID, PATCH, GRID, PATCH, N_PLANES)
    return h.permute(0, 1, 3, 2, 4, 5).reshape(n, N_TOKENS, PATCH * PATCH * N_PLANES)


def _unshuffle(t, dc):
    """(n, N_TOKENS, PATCH*PATCH*Dc) -> (n, B, B, Dc).  Inverse of _patchify."""
    n = t.shape[0]
    h = t.reshape(n, GRID, GRID, PATCH, PATCH, dc).permute(0, 1, 3, 2, 4, 5)
    return h.reshape(n, B, B, dc)


def _block(p, name, h, cfg: NetCfg):
    n, hd, nh = h.shape[0], cfg.head_dim, cfg.heads
    a = _rms(p, name + "n1", h)
    # head-first in one permute, mirroring net.py: both matmuls are then plain
    # batched GEMMs and the scores are upcast for the softmax only.
    qkv = _lin(p, name + "qkv", a).reshape(n, N_SEQ, 3, nh, hd) \
        .permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = _rms(p, name + "qn", q)
    k = _rms(p, name + "kn", k)
    s = (q @ k.transpose(-1, -2)).float() * (hd ** -0.5)
    w = torch.softmax(s, dim=-1).to(v.dtype)
    o = (w @ v).permute(0, 2, 1, 3).reshape(n, N_SEQ, nh * hd)
    h = h + _lin(p, name + "proj", o)

    m = _rms(p, name + "n2", h)
    return h + _lin(p, name + "down", _silu(_lin(p, name + "gate", m))
                    * _lin(p, name + "up", m))


def trunk(p, x, series, cfg: NetCfg):
    """(n, N_PLANES, B, B), (n, N_SERIES, N_TSAMP)
       -> (pooled (n, dim), cells (n, B, B, cell_dim))."""
    n = x.shape[0]
    h = _lin(p, "patch", _patchify(x)) + p["pos"]
    z = _silu(_lin(p, "temp/in", series.reshape(n, -1)))
    t = _lin(p, "temp/out", z).reshape(n, N_TEMPORAL, cfg.dim) + p["ttype"]
    h = torch.cat([h, t], dim=1)                      # (n, N_SEQ, D)
    for i in range(cfg.depth):
        h = _block(p, f"blk{i}/", h, cfg)
    h = _rms(p, "ln_f", h)
    g = h.mean(1)

    # patch tokens only -- the temporal tokens have no cell to un-patchify to
    f = _unshuffle(_lin(p, "dec", h[:, :N_TOKENS]),
                   cfg.cell_dim).permute(0, 3, 1, 2)
    for j in range(cfg.dec_convs):
        f = _silu(_cv(p, f"dcv{j}", f))
    return g, _rms(p, "dec_n", f.permute(0, 2, 3, 1))


def forward(p: dict, x: torch.Tensor, series: torch.Tensor, cfg: NetCfg):
    """(n, N_PLANES, 21, 21), (n, N_SERIES, N_TSAMP) -> (n, N_SRC, N_INT) raw."""
    n = x.shape[0]
    _, f = trunk(p, x, series, cfg)
    return _lin(p, "cell_int", f).reshape(n, N_SRC, N_INT)


def act(p: dict, x: torch.Tensor, series: torch.Tensor, legal, cfg: NetCfg):
    """Masked argmax over the (cell, intent) grid.

    Deployment is greedy, so evaluation must be greedy too or we measure a
    policy we do not ship.  One head means one argmax over 4,410 masked logits;
    with the exact grid, whatever it picks is an action the engine executes.
    """
    logits = forward(p, x, series, cfg)[0]
    filled = torch.where(legal, logits, torch.full_like(logits, NEG_INF))
    a = int(filled.reshape(-1).argmax())
    return a // N_INT, a % N_INT


def random_params(cfg: NetCfg, seed: int = 0) -> dict[str, torch.Tensor]:
    """Shape-correct random weights, for latency measurement only."""
    from .config import param_shapes
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, shp in param_shapes(cfg).items():
        if k.split("/")[0] in TRAIN_ONLY:
            continue
        if k.endswith("/b"):
            out[k] = torch.zeros(shp)
        elif k.endswith("/g"):
            out[k] = torch.ones(shp)
        else:
            fan_in = 1
            for x in shp[:-1]:
                fan_in *= x
            out[k] = torch.randn(shp, generator=g) * (fan_in ** -0.5)
    return out
