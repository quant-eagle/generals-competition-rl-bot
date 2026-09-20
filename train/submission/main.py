#!/usr/bin/env python3
"""Competition entrypoint: a stdin/stdout bot on one CPU core.

Constraints, in priority order:
  * Never crash and never block: every turn emits exactly one well-formed action
    line, whatever happens.  50 faults in one game is a forfeit.
  * Time guard: the clock starts when the frame finishes reading.  A cheap
    fallback move is computed first, and the model only overwrites it if it
    returns inside the soft deadline.
  * The model is optional: with no weights next to this file the bot plays the
    fallback and still completes games.

Greedy argmax, matching training's evaluation exactly -- the policy is selected
on greedy play, so sampling here would ship a different bot.  One flat head over
the (cell, intent) grid, masked by the exact legality grid, so the argmax is
always an action the engine executes.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SOFT_DEADLINE_S = 0.110      # 150 ms hard; leave headroom for stdout + jitter
WEIGHTS = os.path.join(_HERE, "weights.npz")
PASS_LINE = "1 0 0 0 0"


def _ints(line):
    try:
        return [int(x) for x in line.split()]
    except (ValueError, AttributeError):
        return None


def _read_grid(h, w):
    rows = []
    for _ in range(h):
        line = sys.stdin.readline()
        if not line:
            return None
        v = _ints(line)
        if v is None or len(v) != w:
            return None
        rows.append(v)
    return rows


def fallback(type_grid, owner_grid, army_grid, h, w):
    """Cheapest sane move: push the biggest stack at anything not ours.

    Exists so a slow or missing model still produces legal, non-idle play.
    """
    best, best_army = None, 1
    for r in range(h):
        for c in range(w):
            if owner_grid[r][c] != 1 or army_grid[r][c] <= best_army:
                continue
            for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and type_grid[nr][nc] != 2:
                    best, best_army = (r, c, d), army_grid[r][c]
                    break
    if best is None:
        return PASS_LINE
    return f"0 {best[0]} {best[1]} {best[2]} 0"


class Model:
    """Loads inside the 10 s first-move grace; returns None if unavailable."""

    def __init__(self):
        self.ok = False
        try:
            if not os.path.exists(WEIGHTS):
                return
            import numpy as np
            import torch
            torch.set_num_threads(1)          # one dedicated core
            import config
            import net_torch
            import np_exec
            import np_obs

            d = np.load(WEIGHTS)
            # Config lives under a `cfg::` prefix; everything else is a
            # weight.  Separating by namespace rather than by key shape keeps
            # slash-less names such as `pos`.
            cfg_keys = [k for k in d.files if k.startswith("cfg::")]
            self.cfg = config.NetCfg(bf16=False,
                                     **{k[5:]: int(d[k]) for k in cfg_keys})
            params = {k: torch.from_numpy(d[k].astype("float32"))
                      for k in d.files if not k.startswith("cfg::")}
            self.p = net_torch.prepare(params)
            self.net, self.obs, self.torch, self.np = net_torch, np_obs, torch, np
            self.exec, self.mem = np_exec, np_obs.Memory()
            # The temporal rings are real state, exactly like the fog memory:
            # the trunk consumes two temporal tokens built from them, so they
            # must be pushed every turn or the network sees inputs it never
            # saw in training.
            self.rings = np_obs.Rings()
            with torch.inference_mode():      # warm kernels before turn 0
                x = torch.zeros(1, config.N_PLANES, config.B, config.B)
                z = torch.zeros(1, config.N_SERIES, config.N_TSAMP)
                for _ in range(2):
                    net_torch.forward(self.p, x, z, self.cfg)
            self.ok = True
        except Exception:
            self.ok = False

    def move(self, scalars, type_grid, owner_grid, army_grid, h, w):
        if not self.ok:
            return None
        try:
            np = self.np
            f = self.obs.frame_to_obs(
                scalars[0], scalars[1], scalars[2], scalars[3], scalars[4],
                np.asarray(type_grid, np.int32), np.asarray(owner_grid, np.int32),
                np.asarray(army_grid, np.int32))
            self.mem.update(f)
            self.rings.push(f)
            x = self.torch.from_numpy(self.obs.encode(self.mem, f))[None]
            z = self.torch.from_numpy(self.rings.series())[None]
            lg = self.obs.legal_mask(f)
            with self.torch.inference_mode():
                a_src, a_int = self.net.act(
                    self.p, x, z, self.torch.from_numpy(lg), self.cfg)
            kind, r, c, d, s = self.exec.execute(a_src, a_int, f)
            if kind != 1 and not (0 <= r < h and 0 <= c < w):
                return None                   # never emit an off-board cell
            return f"{kind} {r} {c} {d} {s}"
        except Exception:
            return None


def main():
    hs = _ints(sys.stdin.readline())
    if hs is None or len(hs) != 3:
        return 0
    _, h, w = hs
    model = Model()                            # inside the 10 s grace

    while True:
        line = sys.stdin.readline()
        if not line:
            return 0                           # EOF: game over
        scalars = _ints(line)
        if scalars is None or len(scalars) != 5:
            print(PASS_LINE, flush=True)
            continue
        type_grid = _read_grid(h, w)
        owner_grid = _read_grid(h, w)
        army_grid = _read_grid(h, w)
        if type_grid is None or owner_grid is None or army_grid is None:
            return 0

        t0 = time.perf_counter()
        action = fallback(type_grid, owner_grid, army_grid, h, w)
        if model.ok and time.perf_counter() - t0 < SOFT_DEADLINE_S:
            better = model.move(scalars, type_grid, owner_grid, army_grid, h, w)
            if better is not None:
                action = better
        print(action, flush=True)              # pipes are fully buffered


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)                            # never die noisily mid-match
