"""Build the RL seed bank: expert mid-game states with exact observation memory.

`data/apex/seed_bank.npz` carries boards and scenario tags but no memory, and a
seeded episode started from blank memory presents the policy a state no real
game produces (every cell never-seen while the seat owns 60 of them).  This
rebuilds the same seeds with `train.seeds.GameMemory` attached for both seats,
so injection is exact.

Both seats' memory is stored because the scenario tags are per-seat: at
injection the tagged seat is canonicalized to index 0, where the learner sits.

    python -u tools/build_rl_seeds.py --out data/apex/seed_bank_with_memory.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.config import (B, N_EMA, N_SERIES, SEED_DEPTHS,  # noqa: E402
                        TRING, TSCALES)
from train.seeds import GameMemory              # noqa: E402

# Episode horizon per scenario, in ticks.  castle_decision is long on purpose: a
# castle costs 35 army and repays at +0.5/turn, so its 70-tick payback has to
# fit inside the window with a margin a sparse +-1 terminal reward can see.
# 200 ticks leaves +65 army on the table; a shorter window teaches that building
# is simply a loss.  At the deepest start (SEED_DEPTHS[-1] = 160) an episode is
# 360 ticks, still inside the 512-tick rollout scan.
HORIZONS = {"strike_conversion": 40, "blind_commit": 60,
            "defense_warning": 40, "castle_decision": 200}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/apex/seed_bank.npz")
    ap.add_argument("--index", default="data/apex/index.parquet")
    ap.add_argument("--out", default="data/apex/seed_bank_with_memory.npz")
    ap.add_argument("--max-seeds", type=int, default=0, help="0 = all")
    ap.add_argument("--depths", action="store_true",
                    help="emit SEED_DEPTHS variants per seed: the backward "
                         "curriculum's moving start.  The bank is device-"
                         "resident and replicated per GPU, so pair this with "
                         "--max-seeds to hold memory constant.")
    a = ap.parse_args()

    bank = np.load(a.bank)
    scenarios = [str(s) for s in bank["scenarios"]]
    horizons = np.array([HORIZONS[s] for s in scenarios], np.int16)
    rid, tt, tags = bank["replay_id"], bank["t"], bank["tags"]
    # Every seed is used.  They are only episode starting states for RL and no
    # generalization claim is made from them, so withholding maps would just
    # discard coverage of the state distribution being sampled.
    keep = np.ones(len(rid), bool)
    if a.max_seeds:
        sel = np.flatnonzero(keep)
        keep = np.zeros_like(keep)
        keep[np.random.default_rng(0).choice(sel, a.max_seeds, replace=False)] = True
    idx = np.flatnonzero(keep)
    depths = SEED_DEPTHS if a.depths else (0,)
    print(f"seeds: {len(idx):,} of {len(rid):,} x {len(depths)} depths "
          f"{depths} = {len(idx) * len(depths):,} rows | scenarios {scenarios}",
          flush=True)

    paths = dict(zip(pd.read_parquet(a.index).replay_id,
                     pd.read_parquet(a.index).path))
    by_game = defaultdict(list)
    for i in idx:
        by_game[int(rid[i])].append(i)

    n = len(idx) * len(depths)
    out = {
        "armies": np.zeros((n, B, B), np.int16),
        "owners": np.zeros((n, B, B), np.int8),
        "castles": np.zeros((n, B, B), bool),
        "generals": np.zeros((n, B, B), bool),
        "mountains": np.zeros((n, B, B), bool),
        "t": np.zeros(n, np.int16),
        # provenance: lets tests trace an injected seed back to its replay tick
        "replay_id": np.zeros(n, np.int64),

        "tags": np.zeros((n, 2, len(scenarios)), bool),
        # Backward curriculum.  `t` is the actual start tick and the tagged
        # moment is `t + depth_ticks`.  The episode ends at
        # t + depth_ticks + scen_horizon, the same absolute tick at every
        # depth, so a deeper start is a longer task on the same target.
        "depth_idx": np.zeros(n, np.int8),
        "depth_ticks": np.zeros(n, np.int16),
    }
    for s in range(2):
        out[f"last_seen{s}"] = np.zeros((n, B, B), np.int16)
        out[f"ever_seen{s}"] = np.zeros((n, B, B), bool)
        out[f"ghost_owner{s}"] = np.zeros((n, B, B), bool)
        out[f"ghost_army{s}"] = np.zeros((n, B, B), np.int16)
        out[f"egen_seen{s}"] = np.zeros(n, bool)
        # Temporal state (EMA planes and series rings).  fp16: these are
        # smoothed, already-normalised quantities, and the pair dominates the
        # bank's size.
        out[f"ema{s}"] = np.zeros((n, N_EMA, 2, B, B), np.float16)
        out[f"rings{s}"] = np.zeros((n, N_SERIES, len(TSCALES), TRING), np.float16)

    pos = {int(i): k * len(depths) for k, i in enumerate(idx)}
    t0, done = time.time(), 0
    for gi, (game_id, rows) in enumerate(by_game.items(), 1):
        d = np.load(paths[game_id])
        armies, owners = d["armies"], d["owners"]
        mems = [GameMemory(armies, owners, d["generals"], s, d["castles"])
                for s in (0, 1)]
        for i in rows:
            k0, t_tag = pos[i], int(tt[i])
            for j, dep in enumerate(depths):
                k = k0 + j
                # clamp at the start of the replay; record the actual offset so
                # the horizon still lands on the tagged moment + scen_horizon
                t = max(0, t_tag - dep)
                out["depth_idx"][k] = j
                out["depth_ticks"][k] = t_tag - t
                out["armies"][k] = armies[t]
                out["owners"][k] = owners[t]
                out["castles"][k] = d["castles"][t]
                out["generals"][k] = d["generals"]
                out["mountains"][k] = d["mountains"]
                out["t"][k] = t
                out["replay_id"][k] = game_id
                out["tags"][k] = tags[i]
                for s in (0, 1):
                    m = mems[s].at(t)
                    out[f"last_seen{s}"][k] = m["last_seen_t"]
                    out[f"ever_seen{s}"][k] = m["ever_seen"]
                    out[f"ghost_owner{s}"][k] = m["ghost_owner"]
                    out[f"ghost_army{s}"][k] = m["ghost_army"]
                    out[f"egen_seen{s}"][k] = m["egen_seen"]
                    out[f"ema{s}"][k] = m["ema"]
                    out[f"rings{s}"][k] = m["rings"]
                done += 1
        if gi % 200 == 0 or gi == len(by_game):
            el = time.time() - t0
            print(f"  [{gi}/{len(by_game)} games] {done:,}/{n:,} seeds "
                  f"({el:.0f}s, eta {el * (n / max(done, 1) - 1):.0f}s)",
                  flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # The horizon belongs to the scenario, not the seed: a seed can carry
    # several tags, and the rollout applies the horizon of whichever scenario
    # the seed was drawn as.
    np.savez(a.out, scenarios=np.array(scenarios),
             scen_horizon=horizons, **out)
    mb = Path(a.out).stat().st_size / 1e6
    per = {s: int(out["tags"][:, :, j].any(1).sum())
           for j, s in enumerate(scenarios)}
    print(f"\nwrote {a.out}: {n:,} seeds, {mb:.0f} MB\n  per scenario {per}\n"
          f"  scenario horizons {dict(zip(scenarios, horizons.tolist()))}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
