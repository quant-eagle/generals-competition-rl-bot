"""Parallel shard emission driver: split manifest rows across N worker
processes (replay_id modulo N), each emitting into data/shards_p{i}/, then move
the per-worker shard files into data/shards/ under collision-free names.

A game is kept when its winner's leaderboard elo clears --min-winner-elo, or
when either player is named in --always-keep.

Usage: python -m data.emit_parallel [--workers 8] [--min-winner-elo 2000]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _elo_map() -> dict[str, int]:
    # the ladder snapshot written by `data.fetch fetch-leaderboard`
    lb = json.loads((ROOT / "data/leaderboard.json").read_text())
    return {e["label"]: e["elo"] for e in lb["leaderboard"]}


def _worker(args: tuple[int, int, int, frozenset]) -> dict:
    i, n, min_elo, always = args
    sys.path.insert(0, str(ROOT))
    from data import decode

    elo = _elo_map()

    def keep(row: dict) -> bool:
        if int(row["replay_id"]) % n != i:
            return False
        if row.get("player_a") in always or row.get("player_b") in always:
            return True
        return elo.get(row.get("winner_name", ""), 0) >= min_elo

    return decode.emit_dataset(manifest_filter=keep,
                               out_dir=ROOT / f"data/shards_p{i}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-winner-elo", type=int, default=2000)
    ap.add_argument("--always-keep", default="",
                    help="comma-separated players whose games bypass the elo floor")
    a = ap.parse_args()
    always = frozenset(x.strip() for x in a.always_keep.split(",") if x.strip())

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(_worker, [(i, a.workers, a.min_winner_elo, always)
                                        for i in range(a.workers)]))

    out = ROOT / "data/shards"
    out.mkdir(exist_ok=True)
    moved = 0
    for i in range(a.workers):
        pdir = ROOT / f"data/shards_p{i}"
        if not pdir.exists():
            continue
        for f in sorted(pdir.glob("shard_*.npz")):
            shutil.move(str(f), out / f"shard_p{i}_{f.stem.split('_')[1]}.npz")
            moved += 1
        pdir.rmdir()

    tot = {k: sum(r[k] for r in results)
           for k in ("replays", "skipped_missing", "skipped_invalid",
                     "sequences", "transitions", "ambiguous")}
    tot["shard_files_moved"] = moved
    print(json.dumps(tot, indent=2))


if __name__ == "__main__":
    main()
