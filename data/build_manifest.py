"""Rebuild data/manifest.parquet from the replay cache itself.

Every MANIFEST_SCHEMA column except a_side/created_at is present in the replay
JSON (winner is a player index). Rows already in the existing manifest are kept
verbatim (they carry real created_at from the match lists). Useful when a bulk
fetch stops before its end-of-run manifest write.

Usage: python -m data.build_manifest [--workers 10]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

ROOT = Path(__file__).resolve().parent.parent


def _row(path: str) -> dict | None:
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            Path(path).read_bytes(), max_output_size=1 << 28)
        d = json.loads(raw)
        w = d.get("winner")
        players = d.get("players", ["?", "?"])
        return {
            "replay_id": int(Path(path).name.split(".")[0]),
            "player_a": players[0], "player_b": players[1],
            "a_side": 0,
            "winner_name": players[w] if w in (0, 1) else "",
            "loser_name": players[1 - w] if w in (0, 1) else "",
            "seed": int(d.get("seed") or 0),
            "turns": int(d.get("total_ticks") or len(d.get("ticks", []))),
            "n_ticks": len(d.get("ticks", [])),
            "dims_rows": int(d["dims"]["rows"]), "dims_cols": int(d["dims"]["cols"]),
            "created_at": "", "fetched_at": time.time(),
        }
    except Exception as e:  # corrupt or partial download: skip and report
        print(f"SKIP {path}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    from data.fetch import MANIFEST_SCHEMA  # single source of truth for schema

    mpath = ROOT / "data/manifest.parquet"
    old = pq.read_table(mpath).to_pylist() if mpath.exists() else []
    have = {r["replay_id"] for r in old}

    files = [f for f in glob.glob(str(ROOT / "data/replays/*.json.zst"))
             if int(Path(f).name.split(".")[0]) not in have]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for r in ex.map(_row, files, chunksize=64) if r]

    table = pa.Table.from_pylist(old + rows, schema=MANIFEST_SCHEMA)
    pq.write_table(table, mpath)
    print(f"manifest: {table.num_rows} rows ({len(old)} kept, {len(rows)} rebuilt, "
          f"{len(files) - len(rows)} skipped)")


if __name__ == "__main__":
    main()
