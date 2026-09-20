"""Offline validation and import of staged sprint replay downloads.

No network code exists in this module.  Every replay must pass the bit-exact
engine gate before ``--integrate`` can transcode it into ``data/replays`` and
append it to ``data/manifest.parquet``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING = ROOT / "data" / "sprint" / "replays"
DEFAULT_CACHE = ROOT / "data" / "replays"
DEFAULT_MANIFEST = ROOT / "data" / "manifest.parquet"
SPRINT_ID_BASE = -2_000_000_000  # reserved, deterministic signed-int32 range
SPRINT_ID_SPAN = 100_000_000     # ids live in [BASE, BASE + SPAN); far below ladder ids
ZSTD_LEVEL = 10


def stable_replay_ids(rows: list[dict[str, Any]],
                      taken: set[int] | None = None) -> dict[str, int]:
    """Map each replay URL to a deterministic id in the reserved sprint band.

    Content-addressed, not positional: an id keyed to a URL's index in the
    sorted selection would renumber everything already integrated whenever the
    selection grows.  Hashing the URL makes a replay's id depend only on the
    replay.  `taken` (ids already in the manifest) is probed past, so
    previously integrated rows keep their numbers.
    """
    urls = sorted(row["url"] for row in rows)
    if len(urls) != len(set(urls)):
        raise ValueError("selection contains duplicate replay URLs")
    used = set(taken or ())
    out: dict[str, int] = {}
    for url in urls:
        digest = hashlib.blake2b(url.encode("utf-8"), digest_size=8).digest()
        offset = int.from_bytes(digest, "big") % SPRINT_ID_SPAN
        while SPRINT_ID_BASE + offset in used:
            offset = (offset + 1) % SPRINT_ID_SPAN
        replay_id = SPRINT_ID_BASE + offset
        used.add(replay_id)
        out[url] = replay_id
    return out


def ordered_players_from_manifest(row: dict[str, Any]) -> tuple[str, str]:
    players = (row["player_a"], row["player_b"])
    return players[::-1] if row.get("a_side") == 1 else players


def replay_key(seed: int, players: list[str] | tuple[str, str]) -> tuple[int, str, str]:
    return int(seed), str(players[0]), str(players[1])


def _validate_one(task: tuple[str, str]) -> dict[str, Any]:
    local, path_string = task
    path = Path(path_string)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        replay = json.loads(gzip.decompress(raw))
        # Import inside the spawned worker so the parent never forks a live JAX
        # runtime.  This is the same bit-exact gate the ladder replays pass.
        from data import decode
        report = decode.validate(replay)
        return {
            "local": local, "sha256": sha, "report": report,
            "seed": int(replay["seed"]), "players": replay["players"],
            "winner": replay.get("winner"), "total_ticks": replay.get("total_ticks"),
            "n_ticks": len(replay["ticks"]), "dims": replay["dims"],
        }
    except Exception as exc:  # recorded and excluded; never silently imported
        return {"local": local, "sha256": sha, "error": repr(exc)}


def load_selection(staging: Path) -> list[dict[str, Any]]:
    selection = json.loads((staging / "selection.json").read_text())
    rows = selection.get("replays")
    if not isinstance(rows, list):
        raise ValueError("invalid staging selection")
    for row in rows:
        path = staging / row["local"]
        if not path.is_file():
            raise FileNotFoundError(path)
    return rows


def load_validation(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["local"]] = row
    return out


def validate_all(staging: Path, rows: list[dict[str, Any]], workers: int) -> dict[str, dict[str, Any]]:
    validation_path = staging / "validation.jsonl"
    cached = load_validation(validation_path)
    tasks = []
    for row in rows:
        local = row["local"]
        path = staging / local
        old = cached.get(local)
        if old and old.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
            continue
        tasks.append((local, str(path)))
    if not tasks:
        return cached

    ctx = mp.get_context("spawn")
    with validation_path.open("a") as log, ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(_validate_one, task) for task in tasks]
        done = 0
        for future in as_completed(futures):
            result = future.result()
            cached[result["local"]] = result
            log.write(json.dumps(result, sort_keys=True) + "\n")
            log.flush()
            done += 1
            if done == 1 or done % 10 == 0 or done == len(tasks):
                ok = sum(bool(r.get("report", {}).get("ok")) for r in cached.values())
                print(f"validated {done}/{len(tasks)} new ({ok}/{len(cached)} total exact)",
                      flush=True)
    return cached


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


def integrate(staging: Path, cache_dir: Path, manifest_path: Path,
              selection: list[dict[str, Any]],
              validation: dict[str, dict[str, Any]]) -> tuple[int, int]:
    existing = pq.read_table(manifest_path)
    existing_rows = existing.to_pylist()
    existing_ids = {int(row["replay_id"]) for row in existing_rows}
    existing_keys = {
        replay_key(row["seed"], ordered_players_from_manifest(row))
        for row in existing_rows
    }
    # Probe past ids the manifest already holds, so integrated rows keep the
    # numbers their cache files are named for.
    ids = stable_replay_ids(selection, taken=existing_ids)
    by_local = {row["local"]: row for row in selection}
    new_rows = []
    skipped_duplicate = 0
    fetched_at = time.time()

    for local in sorted(by_local):
        selected = by_local[local]
        result = validation.get(local)
        if not result or not result.get("report", {}).get("ok"):
            continue
        key = replay_key(result["seed"], result["players"])
        if key in existing_keys:
            skipped_duplicate += 1
            continue
        replay_id = ids[selected["url"]]
        if replay_id in existing_ids:
            raise RuntimeError(f"reserved replay id collision: {replay_id}")

        raw_json = gzip.decompress((staging / local).read_bytes())
        target = cache_dir / f"{replay_id}.json.zst"
        compressed = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(raw_json)
        if target.exists():
            prior = zstandard.ZstdDecompressor().decompress(
                target.read_bytes(), max_output_size=1 << 28)
            if prior != raw_json:
                raise RuntimeError(f"existing cache path differs: {target}")
        else:
            atomic_write(target, compressed)

        winner = result["winner"]
        players = result["players"]
        winner_name = players[winner] if winner in (0, 1) else None
        loser_name = players[1 - winner] if winner in (0, 1) else None
        dims = result["dims"]
        new_rows.append({
            "replay_id": replay_id,
            "player_a": players[0], "player_b": players[1], "a_side": 0,
            "winner_name": winner_name, "loser_name": loser_name,
            "seed": result["seed"], "turns": result["total_ticks"],
            "n_ticks": result["n_ticks"],
            "dims_rows": dims["rows"], "dims_cols": dims["cols"],
            "created_at": "", "fetched_at": fetched_at,
        })
        existing_ids.add(replay_id)
        existing_keys.add(key)

    if new_rows:
        combined = sorted(existing_rows + new_rows, key=lambda row: row["replay_id"])
        table = pa.Table.from_pylist(combined, schema=existing.schema)
        tmp = manifest_path.with_suffix(".parquet.part")
        pq.write_table(table, tmp)
        tmp.replace(manifest_path)
    return len(new_rows), skipped_duplicate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--integrate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be in [1, 8]")
    selection = load_selection(args.staging)
    results = validate_all(args.staging, selection, args.workers)
    exact = sum(bool(results.get(row["local"], {}).get("report", {}).get("ok"))
                for row in selection)
    failed = len(selection) - exact
    print(f"engine gate: {exact}/{len(selection)} exact, {failed} rejected")
    if not args.integrate:
        print("validation only; pass --integrate to update cache and manifest")
        return
    added, duplicates = integrate(
        args.staging, args.cache, args.manifest, selection, results)
    print(f"integrated: {added} new, {duplicates} already present, "
          f"{failed} invalid")


if __name__ == "__main__":
    main()
