"""Replay and leaderboard fetcher for the generals.bot competition ladder.

Subcommands:
    fetch-leaderboard
    fetch-matches  --player NAME
    fetch-replays  --players "A,B,C" [--losses-only] [--wins-per-player N]
    fetch-all      --min-elo N

Behavior:
- aiohttp, global token-bucket rate limit <= 4 req/s (all requests share one bucket),
  tunable with --rps and eased automatically whenever the server throttles
- exponential backoff with jitter on 403 / 429 / 5xx / transport errors
  (403 is the server's anti-abuse throttle here, not a permission error)
- User-Agent set on every request; redirects (generals.bot -> www) followed
- replay cache: data/replays/{id}.json.zst (zstandard level 10), skip existing
- manifest: data/manifest.parquet via pyarrow, merged/deduped on replay_id

Matches come in seed-pairs (same seed, sides swapped): we never dedupe by seed,
only by replay id, so both halves of a pair are kept.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

BASE_URL = "https://generals.bot/api/leaderboard"
USER_AGENT = "generals-zero-replay-research/1.0 (competition participant)"

DATA_DIR = Path(__file__).resolve().parent
REPLAYS_DIR = DATA_DIR / "replays"
MATCHES_DIR = DATA_DIR / "matches"
MANIFEST_PATH = DATA_DIR / "manifest.parquet"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.json"

RATE_LIMIT_RPS = 4.0  # hard etiquette cap, TOTAL across all concurrent tasks
MIN_RATE_RPS = 1.0  # floor the adaptive backoff will not go below
THROTTLE_STATUS = (403, 429)  # 403 = anti-abuse throttle, retryable (see Fetcher._slow_down)
# The server allows a burst of replays and then answers 403 for a few minutes,
# regardless of request rate or connection reuse, so the cure is to wait the
# cooldown out, not to keep lowering the rate.
THROTTLE_COOLDOWN_S = 90.0
ZSTD_LEVEL = 10
MAX_RETRIES = 10
REQUEST_TIMEOUT_S = 60
MANIFEST_CHECKPOINT_EVERY = 100  # flush the manifest mid-run so a crash keeps progress

MANIFEST_SCHEMA = pa.schema(
    [
        ("replay_id", pa.int64()),
        ("player_a", pa.string()),
        ("player_b", pa.string()),
        ("a_side", pa.int32()),
        ("winner_name", pa.string()),
        ("loser_name", pa.string()),
        ("seed", pa.int64()),
        ("turns", pa.int64()),
        ("n_ticks", pa.int64()),
        ("dims_rows", pa.int32()),
        ("dims_cols", pa.int32()),
        ("created_at", pa.string()),
        ("fetched_at", pa.float64()),
    ]
)

# --------------------------------------------------------------------------
# Pure logic (unit-tested, no I/O)
# --------------------------------------------------------------------------


def replay_id_from_url(url: str) -> int:
    """Extract the numeric replay id from '/api/leaderboard?replay=130192'-style URLs."""
    qs = parse_qs(urlsplit(url).query)
    vals = qs.get("replay")
    if not vals:
        raise ValueError(f"no replay id in url: {url!r}")
    return int(vals[0])


def match_replay_id(match: dict[str, Any]) -> int:
    """Replay id of a match-list entry ('id' field, falling back to 'replay' URL)."""
    if match.get("id") is not None:
        return int(match["id"])
    return replay_id_from_url(match["replay"])


def match_result_for(match: dict[str, Any], player: str) -> str:
    """'win' | 'loss' | 'draw' from the perspective of `player`.

    `winner` is "A"/"B" relative to a_name/b_name; anything else counts as a draw.
    Raises ValueError if `player` is not in the match.
    """
    if match.get("a_name") == player:
        side = "A"
    elif match.get("b_name") == player:
        side = "B"
    else:
        raise ValueError(f"{player!r} not in match {match.get('id')}")
    winner = match.get("winner")
    if winner not in ("A", "B"):
        return "draw"
    return "win" if winner == side else "loss"


def winner_loser_names(match: dict[str, Any]) -> tuple[str | None, str | None]:
    """(winner_name, loser_name) mapped through a_name/b_name; (None, None) on draw."""
    winner = match.get("winner")
    if winner == "A":
        return match.get("a_name"), match.get("b_name")
    if winner == "B":
        return match.get("b_name"), match.get("a_name")
    return None, None


def _created_at_key(match: dict[str, Any]) -> str:
    # ISO-8601 UTC timestamps sort correctly as text.
    return match.get("created_at") or ""


def select_replay_targets(
    matches_by_player: list[tuple[str, list[dict[str, Any]]]],
    losses_only: bool = False,
    wins_per_player: int | None = None,
) -> list[dict[str, Any]]:
    """Ordered download list, deduped by replay id (not by seed: seed-pairs kept).

    Priority: all losses (players in the given order, newest first within a
    player), then wins newest-first (capped at `wins_per_player` per player,
    None = all). Draws travel with losses (they are equally informative
    failure cases) unless losses_only is set, in which case only true losses.
    """
    seen: set[int] = set()
    out: list[dict[str, Any]] = []

    def _emit(match: dict[str, Any]) -> None:
        rid = match_replay_id(match)
        if rid not in seen:
            seen.add(rid)
            out.append(match)

    for player, matches in matches_by_player:
        losses = [m for m in matches if match_result_for(m, player) == "loss"]
        for m in sorted(losses, key=_created_at_key, reverse=True):
            _emit(m)
    if losses_only:
        return out
    for player, matches in matches_by_player:
        draws = [m for m in matches if match_result_for(m, player) == "draw"]
        for m in sorted(draws, key=_created_at_key, reverse=True):
            _emit(m)
    for player, matches in matches_by_player:
        wins = sorted(
            (m for m in matches if match_result_for(m, player) == "win"),
            key=_created_at_key,
            reverse=True,
        )
        if wins_per_player is not None:
            wins = wins[:wins_per_player]
        for m in wins:
            _emit(m)
    return out


def manifest_row(
    match: dict[str, Any], replay: dict[str, Any], fetched_at: float
) -> dict[str, Any]:
    """One manifest record from a match-list entry plus its replay JSON."""
    winner_name, loser_name = winner_loser_names(match)
    dims = replay.get("dims") or {}
    seed = match.get("seed")
    if seed is None:
        seed = replay.get("seed")
    return {
        "replay_id": match_replay_id(match),
        "player_a": match.get("a_name"),
        "player_b": match.get("b_name"),
        "a_side": int(match["a_side"]) if match.get("a_side") is not None else None,
        "winner_name": winner_name,
        "loser_name": loser_name,
        "seed": int(seed) if seed is not None else None,
        "turns": int(match["turns"]) if match.get("turns") is not None else None,
        "n_ticks": len(replay.get("ticks") or []),
        "dims_rows": dims.get("rows"),
        "dims_cols": dims.get("cols"),
        "created_at": match.get("created_at"),
        "fetched_at": float(fetched_at),
    }


def rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    cols = {name: [r.get(name) for r in rows] for name in MANIFEST_SCHEMA.names}
    return pa.table(cols, schema=MANIFEST_SCHEMA)


def merge_manifest(existing: pa.Table | None, new_rows: list[dict[str, Any]]) -> pa.Table:
    """Merge new rows over an existing manifest; new rows win on replay_id collision."""
    by_id: dict[int, dict[str, Any]] = {}
    if existing is not None:
        for r in existing.to_pylist():
            by_id[r["replay_id"]] = r
    for r in new_rows:
        by_id[r["replay_id"]] = r
    rows = [by_id[k] for k in sorted(by_id)]
    return rows_to_table(rows)


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------


class TokenBucket:
    """Async token bucket: at most `rate` acquisitions per second, globally."""

    def __init__(self, rate: float, capacity: float | None = None):
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self.tokens = self.capacity
        self.stamp = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:  # serializes waiters -> total rate respected
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.stamp) * self.rate)
                self.stamp = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self.tokens) / self.rate)


class Fetcher:
    def __init__(self, session: aiohttp.ClientSession, bucket: TokenBucket):
        self.session = session
        self.bucket = bucket
        self.request_count = 0
        self.throttle_events = 0

    def _slow_down(self) -> None:
        """Ease the global rate a little after a throttle response.

        The dominant cure is the cooldown wait in get_json; this only trims the
        steady-state rate so we approach the burst budget more gently. It stops
        at MIN_RATE_RPS because dropping lower buys nothing -- the server is
        counting volume per window, not instantaneous rate.
        """
        self.throttle_events += 1
        new_rate = max(MIN_RATE_RPS, self.bucket.rate * 0.75)
        if new_rate < self.bucket.rate:
            print(f"  throttled: global rate {self.bucket.rate:.2f} -> {new_rate:.2f} req/s",
                  file=sys.stderr)
            self.bucket.rate = new_rate
            self.bucket.capacity = min(self.bucket.capacity, new_rate)

    async def get_json(self, url: str) -> Any:
        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            await self.bucket.acquire()
            self.request_count += 1
            wait = delay
            try:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json(content_type=None)
                    # 403 here is anti-abuse throttling, not a permission error:
                    # the same URL returns 200 when requested in isolation.
                    if resp.status in THROTTLE_STATUS or resp.status >= 500:
                        retry_after = resp.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after else delay
                        last_err = RuntimeError(f"HTTP {resp.status} for {url}")
                        if resp.status in THROTTLE_STATUS:
                            # Sit out the burst window rather than retrying into it.
                            wait = max(wait, THROTTLE_COOLDOWN_S)
                            self._slow_down()
                    else:
                        resp.raise_for_status()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                wait = delay
            print(
                f"  retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s: {url} ({last_err})",
                file=sys.stderr,
            )
            await asyncio.sleep(wait + random.uniform(0, 0.5))
            delay = min(delay * 2, 120.0)
        raise RuntimeError(f"giving up on {url}: {last_err}")

    async def leaderboard(self) -> dict[str, Any]:
        return await self.get_json(BASE_URL)

    async def matches(self, player: str) -> list[dict[str, Any]]:
        data = await self.get_json(f"{BASE_URL}?matches=1&player={quote(player)}")
        return data.get("matches", [])

    async def replay(self, replay_id: int) -> dict[str, Any]:
        return await self.get_json(f"{BASE_URL}?replay={replay_id}")


def make_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT},
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
    )


# --------------------------------------------------------------------------
# Cache / manifest I/O
# --------------------------------------------------------------------------


def replay_cache_path(replay_id: int) -> Path:
    return REPLAYS_DIR / f"{replay_id}.json.zst"


def save_replay(replay_id: int, replay: dict[str, Any]) -> Path:
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    path = replay_cache_path(replay_id)
    raw = json.dumps(replay, separators=(",", ":")).encode()
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(raw))
    tmp.replace(path)
    return path


def load_replay(replay_id: int) -> dict[str, Any]:
    raw = zstd.ZstdDecompressor().decompress(replay_cache_path(replay_id).read_bytes())
    return json.loads(raw)


def load_manifest() -> pa.Table | None:
    if MANIFEST_PATH.exists():
        return pq.read_table(MANIFEST_PATH)
    return None


def write_manifest(table: pa.Table) -> None:
    tmp = MANIFEST_PATH.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(MANIFEST_PATH)


def save_match_list(player: str, matches: list[dict[str, Any]]) -> Path:
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    safe = quote(player, safe="")
    path = MATCHES_DIR / f"{safe}.json"
    path.write_text(json.dumps({"player": player, "matches": matches}, indent=1))
    return path


# --------------------------------------------------------------------------
# High-level operations
# --------------------------------------------------------------------------


async def fetch_replays_for_targets(
    fetcher: Fetcher, targets: list[dict[str, Any]], fetched_at: float
) -> tuple[int, int]:
    """Download targets (skipping cached), update the manifest. -> (downloaded, cached)."""
    new_rows: list[dict[str, Any]] = []
    downloaded = cached = failed = 0
    for i, match in enumerate(targets):
        rid = match_replay_id(match)
        path = replay_cache_path(rid)
        if path.exists():
            cached += 1
            replay = load_replay(rid)  # still (re)build the manifest row
        else:
            try:
                replay = await fetcher.replay(rid)
            except RuntimeError as e:
                # One dead replay must not abandon the other thousands; it stays
                # uncached, so a rerun retries it.
                failed += 1
                print(f"  [{i + 1}/{len(targets)}] SKIP replay {rid}: {e}", file=sys.stderr)
                continue
            save_replay(rid, replay)
            downloaded += 1
            print(f"  [{i + 1}/{len(targets)}] replay {rid} "
                  f"({match.get('a_name')} vs {match.get('b_name')})"
                  f"  [+{downloaded} ok, {cached} cached, {failed} failed,"
                  f" {fetcher.bucket.rate:.2f} rps]")
        new_rows.append(manifest_row(match, replay, fetched_at))
        if len(new_rows) >= MANIFEST_CHECKPOINT_EVERY:
            write_manifest(merge_manifest(load_manifest(), new_rows))
            new_rows = []
    if new_rows:
        write_manifest(merge_manifest(load_manifest(), new_rows))
    if failed:
        print(f"  {failed} replays failed permanently (rerun to retry them)", file=sys.stderr)
    return downloaded, cached


async def fetch_match_lists(
    fetcher: Fetcher, players: list[str]
) -> list[tuple[str, list[dict[str, Any]]]]:
    result: list[tuple[str, list[dict[str, Any]]]] = []
    for player in players:
        matches = await fetcher.matches(player)
        save_match_list(player, matches)
        n_loss = sum(1 for m in matches if match_result_for(m, player) == "loss")
        print(f"  {player}: {len(matches)} matches in window, {n_loss} losses")
        result.append((player, matches))
    return result


# --------------------------------------------------------------------------
# CLI commands
# --------------------------------------------------------------------------


async def cmd_fetch_leaderboard(args: argparse.Namespace) -> None:
    async with make_session() as session:
        fetcher = Fetcher(session, TokenBucket(getattr(args, 'rps', RATE_LIMIT_RPS)))
        data = await fetcher.leaderboard()
    LEADERBOARD_PATH.write_text(json.dumps(data, indent=1))
    board = data.get("leaderboard", [])
    print(f"wrote {LEADERBOARD_PATH} ({len(board)} entries)")
    for i, e in enumerate(board[:30], 1):
        print(f"  #{i:2d} {e['label']:24s} elo={e['elo']} "
              f"{e['wins']}W-{e['losses']}L-{e['draws']}D"
              + (" (provisional)" if e.get("provisional") else ""))


async def cmd_fetch_matches(args: argparse.Namespace) -> None:
    async with make_session() as session:
        fetcher = Fetcher(session, TokenBucket(getattr(args, 'rps', RATE_LIMIT_RPS)))
        await fetch_match_lists(fetcher, [args.player])


async def cmd_fetch_replays(args: argparse.Namespace) -> None:
    players = [p.strip() for p in args.players.split(",") if p.strip()]
    fetched_at = args.fetched_at if args.fetched_at is not None else time.time()
    async with make_session() as session:
        fetcher = Fetcher(session, TokenBucket(getattr(args, 'rps', RATE_LIMIT_RPS)))
        mbp = await fetch_match_lists(fetcher, players)
        targets = select_replay_targets(
            mbp, losses_only=args.losses_only, wins_per_player=args.wins_per_player
        )
        if args.max_replays is not None:
            targets = targets[: args.max_replays]
        print(f"targets: {len(targets)} replays")
        downloaded, cached = await fetch_replays_for_targets(fetcher, targets, fetched_at)
    print(f"done: {downloaded} downloaded, {cached} already cached, "
          f"{fetcher.request_count} HTTP requests")


async def cmd_fetch_all(args: argparse.Namespace) -> None:
    """Priority fetch: losses first (leaderboard order), then the remaining
    games of every player with elo >= min_elo, newest first."""
    fetched_at = args.fetched_at if args.fetched_at is not None else time.time()
    async with make_session() as session:
        fetcher = Fetcher(session, TokenBucket(getattr(args, 'rps', RATE_LIMIT_RPS)))
        board = (await fetcher.leaderboard()).get("leaderboard", [])
        LEADERBOARD_PATH.write_text(json.dumps({"leaderboard": board}, indent=1))
        elig = [e["label"] for e in board if e["elo"] >= args.min_elo]
        print(f"{len(elig)} players with elo >= {args.min_elo}: {elig}")
        mbp = await fetch_match_lists(fetcher, elig)
        # leaderboard order, so the strongest players' losses lead the queue
        targets = select_replay_targets(
            mbp, losses_only=False, wins_per_player=args.wins_per_player
        )
        if args.max_replays is not None:
            targets = targets[: args.max_replays]
        print(f"targets: {len(targets)} replays")
        downloaded, cached = await fetch_replays_for_targets(fetcher, targets, fetched_at)
    print(f"done: {downloaded} downloaded, {cached} already cached, "
          f"{fetcher.request_count} HTTP requests")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fetch.py", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch-leaderboard", help="download leaderboard -> data/leaderboard.json")
    s.add_argument('--rps', type=float, default=RATE_LIMIT_RPS)
    s.set_defaults(func=cmd_fetch_leaderboard)

    s = sub.add_parser("fetch-matches", help="download one player's match list")
    s.add_argument("--player", required=True)
    s.add_argument('--rps', type=float, default=RATE_LIMIT_RPS,
                   help='global request rate; eased automatically on 403/429 throttling')
    s.set_defaults(func=cmd_fetch_matches)

    s = sub.add_parser("fetch-replays", help="download replays for a set of players")
    s.add_argument("--players", required=True, help='comma-separated, e.g. "Alice,Bob"')
    s.add_argument("--losses-only", action="store_true")
    s.add_argument("--wins-per-player", type=int, default=None,
                   help="cap on most-recent wins per player (default: all)")
    s.add_argument("--max-replays", type=int, default=None)
    s.add_argument("--fetched-at", type=float, default=None,
                   help="unix timestamp recorded in manifest (default: now)")
    s.add_argument('--rps', type=float, default=RATE_LIMIT_RPS,
                   help='global request rate; eased automatically on 403/429 throttling')
    s.set_defaults(func=cmd_fetch_replays)

    s = sub.add_parser("fetch-all", help="priority fetch for all players above an elo floor")
    s.add_argument("--min-elo", type=int, required=True)
    s.add_argument("--wins-per-player", type=int, default=None)
    s.add_argument("--max-replays", type=int, default=None)
    s.add_argument("--fetched-at", type=float, default=None)
    s.add_argument('--rps', type=float, default=RATE_LIMIT_RPS,
                   help='global request rate; eased automatically on 403/429 throttling')
    s.set_defaults(func=cmd_fetch_all)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
