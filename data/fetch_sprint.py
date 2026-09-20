"""Cost-bounded downloader for the public sprint replay archive.

The command is a dry run unless ``--download`` is supplied.  It deliberately
uses no login cookie, makes no HEAD requests, never retries automatically, and
stops immediately on server throttling.  Existing valid files are never
requested again.

Examples:
    python -m data.fetch_sprint --players "Alice,Bob"
    python -m data.fetch_sprint --players "Alice,Bob" --download --rps 1

The archive is hosted on someone else's blob storage, so the hard request and
byte ceilings are set far below one dollar of hosting cost even if every
request is an uncached read.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


MANIFEST_URL = "https://www.generals.bot/assets/sprint-2026-08-08.json"
BLOB_HOST = "vw73zsxrbe113bh0.public.blob.vercel-storage.com"
DATA_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = DATA_DIR / "sprint" / "index.json"
DEFAULT_OUTPUT = DATA_DIR / "sprint" / "replays"

# Safety limits.  They cannot be raised from the CLI.
HARD_MAX_REQUESTS = 1_000
HARD_MAX_BYTES = 256 * 1024 * 1024
HARD_MAX_FILE_BYTES = 8 * 1024 * 1024
HARD_MAX_RPS = 5.0
REQUEST_TIMEOUT_S = 60
USER_AGENT = "generals-zero-replay-research/1.0 (competition participant)"

# Deliberately pessimistic accounting guard: both rates are several times the
# host's published transfer and read prices, leaving room for CDN/origin costs.
ACCOUNTING_DOLLARS_PER_GIB = 0.50
ACCOUNTING_DOLLARS_PER_MILLION_REQUESTS = 1.00

_PAIR_RE = re.compile(r"^[A-Za-z0-9_-]+\|[A-Za-z0-9_-]+$")
_FILE_RE = re.compile(r"^[0-9]+-[01]\.json\.gz$")


class SafetyStop(RuntimeError):
    """Raised before a configured cost or etiquette limit can be exceeded."""


@dataclass
class Budget:
    max_requests: int = HARD_MAX_REQUESTS
    max_bytes: int = HARD_MAX_BYTES
    requests: int = 0
    bytes: int = 0

    def begin_request(self) -> None:
        if self.requests >= self.max_requests:
            raise SafetyStop(f"request ceiling reached ({self.max_requests})")
        self.requests += 1

    def check_content_length(self, size: int, per_file_limit: int) -> None:
        if size > per_file_limit:
            raise SafetyStop(f"response is {size} bytes; per-file ceiling is {per_file_limit}")
        if self.bytes + size > self.max_bytes:
            raise SafetyStop(
                f"response would exceed total byte ceiling ({self.max_bytes})")

    def add_bytes(self, size: int, per_file_bytes: int, per_file_limit: int) -> None:
        if per_file_bytes + size > per_file_limit:
            raise SafetyStop(f"stream exceeded per-file ceiling ({per_file_limit})")
        if self.bytes + size > self.max_bytes:
            raise SafetyStop(f"stream exceeded total byte ceiling ({self.max_bytes})")
        self.bytes += size

    @property
    def conservative_cost_usd(self) -> float:
        return (self.bytes / 1024**3 * ACCOUNTING_DOLLARS_PER_GIB +
                self.requests / 1_000_000 * ACCOUNTING_DOLLARS_PER_MILLION_REQUESTS)


class Client:
    def __init__(self, budget: Budget, rps: float):
        if not 0 < rps <= HARD_MAX_RPS:
            raise ValueError(f"rps must be in (0, {HARD_MAX_RPS}]")
        self.budget = budget
        self.interval = 1.0 / rps
        self.last_start = 0.0

    def get(self, url: str, *, allowed_host: str,
            per_file_limit: int = HARD_MAX_FILE_BYTES) -> bytes:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != allowed_host:
            raise SafetyStop(f"refusing unexpected URL: {url}")
        delay = self.last_start + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.budget.begin_request()
        self.last_start = time.monotonic()
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        try:
            response = urlopen(req, timeout=REQUEST_TIMEOUT_S)
        except HTTPError as exc:
            if exc.code in (403, 429):
                raise SafetyStop(
                    f"server returned {exc.code}; stopping without retry") from exc
            raise SafetyStop(f"HTTP {exc.code}; stopping without retry") from exc
        except URLError as exc:
            raise SafetyStop(f"network error; stopping without retry: {exc}") from exc

        with response:
            final = urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != allowed_host:
                raise SafetyStop(f"refusing redirect to {response.geturl()}")
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None:
                self.budget.check_content_length(int(raw_length), per_file_limit)
            chunks: list[bytes] = []
            file_bytes = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.budget.add_bytes(len(chunk), file_bytes, per_file_limit)
                chunks.append(chunk)
                file_bytes += len(chunk)
        return b"".join(chunks)


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


def load_manifest(path: Path, client: Client) -> dict[str, Any]:
    if path.exists():
        raw = path.read_bytes()
    else:
        raw = client.get(
            MANIFEST_URL, allowed_host="www.generals.bot",
            per_file_limit=HARD_MAX_FILE_BYTES)
        atomic_write(path, raw)
    manifest = json.loads(raw)
    if manifest.get("schema") != "sprint-results/1" or not isinstance(manifest.get("matches"), list):
        raise ValueError("unexpected sprint manifest schema")
    return manifest


def replay_relative_path(url: str) -> Path:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != BLOB_HOST:
        raise ValueError(f"unexpected replay host: {url}")
    parts = PurePosixPath(unquote(parsed.path)).parts
    try:
        marker = parts.index("sprint-replays")
    except ValueError as exc:
        raise ValueError(f"unexpected replay path: {url}") from exc
    rel = parts[marker + 1:]
    if len(rel) != 2 or not _PAIR_RE.fullmatch(rel[0]) or not _FILE_RE.fullmatch(rel[1]):
        raise ValueError(f"unsafe replay path: {url}")
    return Path(rel[0]) / rel[1]


def select_targets(manifest: dict[str, Any], players: tuple[str, ...],
                   include_suspect: bool = False) -> list[dict[str, Any]]:
    wanted = set(players)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for match in manifest["matches"]:
        if match.get("p0_name") not in wanted and match.get("p1_name") not in wanted:
            continue
        if not include_suspect and (match.get("suspect") or match.get("forfeit")):
            continue
        url = match.get("replay_gz")
        if not isinstance(url, str) or url in seen:
            continue
        replay_relative_path(url)  # validate every URL before any network activity
        seen.add(url)
        out.append(match)
    return out


def validate_replay(payload: bytes, match: dict[str, Any]) -> dict[str, Any]:
    try:
        replay = json.loads(gzip.decompress(payload))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid gzip replay") from exc
    required = ("dims", "players", "ticks", "seed", "winner")
    if any(key not in replay for key in required):
        raise ValueError("replay is missing required fields")
    if int(replay["seed"]) != int(match["seed"]):
        raise ValueError("replay seed does not match manifest")
    if set(replay["players"]) != {match["p0_name"], match["p1_name"]}:
        raise ValueError("replay players do not match manifest")
    if len(replay["ticks"]) != int(replay.get("total_ticks", -1)) + 1:
        raise ValueError("replay tick count is inconsistent")
    return replay


def cached_ok(path: Path, match: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    validate_replay(path.read_bytes(), match)
    return True


def write_selection(output: Path, manifest_path: Path,
                    targets: list[dict[str, Any]]) -> None:
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = [{
        "local": str(replay_relative_path(m["replay_gz"])),
        "url": m["replay_gz"],
        "seed": m["seed"], "a_side": m.get("a_side"),
        "p0_name": m["p0_name"], "p1_name": m["p1_name"],
        "winner": m.get("winner"), "turns": m.get("turns"),
        "suspect": bool(m.get("suspect")), "forfeit": bool(m.get("forfeit")),
    } for m in targets]
    payload = json.dumps({"manifest_sha256": manifest_sha, "replays": rows},
                         indent=2, sort_keys=True).encode()
    atomic_write(output / "selection.json", payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--players", required=True,
                        help="comma-separated player names whose games to select")
    parser.add_argument("--include-suspect", action="store_true")
    parser.add_argument("--download", action="store_true",
                        help="actually download missing files; otherwise dry-run")
    parser.add_argument("--rps", type=float, default=1.0,
                        help=f"serial request rate, maximum {HARD_MAX_RPS}")
    parser.add_argument("--max-downloads", type=int, default=HARD_MAX_REQUESTS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0 <= args.max_downloads <= HARD_MAX_REQUESTS:
        raise SystemExit(f"--max-downloads must be in [0, {HARD_MAX_REQUESTS}]")
    budget = Budget()
    client = Client(budget, args.rps)
    manifest = load_manifest(args.manifest, client)
    players = tuple(x.strip() for x in args.players.split(",") if x.strip())
    if not players:
        raise SystemExit("--players must not be empty")
    targets = select_targets(manifest, players, args.include_suspect)

    cached = 0
    missing: list[dict[str, Any]] = []
    for match in targets:
        path = args.output / replay_relative_path(match["replay_gz"])
        if cached_ok(path, match):
            cached += 1
        else:
            missing.append(match)
    write_selection(args.output, args.manifest, targets)

    print(f"manifest matches: {len(manifest['matches'])}")
    print(f"selected unique replays: {len(targets)} ({cached} cached, {len(missing)} missing)")
    print(f"hard ceilings: {HARD_MAX_REQUESTS} requests, "
          f"{HARD_MAX_BYTES / 1024**2:.0f} MiB, {HARD_MAX_RPS:.0f} req/s")
    if not args.download:
        print("dry-run only; pass --download to fetch missing replays")
        return

    todo = missing[:args.max_downloads]
    for i, match in enumerate(todo, 1):
        raw = client.get(match["replay_gz"], allowed_host=BLOB_HOST)
        validate_replay(raw, match)
        path = args.output / replay_relative_path(match["replay_gz"])
        atomic_write(path, raw)
        if i == 1 or i % 25 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {budget.bytes / 1024**2:.2f} MiB, "
                  f"{budget.requests} requests, "
                  f"conservative cost <= ${budget.conservative_cost_usd:.4f}")
    print(f"done: {len(todo)} downloaded, {cached} cached; "
          f"{budget.bytes / 1024**2:.2f} MiB in {budget.requests} requests; "
          f"conservative cost <= ${budget.conservative_cost_usd:.4f}")


if __name__ == "__main__":
    main()
