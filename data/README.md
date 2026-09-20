# data/

The replay pipeline: public competition replays in, engine-verified actions out.
Replay data itself is not distributed with this repository; everything here is
the code that fetches, validates and decodes it.

## Why a pipeline at all

Competition replays store the full board state at every tick and **no actions**.
Anything that needs actions or a player's fogged view (mid-game seed positions
for the training curriculum, play-style statistics) has to reconstruct them, and
a reconstruction that is "roughly right" silently corrupts everything built on
top of it. So the rule throughout is: game logic is never reimplemented. Every
inferred action is accepted only if stepping the real engine with it reproduces
the next recorded tick bit-exactly.

## Stages

| step | module | does |
|---|---|---|
| fetch | `fetch.py` | ladder leaderboard, per-player match lists and replays → `replays/{id}.json.zst` + `manifest.parquet` |
| fetch | `fetch_sprint.py` | tournament replay archive → `sprint/replays/` (staging) |
| validate + import | `import_sprint.py` | bit-exact engine gate over staged replays, then `--integrate` into `replays/` + manifest |
| decode | `decode.py` | state sequences → per-tick action pairs, castle masks, exact fog views, per-player sequence shards |
| index | `build_manifest.py` | rebuild `manifest.parquet` from the replay cache |
| scale out | `emit_parallel.py` | run shard emission across worker processes |

All are run as modules from the repository root, e.g. `python -m data.fetch --help`.

### Decoding

For each transition `t → t+1`, `decode.py` diffs the two states net of the
engine's scheduled growth, enumerates the candidate moves and builds that could
explain the touched cells, and steps the engine over candidate pairs until one
reproduces tick `t+1` exactly. Castle positions are hidden state (ticks carry
only armies and owners), so builds are tracked through the decode and ambiguous
cases are resolved by one-step lookahead. `validate()` then replays the whole
inferred action sequence from the initial state; a replay is used only if every
tick and the final winner match.

### Polite fetching

Both fetchers are written to be cheap for the people hosting the data.

- `fetch.py` shares one token bucket across all requests (≤ 4 req/s), backs off
  exponentially with jitter, and on a throttle response sits out a cooldown
  window instead of retrying into it. Cached replays are never re-requested.
- `fetch_sprint.py` is a **dry run unless `--download` is passed**. It has hard
  ceilings on requests, total bytes, per-file bytes and request rate that cannot
  be raised from the command line, makes no HEAD requests, never retries
  automatically, refuses redirects to unexpected hosts, and stops at the first
  throttle response. It tracks a deliberately pessimistic cost estimate as it
  goes. Keep it that way: it is someone else's bandwidth bill.
