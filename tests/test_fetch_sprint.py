import gzip
import json

import pytest

from data import fetch_sprint as fs


URL = ("https://vw73zsxrbe113bh0.public.blob.vercel-storage.com/"
       "sprint-replays/aaa%7Cbbb/123-0.json.gz")


def _match(**extra):
    row = {"p0_name": "BotA", "p1_name": "Other", "seed": 123,
           "a_side": 0, "winner": "p0", "turns": 2,
           "suspect": False, "forfeit": False, "replay_gz": URL}
    row.update(extra)
    return row


def test_selection_filters_and_deduplicates():
    good = _match()
    duplicate = _match()
    suspect = _match(seed=124, suspect=True,
                     replay_gz=URL.replace("123-0", "124-0"))
    irrelevant = _match(p0_name="A", p1_name="B", seed=125,
                        replay_gz=URL.replace("123-0", "125-0"))
    manifest = {"matches": [good, duplicate, suspect, irrelevant]}
    assert fs.select_targets(manifest, ("BotA", "BotB")) == [good]


def test_replay_path_is_strictly_confined():
    assert fs.replay_relative_path(URL).as_posix() == "aaa|bbb/123-0.json.gz"
    with pytest.raises(ValueError):
        fs.replay_relative_path(URL.replace(fs.BLOB_HOST, "example.com"))
    with pytest.raises(ValueError):
        fs.replay_relative_path(URL.replace("aaa%7Cbbb", "..%2F.."))


def test_budget_stops_before_limits_are_exceeded():
    budget = fs.Budget(max_requests=1, max_bytes=10)
    budget.begin_request()
    with pytest.raises(fs.SafetyStop):
        budget.begin_request()
    budget.add_bytes(6, 0, 8)
    with pytest.raises(fs.SafetyStop):
        budget.add_bytes(5, 6, 8)


def test_replay_validation():
    replay = {"dims": {"rows": 18, "cols": 18},
              "players": ["BotA", "Other"], "seed": 123,
              "winner": 0, "total_ticks": 2, "ticks": [{}, {}, {}]}
    payload = gzip.compress(json.dumps(replay).encode())
    assert fs.validate_replay(payload, _match())["winner"] == 0
    with pytest.raises(ValueError):
        fs.validate_replay(payload, _match(seed=999))
