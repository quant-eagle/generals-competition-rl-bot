from data import import_sprint as imp


def test_stable_ids_are_order_independent_and_int32():
    """Ids are content-addressed, not positional: a replay's id depends on the
    replay alone, so growing the selection never renumbers what is already
    integrated.  Pins the property, not literal values."""
    ids = imp.stable_replay_ids([{"url": "z"}, {"url": "a"}])
    assert ids == imp.stable_replay_ids([{"url": "a"}, {"url": "z"}]), \
        "id depended on input order"
    assert ids == {k: v for k, v in
                   imp.stable_replay_ids([{"url": "z"}, {"url": "a"},
                                          {"url": "m"}]).items()
                   if k in ("a", "z")}, "adding a URL renumbered the others"
    assert len(set(ids.values())) == len(ids), "ids collided"
    base, span = imp.SPRINT_ID_BASE, imp.SPRINT_ID_SPAN
    assert all(base <= v < base + span for v in ids.values()), \
        "id escaped the reserved sprint band"
    assert all(-(2**31) <= v < 2**31 for v in ids.values())


def test_existing_manifest_player_order():
    row = {"player_a": "A", "player_b": "B", "a_side": 1}
    assert imp.ordered_players_from_manifest(row) == ("B", "A")
    assert imp.replay_key(3, ("B", "A")) == (3, "B", "A")
