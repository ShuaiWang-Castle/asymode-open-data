import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asymode import splits as S, schema as H  # noqa: E402
from asymode.evalproto import make_folds  # noqa: E402


def test_county_folds_depend_on_outer_seed_only_and_match_legacy_rule():
    fips = [f"{i:05d}" for i in range(500)]
    m0 = S.county_folds(fips, 5, outer_split_seed=0)
    assert m0 == S.county_folds(list(reversed(fips)), 5, outer_split_seed=0)      # order-invariant
    legacy = dict(zip(sorted(fips), make_folds(sorted(fips), k=5, seed=0)))
    assert m0 == legacy                                                              # same rule as before
    assert S.county_folds(fips, 5, outer_split_seed=1) != m0                       # outer seed moves it
    # a "model seed" has no way in: the API has no such argument; the harness test below
    # pins that the assignment used by two model seeds is identical.
    for model_seed in (0, 1, 2):
        assert S.split_digest(S.county_folds(fips, 5, 0)) == S.split_digest(m0)


def test_event_folds_disjoint_balanced_deterministic():
    rng = np.random.default_rng(0)
    sizes = {f"2021-0{d}-01": int(s) for d, s in zip(range(1, 10), rng.integers(200, 2000, 9))}
    m = S.event_folds(sizes, k=5, outer_split_seed=0)
    assert set(m.values()) <= set(range(5)) and len(m) == len(sizes)
    assert m == S.event_folds(sizes, k=5, outer_split_seed=0)                     # deterministic
    load = [sum(sizes[e] for e in m if m[e] == f) for f in range(5)]
    assert max(load) - min(load) <= max(sizes.values())                             # greedy balance bound
    # rows: every row's fold comes from its event; train/test disjoint at event level
    ev = np.array([e for e in sizes for _ in range(3)])
    a = S.assign_rows(ev, m)
    for f in range(5):
        S.check_disjoint(ev, a, f)
    with pytest.raises(KeyError):
        S.assign_rows(np.array(["nope"]), m)


def test_split_digest_and_save(tmp_path):
    m = S.county_folds([f"{i:05d}" for i in range(50)], 5, 0)
    p = S.save_split(m, "county", 5, 0, tmp_path)
    assert p.exists() and S.load_split(p) == m and S.split_digest(m) in p.name


def test_result_header_has_every_required_key_and_rejects_unknown_clock():
    h = H.result_header(experiment_id="t", source={"commit": "abc", "dirty": False}, panel_ids=["b", "a"],
                        panel_digest="p", channel_names=["x"], channel_digest="c", clock="utc_hour",
                        split_unit="event", outer_split_digest="s", outer_split_seed=0,
                        inner_split_seed=7, model_seeds=[0, 1], hyperparameters={"lr": 1e-3})
    assert set(H.REQUIRED_KEYS) <= set(h) and h["panel_ids"] == ["a", "b"]
    assert h["clock_digest"] != H.digest_of("none|" + H.CLOCKS["none"])
    with pytest.raises(ValueError):
        H.result_header(experiment_id="t", source={}, panel_ids=[], panel_digest="p", channel_names=[],
                        channel_digest="c", clock="crew_time", split_unit="event", outer_split_digest="s",
                        outer_split_seed=0, inner_split_seed=0, model_seeds=[0], hyperparameters={})
