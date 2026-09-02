"""A3 items 9, 10, 12 against the exp05 harness functions (no data needed)."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import exp05_real_dynamics as E  # noqa: E402


def _rows(n_panels=6, n_counties=40, origins=3):
    fips = np.array([f"{c:05d}" for _ in range(n_panels) for _ in range(origins) for c in range(n_counties)])
    panel = np.array([f"2021-0{p+1}-01" for p in range(n_panels) for _ in range(origins) for _ in range(n_counties)])
    return fips, panel


# 9. outer assignment does not change with the model seed ------------------------
@pytest.mark.parametrize("unit", ["event", "county"])
def test_outer_assignment_invariant_to_model_seed(unit):
    fips, panel = _rows()
    outs = []
    for model_seed in (0, 1, 2):
        np.random.seed(model_seed)                     # the only thing a model seed may touch
        assign, mapping, digest, _ = E.outer_assignment(fips, panel, unit, 5, outer_split_seed=0)
        outs.append((assign.tolist(), digest))
    assert outs[0] == outs[1] == outs[2]
    a1, _, d1, _ = E.outer_assignment(fips, panel, unit, 5, outer_split_seed=1)
    assert d1 != outs[0][1]                            # the outer seed does move it


def test_event_split_has_no_event_on_both_sides():
    fips, panel = _rows()
    assign, mapping, _, units = E.outer_assignment(fips, panel, "event", 5, 0)
    for f in range(5):
        te = set(units[assign == f]); tr = set(units[assign != f])
        assert not (te & tr)
        assert te                                      # every fold holds out at least one event


# 10. clock follows the timestamp, does not reset at the origin --------------------
def test_utc_clock_uses_timestamp_and_differs_between_origins():
    n, T, F = 3, 48, 2
    X = np.zeros((n, T, F), np.float32)
    t0 = np.array([5, 17, 5])                          # samples 0 and 2 start at 05 UTC, sample 1 at 17 UTC
    out = E.add_context(X, None, T, t0_hour=t0, clock="utc_hour")
    assert out.shape == (n, T, F + 2)
    for i in range(n):
        h = (t0[i] + np.arange(T)) % 24
        assert np.allclose(out[i, :, F], np.sin(2 * np.pi * h / 24), atol=1e-6)
        assert np.allclose(out[i, :, F + 1], np.cos(2 * np.pi * h / 24), atol=1e-6)
    assert not np.allclose(out[0, :, F], out[1, :, F])           # different origins, different phase
    assert np.allclose(out[0, :, F], out[2, :, F])               # same hour of day, same phase
    old = E.add_context(X, None, T, clock="lead_phase_old")
    assert np.allclose(old[0, :, F], old[1, :, F])               # the legacy channel ignores the origin
    assert E.add_context(X, None, T, clock="none").shape[-1] == F
    with pytest.raises(ValueError):
        E.add_context(X, None, T, clock="utc_hour")               # needs t0_hour


# 12. OOF store: every sample predicted exactly once per model seed --------------
def test_oof_stash_covers_every_sample_once_per_seed():
    n, seeds, n_h, k = 50, [0, 1], 4, 5
    assign = np.arange(n) % k
    oof = {}
    for s_i, seed in enumerate(seeds):
        for f in range(k):
            te = np.where(assign == f)[0]
            E._stash(oof, "arm", seed, f, te, np.full((len(te), n_h), 0.5, np.float32), n, seeds, n_h)
    st = oof["arm"]
    assert st["fold_of"].shape == (len(seeds), n) and (st["fold_of"] >= 0).all()
    assert np.array_equal(st["fold_of"][0], assign) and np.array_equal(st["fold_of"][1], assign)
    assert np.isfinite(st["pred"]).all()
