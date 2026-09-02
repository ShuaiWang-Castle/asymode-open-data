"""A3 item 11: the comparability checker must refuse mismatched split, clock,
panel, channel, mask or metric digests, and accept identical ones."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = {"panel_digest": "p", "channel_digest": "c", "clock_digest": "k", "split_unit": "event",
        "outer_split_digest": "s", "mask_digest": "m", "metric_digest": "q", "horizon": 48,
        "stride": 12, "k": 5, "seeds": [0, 1, 2], "horizons": [1, 6, 24, 48]}


def _run(tmp_path, a, b):
    fa, fb = tmp_path / "a.json", tmp_path / "b.json"
    fa.write_text(json.dumps({"config": a, "rows": []})); fb.write_text(json.dumps({"config": b, "rows": []}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/check_comparable.py"), str(fa), str(fb)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def test_identical_configs_are_comparable(tmp_path):
    rc, _ = _run(tmp_path, BASE, dict(BASE))
    assert rc == 0


def test_each_digest_mismatch_fails_closed(tmp_path):
    for key in ["panel_digest", "channel_digest", "clock_digest", "split_unit", "outer_split_digest",
                "mask_digest", "metric_digest"]:
        b = dict(BASE); b[key] = "DIFFERENT"
        rc, out = _run(tmp_path, BASE, b)
        assert rc != 0, key
        assert key in out


def test_legacy_file_without_v2_digests_is_not_comparable_to_v2(tmp_path):
    legacy = {k: v for k, v in BASE.items() if k in ("panel_digest", "channel_digest", "horizon", "stride", "k", "seeds", "horizons")}
    rc, out = _run(tmp_path, BASE, legacy)
    assert rc != 0 and "clock_digest" in out
