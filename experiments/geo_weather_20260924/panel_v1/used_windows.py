"""Windows whose outcomes were read in this repository before DATASET_DESIGN v1 (section 4.2): every event listed in
the event files of the earlier panels, and the panels of the main line (data/interim/panel_<date>.npz, only their
county list and timestamps are read). Output: data_provenance/used_windows.json."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]


def main() -> None:
    out = []
    files = sorted(glob.glob(str(ROOT / "experiments" / "open_gcrk_20260919" / "selected_events*.json"))
                   + glob.glob(str(EXP / "selected_events*.json")))
    for f in files:
        d = json.loads(Path(f).read_text())
        for e in (d["events"] if isinstance(d, dict) else d):
            fp = e.get("footprint_fips") or []
            out.append(dict(source=str(Path(f).relative_to(ROOT)), event=e["event"],
                            window_start=str(pd.Timestamp(e["window_start_utc"])),
                            window_end=str(pd.Timestamp(e.get("window_end_utc") or pd.Timestamp(e["window_start_utc"])
                                                        + pd.Timedelta(hours=215))),
                            fips=sorted(str(x).zfill(5) for x in fp)))
    for f in sorted(glob.glob(str(ROOT / "data" / "interim" / "panel_*.npz"))):
        z = np.load(f, allow_pickle=True)
        ts = pd.to_datetime(z["ts"])
        out.append(dict(source=str(Path(f).relative_to(ROOT)), event=Path(f).stem.replace("panel_", ""),
                        window_start=str(ts.min()), window_end=str(ts.max()),
                        fips=sorted(str(x).zfill(5) for x in z["fips"])))
    seen, uniq = set(), []
    for w in out:
        k = (w["event"], w["window_start"], tuple(w["fips"]))
        if k not in seen:
            seen.add(k)
            uniq.append(w)
    (EXP / "data_provenance" / "used_windows.json").write_text(json.dumps(uniq, indent=0) + "\n")
    print("used windows", len(uniq), "from", len({w['source'] for w in uniq}), "files;",
          "empty footprints:", sum(not w["fips"] for w in uniq))


if __name__ == "__main__":
    main()
