#!/usr/bin/env python3
"""POST-HOC sensitivity run (not in PLAN_TROPICAL.md): identical to tropical_run.py except a longer budget for all three
models (6,000 updates, patience 2,000), written to results/fits_long. Reason: in the registered run NET's best update
was 0 in fold 4 (all seeds) and 2,100-2,900 of 3,000 elsewhere, so NET may be under-trained by the registered budget."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tropical_run as TR  # noqa: E402
TR.UPDATES, TR.PATIENCE, TR.OUT = 6000, 2000, HERE / 'results' / 'fits_long'
if __name__ == '__main__':
    TR.main()
