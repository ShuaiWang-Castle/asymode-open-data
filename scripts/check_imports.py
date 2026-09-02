"""Import every experiment module the way the harness does, and fail loudly.

Run after any change to a shared module (`src/asymode/*`, `experiments/exp05_*`).
A signature change is visible to grep; a decorator that resolves `sys.modules`
at import time is not -- a `@dataclass` added to a module that downstream files
load via `spec_from_file_location` without registering it in `sys.modules`
broke two experiments at import, invisibly, while the processes already running
were unaffected. Only importing catches that class of fault.

Pair with:  grep -l "from exp05_real_dynamics import\|spec_from_file_location" experiments/*.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))


def main() -> int:
    files = sorted((ROOT / "experiments").glob("*.py"))
    failed = []
    for f in files:
        name = f.stem
        try:
            spec = importlib.util.spec_from_file_location(name, f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            print(f"  OK   {name}")
        except Exception as e:                      # noqa: BLE001 -- report, don't mask
            failed.append(name)
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(files) - len(failed)}/{len(files)} experiment modules import cleanly")
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
