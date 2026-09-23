"""Phase 10 gate on any features.py change: a rebuild of 2002-2025 through the
UNCHANGED entry point ``features.build_game_features()`` (defaults: 2002-2025,
k=4) must be byte-identical (same sha256) to the canonical files.

Writes only to ``data/live_scratch/canonical_rebuild/`` (gitignored); the
canonical ``data/processed/`` files are read, never written (``PROC_DIR`` is
redirected at runtime, as in Phase 9).

Usage::

    python -m scripts.phase10_canonical_rebuild
"""

from __future__ import annotations

import hashlib
import sys

from src import features as F

CANON = F.PROC_DIR
OUT = F.PROJECT_ROOT / "data" / "live_scratch" / "canonical_rebuild"
FILES = ("game_features.parquet", "team_game_log.parquet")


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    before = {f: sha(CANON / f) for f in FILES}
    print(f"features.py sha256 {sha(F.PROJECT_ROOT / 'src' / 'features.py')}")
    F.PROC_DIR = OUT
    F.build_game_features()
    ok = True
    for f in FILES:
        got, want = sha(OUT / f), before[f]
        ok &= got == want
        print(f"{'MATCH' if got == want else 'DIFFER'}  {f}\n  canonical {want}\n  rebuild   {got}")
    after = {f: sha(CANON / f) for f in FILES}
    assert after == before, "canonical files changed during the rebuild"
    print("\nbyte-identical: features.py change is safe for 2002-2025" if ok
          else "\n!! NOT byte-identical -- revert the features.py change")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
