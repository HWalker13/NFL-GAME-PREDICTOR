"""Phase 9 (owner-requested): preserve the pre-season 2026 schedule as a snapshot.

``data/raw/schedules_2026.parquet`` was pulled on 2026-09-08 by nfl_data_py
0.3.3 and is the only remaining record of early-September 2026 lines (nflverse
overwrites lines in place). The first ``--refresh-season 2026`` overwrites it,
so this script COPIES it, byte for byte, into ``data/raw/snapshots/`` under the
standard name ``schedules_2026_<UTC>.parquet``. The original is only read.

Pull time: from ``pull_metadata.json`` if the file has an entry, otherwise the
file's mtime, recorded as ``pulled_at_precision: "approximate (mtime)"``.

Publishing uses the same no-overwrite rule as ``src.data_ingest``: temp copy,
then ``os.link`` (fails if the target exists), then read-only. Safe to re-run:
it refuses if the snapshot already exists.

Usage (project root)::

    python -m scripts.phase9_preserve_sep8_snapshot
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import stat

import pandas as pd

from src import data_ingest as DI

SEASON = 2026


def main() -> None:
    src = DI._raw_path("schedules", SEASON)
    meta = DI.load_metadata()
    if src.name in meta:
        pulled = dt.datetime.strptime(meta[src.name]["pulled_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
        pulled = pulled.replace(tzinfo=dt.timezone.utc)
        precision, library, lib_version = ("exact (pull_metadata)", meta[src.name]["library"],
                                           meta[src.name]["library_version"])
    else:
        pulled = dt.datetime.fromtimestamp(src.stat().st_mtime, tz=dt.timezone.utc)
        precision = "approximate (mtime)"
        # Provenance per docs/PHASE7_REVIEW.md 1.1: pulled 2026-09-08 via nfl_data_py.
        library, lib_version = "nfl_data_py", "0.3.3"

    snap_dir = DI._snapshot_dir()
    snap_dir.mkdir(parents=True, exist_ok=True)
    final = snap_dir / f"schedules_{SEASON}_{pulled.strftime('%Y%m%dT%H%M%SZ')}.parquet"
    if final.exists():
        raise FileExistsError(f"{final.name} already exists -- snapshots are never overwritten")

    src_sha = DI._sha256(src)
    tmp = snap_dir / f".{final.name}.tmp"
    shutil.copyfile(src, tmp)
    try:
        os.link(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    final.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    assert DI._sha256(final) == src_sha, "copy is not byte-identical"
    assert DI._sha256(src) == src_sha, "original changed"

    rows = len(pd.read_parquet(final))
    meta = DI.load_metadata()
    meta[f"{DI.SNAPSHOT_SUBDIR}/{final.name}"] = {
        "pulled_at_utc": pulled.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pulled_at_precision": precision,
        "library": library,
        "library_version": lib_version,
        "rows": rows,
        "sha256": src_sha,
        "snapshot_of": src.name,
        "note": "byte-identical copy of the pre-Phase 9 cache file, preserved before the first "
                "--refresh-season 2026 overwrites it (only record of early-Sept 2026 lines)",
    }
    tmp_meta = DI.RAW_DIR / f".{DI.METADATA_FILE}.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.replace(tmp_meta, DI.RAW_DIR / DI.METADATA_FILE)
    print(f"{src.name} -> {DI.SNAPSHOT_SUBDIR}/{final.name}  ({rows} rows, sha256 {src_sha[:12]}..., "
          f"{precision})")


if __name__ == "__main__":
    main()
