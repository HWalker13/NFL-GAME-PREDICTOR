"""Data ingestion for the NFL win/loss predictor (SPEC Section 3; Phase 9).

Thin wrapper around ``nflreadpy.load_pbp`` and ``nflreadpy.load_schedules``
(migrated from the deprecated ``nfl_data_py`` in Phase 9 after a full
2002-2025 equivalence test -- see ``docs/PHASE9_DATA_LAYER.md``). nflreadpy
returns polars; conversion to pandas happens here, at the ingest boundary
(``src.nflreadpy_boundary``), and reproduces the dtypes nfl_data_py wrote, so
nothing downstream of ``data/raw/`` changes. Every raw pull is cached to
``data/raw/`` as Parquet.

Historical range per SPEC Section 3.4: **2002 to present** (the modern 32-team
era). 1999-2001 are deliberately excluded and flagged in the SPEC as an open
question.

Cache rules (Phase 9)
---------------------
* **Completed seasons** (every season before nflreadpy's current season) are
  pulled once and never re-downloaded unless ``--force`` is given explicitly.
* **The in-progress season** is refreshed only on demand, with
  ``--refresh-season <season>``, which re-pulls that season's pbp and schedule
  and touches no other file.
* **Schedule snapshots.** Every pull of the in-progress season's schedule also
  writes an immutable copy to
  ``data/raw/snapshots/schedules_<season>_<UTC timestamp>.parquet``. A
  snapshot is never overwritten: it is published with an atomic hard link that
  fails if the name exists, then made read-only. Snapshots are how Phase 10
  captures the Vegas line available at prediction time (the Phase 7
  line-timing caveat). They live in a subdirectory, so
  ``evaluate.load_schedules`` (a non-recursive ``schedules_*.parquet`` glob)
  never sees them.
* **Pull metadata.** Every file this module writes gets an entry in
  ``data/raw/pull_metadata.json``: UTC pull time, library + version, rows,
  sha256. Files pulled before Phase 9 (nfl_data_py 0.3.3, 2026-09-08) have no
  entry; ``--status`` reports them as legacy.

Usage (project root, venv active)::

    python -m src.data_ingest                        # fill any missing season files
    python -m src.data_ingest --refresh-season 2026  # re-pull the in-progress season only
    python -m src.data_ingest --status               # list files, metadata, snapshots
    python -m src.data_ingest --start 2010 --end 2012 --force   # re-download completed seasons

This module does **no** feature engineering (SPEC Section 5 / Phase 2). It only
fetches and caches.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import stat
from importlib.metadata import version as _pkg_version
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - import shape depends on install
    import nflreadpy
    from nflreadpy.config import update_config
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "nflreadpy is not installed. Activate the venv and "
        "`pip install -r requirements.txt`."
    ) from exc

from src.nflreadpy_boundary import pbp_to_raw_contract, schedules_to_raw_contract

# Always download: a refresh or snapshot must never be served from nflreadpy's
# own (default 24h, in-memory) cache. data/raw/ is the only cache.
update_config(cache_mode="off")

LIBRARY = "nflreadpy"
LIBRARY_VERSION = _pkg_version("nflreadpy")

# SPEC Section 3.4: modern 32-team era starts in 2002.
START_YEAR = 2002
# "to present" -- the NFL season in progress (nflverse's own rule: the new
# season starts the Thursday after Labor Day), not the calendar year, so a
# January run doesn't reach for a season that hasn't started.
END_YEAR = nflreadpy.get_current_season()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SNAPSHOT_SUBDIR = "snapshots"
METADATA_FILE = "pull_metadata.json"

KINDS = ("pbp", "schedules")


# --------------------------------------------------------------------------- #
# Paths, time, metadata
# --------------------------------------------------------------------------- #
def _raw_path(kind: str, year: int) -> Path:
    return RAW_DIR / f"{kind}_{year}.parquet"


def _snapshot_dir() -> Path:
    return RAW_DIR / SNAPSHOT_SUBDIR


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def current_season() -> int:
    """The in-progress season. Every earlier season counts as completed."""
    return nflreadpy.get_current_season()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata() -> dict:
    p = RAW_DIR / METADATA_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def _record_metadata(path: Path, pulled_at: _dt.datetime, rows: int, **extra) -> None:
    meta = load_metadata()
    meta[path.relative_to(RAW_DIR).as_posix()] = {
        "pulled_at_utc": pulled_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "library": LIBRARY,
        "library_version": LIBRARY_VERSION,
        "rows": int(rows),
        "sha256": _sha256(path),
        **extra,
    }
    tmp = RAW_DIR / f".{METADATA_FILE}.tmp"
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.replace(tmp, RAW_DIR / METADATA_FILE)


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp name, then rename, so a crash never leaves a torn cache file."""
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Fetching (the only functions that touch the network)
# --------------------------------------------------------------------------- #
def _fetch(kind: str, year: int) -> pd.DataFrame | None:
    """Download one season via nflreadpy, converted to the raw-file contract.

    Returns ``None`` if the season has no data (future season / not started).
    """
    try:
        if kind == "pbp":
            df = pbp_to_raw_contract(nflreadpy.load_pbp([year]))
        elif kind == "schedules":
            df = schedules_to_raw_contract(nflreadpy.load_schedules([year]))
        else:  # pragma: no cover - programming error
            raise ValueError(f"unknown dataset kind: {kind!r}")
    except (ValueError, ConnectionError) as exc:
        print(f"  [{kind} {year}] no data available ({exc.__class__.__name__}: {exc})")
        return None
    if df is None or df.empty:
        print(f"  [{kind} {year}] returned empty - skipping (season not started?)")
        return None
    return df


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
def write_schedule_snapshot(df: pd.DataFrame, season: int, pulled_at: _dt.datetime) -> Path:
    """Immutable copy of an in-progress season's schedule. Never overwrites.

    Published with ``os.link`` (atomic, fails with ``FileExistsError`` if the
    target exists), then chmod'ed read-only.
    """
    snap_dir = _snapshot_dir()
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = pulled_at.strftime("%Y%m%dT%H%M%SZ")
    final = snap_dir / f"schedules_{season}_{ts}.parquet"
    if final.exists():
        raise FileExistsError(f"snapshot {final.name} already exists -- snapshots are never overwritten")
    tmp = snap_dir / f".{final.name}.tmp"
    df.to_parquet(tmp, index=False)
    try:
        os.link(tmp, final)   # atomic create-if-absent
    finally:
        tmp.unlink(missing_ok=True)
    final.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    _record_metadata(final, pulled_at, len(df), snapshot_of=f"schedules_{season}.parquet")
    print(f"  [schedules {season}] snapshot    -> {SNAPSHOT_SUBDIR}/{final.name}")
    return final


# --------------------------------------------------------------------------- #
# Pull one season file
# --------------------------------------------------------------------------- #
def _pull_year(kind: str, year: int, force: bool = False, refresh: bool = False) -> pd.DataFrame | None:
    """Return one season of one dataset, from cache unless a download is warranted.

    Downloads only if the file is missing, or ``force`` (explicit re-download,
    any season), or ``refresh`` (the in-progress season only). Returns ``None``
    if that season has no data yet.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(kind, year)
    if refresh and year != current_season():
        raise ValueError(f"refusing to refresh {kind} {year}: only the in-progress season "
                         f"({current_season()}) is refreshable; completed seasons need --force")

    if path.exists() and not (force or refresh):
        df = pd.read_parquet(path)
        print(f"  [{kind} {year}] cache hit  -> {len(df):>7,} rows  ({path.name})")
        return df

    print(f"  [{kind} {year}] downloading ...")
    pulled_at = _utcnow()
    df = _fetch(kind, year)
    if df is None:
        return None

    _write_parquet_atomic(df, path)
    _record_metadata(path, pulled_at, len(df))
    print(f"  [{kind} {year}] saved       -> {len(df):>7,} rows  ({path.name})")
    if kind == "schedules" and year == current_season():
        write_schedule_snapshot(df, year, pulled_at)
    return df


def refresh_season(season: int) -> dict[str, pd.DataFrame | None]:
    """Re-pull the in-progress season's pbp + schedule; touch nothing else."""
    return {kind: _pull_year(kind, season, refresh=True) for kind in KINDS}


def _pull_range(kind: str, years: range, force: bool) -> pd.DataFrame:
    frames = []
    for year in years:
        df = _pull_year(kind, year, force)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"no {kind} data pulled for any year in {years}")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def get_pbp(start: int = START_YEAR, end: int = END_YEAR, force: bool = False) -> pd.DataFrame:
    """Return play-by-play data for ``start..end`` inclusive, cached per season."""
    return _pull_range("pbp", range(start, end + 1), force)


def get_schedules(start: int = START_YEAR, end: int = END_YEAR, force: bool = False) -> pd.DataFrame:
    """Return schedule/game data for ``start..end`` inclusive, cached per season."""
    return _pull_range("schedules", range(start, end + 1), force)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _summarize(name: str, df: pd.DataFrame, season_col: str) -> None:
    seasons = sorted(df[season_col].dropna().unique().tolist())
    print(f"\n{name}:")
    print(f"  rows      : {len(df):,}")
    print(f"  columns   : {len(df.columns)}")
    print(f"  seasons   : {int(seasons[0])}..{int(seasons[-1])} ({len(seasons)} seasons)")


def print_status() -> None:
    meta = load_metadata()
    print(f"Raw cache dir: {RAW_DIR}   (in-progress season: {current_season()})")
    for p in sorted(RAW_DIR.glob("*.parquet")):
        m = meta.get(p.name)
        src = (f"{m['library']} {m['library_version']} at {m['pulled_at_utc']}" if m
               else "legacy: no metadata (pre-Phase 9 pull)")
        print(f"  {p.name:<28} {src}")
    snaps = sorted(_snapshot_dir().glob("*.parquet")) if _snapshot_dir().exists() else []
    print(f"snapshots ({len(snaps)}):")
    for p in snaps:
        m = meta.get(f"{SNAPSHOT_SUBDIR}/{p.name}", {})
        print(f"  {p.name:<44} rows={m.get('rows', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=START_YEAR)
    parser.add_argument("--end", type=int, default=END_YEAR)
    parser.add_argument("--force", action="store_true",
                        help="re-download every season in --start..--end, completed ones included")
    parser.add_argument("--refresh-season", type=int, metavar="SEASON",
                        help="re-pull ONLY this in-progress season's pbp + schedule (and snapshot it)")
    parser.add_argument("--status", action="store_true", help="list cached files, metadata, snapshots")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.refresh_season is not None:
        if args.force:
            parser.error("--refresh-season and --force are separate operations; use one")
        if args.refresh_season != current_season():
            parser.error(f"{args.refresh_season} is not the in-progress season "
                         f"({current_season()}); completed seasons are only re-pulled with --force")
        print(f"Refreshing in-progress season {args.refresh_season} only ({LIBRARY} {LIBRARY_VERSION})\n")
        out = refresh_season(args.refresh_season)
        for kind, df in out.items():
            if df is not None:
                _summarize(f"{kind} {args.refresh_season}", df, "season")
        return

    if args.start < 2002:
        parser.error("SPEC Section 3.4 sets the floor at 2002; flag before going earlier.")

    print(f"Raw cache dir: {RAW_DIR}")
    print(f"Pulling seasons {args.start}..{args.end} (force={args.force}, {LIBRARY} {LIBRARY_VERSION})\n")

    print("Play-by-play (nflreadpy.load_pbp):")
    pbp = get_pbp(args.start, args.end, args.force)

    print("\nSchedules (nflreadpy.load_schedules):")
    sched = get_schedules(args.start, args.end, args.force)

    _summarize("play-by-play", pbp, "season")
    _summarize("schedules", sched, "season")
    print("\nDone. Raw pulls cached under data/raw/.")


if __name__ == "__main__":
    main()
