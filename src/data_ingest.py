"""Data ingestion for the NFL win/loss predictor (SPEC Section 3, Phase 1).

Thin wrapper around ``nfl_data_py.import_pbp_data`` and
``nfl_data_py.import_schedules``. Every raw pull is cached, unmodified, to
``data/raw/`` as Parquet so downstream code (and re-runs of this script) never
re-hit the network unless asked to.

Historical range per SPEC Section 3.4: **2002 to present** (the modern 32-team
era). 1999-2001 are deliberately excluded and flagged in the SPEC as an open
question.

Usage::

    python -m src.data_ingest                # pull 2002..current year, cached
    python -m src.data_ingest --start 2015   # narrower range
    python -m src.data_ingest --force        # ignore cache, re-download

This module does **no** feature engineering (SPEC Section 5 / Phase 2). It only
fetches and caches.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - import shape depends on install
    from nfl_data_py import import_pbp_data, import_schedules
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "nfl_data_py is not installed. Activate the venv and "
        "`pip install -r requirements.txt`."
    ) from exc


# SPEC Section 3.4: modern 32-team era starts in 2002.
START_YEAR = 2002
# "to present" - the current calendar year. Seasons with no data yet are
# skipped gracefully (see _pull_year).
END_YEAR = _dt.date.today().year

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _raw_path(kind: str, year: int) -> Path:
    return RAW_DIR / f"{kind}_{year}.parquet"


def _pull_year(kind: str, year: int, force: bool) -> pd.DataFrame | None:
    """Fetch one season of one dataset, using the on-disk cache when possible.

    Returns the season's DataFrame, or ``None`` if that season simply has no
    data yet (e.g. a future season the nflverse files don't cover).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(kind, year)

    if path.exists() and not force:
        df = pd.read_parquet(path)
        print(f"  [{kind} {year}] cache hit  -> {len(df):>7,} rows  ({path.name})")
        return df

    print(f"  [{kind} {year}] downloading ...")
    try:
        if kind == "pbp":
            df = import_pbp_data([year], downcast=True, cache=False)
        elif kind == "schedules":
            df = import_schedules([year])
        else:  # pragma: no cover - programming error
            raise ValueError(f"unknown dataset kind: {kind!r}")
    except Exception as exc:  # noqa: BLE001 - nfl_data_py raises bare errors
        print(f"  [{kind} {year}] no data available ({exc.__class__.__name__}: {exc})")
        return None

    if df is None or df.empty:
        print(f"  [{kind} {year}] returned empty - skipping (season not started?)")
        return None

    df.to_parquet(path, index=False)
    print(f"  [{kind} {year}] saved       -> {len(df):>7,} rows  ({path.name})")
    return df


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


def _summarize(name: str, df: pd.DataFrame, season_col: str) -> None:
    seasons = sorted(df[season_col].dropna().unique().tolist())
    print(f"\n{name}:")
    print(f"  rows      : {len(df):,}")
    print(f"  columns   : {len(df.columns)}")
    print(f"  seasons   : {int(seasons[0])}..{int(seasons[-1])} ({len(seasons)} seasons)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=START_YEAR)
    parser.add_argument("--end", type=int, default=END_YEAR)
    parser.add_argument(
        "--force", action="store_true", help="ignore cache and re-download"
    )
    args = parser.parse_args()

    if args.start < 2002:
        parser.error("SPEC Section 3.4 sets the floor at 2002; flag before going earlier.")

    print(f"Raw cache dir: {RAW_DIR}")
    print(f"Pulling seasons {args.start}..{args.end} (force={args.force})\n")

    print("Play-by-play (import_pbp_data):")
    pbp = get_pbp(args.start, args.end, args.force)

    print("\nSchedules (import_schedules):")
    sched = get_schedules(args.start, args.end, args.force)

    _summarize("play-by-play", pbp, "season")
    _summarize("schedules", sched, "season")
    print("\nDone. Raw pulls cached under data/raw/.")


if __name__ == "__main__":
    main()
