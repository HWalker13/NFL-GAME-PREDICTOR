"""Phase 8, Part B: SPEC 5.5 check 1 (timestamp assertion) on the 2025
holdout rows of game_features.parquet -- WITHOUT reading any 2025 label.

- ``game_features.parquet`` is read with an explicit column list that
  excludes ``home_win``; the label column is never loaded.
- The schedule is read for ``game_id``/``season``/``game_type`` only (no
  scores) to confirm the 2025 row count against the 272-game schedule.
- ``check_feature_timestamps`` also needs ``team_game_log.parquet``: per-game
  EPA inputs used to build features (no scores, no win label). Its oracle
  recomputes every in-season rolling feature from strictly-prior games, and
  its tripwire correlates each rolled feature with the current game's own raw
  metric.

NOT run here: ``check_elo_point_in_time``. It recomputes post-game Elo from
final scores, i.e. it reads 2025 outcomes, so it is deferred to Part D
(after pre-registration approval).

Usage::

    python -m scripts.phase8_leakage_2025
"""

from __future__ import annotations

import sys

import fastparquet
import pandas as pd

from src import evaluate as E, leakage_checks as LC, train_winner as TW

SEASON = 2025


def main() -> int:
    gf_path = TW.PROC_DIR / "game_features.parquet"
    cols = [c for c in fastparquet.ParquetFile(gf_path).columns if c != "home_win"]
    frame = pd.read_parquet(gf_path, columns=cols)
    assert "home_win" not in frame.columns
    frame = frame[frame["season"] == SEASON].reset_index(drop=True)
    tgl = pd.read_parquet(TW.PROC_DIR / "team_game_log.parquet")

    sched_paths = sorted(E.RAW_DIR.glob(f"schedules_{SEASON}.parquet"))
    sched = pd.read_parquet(sched_paths[0], columns=["game_id", "season", "game_type"])
    reg_ids = set(sched.loc[sched["game_type"] == "REG", "game_id"])
    missing = sorted(reg_ids - set(frame["game_id"]))
    extra = sorted(set(frame["game_id"]) - reg_ids)

    print(f"2025 feature rows: {len(frame)}  (schedule REG games: {len(reg_ids)}; "
          f"schedule games absent from frame: {missing}; frame rows not in schedule: {extra})")
    print(f"columns loaded: {len(frame.columns)} (home_win excluded); "
          f"kickoff range {frame['kickoff'].min()} .. {frame['kickoff'].max()}")
    feat_nulls = int(frame[TW.FEATURE_COLS].isna().sum().sum())
    print(f"null feature cells in 2025 rows: {feat_nulls} "
          f"(HGB handles NaN natively; LR/RF pipelines median-impute)")

    try:
        result = LC.check_feature_timestamps(frame, tgl)
    except AssertionError as exc:
        print(f"\n!! CHECK 1 FAIL on {SEASON} rows: {exc}")
        return 1
    print(f"\nresult: {result}")
    if len(frame) != 271 or extra:
        print(f"\n!! unexpected row count/ids for {SEASON}: {len(frame)} rows, extra={extra}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
