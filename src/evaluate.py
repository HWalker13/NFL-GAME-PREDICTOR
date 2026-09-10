"""Evaluation utilities for the NFL win/loss predictor (SPEC Section 2 / 8).

Phase 1 so far only needs the naive baseline. The rest of Section 8 (log loss,
Brier, ROC-AUC, calibration) gets added once there is a model to evaluate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_schedules(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Concatenate every cached ``schedules_*.parquet`` pull in ``data/raw/``."""
    paths = sorted(raw_dir.glob("schedules_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no cached schedules in {raw_dir} - run `python -m src.data_ingest` first"
        )
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def completed_regular_season(schedules: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games with a final score (SPEC Section 15: REG only).

    Ties are dropped: "always pick home" is undefined for a game nobody won.
    """
    reg = schedules[schedules["game_type"] == "REG"].copy()
    reg = reg.dropna(subset=["home_score", "away_score"])
    reg = reg[reg["home_score"] != reg["away_score"]]
    return reg


def home_baseline_accuracy(
    schedules: pd.DataFrame,
    seasons: Iterable[int] | None = None,
) -> dict[str, float | int]:
    """Accuracy of the naive "always predict the home team" rule.

    Computed directly from the home win rate of the games in ``seasons`` (or all
    available seasons if ``None``). This MUST be used in place of a hardcoded
    constant when reporting whether a model beats baseline (SPEC Section 2).

    Why compute it per-split instead of using SPEC Section 2's ~57-58% figure:
    NFL home-field advantage has measurably declined since 2019 (no-crowd 2020,
    rule/officiating shifts, better travel/recovery science). In this dataset the
    2023-2024 test seasons come in at ~54.4% home win rate, not 58% - so a fixed
    historical number would set the model's "beat baseline" bar in the wrong
    place for the era actually being tested.
    """
    games = completed_regular_season(schedules)
    if seasons is not None:
        games = games[games["season"].isin(list(seasons))]
    if len(games) == 0:
        raise ValueError(f"no completed REG games for seasons={seasons}")

    home_wins = int((games["home_score"] > games["away_score"]).sum())
    n = int(len(games))
    return {
        "baseline_accuracy": home_wins / n,
        "home_wins": home_wins,
        "n_games": n,
        "season_min": int(games["season"].min()),
        "season_max": int(games["season"].max()),
    }


def season_range(start: int, end: int) -> range:
    """Inclusive season range helper, e.g. ``season_range(2023, 2024)``."""
    return range(start, end + 1)


if __name__ == "__main__":
    sched = load_schedules()
    for label, yrs in {
        "train 2002-2021": season_range(2002, 2021),
        "val 2022": season_range(2022, 2022),
        "test 2023-2024": season_range(2023, 2024),
        "2025": season_range(2025, 2025),
        "all available": None,
    }.items():
        r = home_baseline_accuracy(sched, yrs)
        print(
            f"{label:<18} baseline={r['baseline_accuracy']:.4f}  "
            f"({r['home_wins']}/{r['n_games']} home wins, "
            f"seasons {r['season_min']}-{r['season_max']})"
        )
