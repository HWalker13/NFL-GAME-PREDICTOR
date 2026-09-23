"""Phase 10 live feature frame: 2002-2026, including rows for UNPLAYED games.

``features.py`` is NOT modified (Phase 10 rule: it may change only if strictly
needed). Everything here reuses its functions:

* **Played games** come from ``features.build_game_features`` itself -- the
  frozen, canonical code path -- run over 2002..season with raw data injected
  in memory (same technique Phase 9 used to redirect it, no source edit).
* **Unplayed games** (REG, no score yet) get a feature row computed exactly as
  if the game were the next row in each team's history. They are processed one
  WEEK at a time: the week's unplayed games are appended to the played
  team-game log as rows with NaN raw metrics, the unchanged
  ``features.roll_features`` is run, and only that week's rows are kept. Each
  team plays at most once per week, so a week's unplayed row only ever sees
  PLAYED games in its window (closed='left' / shift(1) exclude the row itself;
  later unplayed weeks are not in the batch). The rows therefore use exactly
  the games completed before their kickoff, and nothing else.
* **Elo for unplayed games** uses the unchanged ``features.build_elo_table``.
  That function only admits games with a final score, so the batch's unplayed
  games carry a PLACEHOLDER score. The pre-game rating is read from the running
  dict BEFORE a game's own result is applied (Pattern D), and within a week no
  team plays twice, so the placeholder can never reach a row that is kept.
  ``build_live_frame(verify_placeholder=True)`` proves it: the unplayed rows are
  built twice with opposite placeholder results and must be identical.
  ``home_win`` is NaN on every unplayed row.

Correctness gates (``python -m src.live_features``):
  1. 2002-2025 rows == canonical ``data/processed/game_features.parquet``
     (exact, dtypes included);
  2. SPEC 5.5 check 1 (timestamp assertion) on every 2026 row, played and
     unplayed, plus the Elo point-in-time check on the played rows;
  3. unplayed rows carry no outcome-derived values (placeholder invariance,
     NaN label, no raw-metric / Vegas columns);
  4. masked-week equivalence: hide a played week (and everything after it),
     build its rows through the UNPLAYED path, and require them to equal the
     rows the canonical PLAYED path produced for the same games.

Usage::

    python -m src.live_features              # build + verify + write game_features_live.parquet
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_ingest, evaluate, features as F, leakage_checks as LC

LIVE_SEASON = 2026
START_SEASON = F.DEFAULT_START
LIVE_PATH = F.PROC_DIR / "game_features_live.parquet"
CANONICAL_PATH = F.PROC_DIR / "game_features.parquet"

# The only pbp columns features.build_team_game_log reads (Phase 9 doc §2.2),
# plus the keys used here to filter. Loading a column subset leaves every
# value the pipeline sees unchanged; gate 1 verifies the result is identical.
PBP_COLS = ["game_id", "season", "week", "posteam", "defteam", "pass", "rush",
            "qb_dropback", "sack", "qb_scramble", "epa", "qb_epa", "success",
            "cpoe", "interception", "air_yards", "down", "qb_hit"]

# Placeholder final score for unplayed games, used ONLY so the unchanged
# build_game_index / build_elo_table admit them (see module docstring).
PLACEHOLDER_HOME_WIN = (1.0, 0.0)
PLACEHOLDER_AWAY_WIN = (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_raw(end_season: int = LIVE_SEASON, raw_dir: Path | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Schedules (every cached season) + pbp column subset for START..end.

    Reads the cached files directly and never downloads: a missing pbp file is
    an error here, not a silent network pull.
    """
    raw_dir = raw_dir or data_ingest.RAW_DIR
    schedules = evaluate.load_schedules(raw_dir)
    parts = []
    for yr in range(START_SEASON, end_season + 1):
        p = raw_dir / f"pbp_{yr}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- refresh first (python -m src.data_ingest ...)")
        parts.append(pd.read_parquet(p, columns=PBP_COLS))
    return schedules, pd.concat(parts, ignore_index=True)


@contextlib.contextmanager
def _injected(schedules: pd.DataFrame, pbp: pd.DataFrame):
    """Run features.build_game_features on in-memory data (no source edit)."""
    orig_sched, orig_pbp = F.evaluate.load_schedules, F.data_ingest.get_pbp
    F.evaluate.load_schedules = lambda *a, **k: schedules.copy()
    F.data_ingest.get_pbp = lambda start, end, force=False: (
        pbp[pbp["season"].between(start, end)].reset_index(drop=True))
    try:
        yield
    finally:
        F.evaluate.load_schedules, F.data_ingest.get_pbp = orig_sched, orig_pbp


@contextlib.contextmanager
def elo_home_adv(value: float | None):
    """Temporarily set features.ELO_HOME_ADV (read at call time by
    build_elo_table and by the Elo point-in-time oracle). None = unchanged."""
    orig = F.ELO_HOME_ADV
    if value is not None:
        F.ELO_HOME_ADV = float(value)
    try:
        yield
    finally:
        F.ELO_HOME_ADV = orig


# --------------------------------------------------------------------------- #
# Unplayed games
# --------------------------------------------------------------------------- #
def unplayed_schedule(schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    """REG games of ``season`` with no final score yet."""
    s = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    return s[s["home_score"].isna() & s["away_score"].isna()].copy()


def _with_placeholder(rows: pd.DataFrame, placeholder: tuple[float, float]) -> pd.DataFrame:
    out = rows.copy()
    out["home_score"], out["away_score"] = placeholder
    return out


def season_schedule_with_kickoff(schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    """Every REG game of ``season`` (scored, tied or unplayed) with its UTC
    kickoff, computed by the unchanged features.build_game_index (which only
    admits decided games, hence the placeholder on a copy)."""
    s = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    gi = F.build_game_index(_with_placeholder(s, PLACEHOLDER_HOME_WIN))
    out = s.merge(gi[["game_id", "kickoff"]], on="game_id", how="left", validate="1:1")
    assert out["kickoff"].notna().all(), "kickoff could not be computed for some games"
    return out


def _placeholder_team_rows(gi: pd.DataFrame) -> pd.DataFrame:
    """Two team-game-log rows per unplayed game, raw metrics NaN."""
    parts = []
    for side, opp in (("home", "away"), ("away", "home")):
        parts.append(pd.DataFrame({
            "game_id": gi["game_id"].to_numpy(),
            "team": F._canon(gi[f"{side}_team"]).to_numpy(),
            "season": gi["season"].to_numpy(),
            "week": gi["week"].to_numpy(),
            "kickoff": gi["kickoff"].to_numpy(),
            "is_home": side == "home",
            "opponent": F._canon(gi[f"{opp}_team"]).to_numpy(),
        }))
    rows = pd.concat(parts, ignore_index=True)
    for c in ["n_dropbacks", "n_def_plays"] + F.RAW_METRICS:
        rows[c] = np.nan
    return rows


def _assemble(rolled: pd.DataFrame, gi: pd.DataFrame, ctx: pd.DataFrame,
              elo: pd.DataFrame) -> pd.DataFrame:
    """Mirror of build_game_features' assembly for a set of games (gate 4
    checks it against the canonical function)."""
    model_cols = (["game_id", "season", "week", "kickoff", "games_played_sd"]
                  + F.MODEL_SHRUNK + F.MODEL_EWM)
    home = (rolled[rolled["is_home"]][["game_id"] + [c for c in model_cols if c != "game_id"]]
            .add_prefix("home_").rename(columns={"home_game_id": "game_id"}))
    away = (rolled[~rolled["is_home"]][["game_id"] + F.MODEL_SHRUNK + F.MODEL_EWM + ["games_played_sd"]]
            .add_prefix("away_").rename(columns={"away_game_id": "game_id"}))
    frame = home.merge(away, on="game_id", how="inner")
    frame = frame.drop(columns=["home_season", "home_week", "home_kickoff"])
    frame = frame.merge(ctx, on="game_id", how="inner")
    frame = frame.merge(elo, on="game_id", how="inner")
    frame["elo_diff"] = frame["home_elo_pre"] - frame["away_elo_pre"]
    frame = frame.merge(gi[["game_id", "season", "week", "kickoff"]], on="game_id", how="inner")
    frame["home_win"] = np.nan  # no label: the game has not been played
    for name, (a, b) in F.DIFF_FEATURES.items():
        frame[name] = frame[a] - frame[b]
    return frame


def unplayed_frame(schedules: pd.DataFrame, tgl_played: pd.DataFrame, season: int,
                   placeholder: tuple[float, float] = PLACEHOLDER_HOME_WIN,
                   k: int = F.DEFAULT_K) -> pd.DataFrame:
    """Feature rows for every unplayed REG game of ``season``, week by week."""
    todo = unplayed_schedule(schedules, season)
    if todo.empty:
        return pd.DataFrame()
    played_sched = schedules.drop(index=todo.index)
    home_stadium = F.team_home_stadium(schedules)
    out = []
    for week, wk in todo.groupby("week", sort=True):
        wk_ph = _with_placeholder(wk, placeholder)
        gi = F.build_game_index(wk_ph).drop(columns=["home_win"])
        teams = pd.concat([F._canon(gi["home_team"]), F._canon(gi["away_team"])])
        assert not teams.duplicated().any(), f"week {week}: a team has two unplayed games"
        tgl = pd.concat([tgl_played, _placeholder_team_rows(gi)], ignore_index=True)
        rolled = F.roll_features(tgl, k=k)
        rolled = rolled[rolled["game_id"].isin(gi["game_id"])]
        ctx = F.contextual_features(gi, F.STADIUM_GEO, home_stadium)
        elo_sched = pd.concat([played_sched, wk_ph], ignore_index=True)
        elo = F.build_elo_table(elo_sched, START_SEASON, season)
        elo = elo[elo["game_id"].isin(gi["game_id"])]
        out.append(_assemble(rolled, gi, ctx, elo))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Full live frame
# --------------------------------------------------------------------------- #
def played_frame(schedules: pd.DataFrame, pbp: pd.DataFrame, end_season: int) -> pd.DataFrame:
    """Canonical features.build_game_features over START..end_season (no writes)."""
    with _injected(schedules, pbp):
        return F.build_game_features(START_SEASON, end_season, cache=False)


def played_team_log(schedules: pd.DataFrame, pbp: pd.DataFrame, end_season: int) -> pd.DataFrame:
    gi = F.build_game_index(schedules)
    gi = gi[gi["season"].between(START_SEASON, end_season)].reset_index(drop=True)
    return F.build_team_game_log(pbp[pbp["season"].between(START_SEASON, end_season)], gi)


def build_live_frame(schedules: pd.DataFrame, pbp: pd.DataFrame, season: int = LIVE_SEASON,
                     verify_placeholder: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns ``(frame, tgl_played)``. ``frame`` has every played REG game
    2002..season plus every unplayed REG game of ``season``, sorted by
    (kickoff, game_id), with an ``is_played`` column and NaN ``home_win`` on
    unplayed rows."""
    played = played_frame(schedules, pbp, season)
    tgl = played_team_log(schedules, pbp, season)
    unplayed = unplayed_frame(schedules, tgl, season, PLACEHOLDER_HOME_WIN)
    if verify_placeholder and len(unplayed):
        other = unplayed_frame(schedules, tgl, season, PLACEHOLDER_AWAY_WIN)
        pd.testing.assert_frame_equal(unplayed, other, check_exact=True)
    played = played.assign(home_win=played["home_win"].astype("float64"), is_played=True)
    frames = [played]
    if len(unplayed):
        unplayed = unplayed[played.columns.drop("is_played")].assign(is_played=False)
        frames.append(unplayed)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["kickoff", "game_id"]).reset_index(drop=True)
    assert not frame["game_id"].duplicated().any()
    return frame, tgl


def canonical_view(frame: pd.DataFrame, max_season: int) -> pd.DataFrame:
    """Rows <= max_season in the canonical file's exact shape and dtypes."""
    sub = frame[frame["season"] <= max_season].drop(columns=["is_played"]).reset_index(drop=True)
    assert sub["home_win"].notna().all()
    return sub.assign(home_win=sub["home_win"].astype("int64"))


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Played rows with an integer label, ready for model fitting."""
    played = frame[frame["is_played"]].drop(columns=["is_played"]).reset_index(drop=True)
    return played.assign(home_win=played["home_win"].astype("int64"))


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def assert_matches_canonical(frame: pd.DataFrame, canonical_path: Path = CANONICAL_PATH) -> None:
    canon = pd.read_parquet(canonical_path)
    got = canonical_view(frame, int(canon["season"].max()))
    pd.testing.assert_frame_equal(got, canon, check_exact=True)


def assert_no_outcome_in_unplayed(frame: pd.DataFrame) -> None:
    un = frame[~frame["is_played"]]
    assert un["home_win"].isna().all(), "unplayed row carries a label"
    raw_cols = [c for c in frame.columns
                if any(c == f"{s}_{m}" for s in ("home", "away") for m in F.RAW_METRICS)]
    assert not raw_cols, f"raw outcome-derived metrics present: {raw_cols}"
    vegas = [c for c in frame.columns if any(v in c for v in F._VEGAS_COLS)]
    assert not vegas, f"Vegas columns present: {vegas}"


def check1_live(frame: pd.DataFrame, tgl_played: pd.DataFrame, schedules: pd.DataFrame,
                season: int = LIVE_SEASON) -> dict:
    """SPEC 5.5 check 1 on every row of ``season``, played and unplayed.

    The unplayed games are added to the team-game log as NaN-metric rows so the
    check can see each team's previous game and prior-season data for them.
    """
    rows = frame[frame["season"] == season]
    un = rows[~rows["is_played"]]
    tgl = tgl_played
    if len(un):
        gi = un[["game_id", "season", "week", "kickoff"]].merge(
            unplayed_schedule(schedules, season)[["game_id", "home_team", "away_team"]],
            on="game_id", how="left")
        tgl = pd.concat([tgl_played, _placeholder_team_rows(gi)], ignore_index=True)
    return LC.check_feature_timestamps(rows.drop(columns=["is_played"]), tgl)


def masked_week_equivalence(schedules: pd.DataFrame, pbp: pd.DataFrame,
                            full_frame: pd.DataFrame, season: int, week: int) -> int:
    """Hide ``season``/``week`` and everything after it; rebuild its rows via
    the UNPLAYED path; require equality with the PLAYED-path rows in
    ``full_frame``. Returns the number of rows compared."""
    first_kick = full_frame.loc[(full_frame["season"] == season) & (full_frame["week"] == week),
                                "kickoff"].min()
    reg_idx = F.build_game_index(schedules)[["game_id", "kickoff"]]
    hidden = set(reg_idx.loc[reg_idx["kickoff"] >= first_kick, "game_id"])
    hidden |= set(schedules.loc[schedules["season"] > season, "game_id"])
    masked_sched = schedules.copy()
    in_season_hidden = masked_sched["game_id"].isin(hidden) & (masked_sched["season"] == season)
    masked_sched.loc[in_season_hidden, ["home_score", "away_score"]] = np.nan
    masked_sched = masked_sched[masked_sched["season"] <= season]
    masked_pbp = pbp[~pbp["game_id"].isin(hidden) & (pbp["season"] <= season)]
    tgl = played_team_log(masked_sched, masked_pbp, season)
    un = unplayed_frame(masked_sched, tgl, season)
    un = un[un["week"] == week].sort_values("game_id").reset_index(drop=True)
    ref = full_frame[(full_frame["season"] == season) & (full_frame["week"] == week)]
    ref = ref[ref["game_id"].isin(un["game_id"])].sort_values("game_id").reset_index(drop=True)
    assert len(un) and len(un) == len(ref), (len(un), len(ref))
    cols = [c for c in F.feature_columns()] + ["season", "week", "kickoff"]
    pd.testing.assert_frame_equal(un[cols], ref[cols], check_exact=True)
    return len(un)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_and_verify(season: int = LIVE_SEASON, write_path: Path | None = LIVE_PATH,
                     elo_adv: float | None = None, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the live frame and run gates 1-3 (gate 1 only at the frozen
    ELO_HOME_ADV -- a different HFA changes historical Elo by design)."""
    schedules, pbp = load_raw(season)
    with elo_home_adv(elo_adv):
        frame, tgl = build_live_frame(schedules, pbp, season, verify_placeholder=True)
        if elo_adv is None:
            assert_matches_canonical(frame)
            if verbose:
                print("GATE 1 PASS: 2002-2025 rows identical to canonical game_features.parquet "
                      "(exact values and dtypes)")
        assert_no_outcome_in_unplayed(frame)
        if verbose:
            print("GATE 3 PASS: unplayed rows carry no label / raw metric / Vegas column; "
                  "placeholder-result invariance verified (both placeholders give identical rows)")
        check1_live(frame, tgl, schedules, season)
        LC.check_elo_point_in_time(frame[frame["is_played"]], schedules)
    if write_path is not None:
        frame.to_parquet(write_path, index=False)
        if verbose:
            print(f"wrote {write_path}  sha256 {sha256_file(write_path)}")
    return frame, tgl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=LIVE_SEASON)
    ap.add_argument("--masked-week", type=int, nargs=2, action="append", metavar=("SEASON", "WEEK"),
                    help="also run the masked-week equivalence gate (repeatable)")
    args = ap.parse_args()
    frame, _ = build_and_verify(args.season)
    rows = frame[frame["season"] == args.season]
    print(f"\n{args.season} rows: {len(rows)}  (played {int(rows.is_played.sum())}, "
          f"unplayed {int((~rows.is_played).sum())})")
    for s, w in args.masked_week or []:
        schedules, pbp = load_raw(args.season)
        n = masked_week_equivalence(schedules, pbp, frame, s, w)
        print(f"GATE 4 PASS: masked-week {s} wk{w}: {n} rows via the unplayed path == played path")


if __name__ == "__main__":
    main()
