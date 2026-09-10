"""Phase 2 feature pipeline for the NFL win/loss predictor.

Implements SPEC.md Sections 3.3 (features), 4 (shrinkage) and 5.4 (point-in-time
computation). Win/loss only -- SPEC Section 12 (spread/total) is out of scope.

Pipeline
--------
1. ``build_team_game_log``  -- one row per (game_id, team) holding that game's
   RAW QB / defensive metrics. These are outcome-derived (SPEC 5.3, target
   leakage) and live ONLY in ``data/processed/team_game_log.parquet``.
2. ``roll_features``        -- rolls every raw metric forward so a game only ever
   sees PRIOR games, then blends it with a prior-season / league prior via the
   SPEC Section 4 shrinkage formula. Produces ``{metric}_shrunk`` columns.
3. ``contextual_features``  -- rest / home / divisional / travel / timezone /
   roof, all known before kickoff (SPEC 5.3, allowed).
4. ``build_game_features`` -- assembles the game-level modeling frame
   (``data/processed/game_features.parquet``) plus ``feature_manifest.json``.

Leakage discipline (SPEC 5.4, enforced by CLAUDE.md)
---------------------------------------------------
EVERY rolling/aggregated feature uses one of exactly two patterns:

* **Pattern A** -- ``.rolling(window=..., closed='left')`` (the ``closed='left'``
  argument is what excludes the current row from its own window).
* **Pattern B** -- ``pd.merge_asof(..., direction='backward',
  allow_exact_matches=False)`` on the ``season`` axis, so a row can only ever be
  matched to a strictly-earlier season.

Plain ``.expanding().mean()`` / ``.rolling().mean()`` without ``closed='left'``
does not appear anywhere in this module. ``feature_manifest.json`` records the
pattern used for every single feature.

Usage::

    python -m src.features                       # 2002..2025, k=4
    python -m src.features --start 2006 --k 4
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src import data_ingest, evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_ROOT / "data" / "processed"

# SPEC Section 4: k ~= 4 games of prior weight, tunable.
DEFAULT_K = 4
# Large enough that a full 17/18-game (+ 21-game 32-team) season never truncates
# it; with closed='left' this rolling mean IS the in-season-to-date average.
IN_SEASON_WINDOW = 22
DEFAULT_START = 2002
DEFAULT_END = 2025

# Schedules use era team codes (STL/SD/OAK); pbp already uses the modern codes.
# Join is on game_id, but where we read a team code straight off the schedule we
# normalise it through this map first.
TEAM_ABBR_CANON = {
    "SD": "LAC", "SDG": "LAC",
    "STL": "LA", "LAR": "LA", "SL": "LA",
    "OAK": "LV",
    "JAC": "JAX", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "WSH": "WAS",
}

# The 11 raw per-game metrics. Order is the report/manifest order.
QB_METRICS = [
    "qb_epa_per_db",
    "qb_success_rate",
    "cpoe",
    "sack_rate",
    "int_rate",
    "air_yards_per_att",
]
DEF_METRICS = [
    "def_epa_per_play",
    "def_success_allowed",
    "def_pressure_rate",
    "def_epa_early",
    "def_epa_money",
]
RAW_METRICS = QB_METRICS + DEF_METRICS

# Vegas columns that MUST NOT reach the modeling frame (SPEC 5.2 category 4).
_VEGAS_COLS = (
    "spread_line", "total_line", "away_moneyline", "home_moneyline",
    "away_spread_odds", "home_spread_odds", "under_odds", "over_odds",
    "result", "total",
)

# --------------------------------------------------------------------------- #
# Static stadium geo table (SPEC 3.3: travel / timezone -- "free, no data cost")
# --------------------------------------------------------------------------- #
# Keyed by schedules.stadium_id. One coordinate per id (an id groups successive
# venues on the same site / city). Coordinates are public facts (Wikipedia/OSM);
# tz is an IANA name so the UTC offset is computed DST-correctly at kickoff date.
STADIUM_GEO: dict[str, dict] = {
    "ATL00": {"name": "Georgia Dome",              "lat": 33.758, "lon": -84.401, "tz": "America/New_York"},
    "ATL97": {"name": "Mercedes-Benz Stadium",     "lat": 33.755, "lon": -84.401, "tz": "America/New_York"},
    "BAL00": {"name": "M&T Bank Stadium",          "lat": 39.278, "lon": -76.623, "tz": "America/New_York"},
    "BOS00": {"name": "Gillette Stadium",          "lat": 42.091, "lon": -71.264, "tz": "America/New_York"},
    "BRG00": {"name": "Tiger Stadium (LSU)",       "lat": 30.412, "lon": -91.184, "tz": "America/Chicago"},
    "BUF00": {"name": "Highmark Stadium",          "lat": 42.774, "lon": -78.787, "tz": "America/New_York"},
    "BUF01": {"name": "Rogers Centre",             "lat": 43.641, "lon": -79.389, "tz": "America/Toronto"},
    "CAR00": {"name": "Bank of America Stadium",   "lat": 35.226, "lon": -80.853, "tz": "America/New_York"},
    "CHI98": {"name": "Soldier Field",             "lat": 41.862, "lon": -87.617, "tz": "America/Chicago"},
    "CHI99": {"name": "Memorial Stadium (Champaign)", "lat": 40.100, "lon": -88.236, "tz": "America/Chicago"},
    "CIN00": {"name": "Paycor Stadium",            "lat": 39.095, "lon": -84.516, "tz": "America/New_York"},
    "CLE00": {"name": "Cleveland Browns Stadium",  "lat": 41.506, "lon": -81.700, "tz": "America/New_York"},
    "DAL00": {"name": "AT&T Stadium",              "lat": 32.748, "lon": -97.093, "tz": "America/Chicago"},
    "DAL99": {"name": "Texas Stadium",             "lat": 32.840, "lon": -96.914, "tz": "America/Chicago"},
    "DEN00": {"name": "Empower Field at Mile High","lat": 39.744, "lon": -105.020, "tz": "America/Denver"},
    "DET00": {"name": "Ford Field",                "lat": 42.340, "lon": -83.046, "tz": "America/Detroit"},
    "FRA00": {"name": "Deutsche Bank Park",        "lat": 50.068, "lon": 8.646,   "tz": "Europe/Berlin"},
    "GER00": {"name": "Allianz Arena",             "lat": 48.219, "lon": 11.625,  "tz": "Europe/Berlin"},
    "GNB00": {"name": "Lambeau Field",             "lat": 44.501, "lon": -88.062, "tz": "America/Chicago"},
    "HOU00": {"name": "NRG Stadium",               "lat": 29.685, "lon": -95.411, "tz": "America/Chicago"},
    "IND00": {"name": "Lucas Oil Stadium",         "lat": 39.760, "lon": -86.164, "tz": "America/Indiana/Indianapolis"},
    "IND99": {"name": "RCA Dome",                  "lat": 39.766, "lon": -86.163, "tz": "America/Indiana/Indianapolis"},
    "JAX00": {"name": "EverBank Stadium",          "lat": 30.324, "lon": -81.638, "tz": "America/New_York"},
    "KAN00": {"name": "Arrowhead Stadium",         "lat": 39.049, "lon": -94.484, "tz": "America/Chicago"},
    "LAX01": {"name": "SoFi Stadium",              "lat": 33.953, "lon": -118.339, "tz": "America/Los_Angeles"},
    "LAX97": {"name": "Dignity Health Sports Park","lat": 33.864, "lon": -118.261, "tz": "America/Los_Angeles"},
    "LAX99": {"name": "Los Angeles Memorial Coliseum", "lat": 34.014, "lon": -118.288, "tz": "America/Los_Angeles"},
    "LON00": {"name": "Wembley Stadium",           "lat": 51.556, "lon": -0.279,  "tz": "Europe/London"},
    "LON01": {"name": "Twickenham Stadium",        "lat": 51.456, "lon": -0.342,  "tz": "Europe/London"},
    "LON02": {"name": "Tottenham Hotspur Stadium", "lat": 51.604, "lon": -0.066,  "tz": "Europe/London"},
    "MEX00": {"name": "Estadio Azteca",            "lat": 19.303, "lon": -99.150, "tz": "America/Mexico_City"},
    "MIA00": {"name": "Hard Rock Stadium",         "lat": 25.958, "lon": -80.239, "tz": "America/New_York"},
    "MIN00": {"name": "Metrodome",                 "lat": 44.974, "lon": -93.258, "tz": "America/Chicago"},
    "MIN01": {"name": "U.S. Bank Stadium",         "lat": 44.974, "lon": -93.258, "tz": "America/Chicago"},
    "MIN98": {"name": "TCF Bank Stadium",          "lat": 44.976, "lon": -93.225, "tz": "America/Chicago"},
    "NAS00": {"name": "Nissan Stadium",            "lat": 36.166, "lon": -86.771, "tz": "America/Chicago"},
    "NOR00": {"name": "Caesars Superdome",         "lat": 29.951, "lon": -90.081, "tz": "America/Chicago"},
    "NYC00": {"name": "Giants Stadium",            "lat": 40.812, "lon": -74.077, "tz": "America/New_York"},
    "NYC01": {"name": "MetLife Stadium",           "lat": 40.814, "lon": -74.074, "tz": "America/New_York"},
    "OAK00": {"name": "Oakland Coliseum",          "lat": 37.752, "lon": -122.201, "tz": "America/Los_Angeles"},
    "PHI00": {"name": "Lincoln Financial Field",   "lat": 39.901, "lon": -75.168, "tz": "America/New_York"},
    "PHI99": {"name": "Veterans Stadium",          "lat": 39.906, "lon": -75.171, "tz": "America/New_York"},
    "PHO00": {"name": "State Farm Stadium",        "lat": 33.528, "lon": -112.263, "tz": "America/Phoenix"},
    "PHO99": {"name": "Sun Devil Stadium",         "lat": 33.426, "lon": -111.933, "tz": "America/Phoenix"},
    "PIT00": {"name": "Acrisure Stadium",          "lat": 40.447, "lon": -80.016, "tz": "America/New_York"},
    "SAN00": {"name": "Alamodome",                 "lat": 29.417, "lon": -98.479, "tz": "America/Chicago"},
    "SAO00": {"name": "Neo Química Arena",         "lat": -23.545, "lon": -46.474, "tz": "America/Sao_Paulo"},
    "SDG00": {"name": "Qualcomm Stadium",          "lat": 32.783, "lon": -117.119, "tz": "America/Los_Angeles"},
    "SEA00": {"name": "Lumen Field",               "lat": 47.595, "lon": -122.332, "tz": "America/Los_Angeles"},
    "SFO00": {"name": "Candlestick Park",          "lat": 37.713, "lon": -122.386, "tz": "America/Los_Angeles"},
    "SFO01": {"name": "Levi's Stadium",            "lat": 37.403, "lon": -121.970, "tz": "America/Los_Angeles"},
    "STL00": {"name": "Edward Jones Dome",         "lat": 38.633, "lon": -90.189, "tz": "America/Chicago"},
    "TAM00": {"name": "Raymond James Stadium",     "lat": 27.976, "lon": -82.503, "tz": "America/New_York"},
    "VEG00": {"name": "Allegiant Stadium",         "lat": 36.091, "lon": -115.184, "tz": "America/Los_Angeles"},
    "WAS00": {"name": "Commanders Field",          "lat": 38.908, "lon": -76.864, "tz": "America/New_York"},
}


# --------------------------------------------------------------------------- #
# 1. Raw per-(game, team) metrics
# --------------------------------------------------------------------------- #
def _rate(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Elementwise numer/denom, with 0-denominator -> NaN (not inf)."""
    out = numer / denom.replace(0, np.nan)
    return out


def build_team_game_log(pbp: pd.DataFrame, game_index: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team) with that game's RAW QB and defensive metrics.

    RAW = this game's own numbers. Outcome-derived, so SPEC 5.3 target leakage:
    these columns are cached here but are NEVER merged into the modeling frame
    (``build_game_features`` asserts their absence).
    """
    keep_games = set(game_index["game_id"])
    df = pbp[pbp["game_id"].isin(keep_games)].copy()

    # nflfastR play-type indicators (all seasons): `pass`==1 covers dropbacks
    # (throws + sacks + scrambles); `rush`==1 covers designed runs.
    is_pass = df["pass"].eq(1)
    is_rush = df["rush"].eq(1)
    is_db = df["qb_dropback"].eq(1)
    is_sack = df["sack"].eq(1)
    is_scramble = df.get("qb_scramble", pd.Series(0, index=df.index)).eq(1)
    is_throw = is_db & ~is_sack & ~is_scramble          # actual pass attempts
    is_play = (is_pass | is_rush) & df["epa"].notna()   # scrimmage plays

    # ---- offense, grouped by posteam -------------------------------------- #
    o = df[is_db].groupby(["game_id", "posteam"])
    off = pd.DataFrame({
        "n_dropbacks": o.size(),
        "qb_epa_per_db": o["qb_epa"].mean(),
        "qb_success_rate": o["success"].mean(),
        "sack_rate": o["sack"].mean(),
    })
    t = df[is_throw].groupby(["game_id", "posteam"])
    off["cpoe"] = t["cpoe"].mean()
    off["int_rate"] = t["interception"].mean()
    off["air_yards_per_att"] = t["air_yards"].mean()
    off = (off.reset_index().rename(columns={"posteam": "team"})
           [["game_id", "team", "n_dropbacks"] + QB_METRICS])

    # ---- defense, grouped by defteam ------------------------------------- #
    dplay = df[is_play].groupby(["game_id", "defteam"])
    dfn = pd.DataFrame({
        "n_def_plays": dplay.size(),
        "def_epa_per_play": dplay["epa"].mean(),
        "def_success_allowed": dplay["success"].mean(),
    })
    dfn["def_epa_early"] = (df[is_play & df["down"].isin([1, 2])]
                            .groupby(["game_id", "defteam"])["epa"].mean())
    dfn["def_epa_money"] = (df[is_play & df["down"].isin([3, 4])]
                            .groupby(["game_id", "defteam"])["epa"].mean())

    dbf = df[is_db].copy()
    dbf["_pressure"] = (dbf["qb_hit"].eq(1) | dbf["sack"].eq(1)).astype("float64")
    pg = dbf.groupby(["game_id", "defteam"]).agg(
        db_faced=("_pressure", "size"), pressure_sum=("_pressure", "sum"))
    dfn = dfn.join(pg, how="left")
    dfn["def_pressure_rate"] = _rate(dfn["pressure_sum"], dfn["db_faced"])

    dfn = (dfn.reset_index().rename(columns={"defteam": "team"})
           [["game_id", "team", "n_def_plays"] + DEF_METRICS])

    # ---- combine + attach game context --------------------------------- #
    tgl = off.merge(dfn, on=["game_id", "team"], how="outer")
    gi = game_index[["game_id", "season", "week", "kickoff", "home_team", "away_team"]].copy()
    # game_index carries schedule (era) team codes; pbp `team` is already modern.
    gi["home_team"] = _canon(gi["home_team"])
    gi["away_team"] = _canon(gi["away_team"])
    tgl = tgl.merge(gi, on="game_id", how="inner")
    tgl["is_home"] = tgl["team"].eq(tgl["home_team"])
    tgl["opponent"] = np.where(tgl["is_home"], tgl["away_team"], tgl["home_team"])
    tgl = tgl.drop(columns=["home_team", "away_team"])
    tgl = tgl.sort_values(["team", "kickoff", "game_id"]).reset_index(drop=True)
    return tgl


# --------------------------------------------------------------------------- #
# 2. Roll forward + shrinkage
# --------------------------------------------------------------------------- #
def roll_left(df: pd.DataFrame, col: str, window: int, by: list[str],
              agg: str = "mean") -> pd.Series:
    """Pattern A. Rolling aggregate over PRIOR rows only.

    ``closed='left'`` drops the current row from its own window -- this is the
    single argument that prevents SPEC 5.2 aggregation leakage. ``df`` must
    already be sorted so that each group's rows are in ascending time order.
    """
    grp = df.groupby(by, sort=False)[col]
    rolled = grp.rolling(window=window, min_periods=1, closed="left").agg(agg)
    rolled = rolled.reset_index(level=list(range(len(by))), drop=True)
    # groupby reorders rows by group; restore the caller's row order.
    return rolled.reindex(df.index)


def prior_season_mean(tgl: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pattern B. Each (team, season) -> that TEAM's mean of ``metric`` over its
    most recent strictly-earlier season (``merge_asof`` on the season axis,
    ``allow_exact_matches=False``)."""
    per = (tgl.groupby(["team", "season"])[metric].mean()
              .reset_index().rename(columns={metric: f"{metric}_prior"}))
    left = (tgl[["team", "season"]].drop_duplicates()
               .sort_values("season").reset_index(drop=True))
    out = pd.merge_asof(
        left, per.sort_values("season"),
        on="season", by="team",
        direction="backward", allow_exact_matches=False,
    )
    return out


def league_prior_mean(tgl: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pattern B (+ Pattern A seed for the earliest season).

    Each season -> the all-team mean of ``metric`` over strictly-earlier seasons
    (``merge_asof``, ``allow_exact_matches=False``). The earliest season in the
    dataset has no prior season, so it is seeded with a within-season,
    date-ordered ``closed='left'`` expanding league mean -- still leakage-free,
    still Pattern A.
    """
    per = (tgl.groupby("season")[metric].mean()
              .reset_index().rename(columns={metric: f"{metric}_league"}))
    seasons = pd.DataFrame({"season": sorted(tgl["season"].unique())})
    out = pd.merge_asof(
        seasons, per.sort_values("season"),
        on="season", direction="backward", allow_exact_matches=False,
    )
    first = seasons["season"].min()
    if out.loc[out["season"].eq(first), f"{metric}_league"].isna().all():
        g = tgl[tgl["season"].eq(first)].sort_values(["kickoff", "game_id"])
        seed = g[metric].rolling(window=len(g) + 1, min_periods=1,
                                 closed="left").mean().iloc[-1]
        out.loc[out["season"].eq(first), f"{metric}_league"] = seed
    return out


def shrink(n, in_season_avg, prior_avg, league_avg, k: int = DEFAULT_K):
    """SPEC Section 4 shrinkage, vectorised and pure.

    ``(n * in_season_avg + k * prior_avg) / (n + k)`` -- blends the in-season
    sample (weight = games played so far) with a prior (weight k). ``prior_avg``
    falls back to ``league_avg`` when the team has no prior season.

    Returns NaN only when a row has *no* usable information at all (no prior
    in-season games AND no prior-season/league value -- e.g. ``cpoe`` before
    2006, which the data simply does not carry). Those NaNs are left for the
    model's imputer (SPEC deviation note 1); they are never silently coerced to
    a fake 0.
    """
    n = np.asarray(n, dtype="float64")
    in_season_avg = np.asarray(in_season_avg, dtype="float64")
    prior = np.asarray(prior_avg, dtype="float64")
    league = np.asarray(league_avg, dtype="float64")

    prior = np.where(np.isnan(prior), league, prior)          # league fallback
    n0 = np.where(np.isnan(n), 0.0, n)
    have_in = (n0 > 0) & ~np.isnan(in_season_avg)
    have_prior = ~np.isnan(prior)

    num = (np.where(have_in, n0 * np.nan_to_num(in_season_avg), 0.0)
           + np.where(have_prior, k * np.nan_to_num(prior), 0.0))
    den = (np.where(have_in, n0, 0.0)
           + np.where(have_prior, float(k), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


def roll_features(tgl: pd.DataFrame, k: int = DEFAULT_K) -> pd.DataFrame:
    """Produce the model-facing rolled feature table, one row per (game_id, team).

    Per raw metric m:
      {m}_sd      Pattern A  rolling(window=22, closed='left').mean()  by [team, season]
      {m}_sd_n    Pattern A  rolling(window=22, closed='left').count() by [team, season]
      {m}_prior   Pattern B  merge_asof(on='season', by='team', allow_exact_matches=False)
      {m}_league  Pattern B  merge_asof(on='season', allow_exact_matches=False)
      {m}_shrunk  A + B      shrink({m}_sd_n, {m}_sd, {m}_prior, {m}_league, k)
    plus one shared:
      games_played_sd  Pattern A  rolling(window=22, closed='left').count() of game rows
    """
    df = tgl.sort_values(["team", "kickoff", "game_id"]).reset_index(drop=True)

    df["games_played_sd"] = (
        roll_left(df.assign(_one=1.0), "_one", IN_SEASON_WINDOW,
                  ["team", "season"], "count").fillna(0.0)
    )

    out_cols = ["game_id", "team", "season", "week", "kickoff", "is_home",
                "opponent", "games_played_sd"]
    keep = df[out_cols].copy()

    for m in RAW_METRICS:
        sd = roll_left(df, m, IN_SEASON_WINDOW, ["team", "season"], "mean")
        sd_n = roll_left(df, m, IN_SEASON_WINDOW, ["team", "season"], "count").fillna(0.0)

        prior_map = prior_season_mean(df, m)
        league_map = league_prior_mean(df, m)
        merged = (df[["team", "season"]]
                  .merge(prior_map, on=["team", "season"], how="left")
                  .merge(league_map, on="season", how="left"))

        keep[f"{m}_shrunk"] = shrink(
            sd_n.to_numpy(), sd.to_numpy(),
            merged[f"{m}_prior"].to_numpy(), merged[f"{m}_league"].to_numpy(), k,
        )
        # kept only for the leakage-check tripwire / manifest, not for the model:
        keep[f"{m}_sd"] = sd.to_numpy()
        keep[f"{m}_sd_n"] = sd_n.to_numpy()

    return keep


# --------------------------------------------------------------------------- #
# 3. Game index (+ kickoff) and contextual features
# --------------------------------------------------------------------------- #
def build_game_index(schedules: pd.DataFrame) -> pd.DataFrame:
    """Completed REG games, one row per game, with a UTC kickoff timestamp.

    Kickoff = ``gameday`` + ``gametime`` interpreted as US/Eastern, converted to
    naive UTC. ``gametime`` has zero nulls in the cached data; the 13:00 ET
    fallback only makes the SPEC 5.5 assertion stricter (earlier assumed
    kickoff), never looser.
    """
    reg = evaluate.completed_regular_season(schedules).copy()
    reg["_gametime"] = reg["gametime"].fillna("13:00")
    reg["_fallback_kickoff"] = reg["gametime"].isna()
    naive = pd.to_datetime(reg["gameday"] + " " + reg["_gametime"],
                           format="%Y-%m-%d %H:%M", errors="coerce")
    et = naive.dt.tz_localize("America/New_York", ambiguous="NaT",
                              nonexistent="shift_forward")
    reg["kickoff"] = et.dt.tz_convert("UTC").dt.tz_localize(None)
    reg["home_win"] = (reg["home_score"] > reg["away_score"]).astype(int)

    cols = ["game_id", "season", "week", "kickoff", "home_team", "away_team",
            "home_rest", "away_rest", "div_game", "roof", "location",
            "stadium_id", "home_win", "_fallback_kickoff"]
    gi = reg[cols].sort_values(["kickoff", "game_id"]).reset_index(drop=True)
    return gi


def _canon(series: pd.Series) -> pd.Series:
    return series.replace(TEAM_ABBR_CANON)


def team_home_stadium(schedules: pd.DataFrame) -> pd.DataFrame:
    """(team, season) -> stadium_id of that team's home venue that season.

    Modal ``stadium_id`` over the team's ``location=='Home'`` games. Used only as
    the AWAY team's travel origin -- a static physical fact, not a performance
    metric (the leakage checks treat it as context).
    """
    reg = schedules[schedules["game_type"].eq("REG")].copy()
    home = reg[reg["location"].eq("Home")][["season", "home_team", "stadium_id"]]
    home = home.rename(columns={"home_team": "team"})
    home["team"] = _canon(home["team"])
    modal = (home.groupby(["team", "season"])["stadium_id"]
                 .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
                 .reset_index())
    # Fill the odd gap (expansion year / all-neutral season) from the same team.
    modal = modal.sort_values(["team", "season"])
    modal["stadium_id"] = (modal.groupby("team")["stadium_id"]
                                .ffill().bfill())
    return modal


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between two arrays of coordinates."""
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, dtype="float64"),
                                              np.asarray(lon1, dtype="float64"),
                                              np.asarray(lat2, dtype="float64"),
                                              np.asarray(lon2, dtype="float64")))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def _utc_offset_hours(tz_name: str, when: pd.Timestamp) -> float:
    if not isinstance(tz_name, str) or pd.isna(when):
        return np.nan
    off = ZoneInfo(tz_name).utcoffset(when.to_pydatetime())
    return off.total_seconds() / 3600.0 if off is not None else np.nan


def contextual_features(game_index: pd.DataFrame, geo: dict,
                        home_stadium: pd.DataFrame) -> pd.DataFrame:
    """Pre-kickoff game facts (SPEC 5.3: all allowed -- known before the game)."""
    gi = game_index.copy()
    gi["home_team"] = _canon(gi["home_team"])
    gi["away_team"] = _canon(gi["away_team"])

    geo_df = pd.DataFrame(geo).T.rename_axis("stadium_id").reset_index()
    geo_df = geo_df.rename(columns={"lat": "g_lat", "lon": "g_lon", "tz": "g_tz"})
    gi = gi.merge(geo_df[["stadium_id", "g_lat", "g_lon", "g_tz"]],
                  on="stadium_id", how="left")

    hs = home_stadium.rename(columns={"team": "away_team",
                                      "stadium_id": "away_home_stadium"})
    gi = gi.merge(hs, on=["away_team", "season"], how="left")
    gi = gi.merge(hs.rename(columns={"away_team": "home_team",
                                     "away_home_stadium": "home_home_stadium"}),
                  on=["home_team", "season"], how="left")

    base = geo_df.set_index("stadium_id")[["g_lat", "g_lon", "g_tz"]]
    for side in ("home", "away"):
        sid = gi[f"{side}_home_stadium"]
        gi[f"{side}_base_lat"] = sid.map(base["g_lat"])
        gi[f"{side}_base_lon"] = sid.map(base["g_lon"])
        gi[f"{side}_base_tz"] = sid.map(base["g_tz"])

    ctx = pd.DataFrame({"game_id": gi["game_id"]})
    ctx["is_neutral"] = gi["location"].eq("Neutral").astype(int)
    ctx["home_rest"] = gi["home_rest"].clip(3, 20)
    ctx["away_rest"] = gi["away_rest"].clip(3, 20)
    ctx["rest_diff"] = ctx["home_rest"] - ctx["away_rest"]
    ctx["div_game"] = gi["div_game"].astype("float64")
    ctx["roof_indoor"] = gi["roof"].map(
        {"dome": 1.0, "closed": 1.0, "outdoors": 0.0, "open": 0.0}
    )
    ctx["travel_dist_home"] = haversine_km(gi["home_base_lat"], gi["home_base_lon"],
                                           gi["g_lat"], gi["g_lon"])
    ctx["travel_dist_away"] = haversine_km(gi["away_base_lat"], gi["away_base_lon"],
                                           gi["g_lat"], gi["g_lon"])
    ctx["travel_diff"] = ctx["travel_dist_away"] - ctx["travel_dist_home"]

    ctx["tz_home_offset"] = [
        _utc_offset_hours(tz, kt) for tz, kt in zip(gi["home_base_tz"], gi["kickoff"])
    ]
    ctx["tz_away_offset"] = [
        _utc_offset_hours(tz, kt) for tz, kt in zip(gi["away_base_tz"], gi["kickoff"])
    ]
    ctx["tz_diff"] = ctx["tz_home_offset"] - ctx["tz_away_offset"]
    return ctx


CONTEXT_FEATURES = [
    "is_neutral", "home_rest", "away_rest", "rest_diff", "div_game",
    "roof_indoor", "travel_dist_home", "travel_dist_away", "travel_diff",
    "tz_home_offset", "tz_away_offset", "tz_diff",
]

# Matchup differentials (SPEC 7.2): pure subtraction of two *_shrunk columns.
DIFF_FEATURES = {
    "mu_qb_epa_vs_def":  ("home_qb_epa_per_db_shrunk",   "away_def_epa_per_play_shrunk"),
    "mu_def_vs_qb_epa":  ("away_qb_epa_per_db_shrunk",   "home_def_epa_per_play_shrunk"),
    "mu_success_edge_h": ("home_qb_success_rate_shrunk",  "away_def_success_allowed_shrunk"),
    "mu_success_edge_a": ("away_qb_success_rate_shrunk",  "home_def_success_allowed_shrunk"),
    "mu_money_down_edge": ("home_qb_epa_per_db_shrunk",   "away_def_epa_money_shrunk"),
    "mu_protection_edge": ("away_def_pressure_rate_shrunk", "home_sack_rate_shrunk"),
}

MODEL_SHRUNK = [f"{m}_shrunk" for m in RAW_METRICS]


# --------------------------------------------------------------------------- #
# 4. Assembly
# --------------------------------------------------------------------------- #
def build_game_features(start_season: int = DEFAULT_START,
                        end_season: int = DEFAULT_END,
                        k: int = DEFAULT_K,
                        cache: bool = True) -> pd.DataFrame:
    """Assemble the game-level modeling frame + write ``feature_manifest.json``."""
    schedules = evaluate.load_schedules()
    game_index = build_game_index(schedules)
    game_index = game_index[game_index["season"].between(start_season, end_season)]
    game_index = game_index.reset_index(drop=True)

    pbp = data_ingest.get_pbp(start_season, end_season)

    tgl = build_team_game_log(pbp, game_index)
    if cache:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        tgl.to_parquet(PROC_DIR / "team_game_log.parquet", index=False)

    rolled = roll_features(tgl, k=k)

    model_cols = ["game_id", "season", "week", "kickoff", "games_played_sd"] + MODEL_SHRUNK
    home = (rolled[rolled["is_home"]][["game_id"] + [c for c in model_cols if c != "game_id"]]
            .add_prefix("home_").rename(columns={"home_game_id": "game_id"}))
    away = (rolled[~rolled["is_home"]][["game_id"] + MODEL_SHRUNK + ["games_played_sd"]]
            .add_prefix("away_").rename(columns={"away_game_id": "game_id"}))
    frame = home.merge(away, on="game_id", how="inner")
    frame = frame.drop(columns=["home_season", "home_week", "home_kickoff"])

    ctx = contextual_features(game_index, STADIUM_GEO, team_home_stadium(schedules))
    frame = frame.merge(ctx, on="game_id", how="inner")

    frame = frame.merge(
        game_index[["game_id", "season", "week", "kickoff", "home_win"]],
        on="game_id", how="inner",
    )

    for name, (a, b) in DIFF_FEATURES.items():
        frame[name] = frame[a] - frame[b]

    # ---- SPEC 5.3 / 5.2 guards ---------------------------------------- #
    raw_leaks = [c for c in frame.columns
                 if any(c == f"{side}_{m}" for side in ("home", "away") for m in RAW_METRICS)]
    assert not raw_leaks, f"raw outcome-derived metrics leaked into frame: {raw_leaks}"
    vegas_leaks = [c for c in frame.columns if any(v in c for v in _VEGAS_COLS)]
    assert not vegas_leaks, f"Vegas columns leaked into frame: {vegas_leaks}"

    frame = frame.sort_values(["kickoff", "game_id"]).reset_index(drop=True)

    if cache:
        frame.to_parquet(PROC_DIR / "game_features.parquet", index=False)
        _write_manifest(k)
    return frame


def feature_columns() -> list[str]:
    """Model-facing feature columns (excludes keys, label, tripwire intermediates)."""
    cols = []
    for side in ("home", "away"):
        cols += [f"{side}_{m}_shrunk" for m in RAW_METRICS]
        cols.append(f"{side}_games_played_sd")
    cols += list(DIFF_FEATURES)
    cols += CONTEXT_FEATURES
    return cols


def _write_manifest(k: int) -> None:
    A = "rolling(window=%d, closed='left')" % IN_SEASON_WINDOW
    B_team = "merge_asof(on='season', by='team', direction='backward', allow_exact_matches=False)"
    B_league = "merge_asof(on='season', direction='backward', allow_exact_matches=False)"
    features: dict[str, dict] = {}

    for side in ("home", "away"):
        for m in RAW_METRICS:
            features[f"{side}_{m}_shrunk"] = {
                "pattern": "A+B",
                "source_metric": m,
                "components": {
                    "in_season_avg": {"pattern": "A", "call": f"{A}.mean()", "by": ["team", "season"]},
                    "n_games": {"pattern": "A", "call": f"{A}.count()", "by": ["team", "season"]},
                    "prior_avg": {"pattern": "B", "call": B_team},
                    "league_avg": {"pattern": "B", "call": B_league,
                                   "note": "earliest season seeded with a within-season closed='left' league mean (A)"},
                },
                "combine": "shrink(n_games, in_season_avg, prior_avg, league_avg, k=%d)" % k,
                "leakage_category": ["3", "2"],
                "leakage_note": "current game excluded by closed='left'; prior/league are strictly earlier seasons",
            }
        features[f"{side}_games_played_sd"] = {
            "pattern": "A", "call": f"{A}.count() of game rows", "by": ["team", "season"],
            "leakage_category": ["3"],
            "leakage_note": "current game excluded by closed='left'",
        }
    for name, (a, b) in DIFF_FEATURES.items():
        features[name] = {"pattern": "A+B (derived diff)", "formula": f"{a} - {b}",
                          "leakage_note": "inherits the two *_shrunk parents"}
    for c in CONTEXT_FEATURES:
        features[c] = {"pattern": "context",
                       "leakage_note": "schedule fact / static geo, fixed before kickoff (SPEC 5.3 allowed)"}

    manifest = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "in_season_window": IN_SEASON_WINDOW,
        "shrinkage_k": k,
        "n_model_features": len(feature_columns()),
        "raw_metrics_never_exposed": RAW_METRICS,
        "features": features,
    }
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    (PROC_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _null_table(frame: pd.DataFrame) -> pd.DataFrame:
    by_season = frame.groupby("season")
    rows = []
    for col in ["home_cpoe_shrunk", "away_cpoe_shrunk",
                "home_air_yards_per_att_shrunk", "away_air_yards_per_att_shrunk"]:
        s = by_season[col].apply(lambda x: float(x.isna().mean()))
        rows.append(s.rename(col))
    return pd.concat(rows, axis=1).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=DEFAULT_END)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    frame = build_game_features(args.start, args.end, k=args.k)
    tgl = pd.read_parquet(PROC_DIR / "team_game_log.parquet")
    gi = build_game_index(evaluate.load_schedules())
    gi = gi[gi["season"].between(args.start, args.end)]

    print(f"\nteam_game_log.parquet : {tgl.shape[0]:,} rows  ({tgl.shape[1]} cols)")
    print(f"game_features.parquet : {frame.shape[0]:,} rows  ({frame.shape[1]} cols)")
    print(f"  expected ~= completed REG games in range: {len(gi):,}")
    print(f"  gametime fallback used: {int(gi['_fallback_kickoff'].sum())}")
    print(f"  model-facing features : {len(feature_columns())}")

    print("\nrows per season:")
    print(frame.groupby("season").size().to_string())

    print("\nNaN fraction, cpoe / air_yards shrunk features (expect ~1.0 for 2002-2005):")
    print(_null_table(frame).to_string())

    print("\nhome_win rate in frame:", round(frame["home_win"].mean(), 4))

    print("\nfeature_manifest.json (pattern per feature):")
    man = json.loads((PROC_DIR / "feature_manifest.json").read_text())
    for name, meta in man["features"].items():
        print(f"  {name:<34} {meta['pattern']}")

    print("\nsample rows (1 early-season, 1 late-season):")
    cols = ["game_id", "season", "week", "home_win", "home_qb_epa_per_db_shrunk",
            "away_def_epa_per_play_shrunk", "mu_qb_epa_vs_def", "rest_diff",
            "travel_diff", "tz_diff", "roof_indoor"]
    wk1 = frame[frame["week"].eq(1)].iloc[0]
    late = frame[frame["week"].ge(15)].iloc[-1]
    print(frame.loc[[wk1.name, late.name], cols].to_string())


if __name__ == "__main__":
    main()
