"""Shared pieces of the Phase 10 shadow-mode scripts (predict / grade / scorecard).

Everything that decides a number or guards the official record lives here, so
the tests (``scripts/test_phase10_live.py``) exercise the exact functions the
scripts use, by direct calls on temporary files.

Output locations:
  scratch (default)  data/live_scratch/<season>/...   gitignored
  official (--live)  data/live/<season>/...           tracked; owner-run only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SNAPSHOT_DIR = RAW_DIR / "snapshots"
METADATA_PATH = RAW_DIR / "pull_metadata.json"
OFFICIAL_ROOT = PROJECT_ROOT / "data" / "live"
SCRATCH_ROOT = PROJECT_ROOT / "data" / "live_scratch"
FEATURES_PY = PROJECT_ROOT / "src" / "features.py"

# Frozen live models (Phase 10 B4). Any other bytes -> refuse.
FROZEN_MODELS = {
    "champion": ("models/live_2026_champion.joblib",
                 "f813a2b0cb58d3202c547ca30c8dc90e242a84a8a9b74d5a82f43293571c1006"),
    "challenger": ("models/live_2026_challenger_calibrated.joblib",
                   "185d565f2c1202cb693a527766f875d259b2c118140933efe776621c13daad28"),
}
# The challenger's bets are computed but hypothetical. Changed ONLY by the
# pre-registered week-9 checkpoint (docs/PHASE10_PREREG.md section 5), effective
# from week 10; every logged row records its own bet_status, so earlier weeks
# are never rewritten.
OFFICIAL_MODEL = "champion"

# --live refuses until the pre-registration carries this EXACT line (whole-line
# match -- the Phase 8 substring bug is why).
PREREG = PROJECT_ROOT / "docs" / "PHASE10_PREREG.md"
APPROVED_MARKER = "**Status:** APPROVED"

# --live refuses a prediction snapshot older than this at run time (D3): the
# relative freshness check alone would accept a Monday refresh on Wednesday.
MAX_LIVE_SNAPSHOT_AGE_HOURS = 6.0

EDGE_THRESHOLD = 0.04
# Edges are compared after rounding to 10 dp, so an edge that is exactly 0.04
# on paper (e.g. 0.57 - 0.53 = 0.03999999999999992 in binary floating point)
# counts as 0.04. No real-world edge is decided by the 11th decimal.
EDGE_DECIMALS = 10
STAKE = 1.0

SNAPSHOT_RE = re.compile(r"^schedules_(\d{4})_(\d{8}T\d{6}Z)\.parquet$")


class GuardError(RuntimeError):
    """A pre-condition for writing output failed. Nothing was written."""


# --------------------------------------------------------------------------- #
# Paths / CLI
# --------------------------------------------------------------------------- #
def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--live", action="store_true", default=False,
                    help="write the OFFICIAL record under data/live/ (owner only). "
                         "Default: scratch under data/live_scratch/.")


def root_dir(live: bool) -> Path:
    return OFFICIAL_ROOT if live else SCRATCH_ROOT


def predictions_path(root: Path, season: int, week: int) -> Path:
    return root / str(season) / "predictions" / f"week_{week:02d}_predictions.csv"


def graded_path(root: Path, season: int, week: int) -> Path:
    return root / str(season) / "graded" / f"week_{week:02d}_graded.csv"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts) -> str:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or pd.isna(ts):
        return ""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def naive_utc(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prereg_approved(path: Path = PREREG) -> bool:
    """True only if some line of ``path``, stripped, equals APPROVED_MARKER."""
    path = Path(path)
    if not path.exists():
        return False
    return any(line.strip() == APPROVED_MARKER for line in path.read_text().splitlines())


# --------------------------------------------------------------------------- #
# Write-once output
# --------------------------------------------------------------------------- #
def write_once_readonly(write_fn, path: Path) -> str:
    """Create ``path`` via ``write_fn(tmp_path)``; refuse if it exists; make it
    read-only. Published with ``os.link`` (atomic, fails if the name exists),
    so even a race cannot overwrite. Returns the sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise GuardError(f"{path} already exists -- official predictions are never overwritten")
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.unlink(missing_ok=True)
    write_fn(tmp)
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise GuardError(f"{path} already exists -- official predictions are never overwritten") from exc
    finally:
        tmp.unlink(missing_ok=True)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return sha256_file(path)


# --------------------------------------------------------------------------- #
# Snapshots + freshness
# --------------------------------------------------------------------------- #
def snapshot_time(path: Path) -> pd.Timestamp:
    m = SNAPSHOT_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"not a snapshot filename: {path}")
    return pd.Timestamp(datetime.strptime(m.group(2), "%Y%m%dT%H%M%SZ"))


def list_snapshots(season: int, snap_dir: Path = SNAPSHOT_DIR) -> list[Path]:
    """Snapshots of ``season``, oldest first (by the UTC time in the name)."""
    snaps = [p for p in snap_dir.glob(f"schedules_{season}_*.parquet") if SNAPSHOT_RE.match(p.name)]
    return sorted(snaps, key=snapshot_time)


def latest_snapshot(season: int, snap_dir: Path = SNAPSHOT_DIR) -> Path:
    snaps = list_snapshots(season, snap_dir)
    if not snaps:
        raise GuardError(f"no schedule snapshot for {season} in {snap_dir}")
    return snaps[-1]


def check_freshness(season: int, raw_dir: Path = RAW_DIR, snap_dir: Path = SNAPSHOT_DIR,
                    metadata_path: Path = METADATA_PATH) -> Path:
    """Refuse unless the latest snapshot is at least as new as the latest raw
    pull of this season (pbp and schedule) AND is byte-identical to the
    schedule file the features are built from. Returns the snapshot path."""
    snap = latest_snapshot(season, snap_dir)
    meta = json.loads(Path(metadata_path).read_text()) if Path(metadata_path).exists() else {}
    pulls = {}
    for kind in ("pbp", "schedules"):
        name = f"{kind}_{season}.parquet"
        if not (raw_dir / name).exists():
            raise GuardError(f"{name} missing from {raw_dir} -- refresh first")
        if name not in meta:
            raise GuardError(f"{name} has no pull metadata -- refresh first")
        pulls[name] = pd.Timestamp(meta[name]["pulled_at_utc"]).tz_localize(None)
    newest_pull = max(pulls.values())
    if snapshot_time(snap) < newest_pull:
        raise GuardError(f"latest snapshot {snap.name} ({iso(snapshot_time(snap))}) is older than the "
                         f"latest raw pull ({iso(newest_pull)}) -- refresh again")
    if sha256_file(snap) != sha256_file(raw_dir / f"schedules_{season}.parquet"):
        raise GuardError(f"latest snapshot {snap.name} is not byte-identical to "
                         f"schedules_{season}.parquet -- refresh again")
    return snap


def check_snapshot_age(snap: Path, now: pd.Timestamp,
                       max_hours: float = MAX_LIVE_SNAPSHOT_AGE_HOURS) -> float:
    """Refuse if the snapshot was pulled more than ``max_hours`` before ``now``
    (naive UTC). Returns the age in hours."""
    age = (now - snapshot_time(snap)).total_seconds() / 3600.0
    if age > max_hours:
        raise GuardError(f"snapshot {Path(snap).name} is {age:.1f} h old (> {max_hours:g} h) -- "
                         "refresh immediately before predicting")
    return age


def check_pbp_complete(schedule: pd.DataFrame, pbp_game_ids: set, season: int, week: int,
                       now: pd.Timestamp) -> list[str]:
    """Refuse unless every completed REG game of ``season`` before ``week`` has
    play-by-play, and no earlier-week game is past kickoff without a final
    score (result not in yet). Returns game_ids of earlier-week games that are
    not played and not yet due (postponed/rescheduled) -- logged, not fatal.
    ``schedule`` needs ``kickoff`` (naive UTC)."""
    s = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")
                 & (schedule["week"] < week)]
    scored = s["home_score"].notna() & s["away_score"].notna()
    missing = sorted(set(s.loc[scored, "game_id"]) - set(pbp_game_ids))
    if missing:
        raise GuardError(f"pbp missing for {len(missing)} completed game(s) before week {week}: {missing}")
    pending = s[~scored & (s["kickoff"] <= now)]
    if len(pending):
        raise GuardError(f"{len(pending)} game(s) before week {week} are past kickoff with no final "
                         f"score yet: {sorted(pending['game_id'])} -- wait and refresh")
    return sorted(s.loc[~scored, "game_id"])


def check_frozen_models(project_root: Path = PROJECT_ROOT, models: dict = FROZEN_MODELS) -> dict:
    """Refuse unless every live model file has its pre-registered sha256.
    Returns {name: (path, sha)}."""
    out = {}
    for name, (rel, want) in models.items():
        p = project_root / rel
        if not p.exists():
            raise GuardError(f"{rel} missing")
        got = sha256_file(p)
        if got != want:
            raise GuardError(f"{rel} sha256 {got} != pre-registered {want} -- the frozen model changed")
        out[name] = (p, got)
    return out


def check_features_py(recorded_sha: str, path: Path = FEATURES_PY) -> None:
    got = sha256_file(path)
    if got != recorded_sha:
        raise GuardError(f"src/features.py sha256 {got} != the one the live models were trained "
                         f"with ({recorded_sha}) -- the feature code changed mid-season")


# --------------------------------------------------------------------------- #
# Lines + bet rule
# --------------------------------------------------------------------------- #
def american_to_prob(ml) -> np.ndarray:
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):   # np.where evaluates both branches
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def vig_free_home(home_ml, away_ml) -> np.ndarray:
    """Vig-free implied P(home): raw implied probs normalised to sum to 1
    (Phase 7 convention). NaN if either line is missing."""
    rh, ra = american_to_prob(home_ml), american_to_prob(away_ml)
    return rh / (rh + ra)


def profit_per_unit(price) -> np.ndarray:
    """Profit on a 1-unit winning bet at American odds (stake returned on top)."""
    price = np.asarray(price, dtype=float)
    return np.where(price > 0, price / 100.0, 100.0 / np.abs(price))


def edge_qualifies(edge: float, threshold: float = EDGE_THRESHOLD) -> bool:
    """edge >= threshold, compared at EDGE_DECIMALS so float noise in an
    edge that is exactly the threshold on paper cannot flip the decision."""
    return round(float(edge), EDGE_DECIMALS) >= threshold


def paper_bet(p_home: float, home_ml: float, away_ml: float,
              threshold: float = EDGE_THRESHOLD) -> dict:
    """The pre-registered rule for one game and one model.

    Bet 1 unit on the team whose model probability exceeds its vig-free
    implied probability by >= ``threshold``, at that team's moneyline (vig
    included). The two edges sum to zero, so at most one side qualifies.
    Missing line -> no bet.
    """
    if pd.isna(home_ml) or pd.isna(away_ml):
        return {"implied_p_home": np.nan, "edge_home": np.nan, "edge_away": np.nan,
                "paper_bet": "none", "no_bet_reason": "missing_line",
                "bet_price": np.nan, "stake": 0.0}
    imp = float(vig_free_home(home_ml, away_ml))
    edge_home = float(p_home) - imp
    edge_away = (1.0 - float(p_home)) - (1.0 - imp)
    out = {"implied_p_home": imp, "edge_home": edge_home, "edge_away": edge_away}
    if edge_qualifies(edge_home, threshold):
        out.update(paper_bet="home", no_bet_reason="", bet_price=float(home_ml), stake=STAKE)
    elif edge_qualifies(edge_away, threshold):
        out.update(paper_bet="away", no_bet_reason="", bet_price=float(away_ml), stake=STAKE)
    else:
        out.update(paper_bet="none", no_bet_reason="edge_below_threshold", bet_price=np.nan, stake=0.0)
    return out


def bet_units(paper_bet_side: str, bet_price: float, home_score, away_score) -> tuple[str, float]:
    """(outcome, units) for one paper bet. Ties void (stake returned, 0 units)."""
    if paper_bet_side not in ("home", "away"):
        return "none", 0.0
    if pd.isna(home_score) or pd.isna(away_score):
        return "pending", np.nan
    if home_score == away_score:
        return "void", 0.0
    home_won = home_score > away_score
    won = home_won if paper_bet_side == "home" else not home_won
    return ("win", float(profit_per_unit(bet_price))) if won else ("loss", -STAKE)


def bet_clv(paper_bet_side: str, p_pred_home: float, p_late_home: float) -> float:
    """Change in vig-free implied probability of the BET side, prediction
    snapshot -> last pre-kickoff snapshot. Positive = the line moved toward
    the bet (the market came to agree more with it)."""
    if paper_bet_side == "home":
        return p_late_home - p_pred_home
    if paper_bet_side == "away":
        return (1.0 - p_late_home) - (1.0 - p_pred_home)
    return np.nan


def line_move_toward_model(p_model_home: float, p_pred_home: float, p_late_home: float) -> float:
    """(p_late - p_pred) * sign(p_model - p_pred), all in home terms. Positive
    = the market moved toward the model's view, for every game (bet or not)."""
    return (p_late_home - p_pred_home) * float(np.sign(p_model_home - p_pred_home))


def vegas_favorite(spread_line, home_ml, away_ml) -> str:
    """Phase 7 convention: side favored by spread_line (positive = home); a
    pick'em falls back to the lower moneyline; else unresolved."""
    if pd.notna(spread_line) and spread_line != 0:
        return "home" if spread_line > 0 else "away"
    if pd.notna(home_ml) and pd.notna(away_ml) and home_ml != away_ml:
        return "home" if home_ml < away_ml else "away"
    return "unresolved"
