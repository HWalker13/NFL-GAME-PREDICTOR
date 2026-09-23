"""Phase 10: grade one week's logged predictions.

    python -m scripts.live.grade_week --season 2026 --week N           # scratch
    python -m scripts.live.grade_week --season 2026 --week N --live    # OFFICIAL record

Reads week_NN_predictions.csv (never modified) and final scores from the
latest schedule snapshot (run ``python -m src.data_ingest --refresh-season``
first; the same freshness guard as predict_week applies). Writes
week_NN_graded.csv -- regenerable: grading bugs are fixed here and the file
is rebuilt; the predictions file is never touched.

Per game x model:
  correct / log loss / Brier (ties: excluded, like every evaluation in this
  project); paper-bet outcome in units (win = profit at the logged price,
  loss = -1, tie = void, stake returned, 0 units); and line movement.

Line movement is measured to the **last pre-kickoff snapshot**: the latest
snapshot taken strictly before that game's kickoff. It is NOT the closing
line -- snapshots exist only when the owner refreshes. If that snapshot is
not later than the prediction snapshot (e.g. Thursday games when nothing was
pulled in between), CLV is excluded for that game and counted.
  bet CLV            = vig-free implied prob of the BET side, last pre-kickoff
                       snapshot minus prediction snapshot (positive = the line
                       moved toward the bet)
  line move -> model = (p_late - p_pred) * sign(p_model - p_pred), home terms,
                       every game (bet or not)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.live import common as C

CLV_MEASURED = "measured"
CLV_NO_LATER = "excluded: no pre-kickoff snapshot after the prediction snapshot"
CLV_NO_LINE_PRED = "excluded: missing line at prediction"
CLV_NO_LINE_LATE = "excluded: missing line in last pre-kickoff snapshot"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--week", type=int, required=True)
    return ap.parse_args(argv)


def _lines(path: Path, cache: dict) -> pd.DataFrame:
    if path not in cache:
        cache[path] = pd.read_parquet(path)[["game_id", "home_moneyline", "away_moneyline",
                                             "home_score", "away_score"]].set_index("game_id")
    return cache[path]


def last_prekickoff_snapshot(snapshots: list[Path], kickoff: pd.Timestamp) -> Path | None:
    before = [s for s in snapshots if C.snapshot_time(s) < kickoff]
    return before[-1] if before else None


def grade(pred: pd.DataFrame, snapshots: list[Path], scores_snapshot: Path,
          now: pd.Timestamp) -> pd.DataFrame:
    """Pure grading of a predictions frame. ``snapshots`` oldest first."""
    cache: dict = {}
    scores = _lines(scores_snapshot, cache)
    rows = []
    for r in pred.itertuples(index=False):
        g = r._asdict()
        kickoff = pd.Timestamp(r.kickoff_utc).tz_localize(None) if pd.Timestamp(r.kickoff_utc).tzinfo \
            else pd.Timestamp(r.kickoff_utc)
        hs = scores.at[r.game_id, "home_score"] if r.game_id in scores.index else np.nan
        as_ = scores.at[r.game_id, "away_score"] if r.game_id in scores.index else np.nan
        g.update(home_score=hs, away_score=as_, scores_snapshot_file=scores_snapshot.name)

        if pd.isna(hs) or pd.isna(as_):
            g.update(result="pending", home_win=np.nan)
        elif hs == as_:
            g.update(result="tie", home_win=np.nan)
        else:
            g.update(result="home" if hs > as_ else "away", home_win=float(hs > as_))
        y, p = g["home_win"], float(r.p_home)
        if pd.notna(y):
            pc = min(max(p, 1e-15), 1 - 1e-15)
            g.update(correct=float((r.pick == "home") == (y == 1.0)),
                     log_loss=-(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
                     brier=(p - y) ** 2, home_correct=float(y == 1.0))
        else:
            g.update(correct=np.nan, log_loss=np.nan, brier=np.nan, home_correct=np.nan)

        fav = C.vegas_favorite(r.spread_line, r.home_moneyline, r.away_moneyline)
        imp = r.implied_p_home
        g["vegas_fav"] = fav
        if pd.notna(y) and fav != "unresolved":
            g["vegas_correct"] = float((fav == "home") == (y == 1.0))
        else:
            g["vegas_correct"] = np.nan
        if pd.notna(y) and pd.notna(imp):
            ic = min(max(imp, 1e-15), 1 - 1e-15)
            g.update(vegas_log_loss=-(y * np.log(ic) + (1 - y) * np.log(1 - ic)), vegas_brier=(imp - y) ** 2)
        else:
            g.update(vegas_log_loss=np.nan, vegas_brier=np.nan)

        outcome, units = C.bet_units(r.paper_bet, r.bet_price, hs, as_)
        g.update(bet_outcome=outcome, units=units)

        late = last_prekickoff_snapshot(snapshots, kickoff)
        pred_time = C.snapshot_time(Path(r.snapshot_file))
        g.update(last_prekickoff_snapshot_file=late.name if late else "",
                 last_prekickoff_snapshot_pulled_at_utc=C.iso(C.snapshot_time(late)) if late else "",
                 p_late_home=np.nan, bet_clv=np.nan, line_move_toward_model=np.nan)
        if pd.isna(imp):
            g["clv_status"] = CLV_NO_LINE_PRED
        elif late is None or C.snapshot_time(late) <= pred_time:
            g["clv_status"] = CLV_NO_LATER
        else:
            ln = _lines(late, cache)
            hm = ln.at[r.game_id, "home_moneyline"] if r.game_id in ln.index else np.nan
            am = ln.at[r.game_id, "away_moneyline"] if r.game_id in ln.index else np.nan
            if pd.isna(hm) or pd.isna(am):
                g["clv_status"] = CLV_NO_LINE_LATE
            else:
                pl = float(C.vig_free_home(hm, am))
                g.update(clv_status=CLV_MEASURED, p_late_home=pl,
                         bet_clv=C.bet_clv(r.paper_bet, imp, pl),
                         line_move_toward_model=C.line_move_toward_model(p, imp, pl))
        g["graded_at_utc"] = C.iso(now)
        rows.append(g)
    return pd.DataFrame(rows)


def write_graded(df: pd.DataFrame, path: Path) -> None:
    """Regenerable output: atomic replace, overwrite allowed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp, index=False, float_format="%.10g")
    tmp.replace(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = C.root_dir(args.live)
    pred_path = C.predictions_path(root, args.season, args.week)
    out = C.graded_path(root, args.season, args.week)
    print(f"grade_week  season={args.season} week={args.week}  "
          f"mode={'OFFICIAL (--live)' if args.live else 'SCRATCH (no --live)'}")
    if not pred_path.exists():
        print(f"REFUSING: {pred_path} does not exist")
        return 2
    try:
        scores_snap = C.check_freshness(args.season)
    except C.GuardError as exc:
        print(f"REFUSING: {exc}")
        return 2
    pred_sha_before = C.sha256_file(pred_path)
    pred = pd.read_csv(pred_path)
    graded = grade(pred, C.list_snapshots(args.season), scores_snap, C.naive_utc(C.utc_now()))
    assert C.sha256_file(pred_path) == pred_sha_before, "predictions file changed during grading"
    write_graded(graded, out)

    g = graded[graded["model"] == C.OFFICIAL_MODEL]
    print(f"scores from {scores_snap.name}; games: {g['game_id'].nunique()}  "
          f"(final {int(g['result'].isin(['home', 'away']).sum())}, tie {int((g['result'] == 'tie').sum())}, "
          f"pending {int((g['result'] == 'pending').sum())})")
    print("CLV status (champion rows):")
    print(g["clv_status"].value_counts().to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
