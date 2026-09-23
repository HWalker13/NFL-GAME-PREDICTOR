"""Phase 10: log pre-kickoff predictions + paper bets for one week.

    python -m scripts.live.predict_week --season 2026 --week N           # scratch
    python -m scripts.live.predict_week --season 2026 --week N --live    # OFFICIAL (owner only)

Refuses (writes nothing) unless:
  - with --live: docs/PHASE10_PREREG.md has a line that is exactly
    "**Status:** APPROVED";
  - the output file does not exist yet (official predictions are write-once);
  - both frozen model files match their pre-registered sha256, and
    src/features.py matches the sha256 the models were trained with;
  - the latest schedule snapshot is at least as new as the latest raw pull and
    byte-identical to the schedule the features are built from; with --live it
    must also be at most 6 hours old at run time;
  - pbp holds every completed game before week N, and no earlier-week game is
    past kickoff without a final score.

Then, for every week-N game that has NOT kicked off at run time (kicked-off
games are excluded and logged), one row per model: probabilities, the
prediction-snapshot moneylines, vig-free implied probability, edges and the
pre-registered paper bet (champion = official, challenger = hypothetical).

Writes (each write-once, read-only) under data/live_scratch/<season>/predictions/
or, with --live, data/live/<season>/predictions/:
  week_NN_predictions.csv   the record
  week_NN_run.json          guards, hashes, exclusions
  week_NN_features.parquet  the exact feature rows each model saw
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from scripts.live import common as C
from src import features as F, live_features as LF, live_models as LM

MODEL_ORDER = ("champion", "challenger")
COLUMNS = ["season", "week", "game_id", "kickoff_utc", "home_team", "away_team",
           "model", "model_file", "model_sha256", "p_home", "pick",
           "snapshot_file", "snapshot_pulled_at_utc", "home_moneyline", "away_moneyline",
           "spread_line", "implied_p_home", "edge_home", "edge_away",
           "paper_bet", "bet_status", "no_bet_reason", "bet_price", "stake", "predicted_at_utc"]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--week", type=int, required=True)
    return ap.parse_args(argv)


def split_kicked_off(rows: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(to_predict, excluded). A game is excluded if its kickoff is at or
    before ``now`` or it already has a result -- it is never predicted."""
    gone = (rows["kickoff"] <= now) | rows["is_played"].astype(bool)
    return rows[~gone].copy(), rows[gone].copy()


def prediction_rows(rows: pd.DataFrame, lines: pd.DataFrame, model: str, model_file: str,
                    model_sha: str, proba_fn, cols: list[str], snap_name: str,
                    snap_time: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    """``proba_fn(X) -> P(home)``: the saved model's pipeline, followed by its
    calibrator if it has one (live_models.calibrated_proba)."""
    df = rows.merge(lines, on="game_id", how="left", validate="1:1")
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    p = proba_fn(df[cols])
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        bet = C.paper_bet(p[i], r.home_moneyline, r.away_moneyline)
        out.append({
            "season": int(r.season), "week": int(r.week), "game_id": r.game_id,
            "kickoff_utc": C.iso(r.kickoff), "home_team": r.home_team, "away_team": r.away_team,
            "model": model, "model_file": model_file, "model_sha256": model_sha,
            "p_home": float(p[i]), "pick": "home" if p[i] > 0.5 else "away",
            "snapshot_file": snap_name, "snapshot_pulled_at_utc": C.iso(snap_time),
            "home_moneyline": r.home_moneyline, "away_moneyline": r.away_moneyline,
            "spread_line": r.spread_line,
            "bet_status": ("official" if model == C.OFFICIAL_MODEL else "hypothetical")
                          if bet["paper_bet"] != "none" else "",
            "predicted_at_utc": C.iso(now), **bet,
        })
    return pd.DataFrame(out)[COLUMNS]


def main(argv=None) -> int:
    args = parse_args(argv)
    season, week = args.season, args.week
    root = C.root_dir(args.live)
    out_csv = C.predictions_path(root, season, week)
    out_run = out_csv.with_name(f"week_{week:02d}_run.json")
    out_feat = out_csv.with_name(f"week_{week:02d}_features.parquet")
    mode = "OFFICIAL (--live)" if args.live else "SCRATCH (no --live)"
    print(f"predict_week  season={season} week={week}  mode={mode}\n  output: {out_csv}")

    try:
        if args.live and not C.prereg_approved():
            raise C.GuardError(f"{C.PREREG.name} has no line '{C.APPROVED_MARKER}' -- the official "
                               "record starts only after the owner approves the pre-registration")
        for p in (out_csv, out_run, out_feat):
            if p.exists():
                raise C.GuardError(f"{p} already exists -- predictions are write-once, never overwritten")
        now = C.naive_utc(C.utc_now())
        models = C.check_frozen_models()
        loaded = {name: joblib.load(path) for name, (path, _) in models.items()}
        for name, d in loaded.items():
            C.check_features_py(d["features_py_sha256"])
        assert loaded["champion"]["elo_home_adv"] == F.ELO_HOME_ADV, "champion must use the frozen ELO_HOME_ADV"
        snap = C.check_freshness(season)
        snap_time = C.snapshot_time(snap)
        if args.live:
            C.check_snapshot_age(snap, now)
        schedules, pbp = LF.load_raw(season)
        sched = LF.season_schedule_with_kickoff(schedules, season)
        postponed = C.check_pbp_complete(sched, set(pbp.loc[pbp["season"] == season, "game_id"]),
                                         season, week, now)
    except (C.GuardError, AssertionError) as exc:
        print(f"REFUSING: {exc}")
        return 2
    print(f"  guards passed: output absent; frozen models + features.py sha256 match; "
          f"snapshot {snap.name} is fresh; pbp complete before week {week}")
    if postponed:
        print(f"  logged: earlier-week games not played and not yet due: {postponed}")

    lines = pd.read_parquet(snap)[["game_id", "home_team", "away_team",
                                   "home_moneyline", "away_moneyline", "spread_line"]]
    preds, feats, excluded, week_ids, frames = [], [], None, None, {}
    for name in MODEL_ORDER:
        d = loaded[name]
        adv = None if d["elo_home_adv"] == F.ELO_HOME_ADV else d["elo_home_adv"]
        if adv not in frames:
            print(f"\n--- features (ELO_HOME_ADV={d['elo_home_adv']:.4f}) ---")
            frames[adv], _ = LF.build_and_verify(season, write_path=None, elo_adv=adv)
        frame = frames[adv]
        rows = frame[(frame["season"] == season) & (frame["week"] == week)]
        if week_ids is None:
            week_ids = set(sched.loc[sched["week"] == week, "game_id"])
            assert set(rows["game_id"]) == week_ids - set(
                sched.loc[(sched["week"] == week) & (sched["home_score"] == sched["away_score"]), "game_id"]), \
                "feature rows do not cover the week's games"
        keep, gone = split_kicked_off(rows, now)
        excluded = gone if excluded is None else excluded
        assert list(d["feature_cols"]) == F.feature_columns()
        preds.append(prediction_rows(keep, lines, name, models[name][0].relative_to(C.PROJECT_ROOT).as_posix(),
                                     models[name][1], lambda X, d=d: LM.calibrated_proba(d, X),
                                     list(d["feature_cols"]),
                                     snap.name, snap_time, now))
        feats.append(keep.assign(model=name))
    pred = pd.concat(preds, ignore_index=True)
    if pred.empty:
        print(f"REFUSING: no week-{week} game is still to be played -- nothing to predict.")
        return 2

    excl = [{"game_id": r.game_id, "kickoff_utc": C.iso(r.kickoff), "reason": "kicked off before run"}
            for r in excluded.itertuples()]
    missing = sorted(pred.loc[pred["no_bet_reason"] == "missing_line", "game_id"].unique())
    record = {
        "season": season, "week": week, "live": bool(args.live), "predicted_at_utc": C.iso(now),
        "snapshot_file": snap.name, "snapshot_pulled_at_utc": C.iso(snap_time),
        "snapshot_sha256": C.sha256_file(snap),
        "snapshot_age_hours_at_run": round((now - snap_time).total_seconds() / 3600, 2),
        "pbp_sha256": C.sha256_file(C.RAW_DIR / f"pbp_{season}.parquet"),
        "models": {n: {"file": p.relative_to(C.PROJECT_ROOT).as_posix(), "sha256": s,
                       "elo_home_adv": loaded[n]["elo_home_adv"],
                       "calibrator": loaded[n].get("calibrator")} for n, (p, s) in models.items()},
        "features_py_sha256": C.sha256_file(C.FEATURES_PY),
        "script_sha256": {f: C.sha256_file(C.PROJECT_ROOT / "scripts" / "live" / f)
                          for f in ("predict_week.py", "common.py")},
        "n_games_predicted": int(pred["game_id"].nunique()),
        "excluded_kicked_off": excl, "earlier_week_unplayed_not_due": postponed,
        "no_bet_missing_line": missing,
        "official_bets": int(((pred["model"] == C.OFFICIAL_MODEL) & (pred["paper_bet"] != "none")).sum()),
        "hypothetical_bets": int(((pred["model"] != C.OFFICIAL_MODEL) & (pred["paper_bet"] != "none")).sum()),
    }
    try:
        sha_csv = C.write_once_readonly(lambda t: pred.to_csv(t, index=False, float_format="%.10g"), out_csv)
        record["predictions_sha256"] = sha_csv
        C.write_once_readonly(lambda t: pd.concat(feats, ignore_index=True).to_parquet(t, index=False), out_feat)
        C.write_once_readonly(lambda t: t.write_text(json.dumps(record, indent=2, default=str)), out_run)
    except C.GuardError as exc:
        print(f"REFUSING: {exc}")
        return 2

    print(f"\npredicted {record['n_games_predicted']} game(s); excluded (kicked off): "
          f"{[e['game_id'] for e in excl] or 'none'}; missing lines: {missing or 'none'}")
    print(f"official bets: {record['official_bets']}   hypothetical (challenger) bets: {record['hypothetical_bets']}")
    print(f"snapshot age at run: {record['snapshot_age_hours_at_run']} h")
    print(f"wrote (read-only): {out_csv}\n  sha256 {sha_csv}\n  {out_feat.name}, {out_run.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
