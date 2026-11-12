"""Audit: leakage spot-check (read-only; trains nothing, writes only audit/out/).

    venv/bin/python -W ignore -m audit.scripts.leakage_spotcheck

1. Vegas columns: none of the model feature columns (saved LR-tuned, frozen
   2026 champion, src.features.feature_columns()) name a line/odds/score field.
2. Perturbation tests, at the function level (the exact functions
   src/features.py uses to build the table):
   a. EWM (Pattern C, shift(1) then ewm): change every team's metric in ONE
      game week; that week's own feature values must not move, the next
      week's must.
   b. In-season rolling mean (Pattern A, closed='left'): same test.
   c. Elo (Pattern D): change the final scores of one week's games; that
      week's home_elo_pre/away_elo_pre must not move, the next week's must.
3. Independent recompute of one EWM column from team_game_log.parquet with a
   plain Python loop over strictly-earlier games, vs. the stored feature table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import evaluate as E, features as F  # noqa: E402

OUT = ROOT / "audit" / "out"
LINE_WORDS = ("spread", "moneyline", "total", "odds", "line", "result", "score", "vegas", "implied")


def check_vegas() -> dict:
    lists = {
        "features.feature_columns()": F.feature_columns(),
        "models/logreg_winner_tuned.joblib": list(joblib.load(ROOT / "models/logreg_winner_tuned.joblib")["feature_cols"]),
        "models/live_2026_champion.joblib": list(joblib.load(ROOT / "models/live_2026_champion.joblib")["feature_cols"]),
    }
    out = {}
    for k, cols in lists.items():
        hits = [c for c in cols if any(w in c.lower() for w in LINE_WORDS)]
        out[k] = {"n_features": len(cols), "line_like_names": hits}
    assert lists["features.feature_columns()"] == lists["models/logreg_winner_tuned.joblib"] \
        == lists["models/live_2026_champion.joblib"], "feature lists differ"
    frame = pd.read_parquet(ROOT / "data/processed/game_features.parquet")
    out["game_features.parquet_line_like_columns"] = [c for c in frame.columns
                                                      if any(w in c.lower() for w in LINE_WORDS)]
    return out


def perturb_rolling() -> dict:
    tgl = pd.read_parquet(ROOT / "data/processed/team_game_log.parquet")
    tgl = tgl.sort_values(["team", "kickoff", "game_id"]).reset_index(drop=True)
    m = "qb_epa_per_db"
    season, week = 2023, 10
    hit = (tgl["season"] == season) & (tgl["week"] == week)
    nxt = (tgl["season"] == season) & (tgl["week"] == week + 1)
    bumped = tgl.copy()
    bumped.loc[hit, m] = bumped.loc[hit, m] + 5.0   # huge change to this week's OWN outcomes
    res = {}
    for label, fn in (("ewm (shift(1).ewm)", lambda d: F.roll_ewm(d, m, ["team"], F.EWM_HALFLIFE)),
                      ("in-season mean (rolling closed='left')",
                       lambda d: F.roll_left(d, m, F.IN_SEASON_WINDOW, ["team", "season"]))):
        a, b = fn(tgl), fn(bumped)
        same_wk = float(np.nanmax(np.abs(a[hit] - b[hit])))
        next_wk = float(np.nanmax(np.abs(a[nxt] - b[nxt])))
        res[label] = {"max_change_same_week": same_wk, "max_change_next_week": next_wk,
                      "PASS": same_wk == 0.0 and next_wk > 0}
    # leaky contrast: the same EWM WITHOUT shift(1) does move on its own week
    leaky = lambda d: d.groupby("team")[m].transform(lambda s: s.ewm(halflife=F.EWM_HALFLIFE).mean())
    res["contrast: ewm WITHOUT shift(1) (deliberately leaky)"] = {
        "max_change_same_week": float(np.nanmax(np.abs(leaky(tgl)[hit] - leaky(bumped)[hit])))}
    return res


def perturb_elo() -> dict:
    sched = E.load_schedules()
    season, week = 2023, 10
    a = F.build_elo_table(sched, 2002, 2024).set_index("game_id")
    s2 = sched.copy()
    w = (s2["season"] == season) & (s2["week"] == week) & (s2["game_type"] == "REG")
    # flip every result of that week (swap scores) -- ties stay ties
    s2.loc[w, ["home_score", "away_score"]] = s2.loc[w, ["away_score", "home_score"]].to_numpy()
    b = F.build_elo_table(s2, 2002, 2024).set_index("game_id")
    ids_w = sched.loc[w, "game_id"]
    ids_n = sched.loc[(sched["season"] == season) & (sched["week"] == week + 1) & (sched["game_type"] == "REG"), "game_id"]
    ids_w, ids_n = [i for i in ids_w if i in a.index], [i for i in ids_n if i in a.index]
    d_same = float((a.loc[ids_w] - b.loc[ids_w]).abs().to_numpy().max())
    d_next = float((a.loc[ids_n] - b.loc[ids_n]).abs().to_numpy().max())
    return {"max_change_same_week": d_same, "max_change_next_week": d_next, "PASS": d_same == 0.0 and d_next > 0}


def independent_ewm() -> dict:
    """Plain-loop EWM over strictly earlier games (pandas ewm adjust=True weights)."""
    tgl = pd.read_parquet(ROOT / "data/processed/team_game_log.parquet").sort_values(["team", "kickoff", "game_id"])
    frame = pd.read_parquet(ROOT / "data/processed/game_features.parquet").set_index("game_id")
    m = "def_epa_per_play"
    alpha = 1 - np.exp(np.log(0.5) / F.EWM_HALFLIFE)
    rng = np.random.default_rng(0)
    sample = frame[frame["season"].isin([2023, 2024])].sample(40, random_state=0)
    worst, n = 0.0, 0
    for gid, row in sample.iterrows():
        for side in ("home", "away"):
            team_rows = tgl[tgl["game_id"] == gid]
            is_home = side == "home"
            t = team_rows.loc[team_rows["is_home"] == is_home, "team"].iat[0]
            hist = tgl[(tgl["team"] == t) & (tgl["kickoff"] < row["kickoff"])][m].dropna().to_numpy()
            if len(hist) == 0:
                continue
            wts = (1 - alpha) ** np.arange(len(hist))[::-1]
            mine = float((wts * hist).sum() / wts.sum())
            worst = max(worst, abs(mine - row[f"{side}_{m}_ewm"]))
            n += 1
    return {"metric": m, "team_games_checked": n, "max_abs_diff": worst, "PASS": worst < 1e-9}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"vegas_columns": check_vegas(), "rolling_perturbation": perturb_rolling(),
           "elo_perturbation": perturb_elo(), "independent_ewm_recompute": independent_ewm()}
    (OUT / "leakage_spotcheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
