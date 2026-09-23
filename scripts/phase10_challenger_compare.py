"""Phase 10 D1: label-free comparison of the frozen champion and the calibrated
challenger on every UNPLAYED 2026 game (no outcome exists, so this cannot be
an evaluation). Reports the calibrator, probability differences, pick
differences, and paper-bet decision differences using the latest snapshot's
moneylines (games without lines are "no bet" for both models).

Writes nothing.

Usage::

    python -m scripts.phase10_challenger_compare
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

from scripts.live import common as C
from src import live_features as LF, live_models as LM

SEASON = 2026


def main() -> int:
    models = C.check_frozen_models()
    d = {n: joblib.load(p) for n, (p, _) in models.items()}
    schedules, pbp = LF.load_raw(SEASON)
    frame, _ = LF.build_live_frame(schedules, pbp, SEASON)
    un = frame[(frame["season"] == SEASON) & ~frame["is_played"]].reset_index(drop=True)
    assert un["home_win"].isna().all()
    cols = list(d["champion"]["feature_cols"])
    p0 = LM.calibrated_proba(d["champion"], un[cols])
    p1 = LM.calibrated_proba(d["challenger"], un[cols])
    diff = p1 - p0

    snap = C.latest_snapshot(SEASON)
    lines = pd.read_parquet(snap)[["game_id", "home_moneyline", "away_moneyline"]]
    lines = un[["game_id", "week"]].merge(lines, on="game_id", how="left", validate="1:1")
    bets = []
    for i, r in lines.iterrows():
        b0 = C.paper_bet(p0[i], r.home_moneyline, r.away_moneyline)["paper_bet"]
        b1 = C.paper_bet(p1[i], r.home_moneyline, r.away_moneyline)["paper_bet"]
        bets.append((r.week, pd.notna(r.home_moneyline) and pd.notna(r.away_moneyline), b0, b1))
    bets = pd.DataFrame(bets, columns=["week", "has_line", "champion", "challenger"])

    c = d["challenger"]["calibrator"]
    print(f"calibrator (T={SEASON}, OOF {d['challenger']['calibration']['oof_seasons']}): "
          f"slope={c['slope']:.6f} intercept={c['intercept']:+.6f} n_fit={c['n_fit']}")
    print(f"unplayed {SEASON} games: {len(un)} (weeks {un.week.min()}-{un.week.max()}); NO labels")
    print(f"mean p_home: champion={p0.mean():.4f} challenger={p1.mean():.4f}")
    print(f"challenger - champion: mean={diff.mean():+.4f}  mean|diff|={np.abs(diff).mean():.4f}  "
          f"max|diff|={np.abs(diff).max():.4f}  min={diff.min():+.4f}  max={diff.max():+.4f}")
    flips = (p0 > 0.5) != (p1 > 0.5)
    print(f"picks differ: {int(flips.sum())}/{len(un)}  "
          f"(champion home picks {int((p0 > 0.5).sum())}, challenger {int((p1 > 0.5).sum())})")
    print(f"games with lines in {snap.name}: {int(bets.has_line.sum())} "
          f"(weeks {sorted(bets.loc[bets.has_line, 'week'].unique().tolist())})")
    wl = bets[bets.has_line]
    print(f"paper-bet decision differs: {int((wl.champion != wl.challenger).sum())}/{len(wl)} games with lines")
    print("bet decisions (champion -> challenger), games with lines:")
    print(pd.crosstab(wl.champion, wl.challenger).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
