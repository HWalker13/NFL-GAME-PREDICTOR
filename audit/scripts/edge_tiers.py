"""Audit: edge-size counts behind the Amendment 7 display tiers (Lean / ★ / ★★).

    venv/bin/python -W ignore -m audit.scripts.edge_tiers

READ-ONLY. Edge sizes and counts only. No outcome column (home_win, result,
scores, grades) is loaded: every read below names its columns explicitly.
Writes nothing.

Provenance: first run on 2026-11-17 from the session scratchpad (sha256
668c58f0...). This repo copy keeps that computation unchanged. It adds three
things: repo-root-anchored paths, the tier-count section at 10-decimal
rounding (the Effect table in Amendment 7), and full-precision edges for
games near the cutoffs.

Inputs:
  data/live/2026/predictions/week_*_predictions.csv   (champion rows; 2026 live)
  data/processed/walkforward_test_predictions.parquet (LR-tuned probs, 2023-24)
  data/raw/schedules_{2023,2024}.parquet              (moneylines only)
  data/processed/holdout_2025_predictions.parquet     (raw LR-tuned probs + lines, 2025)
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.live import common as C  # noqa: E402  (read-only reuse)

LEAN_MIN, ONE_STAR_MIN, TWO_STAR_MIN = 0.04, 0.11, 0.16   # Amendment 7


def side_edges(p_home, hml, aml):
    imp = C.vig_free_home(hml, aml)
    eh = np.asarray(p_home, float) - imp
    return np.abs(eh)          # edge of the side with positive edge (two edges sum to zero)


# --- 2026 live, official champion rows
live = pd.concat([pd.read_csv(f, usecols=["week", "game_id", "model", "p_home", "home_moneyline", "away_moneyline",
                                          "edge_home", "edge_away", "paper_bet", "bet_status"])
                  for f in sorted(glob.glob(str(ROOT / "data/live/2026/predictions/week_*_predictions.csv")))])
live = live[live.model == "champion"].copy()
live["edge"] = live[["edge_home", "edge_away"]].max(axis=1)
assert np.allclose(live.edge, side_edges(live.p_home, live.home_moneyline, live.away_moneyline))
qual = live.edge.round(C.EDGE_DECIMALS) >= C.EDGE_THRESHOLD
assert (qual == (live.paper_bet != "none")).all() and (qual == (live.bet_status == "official")).all()
print("2026 weeks:", sorted(live.week.unique()), "missing lines:", live.home_moneyline.isna().sum())

# --- 2023-24: LR-tuned walk-forward probs + schedule lines (no result columns read)
wf = pd.read_parquet(ROOT / "data/processed/walkforward_test_predictions.parquet",
                     columns=["game_id", "season", "proba_LR-tuned"])
sch = pd.concat([pd.read_parquet(ROOT / f"data/raw/schedules_{s}.parquet",
                                 columns=["game_id", "home_moneyline", "away_moneyline"]) for s in (2023, 2024)])
h1 = wf.merge(sch, on="game_id", how="left", validate="1:1")
assert len(h1) == 544 and h1[["home_moneyline", "away_moneyline"]].notna().all().all()
h1["edge"] = side_edges(h1["proba_LR-tuned"], h1.home_moneyline, h1.away_moneyline)
# --- 2025: raw LR-tuned (deployed raw) + its stored lines
h2 = pd.read_parquet(ROOT / "data/processed/holdout_2025_predictions.parquet",
                     columns=["game_id", "season", "proba_raw_LR-tuned", "home_moneyline", "away_moneyline"])
assert h2[["home_moneyline", "away_moneyline"]].notna().all().all()
h2["edge"] = side_edges(h2["proba_raw_LR-tuned"], h2.home_moneyline, h2.away_moneyline)

sets = {"2026 live": live, "2023-24": h1, "2025": h2, "held-out 2023-25": pd.concat([h1, h2])}
bins = list(range(4, 20))


def report(name, d):
    e = d.edge.values * 100; n = len(e); v = np.round(d.edge.values, C.EDGE_DECIMALS) >= C.EDGE_THRESHOLD
    print(f"\n=== {name}: games={n}  value calls={v.sum()} ({v.mean():.1%})")
    ev = e[v]
    rows = []
    for b in bins:
        c = ((ev >= b) & (ev < b + 1)).sum() if b > 4 else ((ev < 5)).sum()
        rows.append((f"{b}-{b+1}", c, c / n))
    c = (ev >= 20).sum(); rows.append(("20+", c, c / n))
    for lab, c, p in rows: print(f"  {lab:>6} {c:4d} {p:6.1%}")
    for q in (0.25, 0.10):
        print(f"  edge at top {q:.0%} of ALL games: {np.quantile(e, 1 - q):.2f} pts")
    return e


E = {k: report(k, d) for k, d in sets.items()}
print("\n=== whole-number cutoffs: share of ALL games with edge >= c")
print("   c  " + "  ".join(f"{k:>16}" for k in E))
for c in range(4, 19):
    print(f"  {c:2d}  " + "  ".join(f"{(np.round(e, 6) >= c).mean():16.1%}" for e in E.values()))

print("\n=== Amendment 7 tiers, edges compared at EDGE_DECIMALS")
for k, d in sets.items():
    e = d.edge.round(C.EDGE_DECIMALS); n = len(e)
    lean = ((e >= LEAN_MIN) & (e < ONE_STAR_MIN)).sum()
    one = ((e >= ONE_STAR_MIN) & (e < TWO_STAR_MIN)).sum()
    two = (e >= TWO_STAR_MIN).sum()
    print(f"  {k:>16}: games={n} calls={lean + one + two} lean={lean} one_star={one} two_star={two} "
          f"starred={(one + two) / n:.1%} two_star={two / n:.1%}")

print("\n=== games within 0.2 pts of 11 or 16 (full-precision edge)")
for k, d in [("2026", live), ("2023-24", h1), ("2025", h2)]:
    for c in (11, 16):
        m = d[(d.edge * 100 - c).abs() <= 0.2]
        print(f"  {k} @{c}: {len(m)}  " + ", ".join(f"{g} {x:.6f}" for g, x in zip(m.game_id, m.edge)))
    e = d.edge.round(C.EDGE_DECIMALS)
    print(f"  {k} exactly at 0.11 / 0.16: {(e == ONE_STAR_MIN).sum()} / {(e == TWO_STAR_MIN).sum()}")
