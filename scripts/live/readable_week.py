"""Phase 10: a plain-English picks page for one week (derived view).

    python -m scripts.live.readable_week --season 2026 --week N           # -> data/live_scratch/2026/week_NN_picks.md
    python -m scripts.live.readable_week --season 2026 --week N --live    # -> docs/live/week_NN_picks.md

Regenerable, like the scorecard. It READS the official record --
data/live/<season>/predictions/week_NN_predictions.csv and, if it exists,
data/live/<season>/graded/week_NN_graded.csv -- in both modes; ``--live``
changes only where the page is written. It never writes under data/live/.

Every number on the page is copied from the logged columns (p_home, pick,
implied_p_home, spread_line, paper_bet, bet_price, and the graded outcome
columns); nothing is re-predicted or re-priced. Conventions, verified:
  spread_line  > 0 = home favored by that many points, < 0 = away favored
               (nflverse dictionary_schedules.csv; also sign-matches the
               moneyline favorite in 5,078/5,099 2002-2026 games)
  moneylines   American odds: negative = favorite (risk |ml| to win 100),
               positive = underdog (risk 100 to win ml)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.live import common as C

PACIFIC = ZoneInfo("America/Los_Angeles")
OFFICIAL_MODEL_LABEL = "champion"
# Graded columns must agree with the predictions they grade (same logged row).
MATCH_COLS = ["p_home", "pick", "implied_p_home", "paper_bet", "bet_price", "snapshot_file", "predicted_at_utc"]

LEGEND = """## How to read this page

- **This is paper trading only.** No real money is bet, and nothing here is betting advice. The
  project is testing, in public and in advance, whether a statistical model knows anything the
  betting market doesn't. The honest expectation is that it doesn't.
- **What the model predicts: who wins.** Each game gets the model's probability that each team
  wins outright. It does **not** predict the score or the margin, so it says **nothing** about the
  point spread ("covering") or the over/under.
- **Model's chance**: the model's probability that the team it picks wins.
- **Vegas favorite**: who the betting market favors, and by how many points ("CIN by 3.5" means the
  market expects CIN to win by about 3.5). Shown for context only; the model does not use it.
- **Vegas's chance**: the market's probability for the **same team the model picks**, worked out
  from the betting odds with the bookmaker's built-in cut removed.
- **Gap**: model's chance minus Vegas's chance for that team, in percentage points. It is
  calculated before rounding, so it can differ by a point from subtracting the two whole numbers.
- **Paper bet**: an imaginary $1 **moneyline** bet (a bet on who wins, not on the spread or the
  total). The rule, fixed before the season: bet on a team whenever the model gives it a chance at
  least **4 points higher** than Vegas does. Otherwise, no bet. The number in brackets, e.g.
  "Bet PIT (+18.5 pts)", is that gap for **the team bet on**, so it is always +4.0 or more.
- **Why a bet can be on the team the model doesn't pick** (marked †): the model can think a team
  will probably lose but still rate it much higher than the market does. Example: the model gives an
  underdog 39% and Vegas gives it 16%. The model still picks the favorite to win, but the underdog is
  the side it thinks is underpriced, so that is where the bet goes. On those rows the Gap column
  (for the picked team) is 4 points or more *below* zero, and the bet cell shows the same gap from
  the bet team's side, with the sign flipped (e.g. Gap -23.0, "Bet MIA † (+23.0 pts)").
- **If the bet wins**: profit on a $1 bet at the odds when the prediction was logged. A losing bet
  loses the $1. A tie refunds it.
- **Champion vs challenger**: the **champion** model's bets are the official record. The
  **challenger** (the same model with a small correction for over-rating home teams) is tracked for
  comparison only; it appears below only where it disagrees with the champion.
"""


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--week", type=int, required=True)
    return ap.parse_args(argv)


def output_path(live: bool, season: int, week: int) -> Path:
    if live:
        return C.PROJECT_ROOT / "docs" / "live" / f"week_{week:02d}_picks.md"
    return C.SCRATCH_ROOT / str(season) / f"week_{week:02d}_picks.md"


# --------------------------------------------------------------------------- #
# Formatting helpers (each one tested)
# --------------------------------------------------------------------------- #
def kickoff_pacific(kickoff_utc: str) -> str:
    """'2026-09-27T17:00:00Z' -> 'Sun 9/27 10:00 AM' (US Pacific, DST-aware)."""
    t = pd.Timestamp(kickoff_utc)
    t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(PACIFIC)
    hour = t.hour % 12 or 12
    return f"{t:%a} {t.month}/{t.day} {hour}:{t:%M} {'AM' if t.hour < 12 else 'PM'}"


def _num(x: float) -> str:
    return f"{x:g}" if float(x).is_integer() else f"{x:.1f}"


def vegas_favorite_text(spread_line, home: str, away: str) -> str:
    """nflverse convention: positive spread_line = HOME favored by that many points."""
    if pd.isna(spread_line):
        return "–"
    if spread_line == 0:
        return "Pick'em"
    return f"{home if spread_line > 0 else away} by {_num(abs(spread_line))}"


def payout_text(price) -> str:
    """Profit on a winning $1 bet at American odds, e.g. +154 -> '+$1.54', -250 -> '+$0.40'."""
    return f"+${float(C.profit_per_unit(price)):.2f} per $1 risked"


def pct(p) -> str:
    return "–" if pd.isna(p) else f"{100 * p:.0f}%"


def gap_text(g) -> str:
    return "–" if pd.isna(g) else f"{100 * g:+.1f}"


def units_text(u) -> str:
    return f"{u:+.2f}"


def picked_team_view(row: pd.Series) -> dict:
    """Everything from the perspective of the team the model picks (home or away)."""
    home_pick = row["pick"] == "home"
    team = row["home_team"] if home_pick else row["away_team"]
    p_model = row["p_home"] if home_pick else 1.0 - row["p_home"]
    p_vegas = row["implied_p_home"] if home_pick else 1.0 - row["implied_p_home"]
    side = row["paper_bet"]
    if side in ("home", "away"):
        bet_team = row["home_team"] if side == "home" else row["away_team"]
        bet_gap = row["edge_home"] if side == "home" else row["edge_away"]   # the bet team's own gap
        bet = f"Bet {bet_team}" + (" †" if side != row["pick"] else "") + f" ({gap_text(bet_gap)} pts)"
        payout = payout_text(row["bet_price"])
    else:
        bet = "No bet (no line)" if row.get("no_bet_reason") == "missing_line" else "No bet"
        payout = "–"
    return {"team": team, "p_model": p_model, "p_vegas": p_vegas,
            "gap": p_model - p_vegas, "bet": bet, "payout": payout}


def final_score_text(row: pd.Series) -> str:
    if row["result"] == "pending":
        return "Not played yet"
    h, a = int(row["home_score"]), int(row["away_score"])
    return f"{row['away_team']} {a}, {row['home_team']} {h}"


def pick_right_text(row: pd.Series) -> str:
    return {"pending": "–", "tie": "Tie"}.get(row["result"], "Yes" if row["correct"] == 1 else "No")


def bet_result_text(row: pd.Series) -> str:
    o = row["bet_outcome"]
    if o == "none":
        return "–"
    if o == "pending":
        return "Pending"
    if o == "void":
        return "Void (tie): 0.00"
    return f"{'Won' if o == 'win' else 'Lost'}: {units_text(row['units'])}"


# --------------------------------------------------------------------------- #
# Loading (read-only) + consistency checks
# --------------------------------------------------------------------------- #
def load_inputs(pred_path: Path, graded_path: Path, run_json_path: Path | None) -> tuple[pd.DataFrame, bool]:
    """Read the logged predictions (+ graded outcomes if present). Refuses if the
    predictions file no longer matches the sha256 its run record logged, or if
    the graded file does not grade exactly these logged rows."""
    if not pred_path.exists():
        raise C.GuardError(f"{pred_path} not found -- nothing to show")
    if run_json_path is not None and run_json_path.exists():
        want = json.loads(run_json_path.read_text()).get("predictions_sha256")
        if want and C.sha256_file(pred_path) != want:
            raise C.GuardError(f"{pred_path.name} sha256 does not match {run_json_path.name}")
    pred = pd.read_csv(pred_path)
    if not graded_path.exists():
        return pred, False
    graded = pd.read_csv(graded_path)
    key = ["game_id", "model"]
    m = pred.merge(graded, on=key, how="outer", suffixes=("", "_g"), indicator=True)
    if (m["_merge"] != "both").any():
        raise C.GuardError(f"{graded_path.name} and {pred_path.name} cover different games/models")
    for col in MATCH_COLS:
        a, b = m[col], m[f"{col}_g"]
        same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            raise C.GuardError(f"{graded_path.name} does not grade the logged predictions ({col} differs)")
    extra = [c for c in graded.columns if c not in pred.columns]
    return pred.merge(graded[key + extra], on=key, how="left"), True


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def champion_table(df: pd.DataFrame, graded: bool) -> str:
    header = ["Game", "Kickoff (Pacific)", "Model picks", "Model's chance", "Vegas favorite",
              "Vegas's chance", "Gap (pts)", "Paper bet", "If the bet wins"]
    if graded:
        header += ["Final score", "Pick right?", "Bet result (units)"]
    rows = []
    for _, r in df.iterrows():
        v = picked_team_view(r)
        row = [f"{r['away_team']} @ {r['home_team']}", kickoff_pacific(r["kickoff_utc"]), v["team"],
               pct(v["p_model"]), vegas_favorite_text(r["spread_line"], r["home_team"], r["away_team"]),
               pct(v["p_vegas"]), gap_text(v["gap"]), v["bet"], v["payout"]]
        if graded:
            row += [final_score_text(r), pick_right_text(r), bet_result_text(r)]
        rows.append(row)
    return _table(header, rows)


def week_summary(champ: pd.DataFrame, graded: bool) -> str:
    bets = champ[champ["paper_bet"].isin(["home", "away"])]
    against = int((bets["paper_bet"] != bets["pick"]).sum())
    lines = [f"- **{len(champ)} games**, **{len(bets)} official paper bets** "
             f"({against} on the team the model does not pick to win, marked †)."]
    if graded:
        dec = champ[champ["result"].isin(["home", "away"])]
        settled = bets[bets["bet_outcome"].isin(["win", "loss", "void"])]
        lines.append(f"- Picks right so far: **{int(dec['correct'].sum())} of {len(dec)}** decided games "
                     f"({int((champ['result'] == 'pending').sum())} not played yet, "
                     f"{int((champ['result'] == 'tie').sum())} ties).")
        lines.append(f"- Paper bets so far: **{int((settled['bet_outcome'] == 'win').sum())} won, "
                     f"{int((settled['bet_outcome'] == 'loss').sum())} lost**, "
                     f"{int((settled['bet_outcome'] == 'void').sum())} void; "
                     f"net **{units_text(float(settled['units'].sum()))} units** "
                     f"(1 unit = the $1 risked per bet). One week says almost nothing about skill.")
    return "\n".join(lines)


def challenger_section(df: pd.DataFrame, graded: bool) -> str:
    ch = df[df["model"] == "champion"].set_index("game_id")
    cl = df[df["model"] == "challenger"].set_index("game_id")
    common = [g for g in ch.index if g in cl.index]
    diff = [g for g in common if ch.at[g, "pick"] != cl.at[g, "pick"]
            or ch.at[g, "paper_bet"] != cl.at[g, "paper_bet"]]
    head = ("## Challenger -- where it differs (comparison only, not the official record)\n\n"
            f"The challenger agreed with the champion on {len(common) - len(diff)} of {len(common)} games.")
    if not diff:
        return head
    header = ["Game", "Champion: pick / bet", "Challenger picks", "Challenger's chance",
              "Vegas's chance", "Gap (pts)", "Challenger's bet (hypothetical)"]
    if graded:
        header.append("Challenger bet result")
    rows = []
    for g in diff:
        a, b = picked_team_view(ch.loc[g]), picked_team_view(cl.loc[g])
        row = [f"{cl.at[g, 'away_team']} @ {cl.at[g, 'home_team']}", f"{a['team']} / {a['bet']}",
               b["team"], pct(b["p_model"]), pct(b["p_vegas"]), gap_text(b["gap"]), b["bet"]]
        if graded:
            row.append(bet_result_text(cl.loc[g]))
        rows.append(row)
    return head + "\n\n" + _table(header, rows)


def build_markdown(df: pd.DataFrame, graded: bool, season: int, week: int,
                   csv_link: str, now: str) -> str:
    champ = df[df["model"] == OFFICIAL_MODEL_LABEL].sort_values(["kickoff_utc", "game_id"])
    pred_at = kickoff_pacific(champ["predicted_at_utc"].iloc[0])
    snap_at = kickoff_pacific(champ["snapshot_pulled_at_utc"].iloc[0])
    status = "with results" if graded else "before kickoff, no results yet"
    parts = [
        f"# {season} Week {week} -- model picks ({status})",
        f"_Predictions logged {pred_at} Pacific, against betting odds pulled {snap_at} Pacific. "
        f"This page is regenerated from the official record and is not itself the record. "
        f"Generated {now}._",
        LEGEND,
        "## This week",
        week_summary(champ, graded),
        "## Official picks (champion model)",
        champion_table(champ, graded),
        "† Bet on the team the model does not pick to win; see \"Why a bet can be on the team the model "
        "doesn't pick\" above.",
        challenger_section(df, graded),
        "## The official record",
        f"The official, write-once record is [`week_{week:02d}_predictions.csv`]({csv_link}). It holds the "
        "full proof columns this page leaves out: exact probabilities, model file fingerprints (sha256), "
        "the betting-odds snapshot file and its pull time, both moneylines, and the prediction timestamp. "
        "Season totals and statistics: `docs/live/SCORECARD_2026.md`. Rules: `docs/PHASE10_PREREG.md`.",
    ]
    return "\n\n".join(parts) + "\n"


def write_page(md: str, out: Path, protected: list[Path]) -> None:
    """Write ``out`` atomically; refuse any path under the official data dir or
    equal to an input file."""
    out = out.resolve()
    if out.is_relative_to(C.OFFICIAL_ROOT.resolve()) or any(out == p.resolve() for p in protected):
        raise C.GuardError(f"refusing to write {out}: it is (or is inside) the official record")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(md)
    os.replace(tmp, out)


def main(argv=None) -> int:
    args = parse_args(argv)
    pred_path = C.predictions_path(C.OFFICIAL_ROOT, args.season, args.week)
    graded_path = C.graded_path(C.OFFICIAL_ROOT, args.season, args.week)
    run_json = pred_path.with_name(f"week_{args.week:02d}_run.json")
    out = output_path(args.live, args.season, args.week)
    try:
        df, graded = load_inputs(pred_path, graded_path, run_json)
        link = os.path.relpath(pred_path, out.parent)
        md = build_markdown(df, graded, args.season, args.week, link, C.iso(C.utc_now()))
        write_page(md, out, [pred_path, graded_path, run_json])
    except C.GuardError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out} ({'with' if graded else 'without'} results)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
