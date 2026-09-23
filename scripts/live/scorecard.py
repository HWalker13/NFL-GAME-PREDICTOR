"""Phase 10: cumulative + weekly scorecard from the graded files.

    python -m scripts.live.scorecard --season 2026           # -> data/live_scratch/2026/SCORECARD_2026.md
    python -m scripts.live.scorecard --season 2026 --live    # -> docs/live/SCORECARD_2026.md

Regenerable. Reads only week_NN_graded.csv files. Rules and definitions:
docs/PHASE10_PREREG.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.live import common as C

CHECKPOINT_WEEK = 9
FIRST_LIVE_WEEK = 4
ALPHA = 0.05
EDGE_BUCKETS = ((0.04, 0.06), (0.06, 0.08), (0.08, np.inf))   # [lo, hi), descriptive only
SMALL_BUCKET_N = 30
EDGE_BREAKDOWN_NOTE = (
    "This breakdown is descriptive only. It cannot change the season-end verdict (Section 4), the "
    "paper-bet threshold, or the checkpoint decision. Its purpose: if the model carries real information, "
    "larger edges should perform better; a flat or non-monotonic pattern indicates edges are mostly model "
    "error. Small per-bucket samples are expected and will be labelled."
)
CHECKPOINT_MARGIN = 0.010   # same as Phase 8's log-loss rule: mean diff must be < -0.010
C_MEASURED = "measured"   # grade_week.CLV_MEASURED

HEADER = """# {season} shadow-mode scorecard

_Generated {now} from {n_files} graded week file(s): {weeks}. Paper trading only -- no real money.
Rules: `docs/PHASE10_PREREG.md`. Regenerated from the graded files; the logged predictions are never edited._

## How to read this

- **Champion** is the frozen deployment model (LR-tuned, trained 2002-2025). Its paper bets are the
  official record. **Challenger** is the same frozen model with a sigmoid probability calibrator (fit
  out-of-fold on 2021-2025) that corrects its home-win bias; its bets are hypothetical and exist for the
  midseason comparison.
- **Home baseline** always picks the home team. **Vegas favorite** is the side favored by the spread in
  the snapshot the prediction was logged against; **Vegas implied** is the vig-free moneyline probability
  from that same snapshot.
- **Accuracy**: share of games picked correctly. **Log loss**: penalises confident wrong probabilities;
  lower is better; ~0.693 is a coin flip. **Brier**: mean squared error of the probability; lower is
  better; 0.25 is a coin flip.
- **Units / ROI**: profit from flat 1-unit paper bets at the logged moneyline (vig included); ROI =
  units / units staked (void ties excluded).
- **Bet CLV**: how much the vig-free implied probability of the side bet moved between the prediction
  snapshot and the **last pre-kickoff snapshot** (not the true close). Positive = the market moved
  toward the bet. **Line move -> model**: the same movement for every game, signed toward the model's view.
- **Small samples mislead.** A season is ~270 games; this rule bets only a fraction of them. With 50
  bets, a ROI 95% interval is roughly +/-25 points wide, so a hot or cold streak says almost nothing.
  CLV is the less noisy signal, and it is the only one the pre-registered verdict accepts as evidence.
  Every number below comes with its 95% interval; read the interval, not the point estimate.
- Ties are excluded from accuracy / log loss / Brier and are void (0 units) for bets. Pending games
  (no final score yet) are excluded everywhere and counted.
"""


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    return ap.parse_args(argv)


def output_path(live: bool, season: int) -> Path:
    if live:
        return C.PROJECT_ROOT / "docs" / "live" / f"SCORECARD_{season}.md"
    return C.SCRATCH_ROOT / str(season) / f"SCORECARD_{season}.md"


def load_graded(root: Path, season: int) -> pd.DataFrame:
    files = sorted((root / str(season) / "graded").glob("week_*_graded.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mean_ci(x) -> tuple[float, float, float, int]:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    m = float(x.mean())
    if n < 2:
        return (m, np.nan, np.nan, n)
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return (m, m - h, m + h, n)


def one_sided_t(x, alternative: str) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 2 or np.all(x == x[0]):
        return np.nan
    return float(stats.ttest_1samp(x, 0.0, alternative=alternative).pvalue)


def fmt(v, d=3) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fmt_ci(m, lo, hi, d=3) -> str:
    if np.isnan(m):
        return "–"
    if np.isnan(lo):
        return f"{m:.{d}f}"
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def decided(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["result"].isin(["home", "away"])]


def predictor_rows(df: pd.DataFrame) -> list[dict]:
    """Champion, challenger, home baseline, Vegas favorite/implied."""
    out = []
    for model in ("champion", "challenger"):
        d = decided(df[df["model"] == model])
        k, n = int(d["correct"].sum()), len(d)
        out.append({"predictor": model.capitalize(), "n": n, "acc": (k, n),
                    "ll": mean_ci(d["log_loss"]), "brier": mean_ci(d["brier"])})
    base = decided(df[df["model"] == "champion"])
    out.append({"predictor": "Home baseline", "n": len(base),
                "acc": (int(base["home_correct"].sum()), len(base)), "ll": None, "brier": None})
    v = base[base["vegas_correct"].notna()]
    out.append({"predictor": "Vegas favorite (prediction snapshot)", "n": len(v),
                "acc": (int(v["vegas_correct"].sum()), len(v)), "ll": None, "brier": None})
    vi = base[base["vegas_log_loss"].notna()]
    out.append({"predictor": "Vegas implied (prediction snapshot)", "n": len(vi), "acc": None,
                "ll": mean_ci(vi["vegas_log_loss"]), "brier": mean_ci(vi["vegas_brier"])})
    return out


def predictor_table(df: pd.DataFrame) -> str:
    lines = ["| Predictor | n | Accuracy [95% CI] | Log loss [95% CI] | Brier [95% CI] |",
             "|---|---:|---|---|---|"]
    for r in predictor_rows(df):
        if r["acc"]:
            k, n = r["acc"]
            lo, hi = wilson(k, n)
            acc = f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]" if n else "–"
        else:
            acc = "–"
        ll = fmt_ci(*r["ll"][:3]) if r["ll"] else "–"
        br = fmt_ci(*r["brier"][:3]) if r["brier"] else "–"
        lines.append(f"| {r['predictor']} | {r['n']} | {acc} | {ll} | {br} |")
    return "\n".join(lines)


def _bet_summary(label: str, d: pd.DataFrame, bets: pd.DataFrame) -> dict:
    """``d`` = the logged rows (for all-game line movement), ``bets`` = the bets."""
    settled = bets[bets["bet_outcome"].isin(["win", "loss"])]
    clv = bets[bets["clv_status"] == C_MEASURED]
    # line movement: every logged game (result not needed); none for the official-record row
    move = mean_ci(d.loc[d["clv_status"] == C_MEASURED, "line_move_toward_model"]) if len(d) \
        else (np.nan, np.nan, np.nan, 0)
    return {
        "label": label, "bets": len(bets),
        "w": int((bets["bet_outcome"] == "win").sum()), "l": int((bets["bet_outcome"] == "loss").sum()),
        "void": int((bets["bet_outcome"] == "void").sum()), "pending": int((bets["bet_outcome"] == "pending").sum()),
        "units": float(settled["units"].sum()),
        "roi": mean_ci(settled["units"]),            # mean profit per unit staked = ROI
        "home_bets": int((bets["paper_bet"] == "home").sum()),
        "clv": mean_ci(clv["bet_clv"]), "clv_p": one_sided_t(clv["bet_clv"], "greater"),
        "clv_excluded": int((bets["clv_status"] != C_MEASURED).sum()),
        "move": move,
    }


def official_bets(df: pd.DataFrame) -> pd.DataFrame:
    """The official record, by what was LOGGED (``bet_status``), so a week-9
    checkpoint swap moves the official record to the challenger from week 10
    without rewriting any earlier row."""
    return df[df["paper_bet"].isin(["home", "away"]) & (df["bet_status"] == "official")]


def betting_rows(df: pd.DataFrame) -> list[dict]:
    out = []
    for model in ("champion", "challenger"):
        d = df[df["model"] == model]
        out.append(_bet_summary(model.capitalize(), d, d[d["paper_bet"].isin(["home", "away"])]))
    out.append(_bet_summary("Official record (bet_status = official)", pd.DataFrame(), official_bets(df)))
    return out


def betting_table(df: pd.DataFrame) -> str:
    lines = ["| Model | Bets (home) | W-L-Void (pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n, excluded) "
             "| One-sided p (CLV>0) | Line move -> model [95% CI] (n) |",
             "|---|---:|---|---:|---|---|---:|---|"]
    for r in betting_rows(df):
        m, lo, hi, n = r["clv"]
        mv, mlo, mhi, mn = r["move"]
        move_txt = f"{fmt_ci(mv, mlo, mhi, 4)} (n={mn})" if mn else "–"
        lines.append(
            f"| {r['label']} | {r['bets']} ({r['home_bets']}) | "
            f"{r['w']}-{r['l']}-{r['void']} ({r['pending']}) | {r['units']:+.2f} | "
            f"{fmt_ci(*r['roi'][:3])} | {fmt_ci(m, lo, hi, 4)} (n={n}, excl {r['clv_excluded']}) | "
            f"{fmt(r['clv_p'], 4)} | {move_txt} |")
    return "\n".join(lines)


def edge_bucket(edge: float) -> str | None:
    """Bucket label for the BET side's edge, compared at the same rounding as
    the bet rule (common.EDGE_DECIMALS), so exactly 0.06 / 0.08 fall in the
    upper bucket. None below the 0.04 threshold (not a bet)."""
    if pd.isna(edge):
        return None
    e = round(float(edge), C.EDGE_DECIMALS)
    for lo, hi in EDGE_BUCKETS:
        if lo <= e < hi:
            return f"[{lo:.2f}, {hi:.2f})" if np.isfinite(hi) else f"[{lo:.2f}+)"
    return None


def bet_side_edge(bets: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(bets["paper_bet"] == "home", bets["edge_home"], bets["edge_away"]),
                     index=bets.index)


def edge_breakdown_table(df: pd.DataFrame) -> str:
    lines = ["| Model | Edge bucket | Bets | W-L (void, pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n) | Sample |",
             "|---|---|---:|---|---:|---|---|---|"]
    labels = [edge_bucket(lo) for lo, _ in EDGE_BUCKETS]
    for model in ("champion", "challenger"):
        bets = df[(df["model"] == model) & df["paper_bet"].isin(["home", "away"])]
        buckets = bet_side_edge(bets).map(edge_bucket) if len(bets) else pd.Series(dtype=object)
        for label in labels:
            b = bets[buckets == label]
            settled = b[b["bet_outcome"].isin(["win", "loss"])]
            clv = b.loc[b["clv_status"] == C_MEASURED, "bet_clv"]
            m, lo, hi, n = mean_ci(clv)
            lines.append(
                f"| {model} | {label} | {len(b)} | {int((b['bet_outcome'] == 'win').sum())}-"
                f"{int((b['bet_outcome'] == 'loss').sum())} ({int((b['bet_outcome'] == 'void').sum())}, "
                f"{int((b['bet_outcome'] == 'pending').sum())}) | {settled['units'].sum():+.2f} | "
                f"{fmt_ci(*mean_ci(settled['units'])[:3])} | {fmt_ci(m, lo, hi, 4)} (n={n}) | "
                f"{'small sample (n<%d)' % SMALL_BUCKET_N if len(b) < SMALL_BUCKET_N else ''} |")
    return "\n".join(lines)


def weekly_table(df: pd.DataFrame) -> str:
    lines = ["| Week | Model | n | Acc | Log loss | Brier | Home acc | Vegas fav acc | Bets | Units | Mean bet CLV (n) |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for week, w in df.groupby("week"):
        for model in ("champion", "challenger"):
            d = w[w["model"] == model]
            dd = decided(d)
            bets = d[d["paper_bet"].isin(["home", "away"])]
            settled = bets[bets["bet_outcome"].isin(["win", "loss"])]
            clv = bets[bets["clv_status"] == C_MEASURED]["bet_clv"]
            lines.append(
                f"| {week} | {model} | {len(dd)} | {fmt(dd['correct'].mean())} | {fmt(dd['log_loss'].mean(), 4)} | "
                f"{fmt(dd['brier'].mean(), 4)} | {fmt(dd['home_correct'].mean())} | "
                f"{fmt(dd['vegas_correct'].mean())} | {len(bets)} | {settled['units'].sum():+.2f} | "
                f"{fmt(clv.mean(), 4)} ({len(clv)}) |")
    return "\n".join(lines)


def paired_ll(df: pd.DataFrame, max_week: int | None = None) -> pd.Series:
    """Per-game log loss, challenger minus champion, on games both graded."""
    d = decided(df)
    if max_week is not None:
        d = d[d["week"] <= max_week]
    wide = d.pivot_table(index="game_id", columns="model", values="log_loss")
    wide = wide.dropna(subset=[c for c in ("champion", "challenger") if c in wide])
    if not {"champion", "challenger"} <= set(wide.columns):
        return pd.Series(dtype=float)
    return wide["challenger"] - wide["champion"]


def checkpoint_section(df: pd.DataFrame) -> str:
    shadow = df[df["week"].between(FIRST_LIVE_WEEK, CHECKPOINT_WEEK)]
    weeks_done = sorted(shadow["week"].unique().tolist())
    all_weeks = list(range(FIRST_LIVE_WEEK, CHECKPOINT_WEEK + 1))
    pending = int((shadow["result"] == "pending").sum())
    head = (f"## Midseason checkpoint (after week {CHECKPOINT_WEEK})\n\n"
            f"Rule (pre-registered): the challenger replaces the champion for the rest of the season ONLY if "
            f"its mean per-game log loss over all shadow weeks ({FIRST_LIVE_WEEK}-{CHECKPOINT_WEEK}) is lower "
            f"by more than {CHECKPOINT_MARGIN:.3f} (mean challenger - champion < -{CHECKPOINT_MARGIN:.3f}) AND a "
            f"one-sided paired t-test on per-game log loss gives p < {ALPHA} (Phase 8's rule). Otherwise the "
            f"champion continues and the comparison is recorded. A swap at n = 88 is unlikely; the "
            f"comparison's main value is informing the 2027 model.\n\n")
    if weeks_done != all_weeks or pending:
        return head + (f"Not yet evaluable: graded shadow weeks {weeks_done or 'none'}, "
                       f"{pending} pending game(s). The decision is taken once, after week "
                       f"{CHECKPOINT_WEEK} is fully graded.\n")
    d = paired_ll(shadow)
    mean = float(d.mean())
    p = one_sided_t(d, "less")
    swap = bool(mean < -CHECKPOINT_MARGIN and not np.isnan(p) and p < ALPHA)
    return head + (f"n = {len(d)} paired games; mean per-game log-loss difference (challenger - champion) "
                   f"= {mean:+.5f}; one-sided paired t p = {fmt(p, 4)}.\n\n"
                   f"**Decision: {'SWAP -- challenger becomes the official model from week 10' if swap else 'NO SWAP -- champion continues'}.**\n")


def verdict_section(df: pd.DataFrame, season_complete: bool) -> str:
    champ = _bet_summary("official", pd.DataFrame(), official_bets(df))
    m, lo, hi, n = champ["clv"]
    p = champ["clv_p"]
    edge = bool(n >= 2 and m > 0 and not np.isnan(p) and p < ALPHA)
    roi = fmt_ci(*champ["roi"][:3])
    status = "FINAL" if season_complete else "PROVISIONAL -- the verdict is only taken at season end"
    return (f"## Season-end betting verdict ({status})\n\n"
            f"Rule (pre-registered): \"evidence of an edge\" ONLY if the official mean bet CLV > 0 "
            f"with a one-sided t-test p < {ALPHA}; the units ROI 95% CI is reported alongside. ROI alone never "
            f"counts as evidence.\n\n"
            f"Official bets with measured CLV: n = {n}; mean CLV = {fmt_ci(m, lo, hi, 4)}; one-sided p = "
            f"{fmt(p, 4)}. Units ROI = {roi}.\n\n"
            f"**{'Evidence of an edge' if edge else 'No evidence of an edge'}** under the rule "
            f"({'as of the games graded so far' if not season_complete else 'full season'}).\n")


def build_markdown(df: pd.DataFrame, season: int, n_files: int, now: str, season_complete: bool) -> str:
    weeks = ", ".join(str(w) for w in sorted(df["week"].unique())) if len(df) else "none"
    parts = [HEADER.format(season=season, now=now, n_files=n_files, weeks=weeks)]
    if df.empty:
        parts.append("\nNo graded weeks yet.\n")
        return "\n".join(parts)
    g = df[df["model"] == "champion"]
    parts.append(f"Games logged: {g['game_id'].nunique()} (decided {len(decided(g))}, "
                 f"tie {int((g['result'] == 'tie').sum())}, pending {int((g['result'] == 'pending').sum())}).\n")
    parts.append("## Cumulative -- prediction quality\n\n" + predictor_table(df) + "\n")
    parts.append("## Cumulative -- paper bets and line movement\n\n" + betting_table(df) + "\n\n"
                 "`Bets (home)` shows how many bets were on the home side: both models carry the known "
                 "home-win bias (Phase 8 negative calibration intercepts), so a home-heavy bet mix is expected.\n")
    parts.append("## Paper bets by edge size (DESCRIPTIVE ONLY)\n\n" + EDGE_BREAKDOWN_NOTE + "\n\n"
                 "Edge = model probability minus vig-free implied probability for the side bet, at the "
                 "prediction snapshot. ROI = units per unit staked on settled bets.\n\n"
                 + edge_breakdown_table(df) + "\n")
    parts.append("## Weekly\n\n" + weekly_table(df) + "\n")
    parts.append(checkpoint_section(df))
    parts.append(verdict_section(df, season_complete))
    return "\n".join(parts)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = C.root_dir(args.live)
    df = load_graded(root, args.season)
    n_files = len(list((root / str(args.season) / "graded").glob("week_*_graded.csv"))) \
        if (root / str(args.season) / "graded").exists() else 0
    complete = bool(len(df) and df["week"].max() >= 18 and not (df["result"] == "pending").any())
    md = build_markdown(df, args.season, n_files, C.iso(C.utc_now()), complete)
    out = output_path(args.live, args.season)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(md)
    tmp.replace(out)
    print(md)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
