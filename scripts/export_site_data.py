"""Export one JSON file for the portfolio site: exports/site_data.json.

    python -m scripts.export_site_data                 # -> exports/site_data.json
    python -m scripts.export_site_data --out PATH      # write somewhere else (tests)

EXPORT LAYER ONLY. Trains nothing, predicts nothing, grades nothing. It reads:
  audit/out/metrics.json                                (verified held-out metrics: cross-check target)
  data/processed/walkforward_test_predictions.parquet   (2023-24 per-game probabilities)
  data/processed/holdout_2025_predictions.parquet       (2025 pre-registered holdout per-game probabilities)
  data/processed/holdout_2025_run_record.json           (2025 McNemar record: cross-check target)
  data/raw/schedules_{2023,2024,2025}.parquet           (scores, kickoffs, nflverse lines)
  data/live/2026/predictions/*.csv, data/live/2026/graded/*.csv   (the official live record)
  latest data/raw/snapshots/schedules_2026_*.parquet    (2026 schedule: weeks, kickoffs)
  docs/live/SCORECARD_2026.md                           (live numbers: cross-check target)
  audit/out/negative_control.json                       (leakage-check negative control: copied, gated)
and git metadata (HEAD, working-tree state, the commit that added each prediction file).

Every number is recomputed from per-game data with the project's own functions
(scripts/live/common.py bet rule, scripts/live/scorecard.py statistics) and then
compared with audit/out/metrics.json, docs/live/SCORECARD_2026.md and the
recorded McNemar values. ANY disagreement -> nothing is written, every
discrepancy is printed, exit code 2. It never picks one number over another.
The negative-control block is copied (not recomputed) from its source JSON and
refused the same way if that file is marked dirty, the clean condition fails
Check 1 or tops 70% contrast accuracy, or any sabotage condition passes Check 1.

Determinism: sorted keys, floats rounded to 4 dp, games ordered by
(season, week, kickoff_utc, game_id). The live "as of" time is the latest 2026
schedule snapshot's pull time (an input), not the wall clock. ``generated_at``
is the only wall-clock value: it is taken from SOURCE_DATE_EPOCH / --generated-at
if given, and otherwise an existing output whose content is otherwise identical
keeps its generated_at, so re-running on unchanged inputs is byte-identical.

No rolling/aggregated features are computed here (CLAUDE.md flag rule: nothing to flag).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from scripts.live import common as C
from scripts.live import scorecard as SC

ROOT = C.PROJECT_ROOT
SCHEMA_VERSION = "1.1.0"
EXPORT_DIR = ROOT / "exports"
OUT_PATH = EXPORT_DIR / "site_data.json"
METRICS_PATH = ROOT / "audit" / "out" / "metrics.json"
SCORECARD_PATH = ROOT / "docs" / "live" / "SCORECARD_2026.md"
WALKFORWARD_PATH = ROOT / "data" / "processed" / "walkforward_test_predictions.parquet"
HOLDOUT_2025_PATH = ROOT / "data" / "processed" / "holdout_2025_predictions.parquet"
HOLDOUT_2025_RECORD = ROOT / "data" / "processed" / "holdout_2025_run_record.json"
NEGATIVE_CONTROL_PATH = ROOT / "audit" / "out" / "negative_control.json"
NC_CLEAN = "clean"                 # the one unsabotaged condition; every other condition is a sabotage
NC_FEATURE_SETS = {"current_67": "check2_label_shuffle_current_67", "phase2_42": "check2_label_shuffle_phase2_42"}
LIVE_ROOT = C.OFFICIAL_ROOT        # read only
LIVE_SEASON = 2026
FIRST_OFFICIAL_WEEK = 3            # docs/PHASE10_PREREG.md, Amendment 1
MODELS = ("LR-untuned", "LR-tuned", "RandomForest", "HGB-tuned")
SELECTED_MODEL = "LR-tuned"
DP = 4

# docs/PHASE7_REVIEW.md section 1: exact McNemar p-values for the 2023-24 walk-forward test,
# reproduced at Phase 7 (the three non-1.0 pairs; "the other three pairs p = 1.0000").
RECORDED_MCNEMAR_2023_24 = {
    ("LR-untuned", "LR-tuned"): 0.0963, ("LR-tuned", "RandomForest"): 0.3240,
    ("LR-tuned", "HGB-tuned"): 0.3222, ("LR-untuned", "RandomForest"): 1.0,
    ("LR-untuned", "HGB-tuned"): 1.0, ("RandomForest", "HGB-tuned"): 1.0,
}

DEFINITIONS = {
    "accuracy": "Out of the games that had a winner, the share where the team the model said was more "
                "likely to win actually won. Tied games are left out.",
    "log_loss": "A score for how good the win probabilities were, not just the picks. Confident wrong "
                "calls are punished hard. Lower is better; 0.693 is what a coin flip scores.",
    "brier": "Another probability score: the average squared gap between the stated chance and what "
             "happened (1 = won, 0 = lost). Lower is better; 0.25 is what a coin flip scores.",
    "home_baseline": "The simplest possible strategy: always pick the home team.",
    "vegas_favorite": "Always pick the team the betting market favored by the point spread. For "
                      "2023-2025 this is one untimestamped line per game (most likely the closing line); "
                      "for 2026 it is the line saved at the moment the prediction was logged.",
    "vegas_implied": "The betting market's own win probability, read from the moneyline odds with the "
                     "bookmaker's built-in margin removed so both teams add up to 100%.",
    "units": "Profit from paper bets of 1 unit each at the real listed odds (bookmaker margin "
             "included). +1.00 means one stake's worth of profit. No real money was ever bet.",
    "roi": "Return on investment: units won divided by units staked. Tied games are refunded and "
           "left out. +10% means 10 cents profit per dollar bet.",
    "clv": "Closing line value: how far the market's probability for the side bet moved between the "
           "prediction and the last line saved before kickoff. If the market keeps moving toward your "
           "bets, that is the most reliable early sign of a real edge. It is the only evidence the "
           "pre-registered 2026 rule accepts.",
    "bet_rule": "Bet 1 paper unit on a team when the model gives it at least 4 percentage points more "
                "chance to win than the market does (after removing the bookmaker's margin), at that "
                "team's listed odds. Otherwise no bet. Ties are refunded.",
    "ci95": "95% confidence interval: the range the true value plausibly lies in given how few games "
            "there are. Rates use the Wilson interval; averages (log loss, Brier, ROI, CLV) use a "
            "t-interval.",
    "mcnemar_p": "Exact McNemar test comparing two models on the same games, using only the games "
                 "where exactly one of them was right. p below 0.05 would mean the gap is unlikely to "
                 "be luck.",
    "retroactive": "The 2026 betting rule applied after the fact to seasons it was not written for. "
                   "Descriptive only; it was never a live or pre-registered test.",
    "pre_registered": "The rules and pass/fail thresholds were written down and frozen before the "
                      "results existed.",
    "prediction_committed_at": "Author timestamp of the local git commit that added the week's "
                               "prediction file. Set by the owner's computer; it is not proof of "
                               "when the commit was pushed to GitHub.",
}


class Discrepancy(RuntimeError):
    """An exported number disagrees with a verified source. Nothing is written."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def rate(k: int, n: int) -> dict:
    lo, hi = SC.wilson(int(k), int(n))
    return {"k": int(k), "n": int(n), "value": (k / n) if n else None, "ci95": [lo, hi]}


def mean_block(x) -> dict:
    m, lo, hi, n = SC.mean_ci(x)
    return {"mean": m, "ci95": [lo, hi], "n": int(n)}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def utc_iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def source_commit() -> dict:
    """HEAD hash + dirty flag. The exporter's own output (exports/) is ignored,
    so writing the export does not make the next run 'dirty'."""
    head = git("rev-parse", "HEAD").strip()
    changed = []
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:].split(" -> ")[-1].strip('"')
        if not path.startswith("exports/"):
            changed.append(path)
    return {"hash": head, "dirty": bool(changed), "dirty_paths": sorted(changed)}


def file_add_commit(path: Path) -> tuple[str | None, str | None]:
    """(hash, author time UTC) of the commit that added ``path``; (None, None) if uncommitted."""
    rel = path.relative_to(ROOT).as_posix()
    out = git("log", "--diff-filter=A", "--format=%H%x09%aI", "--", rel).strip().splitlines()
    if not out:
        return None, None
    h, t = out[-1].split("\t")
    return h, utc_iso(t)


def bet_record(units: np.ndarray, outcome: np.ndarray, side: np.ndarray) -> dict:
    """outcome per bet in win/loss/void/pending; units per bet (NaN if pending)."""
    settled = np.isin(outcome, ["win", "loss"])
    w, l, v, p = (int((outcome == o).sum()) for o in ("win", "loss", "void", "pending"))
    roi = mean_block(units[settled])
    return {"bets": int(len(outcome)), "home_bets": int((side == "home").sum()),
            "away_bets": int((side == "away").sum()), "wins": w, "losses": l, "voids": v, "pending": p,
            "record": f"{w}-{l}-{v}", "units": float(np.nansum(units[settled])),
            "roi": roi["mean"], "roi_ci95": roi["ci95"]}


# --------------------------------------------------------------------------- #
# Held-out evaluations (2023-24 walk-forward, 2025 pre-registered holdout)
# --------------------------------------------------------------------------- #
def load_schedule(season: int) -> pd.DataFrame:
    from src import live_features as LF   # same kickoff definition predict/grade use
    s = pd.read_parquet(ROOT / "data" / "raw" / f"schedules_{season}.parquet")
    return LF.season_schedule_with_kickoff(s, season)


def historical_frame(preds: pd.DataFrame, seasons) -> tuple[pd.DataFrame, int]:
    sched = pd.concat([load_schedule(s) for s in seasons], ignore_index=True)
    scored = sched.dropna(subset=["home_score", "away_score"])
    n_ties = int((scored["home_score"] == scored["away_score"]).sum())
    cols = ["game_id", "home_team", "away_team", "home_score", "away_score", "spread_line",
            "home_moneyline", "away_moneyline", "kickoff"]
    keep = ["game_id", "season", "week", "home_win"] + [f"proba_{m}" for m in MODELS]
    df = preds[keep].merge(sched[cols], on="game_id", how="left", validate="1:1")
    if df["home_score"].isna().any():
        raise Discrepancy("a held-out prediction has no final score in the schedule")
    y = (df["home_score"] > df["away_score"]).astype(int)
    if not (y == df["home_win"]).all() or (df["home_score"] == df["away_score"]).any():
        raise Discrepancy("stored home_win label disagrees with schedule scores")
    df["implied_p_home"] = C.vig_free_home(df["home_moneyline"], df["away_moneyline"])
    df["vegas_fav"] = [C.vegas_favorite(a, b, c) for a, b, c in
                       zip(df["spread_line"], df["home_moneyline"], df["away_moneyline"])]
    p = df[f"proba_{SELECTED_MODEL}"].to_numpy()
    bets = [C.paper_bet(ph, h, a) for ph, h, a in zip(p, df["home_moneyline"], df["away_moneyline"])]
    df["paper_bet"] = [b["paper_bet"] for b in bets]
    df["bet_price"] = [b["bet_price"] for b in bets]
    ou = [C.bet_units(s, pr, hs, as_) for s, pr, hs, as_ in
          zip(df["paper_bet"], df["bet_price"], df["home_score"], df["away_score"])]
    df["bet_outcome"] = [o for o, _ in ou]
    df["units"] = [u for _, u in ou]
    return df, n_ties


def mcnemar(df: pd.DataFrame) -> list[dict]:
    y = df["home_win"].to_numpy()
    right = {m: ((df[f"proba_{m}"].to_numpy() > 0.5).astype(int) == y) for m in MODELS}
    out = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            ab, ba = int((right[a] & ~right[b]).sum()), int((~right[a] & right[b]).sum())
            pval = float(binomtest(min(ab, ba), ab + ba, 0.5).pvalue) if ab + ba else 1.0
            out.append({"model_a": a, "model_b": b, "only_a_right": ab, "only_b_right": ba, "p": pval})
    return out


def eval_block(df: pd.DataFrame) -> dict:
    y = df["home_win"].to_numpy()
    models = {}
    for m in MODELS:
        p = df[f"proba_{m}"].to_numpy()
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        models[m] = {"accuracy": rate(((p > 0.5).astype(int) == y).sum(), len(y)),
                     "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                     "brier": float(np.mean((p - y) ** 2))}
    fav = df["vegas_fav"].to_numpy()
    res = fav != "unresolved"
    imp = df["implied_p_home"].to_numpy()
    ok = ~np.isnan(imp)
    ic = np.clip(imp[ok], 1e-15, 1 - 1e-15)
    yi = y[ok]
    isbet = df["paper_bet"].isin(["home", "away"]).to_numpy()
    bets = bet_record(df["units"].to_numpy()[isbet], df["bet_outcome"].to_numpy()[isbet],
                      df["paper_bet"].to_numpy()[isbet])
    pick = np.where(df[f"proba_{SELECTED_MODEL}"].to_numpy() > 0.5, "home", "away")
    bets["bets_against_model_pick"] = int((isbet & (df["paper_bet"].to_numpy() != pick)).sum())
    bets["retroactive"] = True
    return {
        "n": int(len(df)),
        "models": models,
        "home_baseline": {"accuracy": rate((y == 1).sum(), len(y))},
        "vegas_favorite": {"accuracy": rate(((fav[res] == "home").astype(int) == y[res]).sum(), int(res.sum())),
                           "unresolved_excluded": int((~res).sum())},
        "vegas_implied": {"accuracy": rate(((imp[ok] > 0.5).astype(int) == yi).sum(), int(ok.sum())),
                          "log_loss": float(-np.mean(yi * np.log(ic) + (1 - yi) * np.log(1 - ic))),
                          "brier": float(np.mean((imp[ok] - yi) ** 2))},
        "retroactive_bets": bets,
    }


def check_hist(block: dict, ref: dict, where: str, errs: list[str]) -> None:
    """Compare one block with the matching audit/out/metrics.json block (4 dp; units 2 dp)."""
    def cmp(name, mine, theirs, d=DP):
        if theirs is None and mine is None:
            return
        if theirs is None or mine is None or round(float(mine), d) != round(float(theirs), d):
            errs.append(f"{where} {name}: export {mine!r} vs metrics.json {theirs!r}")

    def cmp_rate(name, r, ref_r):
        cmp(f"{name}.k", r["k"], ref_r["k"]); cmp(f"{name}.n", r["n"], ref_r["n"])
        cmp(f"{name}.value", r["value"], ref_r["acc"])
        cmp(f"{name}.ci95[0]", r["ci95"][0], ref_r["wilson95"][0])
        cmp(f"{name}.ci95[1]", r["ci95"][1], ref_r["wilson95"][1])

    cmp("n", block["n"], ref["n"])
    for m in MODELS:
        cmp_rate(f"{m}.accuracy", block["models"][m]["accuracy"], ref[m])
        cmp(f"{m}.log_loss", block["models"][m]["log_loss"], ref[m]["log_loss"])
        cmp(f"{m}.brier", block["models"][m]["brier"], ref[m]["brier"])
    cmp_rate("home_baseline", block["home_baseline"]["accuracy"], ref["home_baseline"])
    cmp_rate("vegas_favorite", block["vegas_favorite"]["accuracy"], ref["vegas_favorite"])
    cmp("vegas_favorite.unresolved", block["vegas_favorite"]["unresolved_excluded"],
        ref["vegas_favorite"]["n_unresolved_excluded"])
    vi = ref["vegas_novig_implied"]
    cmp_rate("vegas_implied.accuracy", block["vegas_implied"]["accuracy"], vi["acc_p_gt_0.5"])
    cmp("vegas_implied.log_loss", block["vegas_implied"]["log_loss"], vi["log_loss"])
    cmp("vegas_implied.brier", block["vegas_implied"]["brier"], vi["brier"])
    b, rb = block["retroactive_bets"], ref[f"bets_{SELECTED_MODEL}_RETROACTIVE"]
    cmp("bets", b["bets"], rb["n_bets"]); cmp("bets.home", b["home_bets"], rb["home_bets"])
    cmp("bets.away", b["away_bets"], rb["away_bets"]); cmp("bets.W", b["wins"], rb["W"])
    cmp("bets.L", b["losses"], rb["L"]); cmp("bets.P", b["voids"], rb["P"])
    cmp("bets.units", b["units"], rb["units"], 2); cmp("bets.roi", b["roi"], rb["roi"])
    cmp("bets.against_pick", b["bets_against_model_pick"], rb["bets_against_model_pick"])


def held_out(errs: list[str], metrics: dict) -> tuple[dict, dict, list[dict]]:
    # ---- 2023-24 walk-forward ------------------------------------------------
    wf, ties_wf = historical_frame(pd.read_parquet(WALKFORWARD_PATH), (2023, 2024))
    hw = {"label": "2023-24 walk-forward test (held out)", "seasons": [2023, 2024],
          "game_types": ["REG"], "ties_excluded": ties_wf,
          "train_rule": "Each season predicted by a model trained only on earlier seasons "
                        "(2002-2022 -> 2023; 2002-2023 -> 2024). Hyperparameters and features were fixed "
                        "on the 2022 validation season.",
          "line_timing": "untimestamped nflverse line, most likely the closing line",
          "probabilities": "raw (uncalibrated)",
          **eval_block(wf),
          "per_season": {str(s): eval_block(wf[wf["season"] == s]) for s in (2023, 2024)},
          "selected_model": SELECTED_MODEL}
    mc = mcnemar(wf)
    hw["mcnemar"] = mc
    for row in mc:
        want = RECORDED_MCNEMAR_2023_24[(row["model_a"], row["model_b"])]
        if round(row["p"], 4) != want:
            errs.append(f"2023-24 McNemar {row['model_a']} vs {row['model_b']}: export {row['p']:.4f} "
                        f"vs PHASE7_REVIEW {want}")
    pmin = min(r["p"] for r in mc)
    accs = {m: hw["models"][m]["accuracy"]["value"] for m in MODELS}
    if max(accs, key=accs.get) != SELECTED_MODEL:
        errs.append("selection note assumes the selected model has the highest 2023-24 accuracy; it does not")
    vs_sel = {r["model_b"] if r["model_a"] == SELECTED_MODEL else r["model_a"]: r["p"]
              for r in mc if SELECTED_MODEL in (r["model_a"], r["model_b"])}
    hw["selection_note"] = (
        f"{SELECTED_MODEL} was named the deployment model AFTER these 2023-24 results were known "
        f"(Phase 7), on simplicity grounds, not because it scored highest. The four models are "
        f"statistically tied: no pair differs at p < 0.05 (exact McNemar; {SELECTED_MODEL} vs "
        + ", ".join(f"{m} p = {vs_sel[m]:.4f}" for m in MODELS if m != SELECTED_MODEL)
        + f"; smallest p over all six pairs = {pmin:.4f}). Its {accs[SELECTED_MODEL]:.1%} is the highest "
        f"of four tied scores ({min(accs.values()):.1%}-{max(accs.values()):.1%}), so treat it as an "
        "optimistic pick from that range.")
    hw["retroactive_bets_note"] = (
        "The 2026 betting rule applied after the fact to the 2023-24 predictions at the historical "
        "(most likely closing) moneylines. It was not pre-registered for these seasons and is descriptive only.")
    ref = metrics["heldout_2023_2024"]
    if ref["ties_dropped"] != ties_wf:
        errs.append(f"2023-24 ties: export {ties_wf} vs metrics.json {ref['ties_dropped']}")
    check_hist(hw, ref["pooled"], "2023-24 pooled", errs)
    for s in ("2023", "2024"):
        check_hist(hw["per_season"][s], ref["per_season"][s], f"{s}", errs)

    # ---- 2025 pre-registered holdout ----------------------------------------
    h25 = pd.read_parquet(HOLDOUT_2025_PATH)
    for m in MODELS:   # Phase 8: all four ran raw; proba_<m> must be the raw column
        if not np.allclose(h25[f"proba_{m}"], h25[f"proba_raw_{m}"]):
            errs.append(f"2025 {m}: as-deployed probabilities are not the raw ones")
    df25, ties25 = historical_frame(h25, (2025,))
    stored = h25.set_index("game_id")[["spread_line", "home_moneyline", "away_moneyline"]]
    now = df25.set_index("game_id")[["spread_line", "home_moneyline", "away_moneyline"]].loc[stored.index]
    if not np.allclose(stored.to_numpy(float), now.to_numpy(float)):
        errs.append("2025 lines in holdout_2025_predictions.parquet differ from schedules_2025.parquet")
    p25 = {"label": "2025 pre-registered holdout", "seasons": [2025], "game_types": ["REG"],
           "ties_excluded": ties25,
           "train_rule": "All four models trained on 2002-2024 and scored once on 2025 under rules "
                         "written in advance (docs/PHASE8_PREREG.md).",
           "line_timing": "untimestamped nflverse line, most likely the closing line",
           "probabilities": "raw (uncalibrated); the pre-registered calibration step was not adopted",
           **eval_block(df25), "selected_model": SELECTED_MODEL}
    mc25 = mcnemar(df25)
    p25["mcnemar"] = mc25
    rec = json.loads(HOLDOUT_2025_RECORD.read_text())["mcnemar"]
    for row in mc25:
        r = rec[f"{row['model_a']} vs {row['model_b']}"]
        if (round(row["p"], 6) != round(r["p"], 6) or row["only_a_right"] != r["a_right_b_wrong"]
                or row["only_b_right"] != r["a_wrong_b_right"]):
            errs.append(f"2025 McNemar {row['model_a']} vs {row['model_b']}: export {row} vs run record {r}")
    hgb = next(r for r in mc25 if r["model_b"] == "HGB-tuned" and r["model_a"] == SELECTED_MODEL)
    p25["selection_note"] = (
        f"{SELECTED_MODEL} was already the provisional deployment model, chosen after the 2023-24 results "
        f"were known. This 2025 test then applied a rule written in advance: another model replaces it only "
        f"if it is significantly better. None was. HGB-tuned scored higher "
        f"({p25['models']['HGB-tuned']['accuracy']['k']}/{p25['n']} vs "
        f"{p25['models'][SELECTED_MODEL]['accuracy']['k']}/{p25['n']}), but McNemar p = {hgb['p']:.4f} "
        "missed the 0.05 bar, and its log loss was worse. Of the six pairwise McNemar tests, "
        + ", ".join(f"{r['model_a']} vs {r['model_b']} (p = {r['p']:.4f})" for r in mc25 if r["p"] < 0.05)
        + " fell below 0.05, and only the first survives a correction for running six tests. Disclosure: "
        "the run was executed before the owner's formal approval of the pre-registration (an agent "
        "error). The executed protocol was byte-identical by hash to the one under review (docs/PHASE8_RESULTS.md).")
    p25["retroactive_bets_note"] = (
        "The 2026 betting rule applied after the fact to the 2025 predictions at the historical "
        "(most likely closing) moneylines. It was not part of the 2025 pre-registration and is descriptive only.")
    ref25 = metrics["holdout_2025"]
    if ref25["ties_dropped"] != ties25:
        errs.append(f"2025 ties: export {ties25} vs metrics.json {ref25['ties_dropped']}")
    check_hist(p25, ref25["per_season"]["2025"], "2025", errs)

    games = hist_games(wf, "held_out_2023_24") + hist_games(df25, "prereg_2025")
    return hw, p25, games


def hist_games(df: pd.DataFrame, evaluation: str) -> list[dict]:
    out = []
    p = df[f"proba_{SELECTED_MODEL}"].to_numpy()
    for i, r in enumerate(df.to_dict("records")):
        result = "home" if r["home_score"] > r["away_score"] else "away"
        pick = "home" if p[i] > 0.5 else "away"
        isbet = r["paper_bet"] in ("home", "away")
        out.append({
            "evaluation": evaluation, "season": int(r["season"]), "week": int(r["week"]),
            "game_id": r["game_id"], "away": r["away_team"], "home": r["home_team"],
            "kickoff_utc": C.iso(r["kickoff"]), "model": SELECTED_MODEL,
            "model_home_prob": float(p[i]),
            "vegas_implied_home_prob": None if pd.isna(r["implied_p_home"]) else float(r["implied_p_home"]),
            "model_pick": pick, "vegas_favorite": r["vegas_fav"],
            "bet": isbet, "bet_side": r["paper_bet"] if isbet else None,
            "bet_odds": float(r["bet_price"]) if isbet else None,
            "bet_retroactive": True,
            "result": result, "home_score": int(r["home_score"]), "away_score": int(r["away_score"]),
            "pick_correct": pick == result,
            "bet_units": float(r["units"]) if isbet else None,
        })
    return out


# --------------------------------------------------------------------------- #
# Leakage-check negative control (audit/out/negative_control.json)
# --------------------------------------------------------------------------- #
def negative_control(errs: list[str]) -> dict:
    """Copy the negative-control results; refuse (via errs) a dirty or non-failing control."""
    src = json.loads(NEGATIVE_CONTROL_PATH.read_text())
    where = "negative_control.json"
    if src.get("src_dirty") is not False:      # missing or true -> refuse
        errs.append(f"{where}: src_dirty is {src.get('src_dirty')!r}; re-run the control on a clean src/")
    conds = src["conditions"]
    if NC_CLEAN not in conds:
        errs.append(f"{where}: no '{NC_CLEAN}' condition")
    if not [n for n in conds if n != NC_CLEAN]:
        errs.append(f"{where}: no sabotage conditions")
    gate_verdicts = src["integrity_gates"]["check1_catches_every_sabotage"]["verdicts"]
    out_conds = []
    for name, c in conds.items():          # source order (clean first), deterministic for a given file
        c1 = c["check1_timestamp"]
        sabotage = name != NC_CLEAN
        if sabotage and (c1["verdict"] != "FAIL" or c1["caught"] is not True):
            errs.append(f"{where}: sabotage condition '{name}' PASSES Check 1 "
                        f"(verdict {c1['verdict']!r}, caught {c1['caught']!r}); the control is not failing")
        if not sabotage and (c1["verdict"] != "PASS" or c1["caught"] is not False):
            errs.append(f"{where}: clean condition fails Check 1 (verdict {c1['verdict']!r})")
        if gate_verdicts.get(name) != c1["verdict"]:
            errs.append(f"{where}: integrity_gates verdict for '{name}' {gate_verdicts.get(name)!r} "
                        f"vs condition verdict {c1['verdict']!r}")
        check2 = {}
        for fs, key in NC_FEATURE_SETS.items():
            c2 = c[key]
            if not sabotage and c2["true_label_contrast_acc"] >= 0.70:   # CLAUDE.md / SPEC 5.6
                errs.append(f"{where}: clean contrast accuracy {c2['true_label_contrast_acc']} ({fs}) >= 0.70")
            check2[fs] = {"n_features": c2["n_features"], "shuffled_mean": c2["shuffled_mean"],
                          "contrast_accuracy": c2["true_label_contrast_acc"], "verdict": c2["verdict"]}
        out_conds.append({
            "condition": name, "role": "sabotage" if sabotage else "control",
            "description": c["description"],
            "columns_changed": list(c["model_columns_changed_vs_clean"]),
            "n_columns_changed": len(c["model_columns_changed_vs_clean"]),
            "check1": {"verdict": c1["verdict"],
                       "failing_oracle": c1.get("failed_stage") if c1["verdict"] == "FAIL" else None},
            "check2": check2,
        })
    c2_sab = [v["verdict"] for c in out_conds if c["role"] == "sabotage" for v in c["check2"].values()]
    c2_text = ("Check 2 (label shuffle) passes every sabotage condition, so on its own it cannot detect "
               "this kind of leak." if c2_sab and all(v == "PASS" for v in c2_sab) else
               "Check 2 (label shuffle) verdicts are listed per condition.")
    return {
        "label": "Leakage-check negative control",
        "source": (NEGATIVE_CONTROL_PATH.relative_to(ROOT).as_posix()
                   if NEGATIVE_CONTROL_PATH.is_relative_to(ROOT) else str(NEGATIVE_CONTROL_PATH)),
        "source_git_head": src["git_head"],
        "src_dirty": src.get("src_dirty"),
        "seasons": src["seasons"],
        "conditions": out_conds,
        "note": ("Each sabotage deliberately lets a game's own result leak into its rolling features, to show "
                 "the leakage checks catch it. Check 1 must fail every sabotage. Contrast accuracies under "
                 "sabotage (some above 70%) are what a leak looks like, not model results. " + c2_text),
    }


# --------------------------------------------------------------------------- #
# Live 2026
# --------------------------------------------------------------------------- #
def load_live_predictions() -> pd.DataFrame:
    files = sorted((LIVE_ROOT / str(LIVE_SEASON) / "predictions").glob("week_*_predictions.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def week_status(scheduled: int, predicted: int, graded: int, pending: int, all_kicked_off: bool) -> str:
    """graded | partial | missed | pending (see exports/SCHEMA.md)."""
    if predicted == 0:
        return "missed" if all_kicked_off else "pending"
    if graded < predicted or pending:
        return "pending"
    if predicted < scheduled:
        return "partial" if all_kicked_off else "pending"
    return "graded"


def live(errs: list[str], metrics: dict) -> tuple[dict, list[dict]]:
    graded = SC.load_graded(LIVE_ROOT, LIVE_SEASON)
    preds = load_live_predictions()
    snap = C.latest_snapshot(LIVE_SEASON)
    as_of = C.snapshot_time(snap)                       # naive UTC, an input not the wall clock
    schedule = SC.load_schedule_kickoffs(LIVE_SEASON)
    predicted = SC.load_predicted(LIVE_ROOT, LIVE_SEASON)

    # ---- prediction quality (scorecard's own computations) -------------------
    names = {"Champion": "champion", "Challenger": "challenger", "Home baseline": "home_baseline",
             "Vegas favorite (prediction snapshot)": "vegas_favorite",
             "Vegas implied (prediction snapshot)": "vegas_implied"}
    quality = {}
    for r in SC.predictor_rows(graded):
        q = {"n": int(r["n"])}
        if r["acc"]:
            q["accuracy"] = rate(*r["acc"])
        if r["ll"]:
            q["log_loss"] = {"mean": r["ll"][0], "ci95": [r["ll"][1], r["ll"][2]]}
            q["brier"] = {"mean": r["brier"][0], "ci95": [r["brier"][1], r["brier"][2]]}
        quality[names[r["predictor"]]] = q

    # ---- bets ------------------------------------------------------------------
    rows = {r["label"]: r for r in SC.betting_rows(graded)}

    def bet_out(r: dict) -> dict:
        m, lo, hi, n = r["clv"]
        return {"bets": r["bets"], "home_bets": r["home_bets"], "wins": r["w"], "losses": r["l"],
                "voids": r["void"], "pending": r["pending"], "record": f"{r['w']}-{r['l']}-{r['void']}",
                "units": r["units"], "roi": r["roi"][0], "roi_ci95": [r["roi"][1], r["roi"][2]],
                "clv": {"n": n, "mean": m, "ci95": [lo, hi], "p_one_sided": r["clv_p"],
                        "n_excluded": r["clv_excluded"]}}
    official = bet_out(rows["Official record (bet_status = official)"])
    challenger_bets = bet_out(rows["Challenger"])

    # ---- CLV verdict, exactly as the scorecard states it ----------------------
    complete = bool(len(graded) and graded["week"].max() >= 18 and not (graded["result"] == "pending").any())
    vsec = SC.verdict_section(graded, complete)
    vline = next(l for l in vsec.splitlines() if l.startswith("**") and "edge" in l)
    clv = official["clv"]
    verdict = {"statement": vline.replace("**", ""), "scorecard_line": vline,
               "verdict": re.match(r"\*\*(.+?)\*\*", vline).group(1),
               "final": complete, "n": clv["n"], "mean": clv["mean"], "ci95": clv["ci95"],
               "p_one_sided": clv["p_one_sided"], "n_excluded": clv["n_excluded"],
               "rule": "Evidence of an edge ONLY if the official bets' mean CLV > 0 with a one-sided t-test "
                       "p < 0.05. ROI never counts as evidence (docs/PHASE10_PREREG.md section 4).",
               "clv_window_rule": "Amendment 3: CLV is measured only against a snapshot at least 6 hours after "
                                  "the prediction snapshot and before kickoff; otherwise the bet is excluded and counted."}

    # ---- checkpoint -------------------------------------------------------------
    window = list(range(SC.CHECKPOINT_FIRST_WEEK, SC.CHECKPOINT_WEEK + 1))
    wk_status = {w: SC.checkpoint_week_status(graded, w, predicted, schedule, as_of) for w in window}
    evaluable = all(d for d, _ in wk_status.values())
    shadow = graded[graded["week"].between(SC.CHECKPOINT_FIRST_WEEK, SC.CHECKPOINT_WEEK)]
    d = SC.paired_ll(shadow)
    means = SC.decided(shadow).pivot_table(index="game_id", columns="model", values="log_loss").loc[d.index].mean()
    pcp = SC.one_sided_t(d, "less")
    swap = bool(evaluable and d.mean() < -SC.CHECKPOINT_MARGIN and pcp < SC.ALPHA)
    checkpoint = {
        "window_weeks": window, "evaluable": evaluable,
        "planned_games": int(schedule["week"].isin(window).sum()), "paired_games": int(len(d)),
        "champion_mean_log_loss": float(means["champion"]), "challenger_mean_log_loss": float(means["challenger"]),
        "mean_diff_challenger_minus_champion": float(d.mean()), "p_one_sided": pcp,
        "decision": ("SWAP -- challenger becomes the official model" if swap else "NO SWAP -- champion continues")
                    if evaluable else "not yet evaluable",
        "rule": "The challenger replaces the champion only if its mean per-game log loss over weeks 4-9 is lower "
                "by more than 0.010 AND a one-sided paired t-test gives p < 0.05 (docs/PHASE10_PREREG.md section 5).",
        "week_status": {str(w): s for w, (_, s) in wk_status.items()},
        "governing_amendment": {
            "id": "Amendment 4", "date": "2026-11-12", "source": "docs/PHASE10_PREREG.md",
            "summary": "Week 9 was entirely missed and week 8 partially missed, so 'after week 9 is fully graded' "
                       "could never happen. Amendment 4 changes only WHEN the checkpoint runs: once every window "
                       "week is fully graded or is a missed week (no prediction file, every kickoff past). Missed "
                       "games count as 0 games. The decision rule, margin, test and window (weeks 4-9) are unchanged.",
            "disclosure": "Written after the weeks 4-8 log losses were visible. The decision is NO SWAP under "
                          "either wording: without the amendment the checkpoint never runs and the champion "
                          "continues by default.",
        },
    }

    # ---- weeks ------------------------------------------------------------------
    champ_pred = preds[preds["model"] == "champion"]
    champ_grad = graded[graded["model"] == "champion"]
    last_week = max(max(predicted), int(schedule.loc[schedule["kickoff"] <= as_of, "week"].max()))
    weeks, commits = [], {}
    for w in range(FIRST_OFFICIAL_WEEK, last_week + 1):
        sw = schedule[schedule["week"] == w]
        pf = C.predictions_path(LIVE_ROOT, LIVE_SEASON, w)
        g = champ_grad[champ_grad["week"] == w]
        n_pred = int((champ_pred["week"] == w).sum())
        n_graded = int(g["result"].isin(["home", "away", "tie"]).sum())
        n_pending = int((g["result"] == "pending").sum()) + (n_pred - len(g))
        commit, committed_at = file_add_commit(pf) if pf.exists() else (None, None)
        commits[w] = (commit, committed_at)
        weeks.append({
            "week": w, "scheduled": int(len(sw)), "predicted": n_pred, "graded": n_graded,
            "decided": int(g["result"].isin(["home", "away"]).sum()), "ties": int((g["result"] == "tie").sum()),
            "pending": n_pending,
            "status": week_status(len(sw), n_pred, n_graded, n_pending, bool((sw["kickoff"] <= as_of).all())),
            "first_kickoff_utc": C.iso(sw["kickoff"].min()),
            "prediction_commit": commit, "prediction_committed_at": committed_at,
        })

    # ---- games ------------------------------------------------------------------
    gcols = ["game_id", "home_score", "away_score", "result", "correct", "bet_outcome", "units"]
    merged = champ_pred.merge(champ_grad[gcols], on="game_id", how="left", validate="1:1")
    games = []
    for r in merged.to_dict("records"):
        result = r["result"] if isinstance(r["result"], str) else "pending"
        isbet = r["paper_bet"] in ("home", "away")
        settled = isbet and result in ("home", "away", "tie")
        commit, committed_at = commits[int(r["week"])]
        games.append({
            "evaluation": "live_2026", "season": int(r["season"]), "week": int(r["week"]),
            "game_id": r["game_id"], "away": r["away_team"], "home": r["home_team"],
            "kickoff_utc": r["kickoff_utc"], "model": "champion (LR-tuned, frozen 2026)",
            "model_home_prob": float(r["p_home"]),
            "vegas_implied_home_prob": None if pd.isna(r["implied_p_home"]) else float(r["implied_p_home"]),
            "model_pick": r["pick"],
            "vegas_favorite": C.vegas_favorite(r["spread_line"], r["home_moneyline"], r["away_moneyline"]),
            "bet": isbet, "bet_side": r["paper_bet"] if isbet else None,
            "bet_odds": float(r["bet_price"]) if isbet else None, "bet_retroactive": False,
            "bet_status": r["bet_status"] if isbet else None,
            "result": result,
            "home_score": None if pd.isna(r["home_score"]) else int(r["home_score"]),
            "away_score": None if pd.isna(r["away_score"]) else int(r["away_score"]),
            "pick_correct": (r["pick"] == result) if result in ("home", "away") else None,
            "bet_units": float(r["units"]) if settled else None,
            "predicted_at_utc": r["predicted_at_utc"],
            "prediction_commit": commit, "prediction_committed_at": committed_at,
            "prediction_committed_at_kind": "local git author timestamp (not proof of push time)",
        })

    champ_all = graded[graded["model"] == "champion"]
    block = {
        "label": "2026 live shadow mode (paper trading, no real money)",
        "as_of_utc": C.iso(as_of), "as_of_source": f"latest schedule snapshot {snap.name}",
        "first_official_week": FIRST_OFFICIAL_WEEK,
        "games_logged": int(len(champ_pred)), "games_graded": int(len(champ_all)),
        "games_decided": int(SC.decided(champ_all).shape[0]),
        "ties": int((champ_all["result"] == "tie").sum()),
        "games_pending": int(len(champ_pred) - SC.decided(champ_all).shape[0] - (champ_all["result"] == "tie").sum()),
        "line_timing": "line snapshot saved when the prediction was logged (usually Wednesday), not the close",
        "prediction_quality": quality,
        "official_bets": official,
        "challenger_bets_hypothetical": challenger_bets,
        "clv_verdict": verdict,
        "weeks": weeks,
        "checkpoint": checkpoint,
        "provenance_note": "prediction_committed_at is the local git author time of the commit that added each "
                           "week's prediction file. The owner's computer sets it, so it is not proof of push time.",
    }
    check_live(block, graded, predicted, schedule, as_of, metrics["live_2026"], errs)
    return block, games


# --------------------------------------------------------------------------- #
# Live cross-checks: docs/live/SCORECARD_2026.md and audit/out/metrics.json
# --------------------------------------------------------------------------- #
def parse_scorecard(text: str) -> dict:
    """Pull the cumulative numbers out of the scorecard markdown."""
    out = {"quality": {}, "bets": {}}
    m = re.search(r"Games logged: (\d+) \(decided (\d+), tie (\d+), pending (\d+)\)", text)
    out["games"] = tuple(int(x) for x in m.groups())
    num = r"(-?\d+\.\d+)"
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[1].isdigit() and not cells[0].isdigit():
            acc = re.match(rf"(\d+)/(\d+) = {num} \[{num}, {num}\]", cells[2])
            ll = re.match(rf"{num} \[{num}, {num}\]", cells[3])
            br = re.match(rf"{num} \[{num}, {num}\]", cells[4])
            out["quality"][cells[0]] = {"n": int(cells[1]), "acc": acc.groups() if acc else None,
                                        "ll": ll.groups() if ll else None, "brier": br.groups() if br else None}
        if len(cells) == 8 and re.match(r"\d+ \(\d+\)", cells[1]):
            b = re.match(r"(\d+) \((\d+)\)", cells[1]).groups()
            wl = re.match(r"(\d+)-(\d+)-(\d+) \((\d+)\)", cells[2]).groups()
            roi = re.match(rf"{num} \[{num}, {num}\]", cells[4]).groups()
            clv = re.match(rf"{num} \[{num}, {num}\] \(n=(\d+), excl (\d+)\)", cells[5]).groups()
            out["bets"][cells[0]] = {"bets": b, "wlvp": wl, "units": cells[3], "roi": roi, "clv": clv,
                                     "p": cells[6]}
    v = re.search(r"^(\*\*(?:No evidence|Evidence) of an edge\*\*.*)$", text, re.M)
    out["verdict_line"] = v.group(1) if v else None
    cp = re.search(rf"paired games actually graded for both models: n = (\d+)\.", text)
    cm = re.search(rf"champion {num}, challenger {num}; mean difference \(challenger - champion\) = "
                   rf"([+-]\d+\.\d+); one-sided paired t p = {num}\.", text)
    dec = re.search(r"\*\*Decision: (.+?)\.\*\*", text)
    out["checkpoint"] = {"n": int(cp.group(1)) if cp else None, "nums": cm.groups() if cm else None,
                         "decision": dec.group(1) if dec else None}
    return out


def check_live(block: dict, graded: pd.DataFrame, predicted: dict, schedule: pd.DataFrame,
               as_of, ref: dict, errs: list[str]) -> None:
    text = SCORECARD_PATH.read_text()
    # 1) The scorecard's own sections, regenerated from today's graded files, must appear verbatim.
    for name, sec in (("prediction table", SC.predictor_table(graded)), ("betting table", SC.betting_table(graded)),
                      ("verdict", SC.verdict_section(graded, block["clv_verdict"]["final"])),
                      ("checkpoint", SC.checkpoint_section(graded, predicted, schedule, as_of))):
        if sec not in text:
            errs.append(f"SCORECARD_2026.md {name} does not match the graded files (regenerate the scorecard?)")
    # 2) Exported values, formatted the scorecard's way, must equal the parsed cells.
    sc = parse_scorecard(text)
    f3, f4 = (lambda x: f"{x:.3f}"), (lambda x: f"{x:.4f}")
    if sc["games"] != (block["games_graded"], block["games_decided"], block["ties"], 0):
        errs.append(f"scorecard games logged {sc['games']} vs export graded/decided/ties "
                    f"{block['games_graded'], block['games_decided'], block['ties']}")
    label = {"champion": "Champion", "challenger": "Challenger", "home_baseline": "Home baseline",
             "vegas_favorite": "Vegas favorite (prediction snapshot)",
             "vegas_implied": "Vegas implied (prediction snapshot)"}
    for key, q in block["prediction_quality"].items():
        s = sc["quality"].get(label[key])
        if s is None:
            errs.append(f"scorecard row {label[key]} missing"); continue
        if s["n"] != q["n"]:
            errs.append(f"live {key} n: export {q['n']} vs scorecard {s['n']}")
        if "accuracy" in q:
            a = q["accuracy"]
            mine = (str(a["k"]), str(a["n"]), f3(a["value"]), f3(a["ci95"][0]), f3(a["ci95"][1]))
            if s["acc"] != mine:
                errs.append(f"live {key} accuracy: export {mine} vs scorecard {s['acc']}")
        for met in ("log_loss", "brier"):
            if met in q:
                mine = (f3(q[met]["mean"]), f3(q[met]["ci95"][0]), f3(q[met]["ci95"][1]))
                theirs = s["ll" if met == "log_loss" else "brier"]
                if theirs != mine:
                    errs.append(f"live {key} {met}: export {mine} vs scorecard {theirs}")
    for key, lab in (("official_bets", "Official record (bet_status = official)"),
                     ("challenger_bets_hypothetical", "Challenger")):
        b, s = block[key], sc["bets"].get(lab)
        if s is None:
            errs.append(f"scorecard betting row {lab} missing"); continue
        mine = {"bets": (str(b["bets"]), str(b["home_bets"])),
                "wlvp": (str(b["wins"]), str(b["losses"]), str(b["voids"]), str(b["pending"])),
                "units": f"{b['units']:+.2f}", "roi": (f3(b["roi"]), f3(b["roi_ci95"][0]), f3(b["roi_ci95"][1])),
                "clv": (f4(b["clv"]["mean"]), f4(b["clv"]["ci95"][0]), f4(b["clv"]["ci95"][1]),
                        str(b["clv"]["n"]), str(b["clv"]["n_excluded"])),
                "p": f4(b["clv"]["p_one_sided"])}
        for k, v in mine.items():
            if s[k] != v:
                errs.append(f"live {key} {k}: export {v} vs scorecard {s[k]}")
    if sc["verdict_line"] != block["clv_verdict"]["scorecard_line"]:
        errs.append(f"CLV verdict: export {block['clv_verdict']['scorecard_line']!r} vs scorecard {sc['verdict_line']!r}")
    cp = block["checkpoint"]
    mine = (f"{cp['champion_mean_log_loss']:.5f}", f"{cp['challenger_mean_log_loss']:.5f}",
            f"{cp['mean_diff_challenger_minus_champion']:+.5f}", f4(cp["p_one_sided"]))
    if sc["checkpoint"]["nums"] != mine or sc["checkpoint"]["n"] != cp["paired_games"] \
            or sc["checkpoint"]["decision"] != cp["decision"]:
        errs.append(f"checkpoint: export {mine, cp['paired_games'], cp['decision']} vs scorecard {sc['checkpoint']}")
    # 3) audit/out/metrics.json live block.
    q = block["prediction_quality"]
    pairs = [("champion k", q["champion"]["accuracy"]["k"], ref["champion"]["k"]),
             ("champion n", q["champion"]["accuracy"]["n"], ref["champion"]["n"]),
             ("champion log_loss", q["champion"]["log_loss"]["mean"], ref["champion"]["log_loss"]),
             ("champion brier", q["champion"]["brier"]["mean"], ref["champion"]["brier"]),
             ("home k", q["home_baseline"]["accuracy"]["k"], ref["home_baseline"]["k"]),
             ("vegas fav k", q["vegas_favorite"]["accuracy"]["k"], ref["vegas_favorite"]["k"]),
             ("vegas implied log_loss", q["vegas_implied"]["log_loss"]["mean"], ref["vegas_novig_implied"]["log_loss"]),
             ("vegas implied brier", q["vegas_implied"]["brier"]["mean"], ref["vegas_novig_implied"]["brier"]),
             ("bets", block["official_bets"]["bets"], ref["bets"]["n_bets"]),
             ("W", block["official_bets"]["wins"], ref["bets"]["W"]),
             ("L", block["official_bets"]["losses"], ref["bets"]["L"]),
             ("P", block["official_bets"]["voids"], ref["bets"]["P"]),
             ("roi", block["official_bets"]["roi"], ref["bets"]["roi_per_settled_bet"]),
             ("clv n", block["official_bets"]["clv"]["n"], ref["bets"]["clv_measured_n"]),
             ("clv mean", block["official_bets"]["clv"]["mean"], ref["bets"]["clv_mean"])]
    for name, mine_v, theirs in pairs:
        if round(float(mine_v), DP) != round(float(theirs), DP):
            errs.append(f"live {name}: export {mine_v!r} vs metrics.json {theirs!r}")
    if round(block["official_bets"]["units"], 2) != round(ref["bets"]["units"], 2):
        errs.append(f"live units: export {block['official_bets']['units']} vs metrics.json {ref['bets']['units']}")
    if block["games_graded"] != ref["games_graded"] or block["ties"] != ref["ties"]:
        errs.append("live games graded / ties disagree with metrics.json")


# --------------------------------------------------------------------------- #
# Assembly + serialisation
# --------------------------------------------------------------------------- #
def clean(o):
    """Recursively: floats -> 4 dp (NaN/inf -> null), numpy scalars -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        if not math.isfinite(f):
            return None
        r = round(f, DP)
        return 0.0 if r == 0 else r
    return o


def serialise(payload: dict) -> str:
    return json.dumps(clean(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def build_payload(generated_at: str) -> dict:
    """Everything except writing. Raises Discrepancy listing every mismatch."""
    errs: list[str] = []
    metrics = json.loads(METRICS_PATH.read_text())
    hw, p25, hist_games_ = held_out(errs, metrics)
    live_block, live_games = live(errs, metrics)
    nc = negative_control(errs)
    games = sorted(hist_games_ + live_games, key=lambda g: (g["season"], g["week"], g["kickoff_utc"], g["game_id"]))
    for g in games:   # invariant the site relies on
        if g["bet_units"] is not None and not g["bet"]:
            errs.append(f"{g['game_id']}: bet_units without bet")
    if errs:
        raise Discrepancy("\n".join(errs))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_commit": source_commit(),
        "definitions": DEFINITIONS,
        "held_out_2023_24": hw,
        "prereg_2025": p25,
        "live_2026": live_block,
        "negative_control": nc,
        "games": games,
    }


def resolve_generated_at(cli_value: str | None, out: Path, payload_wo_time: dict) -> str:
    if cli_value:
        return cli_value
    if os.environ.get("SOURCE_DATE_EPOCH"):
        return datetime.fromtimestamp(int(os.environ["SOURCE_DATE_EPOCH"]), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if out.exists():   # unchanged content -> keep the old timestamp (byte-identical re-run)
        try:
            old = json.loads(out.read_text())
            if {k: v for k, v in old.items() if k != "generated_at"} == clean(payload_wo_time):
                return old["generated_at"]
        except (ValueError, KeyError):
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--generated-at", default=None, help="fixed UTC ISO timestamp (reproducible builds)")
    args = ap.parse_args(argv)
    try:
        payload = build_payload(generated_at="")
    except Discrepancy as exc:
        print("DISCREPANCY -- nothing written. Each line: exported value vs verified source.\n", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2
    body = {k: v for k, v in payload.items() if k != "generated_at"}
    payload["generated_at"] = resolve_generated_at(args.generated_at, args.out, body)
    text = serialise(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.tmp")
    tmp.write_text(text)
    tmp.replace(args.out)
    print(f"wrote {args.out} ({len(text.encode())} bytes, {len(payload['games'])} games)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
