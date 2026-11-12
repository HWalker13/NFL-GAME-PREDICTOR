"""Audit: recompute every reported metric from per-game source data.

    venv/bin/python -m audit.scripts.recompute_metrics

READ-ONLY. Trains nothing, predicts nothing. Inputs:
  data/processed/walkforward_test_predictions.parquet  (2023-24 per-game probs;
      written by scripts/reproduce_walkforward_predictions.py, Phase 7)
  data/processed/holdout_2025_predictions.parquet      (2025 per-game probs, Phase 8)
  data/raw/schedules_{2023,2024,2025}.parquet          (scores + nflverse lines)
  data/live/2026/predictions/*.csv, data/live/2026/graded/*.csv (live log)
  latest data/raw/snapshots/schedules_2026_*.parquet   (independent score check)

Writes audit/out/metrics.json (and prints everything).

Independent re-implementations (vig-free prob, Vegas favorite, bet rule, bet
payout) are asserted equal to the project's own functions in
scripts/live/common.py on every game, so any disagreement fails loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.live import common as C  # noqa: E402  (read-only reuse, for cross-checks)

OUT = ROOT / "audit" / "out"
MODELS = ("LR-untuned", "LR-tuned", "RandomForest", "HGB-tuned")
HEADLINE = "LR-tuned"
Z = 1.959964
BREAKEVEN_110 = 110 / 210  # 0.5238


# --------------------------------------------------------------------------- #
# Independent implementations
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def raw_implied(ml: np.ndarray) -> np.ndarray:
    ml = np.asarray(ml, float)
    return np.where(ml < 0, -ml / (100 - ml), 100 / (ml + 100))


def novig_home(hml, aml) -> np.ndarray:
    h, a = raw_implied(hml), raw_implied(aml)
    return h / (h + a)


def vegas_fav(spread, hml, aml) -> np.ndarray:
    """+1 home, -1 away, 0 unresolved. spread_line > 0 = home favored; 0 -> lower ML."""
    s = np.sign(np.asarray(spread, float))
    ml = np.sign(np.asarray(aml, float) - np.asarray(hml, float))
    return np.where(s != 0, s, ml)


def profit(price) -> np.ndarray:
    price = np.asarray(price, float)
    return np.where(price > 0, price / 100, 100 / np.abs(price))


def logloss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def acc_block(correct: np.ndarray) -> dict:
    k, n = int(np.sum(correct)), int(len(correct))
    return {"k": k, "n": n, "acc": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


# --------------------------------------------------------------------------- #
# Bets (the Phase 10 pre-registered rule, applied retroactively where noted)
# --------------------------------------------------------------------------- #
def apply_bet_rule(p_home, hml, aml, thr=0.04) -> pd.DataFrame:
    imp = novig_home(hml, aml)
    eh = np.round(p_home - imp, 10)
    side = np.where(eh >= thr, "home", np.where(-eh >= thr, "away", "none"))
    price = np.where(side == "home", hml, np.where(side == "away", aml, np.nan))
    return pd.DataFrame({"implied_p_home": imp, "edge_home": p_home - imp, "bet": side, "price": price})


def bet_block(df: pd.DataFrame, y: np.ndarray, model_correct: np.ndarray) -> dict:
    """df has bet/price; y = home_win (1/0); ties are not present in historical sets."""
    isbet = (df["bet"] != "none").to_numpy()
    n_games = len(df)
    won = np.where(df["bet"] == "home", y == 1, y == 0)[isbet]
    pr = df["price"].to_numpy()[isbet]
    units = np.where(won, profit(pr), -1.0)
    k, n = int(won.sum()), int(isbet.sum())
    # Break-even win rate for THIS set of prices: the win rate at which the
    # same bets would net exactly 0 units = n / sum(1 + profit_i) if all prices
    # were equal; for mixed prices we report the mean raw (vig-included)
    # implied probability of the prices taken -- the fair per-bet comparator.
    be_mean = float(raw_implied(pr).mean()) if n else float("nan")
    return {
        "n_bets": n, "pct_of_games": round(n / n_games, 4) if n_games else None,
        "home_bets": int((df["bet"] == "home").sum()), "away_bets": int((df["bet"] == "away").sum()),
        "W": k, "L": n - k, "P": 0,
        "win_rate": round(k / n, 4) if n else None, "win_rate_wilson95": wilson(k, n),
        "breakeven_at_minus110": round(BREAKEVEN_110, 4),
        "breakeven_mean_implied_of_prices_taken": round(be_mean, 4),
        "units": round(float(units.sum()), 2), "roi": round(float(units.mean()), 4) if n else None,
        "binom_p_vs_mean_implied": (round(float(binomtest(k, n, be_mean).pvalue), 4) if n else None),
        "model_acc_on_bet_games": acc_block(model_correct[isbet]),
        "model_acc_on_nobet_games": acc_block(model_correct[~isbet]),
    }


# --------------------------------------------------------------------------- #
# Historical held-out sets
# --------------------------------------------------------------------------- #
def load_sched(seasons) -> pd.DataFrame:
    s = pd.concat(pd.read_parquet(ROOT / "data" / "raw" / f"schedules_{y}.parquet") for y in seasons)
    return s[s["game_type"] == "REG"]


def historical(preds: pd.DataFrame, seasons, label: str, prob_col_fmt: str = "proba_{}") -> dict:
    sched = load_sched(seasons)
    all_reg = sched.dropna(subset=["home_score", "away_score"])
    ties = all_reg[all_reg["home_score"] == all_reg["away_score"]]
    lines = sched[["game_id", "home_team", "away_team", "home_score", "away_score",
                   "spread_line", "home_moneyline", "away_moneyline", "gameday"]]
    df = preds.drop(columns=[c for c in ("spread_line", "home_moneyline", "away_moneyline") if c in preds]) \
        .merge(lines, on="game_id", how="left", validate="1:1")
    assert df["home_score"].notna().all()
    y_src = (df["home_score"] > df["away_score"]).astype(int).to_numpy()
    assert (y_src == df["home_win"].to_numpy()).all(), "stored label != label from schedule scores"
    assert (df["home_score"] != df["away_score"]).all()
    y = y_src

    fav = vegas_fav(df["spread_line"], df["home_moneyline"], df["away_moneyline"])
    imp = novig_home(df["home_moneyline"], df["away_moneyline"])
    # cross-check vs project functions
    proj_fav = np.array([{"home": 1, "away": -1, "unresolved": 0}[C.vegas_favorite(a, b, c)]
                         for a, b, c in zip(df["spread_line"], df["home_moneyline"], df["away_moneyline"])])
    assert (proj_fav == fav).all(), "vegas favorite disagrees with scripts/live/common.py"
    assert np.allclose(C.vig_free_home(df["home_moneyline"], df["away_moneyline"]), imp, equal_nan=True)

    res = {"label": label, "seasons": list(seasons),
           "reg_games_with_score": int(len(all_reg)), "ties_dropped": int(len(ties)),
           "tie_game_ids": ties["game_id"].tolist(), "n": int(len(df)),
           "missing_lines": int(df[["spread_line", "home_moneyline", "away_moneyline"]].isna().any(axis=1).sum()),
           "pickem_spread0": int((df["spread_line"] == 0).sum()),
           "unresolved_fav": int((fav == 0).sum()),
           "spread_fav_ne_ml_fav": int(((np.sign(df["spread_line"]) != 0)
                                        & (np.sign(df["away_moneyline"] - df["home_moneyline"]) != 0)
                                        & (np.sign(df["spread_line"]) != np.sign(df["away_moneyline"] - df["home_moneyline"]))).sum()),
           "per_season": {}}

    def block(m: np.ndarray) -> dict:
        yy = y[m]
        out = {"n": int(m.sum()), "home_baseline": acc_block(yy == 1)}
        r = m & (fav != 0)
        out["vegas_favorite"] = acc_block((fav[r] > 0).astype(int) == y[r])
        out["vegas_favorite"]["n_unresolved_excluded"] = int((m & (fav == 0)).sum())
        ok = m & ~np.isnan(imp)
        out["vegas_novig_implied"] = {"acc_p_gt_0.5": acc_block((imp[ok] > 0.5).astype(int) == y[ok]),
                                      "brier": round(brier(y[ok], imp[ok]), 4),
                                      "log_loss": round(logloss(y[ok], imp[ok]), 4)}
        for name in MODELS:
            p = df[prob_col_fmt.format(name)].to_numpy()
            pick = (p > 0.5).astype(int)
            stored_pred = df.get(f"pred_{name}")
            if stored_pred is not None:
                assert (stored_pred.to_numpy() == pick).all(), f"{name}: stored pred != (p>0.5)"
            out[name] = {**acc_block(pick[m] == yy), "brier": round(brier(yy, p[m]), 4),
                         "log_loss": round(logloss(yy, p[m]), 4)}
            # disagreement with Vegas favorite
            dm = m & (fav != 0) & (np.where(fav > 0, 1, 0) != pick)
            k = int((pick[dm] == y[dm]).sum())
            out[name]["disagree_with_vegas"] = {
                "n": int(dm.sum()), "model_right": k,
                "model_right_rate": round(k / dm.sum(), 4) if dm.sum() else None,
                "wilson95": wilson(k, int(dm.sum())),
                "exact_binom_p_vs_0.5": round(float(binomtest(k, int(dm.sum()), 0.5).pvalue), 4) if dm.sum() else None}
        # bets: headline model, retroactive application of the Phase 10 rule
        p = df[prob_col_fmt.format(HEADLINE)].to_numpy()
        bets = apply_bet_rule(p[m], df["home_moneyline"].to_numpy()[m], df["away_moneyline"].to_numpy()[m])
        # cross-check with project rule
        for i, (ph, hm, am) in enumerate(zip(p[m], df["home_moneyline"].to_numpy()[m], df["away_moneyline"].to_numpy()[m])):
            assert C.paper_bet(ph, hm, am)["paper_bet"] == bets["bet"].iat[i], "bet rule mismatch vs common.py"
        mc = ((p[m] > 0.5).astype(int) == yy)
        bb = bet_block(bets, yy, mc)
        pick_side = np.where(p[m] > 0.5, "home", "away")
        bb["bets_against_model_pick"] = int(((bets["bet"] != "none") & (bets["bet"] != pick_side)).sum())
        out[f"bets_{HEADLINE}_RETROACTIVE"] = bb
        return out

    for s in seasons:
        res["per_season"][str(s)] = block((df["season"] == s).to_numpy())
    if len(seasons) > 1:
        res["pooled"] = block(np.ones(len(df), bool))
    return res, df


def sign_convention_examples(df: pd.DataFrame) -> list[dict]:
    ids = ["2023_01_DET_KC", "2023_01_ARI_WAS", "2024_01_BAL_KC"]
    out = []
    for g in ids:
        r = df[df["game_id"] == g].iloc[0]
        out.append({k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in
                    ("game_id", "gameday", "away_team", "home_team", "spread_line", "home_moneyline",
                     "away_moneyline", "away_score", "home_score")})
    return out


# --------------------------------------------------------------------------- #
# Live 2026
# --------------------------------------------------------------------------- #
def live_2026() -> dict:
    pdir, gdir = ROOT / "data/live/2026/predictions", ROOT / "data/live/2026/graded"
    preds = pd.concat([pd.read_csv(f) for f in sorted(pdir.glob("week_*_predictions.csv"))], ignore_index=True)
    graded = pd.concat([pd.read_csv(f) for f in sorted(gdir.glob("week_*_graded.csv"))], ignore_index=True)
    snap = C.latest_snapshot(2026)
    sc = pd.read_parquet(snap)[["game_id", "home_score", "away_score", "gameday", "gametime"]].set_index("game_id")

    champ = graded[graded["model"] == "champion"].copy()
    # independent score check against the latest snapshot
    s2 = sc.loc[champ["game_id"]]
    score_mismatch = int(((s2["home_score"].to_numpy() != champ["home_score"].to_numpy())
                          | (s2["away_score"].to_numpy() != champ["away_score"].to_numpy())).sum())
    dec = champ[champ["home_score"] != champ["away_score"]].copy()
    y = (dec["home_score"] > dec["away_score"]).astype(int).to_numpy()
    p = dec["p_home"].to_numpy()
    pick = (dec["pick"] == "home").astype(int).to_numpy()
    assert ((p > 0.5).astype(int) == pick).all()
    fav = vegas_fav(dec["spread_line"], dec["home_moneyline"], dec["away_moneyline"])
    imp = novig_home(dec["home_moneyline"], dec["away_moneyline"])
    assert np.allclose(imp, dec["implied_p_home"].to_numpy())
    # my bet rule vs logged bet
    mine = apply_bet_rule(champ["p_home"].to_numpy(), champ["home_moneyline"].to_numpy(), champ["away_moneyline"].to_numpy())
    bet_rule_mismatch = int((mine["bet"].to_numpy() != champ["paper_bet"].to_numpy()).sum())
    isbet = champ["paper_bet"].isin(["home", "away"])
    b = champ[isbet]
    tie = b["home_score"] == b["away_score"]
    won = np.where(b["paper_bet"] == "home", b["home_score"] > b["away_score"], b["away_score"] > b["home_score"])
    units = np.where(tie, 0.0, np.where(won, profit(b["bet_price"]), -1.0))
    units_mismatch = float(np.abs(units - b["units"].to_numpy()).max())
    settled = ~tie.to_numpy()
    W, L, P = int((won & settled).sum()), int((~won & settled).sum()), int(tie.sum())
    mc_all = (pick == y)
    dec_isbet = dec["paper_bet"].isin(["home", "away"]).to_numpy()
    dm = (fav != 0) & (np.where(fav > 0, 1, 0) != pick)
    k_dm = int((pick[dm] == y[dm]).sum())
    clv = b.loc[b["clv_status"] == "measured", "bet_clv"]

    weeks = {}
    for w, g in preds[preds["model"] == "champion"].groupby("week"):
        first_kick = g["kickoff_utc"].min()
        weeks[int(w)] = {"games_logged": int(g["game_id"].nunique()), "predicted_at_utc": g["predicted_at_utc"].iloc[0],
                         "first_logged_kickoff_utc": first_kick,
                         "graded": bool((gdir / f"week_{int(w):02d}_graded.csv").exists())}
    return {
        "weeks": weeks,
        "graded_weeks": sorted(int(w) for w in champ["week"].unique()),
        "games_graded": int(len(champ)), "ties": int((champ["home_score"] == champ["away_score"]).sum()),
        "score_mismatch_vs_latest_snapshot": score_mismatch, "latest_snapshot": snap.name,
        "bet_rule_mismatch_vs_logged": bet_rule_mismatch, "units_max_abs_diff_vs_graded": units_mismatch,
        "champion": {**acc_block(mc_all), "brier": round(brier(y, p), 4), "log_loss": round(logloss(y, p), 4)},
        "home_baseline": acc_block(y == 1),
        "vegas_favorite": acc_block((fav[fav != 0] > 0).astype(int) == y[fav != 0]),
        "vegas_novig_implied": {"brier": round(brier(y, imp), 4), "log_loss": round(logloss(y, imp), 4)},
        "disagree_with_vegas": {"n": int(dm.sum()), "model_right": k_dm, "wilson95": wilson(k_dm, int(dm.sum()))},
        "bets": {"n_bets": int(isbet.sum()), "pct_of_games": round(float(isbet.mean()), 4),
                 "W": W, "L": L, "P": P, "win_rate": round(W / (W + L), 4), "win_rate_wilson95": wilson(W, W + L),
                 "breakeven_at_minus110": round(BREAKEVEN_110, 4),
                 "breakeven_mean_implied_of_prices_taken": round(float(raw_implied(b["bet_price"][settled]).mean()), 4),
                 "units": round(float(units.sum()), 2), "roi_per_settled_bet": round(float(units[settled].mean()), 4),
                 "bets_against_model_pick": int((b["paper_bet"] != b["pick"]).sum()),
                 "model_acc_on_bet_games": acc_block(mc_all[dec_isbet]),
                 "model_acc_on_nobet_games": acc_block(mc_all[~dec_isbet]),
                 "clv_measured_n": int(len(clv)), "clv_mean": round(float(clv.mean()), 4) if len(clv) else None},
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    wf = pd.read_parquet(ROOT / "data/processed/walkforward_test_predictions.parquet")
    h25 = pd.read_parquet(ROOT / "data/processed/holdout_2025_predictions.parquet")
    # 2025: all four models ran RAW (Phase 8 rule) -- verify proba_* == proba_raw_*
    for m in MODELS:
        assert np.allclose(h25[f"proba_{m}"], h25[f"proba_raw_{m}"]), f"2025 {m} not raw"
    r2324, df2324 = historical(wf, (2023, 2024), "2023-24 walk-forward test")
    r25, df25 = historical(h25, (2025,), "2025 pre-registered holdout")
    # lines stored in the 2025 parquet vs current schedules_2025 (mutation check)
    h25l = h25.merge(load_sched((2025,))[["game_id", "spread_line", "home_moneyline", "away_moneyline"]],
                     on="game_id", suffixes=("_stored", "_now"))
    r25["lines_changed_since_phase8"] = int(sum(
        (~np.isclose(h25l[f"{c}_stored"], h25l[f"{c}_now"])).sum() for c in ("spread_line", "home_moneyline", "away_moneyline")))
    out = {"heldout_2023_2024": r2324, "holdout_2025": r25, "live_2026": live_2026(),
           "sign_convention_examples": sign_convention_examples(df2324)}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
