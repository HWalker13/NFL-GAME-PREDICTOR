"""Phase 7, Part C: Vegas benchmark on the 2023-2024 walk-forward test games.

DESCRIPTIVE ONLY. No model, feature, hyperparameter, or calibration decision
may be made from anything this script prints -- in Phase 7 or retroactively.
That restriction is what keeps this from being a second use of the
2023-2024 test set. Vegas lines are used here strictly as a benchmark (SPEC
Section 2 stretch goal, Section 5.2 category 4), never as a feature.

Inputs (read-only, nothing is retrained):
  data/processed/walkforward_test_predictions.parquet  (from
      scripts/reproduce_walkforward_predictions.py)
  data/raw/schedules_2023.parquet, schedules_2024.parquet  (spread_line,
      home_moneyline, away_moneyline)

Conventions:
  - Ties are excluded, same as every model evaluation in this project
    (evaluate.completed_regular_season). 2023-2024 had zero REG ties.
  - Vegas favorite = side favored by spread_line (positive = home favored).
    A spread of exactly 0 (pick'em) falls back to the lower moneyline; if the
    moneylines are also equal, the game is counted as unresolved and
    excluded from favorite-based numbers, with the count reported.
  - Vig-removed implied home prob = raw implied home / (raw home + raw away),
    from American moneylines.
  - Model probabilities are UNCALIBRATED raw predict_proba (SPEC 7.3 not yet
    applied), so probability-metric comparisons are provisional.

Usage::

    python -m scripts.vegas_benchmark_2023_2024
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src import evaluate as E, train_winner as TW

MODELS = ("LR-untuned", "LR-tuned", "RandomForest", "HGB-tuned")
TEST_SEASONS = (2023, 2024)


def american_to_prob(ml: pd.Series) -> pd.Series:
    ml = ml.astype(float)
    return pd.Series(np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0)), index=ml.index)


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def prob_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    return {"log_loss": log_loss(y, proba), "brier_score": brier_score_loss(y, proba),
            "roc_auc": roc_auc_score(y, proba)}


def build_frame() -> pd.DataFrame:
    preds = pd.read_parquet(TW.PROC_DIR / "walkforward_test_predictions.parquet")
    sched = E.completed_regular_season(E.load_schedules())
    sched = sched[sched["season"].isin(TEST_SEASONS)]
    df = preds.merge(sched[["game_id", "spread_line", "home_moneyline", "away_moneyline", "result"]],
                     on="game_id", how="left", validate="1:1")
    assert len(df) == 544 and df["result"].notna().all(), "test games failed to join to schedules"
    assert (df["home_win"] == (df["result"] > 0)).all(), "label mismatch vs schedule result"
    assert df[["spread_line", "home_moneyline", "away_moneyline"]].notna().all().all(), "missing lines"

    raw_h = american_to_prob(df["home_moneyline"])
    raw_a = american_to_prob(df["away_moneyline"])
    df["vig"] = raw_h + raw_a - 1.0
    df["vegas_proba"] = raw_h / (raw_h + raw_a)

    fav = np.sign(df["spread_line"])  # +1 home fav, -1 away fav, 0 pick'em
    ml_fav = np.sign(df["away_moneyline"] - df["home_moneyline"])  # lower ML = favorite
    df["spread_fav"] = fav
    df["ml_fav"] = ml_fav
    df["vegas_fav"] = np.where(fav != 0, fav, ml_fav)  # 0 = unresolved
    df["vegas_pick"] = (df["vegas_fav"] > 0).astype(int)
    return df


def main() -> None:
    df = build_frame()
    y = df["home_win"].to_numpy()
    print("=== Phase 7 Part C: Vegas benchmark, 2023-2024 walk-forward test games (DESCRIPTIVE ONLY) ===\n")
    print(f"games: {len(df)}  (2023: {(df.season == 2023).sum()}, 2024: {(df.season == 2024).sum()})")
    print(f"spread_line == 0 (pick'em): {(df.spread_fav == 0).sum()}")
    print(f"unresolved after moneyline fallback: {(df.vegas_fav == 0).sum()}")
    both = (df.spread_fav != 0) & (df.ml_fav != 0)
    print(f"spread favorite != moneyline favorite: {(both & (df.spread_fav != df.ml_fav)).sum()} "
          f"(moneylines equal: {(df.ml_fav == 0).sum()})")
    print(f"bookmaker vig (overround): mean {df.vig.mean():.4f}, min {df.vig.min():.4f}, max {df.vig.max():.4f}")
    print(f"vig-removed implied home prob: min {df.vegas_proba.min():.4f}, max {df.vegas_proba.max():.4f}")

    resolved = df["vegas_fav"] != 0
    yr = y[resolved.to_numpy()]
    vegas_fav_correct = int((df.loc[resolved, "vegas_pick"].to_numpy() == yr).sum())
    implied_pick = (df["vegas_proba"] > 0.5).astype(int).to_numpy()
    print(f"implied-prob pick (p>0.5) == spread favorite pick in "
          f"{int((implied_pick[resolved.to_numpy()] == df.loc[resolved, 'vegas_pick'].to_numpy()).sum())}"
          f"/{int(resolved.sum())} games; implied prob exactly 0.5 in {(df.vegas_proba == 0.5).sum()}")

    rows = {}
    for season_label, mask in (("2023", df.season == 2023), ("2024", df.season == 2024),
                               ("pooled", pd.Series(True, index=df.index))):
        m = mask.to_numpy()
        tbl = {}
        tbl["home_baseline"] = {"accuracy": y[m].mean()}
        for name in MODELS:
            acc = (df[f"pred_{name}"].to_numpy()[m] == y[m]).mean()
            tbl[name] = {"accuracy": acc, **prob_metrics(y[m], df[f"proba_{name}"].to_numpy()[m])}
        rm = m & resolved.to_numpy()
        tbl["Vegas favorite (spread)"] = {"accuracy": (df["vegas_pick"].to_numpy()[rm] == y[rm]).mean()}
        tbl["Vegas implied prob (vig-free ML)"] = {"accuracy": (implied_pick[m] == y[m]).mean(),
                                                   **prob_metrics(y[m], df["vegas_proba"].to_numpy()[m])}
        rows[season_label] = pd.DataFrame(tbl).T[["accuracy", "log_loss", "brier_score", "roc_auc"]]
        rows[season_label]["n"] = int(m.sum())
        print(f"\n--- {season_label} (n={int(m.sum())}) ---")
        print(rows[season_label].to_string(float_format=lambda v: "-" if pd.isna(v) else f"{v:.4f}"))
    print("\nNOTE: model probabilities are UNCALIBRATED (SPEC 7.3 not yet applied); "
          "log_loss/brier comparisons vs Vegas are provisional.")

    lo, hi = wilson_ci(vegas_fav_correct, int(resolved.sum()))
    print(f"\nVegas favorite pooled: {vegas_fav_correct}/{int(resolved.sum())} = "
          f"{vegas_fav_correct / resolved.sum():.4f}  (95% Wilson CI {lo:.4f}-{hi:.4f})")

    print("\n=== Agreement analysis: model pick vs Vegas favorite (pooled, resolved games) ===")
    print("On disagreement games exactly one side is right, so model accuracy there is "
          "also an exact McNemar test of model vs Vegas favorite (H0: 50%).")
    vp = df.loc[resolved, "vegas_pick"].to_numpy()
    agree_rows = []
    for name in MODELS:
        mp = df.loc[resolved, f"pred_{name}"].to_numpy()
        dis = mp != vp
        n_dis = int(dis.sum())
        k = int((mp[dis] == yr[dis]).sum())
        lo, hi = wilson_ci(k, n_dis)
        p = binomtest(k, n_dis, 0.5).pvalue if n_dis else np.nan
        n_agree = int((~dis).sum())
        agree_acc = (mp[~dis] == yr[~dis]).mean()
        dis_home = int((mp[dis] == 1).sum())
        agree_rows.append({"model": name, "agree_n": n_agree, "agree_acc": agree_acc,
                           "disagree_n": n_dis, "model_right": k, "vegas_right": n_dis - k,
                           "model_acc_on_disagree": k / n_dis if n_dis else np.nan,
                           "ci95_lo": lo, "ci95_hi": hi, "mcnemar_p": p,
                           "model_picked_home_in_disagree": dis_home})
    at = pd.DataFrame(agree_rows).set_index("model")
    print(at.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== Disagreement games by season ===")
    for s in TEST_SEASONS:
        sm = (df.loc[resolved, "season"] == s).to_numpy()
        parts = []
        for name in MODELS:
            mp = df.loc[resolved, f"pred_{name}"].to_numpy()
            dis = (mp != vp) & sm
            parts.append(f"{name}: {int((mp[dis] == yr[dis]).sum())}/{int(dis.sum())}")
        print(f"  {s}: " + "   ".join(parts))


if __name__ == "__main__":
    main()
