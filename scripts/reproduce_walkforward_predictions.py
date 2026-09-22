"""Reproduction check (Phase 7, Part B) of the Phase 6 SPEC Section 6
walk-forward test result, persisting PER-GAME predictions this time.

Provenance: the original Phase 6 run (2026-09-22) used a standalone script in
a session scratchpad that only printed aggregate metrics -- no per-game
predictions were saved and the script was never committed. This file
re-implements that exact protocol so it is version-controlled, and saves the
per-game output so it never has to be regenerated again.

Protocol (identical to Phase 6, nothing chosen here):
  Step A: train seasons 2002-2022 -> predict 2023
  Step B: train seasons 2002-2023 -> predict 2024
  Models are sklearn.base.clone() of each SAVED pipeline (same steps, same
  hyperparameters, random_state=0), on the saved 67-feature list.

This is NOT model selection, tuning, or a new evaluation: the regenerated
pooled accuracies MUST match the Phase 6 numbers recorded in CLAUDE.md
exactly (as integer correct-counts out of 544), or the script exits non-zero
and writes nothing. Season 2025 is never read past the season filter.

Usage::

    python -m scripts.reproduce_walkforward_predictions
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone

from src import train_winner as TW

OUT_PATH = TW.PROC_DIR / "walkforward_test_predictions.parquet"

# model label -> saved joblib (RF-untuned skipped: identical hyperparameters
# to RF-tuned, per Phase 6).
SAVED = {
    "LR-untuned": "logreg_winner.joblib",
    "LR-tuned": "logreg_winner_tuned.joblib",
    "RandomForest": "rf_winner_tuned.joblib",
    "HGB-tuned": "hgb_winner_tuned.joblib",
}

# Phase 6 pooled accuracies from CLAUDE.md, expressed as correct/544.
EXPECTED_CORRECT = {"LR-untuned": 342, "LR-tuned": 350, "RandomForest": 343, "HGB-tuned": 342}
EXPECTED_N = 544
# Phase 6 McNemar p-values recorded in CLAUDE.md (4 dp).
EXPECTED_MCNEMAR = {("LR-untuned", "LR-tuned"): 0.0963,
                    ("LR-tuned", "RandomForest"): 0.3240,
                    ("LR-tuned", "HGB-tuned"): 0.3222}

STEPS = ((2022, 2023), (2023, 2024))  # (train_max_season, test_season)


def load_unfitted() -> tuple[dict, list[str]]:
    pipes, feature_cols = {}, None
    for name, fname in SAVED.items():
        d = joblib.load(TW.MODELS_DIR / fname)
        cols = list(d["feature_cols"])
        if feature_cols is None:
            feature_cols = cols
        assert cols == feature_cols, f"{fname}: feature list differs from the other saved models"
        pipe = clone(d["pipeline"])
        assert pipe.steps[-1][1].get_params()["random_state"] == 0, f"{fname}: random_state != 0"
        pipes[name] = pipe
    return pipes, feature_cols


def mcnemar_p(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    ca, cb = a == y, b == y
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = n10 + n01
    return (1.0 if n == 0 else binomtest(n10, n, 0.5).pvalue), n


def main() -> int:
    frame = pd.read_parquet(TW.PROC_DIR / "game_features.parquet")
    pipes, feature_cols = load_unfitted()
    print(f"features: {len(feature_cols)}  models: {list(pipes)}")

    parts = []
    for train_max, test_season in STEPS:
        train = frame[frame["season"] <= train_max]
        test = frame[frame["season"] == test_season]
        out = test[["game_id", "season", "week", "home_win"]].reset_index(drop=True)
        out["train_max_season"] = train_max
        for name, unfitted in pipes.items():
            pipe = clone(unfitted).fit(train[feature_cols], train["home_win"].to_numpy())
            out[f"proba_{name}"] = pipe.predict_proba(test[feature_cols])[:, 1]
            out[f"pred_{name}"] = pipe.predict(test[feature_cols]).astype(int)
        print(f"train 2002-{train_max} (n={len(train):,}) -> test {test_season} (n={len(test)})")
        parts.append(out)
    preds = pd.concat(parts, ignore_index=True)
    y = preds["home_win"].to_numpy()

    print("\n--- reproduction check vs CLAUDE.md Phase 6 pooled accuracy ---")
    ok = len(preds) == EXPECTED_N
    print(f"n = {len(preds)} (expected {EXPECTED_N})")
    for name in pipes:
        correct = int((preds[f"pred_{name}"].to_numpy() == y).sum())
        match = correct == EXPECTED_CORRECT[name]
        ok &= match
        print(f"  {name:<13} {correct}/{len(y)} = {correct / len(y):.4f}   "
              f"expected {EXPECTED_CORRECT[name]}/{EXPECTED_N} = {EXPECTED_CORRECT[name] / EXPECTED_N:.4f}   "
              f"match={match}")

    print("\n--- McNemar exact p-values (all 6 pairs) ---")
    names = list(pipes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p, n_disc = mcnemar_p(y, preds[f"pred_{a}"].to_numpy(), preds[f"pred_{b}"].to_numpy())
            exp = EXPECTED_MCNEMAR.get((a, b))
            tag = "" if exp is None else f"  expected {exp:.4f} match={round(p, 4) == exp}"
            if exp is not None:
                ok &= round(p, 4) == exp
            print(f"  {a:<13} vs {b:<13} p={p:.4f} discordant={n_disc}{tag}")

    if not ok:
        print("\n!! REPRODUCTION MISMATCH -- nothing written. Stop and report; do not debug by changing anything.")
        return 1
    preds.to_parquet(OUT_PATH, index=False)
    print(f"\nall checks match -> wrote {OUT_PATH.relative_to(TW.PROJECT_ROOT)} ({len(preds)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
