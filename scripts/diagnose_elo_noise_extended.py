"""Throwaway diagnostic (not part of the pipeline). Extension of
scripts/diagnose_elo_noise.py: same rotating internal holdout, restricted to
TRAIN seasons (2002-2021) ONLY -- never touches the real 2022 validation
season or the real 2023-2024 test seasons -- but over EVERY season from 2010
through 2021 (12 folds instead of 4), plus per-game flip counts and a sign
test.

Does not modify any file in src/, does not touch season 2022 or 2023-2024
anywhere, does not retrain or resave any real model, does not change any
feature set or model code.

Usage::

    python -m scripts.diagnose_elo_noise_extended
"""

from __future__ import annotations

import sys
from math import comb

import numpy as np
import pandas as pd

from src import train_winner as TW, train_winner_tuned as TWT

PSEUDO_VAL_SEASONS = tuple(range(2010, 2022))  # 2010..2021 inclusive, 12 folds
LR_C = 0.01  # the tuned hyperparameter (best_params_ from GridSearchCV, prior sessions)


def fit_predict(train_slice: pd.DataFrame, pseudo_val: pd.DataFrame,
                feature_cols: list[str]) -> np.ndarray:
    X_train, y_train = train_slice[feature_cols], train_slice["home_win"].to_numpy()
    X_val = pseudo_val[feature_cols]
    pipe = TWT.make_lr_pipeline(C=LR_C, random_state=0).fit(X_train, y_train)
    return pipe.predict(X_val)


def sign_test_p_at_least_k(n_nonzero: int, k_larger: int) -> float:
    """P(X >= k_larger) under X ~ Binomial(n_nonzero, 0.5) -- one-sided sign
    test probability of seeing at least this many outcomes in the majority
    direction by chance alone, if the true effect were zero."""
    if n_nonzero == 0:
        return float("nan")
    total = 0.0
    for x in range(k_larger, n_nonzero + 1):
        total += comb(n_nonzero, x) * (0.5 ** n_nonzero)
    return total


def main() -> int:
    frame = TW.load_train_val_frame()
    # ONLY ever touch season <= TRAIN_MAX_SEASON (2021) from here on -- the
    # season==2022 rows inside `frame` are never sliced out or used.
    train_df = frame[frame["season"] <= TW.TRAIN_MAX_SEASON].reset_index(drop=True)
    print(f"TRAIN-only frame: seasons {train_df['season'].min()}-{train_df['season'].max()}  "
          f"n={len(train_df):,}  (season 2022/2023/2024 never touched in this script)\n")

    feature_cols_full = TW.FEATURE_COLS
    feature_cols_no_elo = [c for c in feature_cols_full if c != "elo_diff"]
    print(f"full feature set: {len(feature_cols_full)} columns")
    print(f"without elo_diff: {len(feature_cols_no_elo)} columns")
    print(f"pseudo-validation seasons: {PSEUDO_VAL_SEASONS}  ({len(PSEUDO_VAL_SEASONS)} folds)\n")

    results = []
    for s in PSEUDO_VAL_SEASONS:
        train_slice = train_df[train_df["season"] < s].reset_index(drop=True)
        pseudo_val = train_df[train_df["season"] == s].reset_index(drop=True)
        n = len(pseudo_val)
        y_val = pseudo_val["home_win"].to_numpy()

        pred_with = fit_predict(train_slice, pseudo_val, feature_cols_full)
        pred_without = fit_predict(train_slice, pseudo_val, feature_cols_no_elo)

        acc_with = float((pred_with == y_val).mean())
        acc_without = float((pred_without == y_val).mean())
        delta = acc_without - acc_with  # same convention as ablation_check: ablated - full

        n_flipped = int((pred_with != pred_without).sum())

        p = acc_with
        se = float(np.sqrt(p * (1 - p) / n))
        within_1se = abs(delta) <= se

        results.append({
            "pseudo_val_season": s, "train_seasons": f"2002-{s - 1}", "n_games": n,
            "acc_with_elo_diff": acc_with, "acc_without_elo_diff": acc_without,
            "delta_(ablated-full)": delta, "n_predictions_flipped": n_flipped,
            "binomial_SE": se, "within_1SE": within_1se,
        })

    tbl = pd.DataFrame(results)
    print("=" * 110)
    print(f"Rotating internal holdout (TRAIN seasons only, {len(PSEUDO_VAL_SEASONS)} folds) -- "
          f"LR (C=0.01), full 67-feature set")
    print("=" * 110)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    deltas = tbl["delta_(ablated-full)"].to_numpy()
    signs = np.where(deltas > 0, "+", np.where(deltas < 0, "-", "0"))
    n_pos = int((deltas > 0).sum())
    n_neg = int((deltas < 0).sum())
    n_zero = int((deltas == 0).sum())

    print(f"\ndeltas (ablated - full), in season order {PSEUDO_VAL_SEASONS}: "
          f"{[round(d, 4) for d in deltas]}")
    print(f"signs: {list(signs)}")

    # ------------------------------------------------------------------ #
    # 1. Positive / negative / exactly-zero counts.
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 110}\n1. Sign counts across {len(PSEUDO_VAL_SEASONS)} folds\n{'=' * 110}")
    print(f"  positive: {n_pos}   negative: {n_neg}   exactly-zero: {n_zero}")

    # ------------------------------------------------------------------ #
    # 2. Per-fold count of individual game predictions that flipped.
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 110}\n2. Individual prediction flips per fold (with-elo_diff vs without)\n{'=' * 110}")
    nonzero_tbl = tbl[tbl["delta_(ablated-full)"] != 0]
    print(tbl[["pseudo_val_season", "n_games", "delta_(ablated-full)", "n_predictions_flipped"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  ({len(nonzero_tbl)} of {len(PSEUDO_VAL_SEASONS)} folds have a non-zero accuracy delta)")

    # ------------------------------------------------------------------ #
    # 3. Sign test: P(at least this many in the majority direction | p=0.5).
    # ------------------------------------------------------------------ #
    n_nonzero = n_pos + n_neg
    k_majority = max(n_pos, n_neg)
    p_at_least_k = sign_test_p_at_least_k(n_nonzero, k_majority) if n_nonzero else float("nan")

    print(f"\n{'=' * 110}\n3. Sign test (two-sided-observed, one-sided probability computed)\n{'=' * 110}")
    print(f"  non-zero folds: {n_nonzero}  (positive={n_pos}, negative={n_neg})")
    print(f"  majority direction: {'positive' if n_pos >= n_neg else 'negative'}  (count={k_majority})")
    print(f"  P(X >= {k_majority} out of {n_nonzero}) under Binomial(n={n_nonzero}, p=0.5): "
          f"{p_at_least_k:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
