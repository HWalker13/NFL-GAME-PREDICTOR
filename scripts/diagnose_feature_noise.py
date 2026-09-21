"""Throwaway diagnostic (not part of the pipeline). Generalized version of
scripts/diagnose_elo_noise_extended.py: same rotating internal holdout,
restricted to TRAIN seasons (2002-2021) ONLY -- never touches the real 2022
validation season or the real 2023-2024 test seasons -- parameterized by
TARGET FEATURE and LogisticRegression C so the same method can be applied to
any SPEC 5.5 check-4-flagged feature, not just elo_diff.

Does not modify any file in src/, does not touch season 2022 or 2023-2024
anywhere, does not retrain or resave any real model, does not change any
feature set or model code.

Usage::

    python -m scripts.diagnose_feature_noise
"""

from __future__ import annotations

import sys
from math import comb

import numpy as np
import pandas as pd

from src import train_winner as TW, train_winner_tuned as TWT

PSEUDO_VAL_SEASONS = tuple(range(2010, 2022))  # 2010..2021 inclusive, 12 folds


def fit_predict(train_slice: pd.DataFrame, pseudo_val: pd.DataFrame,
                feature_cols: list[str], C: float) -> np.ndarray:
    X_train, y_train = train_slice[feature_cols], train_slice["home_win"].to_numpy()
    X_val = pseudo_val[feature_cols]
    pipe = TWT.make_lr_pipeline(C=C, random_state=0).fit(X_train, y_train)
    return pipe.predict(X_val)


def sign_test_p_at_least_k(n_nonzero: int, k_larger: int) -> float:
    """P(X >= k_larger) under X ~ Binomial(n_nonzero, 0.5)."""
    if n_nonzero == 0:
        return float("nan")
    total = 0.0
    for x in range(k_larger, n_nonzero + 1):
        total += comb(n_nonzero, x) * (0.5 ** n_nonzero)
    return total


def run_rotating_holdout(train_df: pd.DataFrame, target_feature: str, C: float,
                         pseudo_val_seasons=PSEUDO_VAL_SEASONS) -> pd.DataFrame:
    feature_cols_full = TW.FEATURE_COLS
    feature_cols_without = [c for c in feature_cols_full if c != target_feature]

    results = []
    for s in pseudo_val_seasons:
        train_slice = train_df[train_df["season"] < s].reset_index(drop=True)
        pseudo_val = train_df[train_df["season"] == s].reset_index(drop=True)
        n = len(pseudo_val)
        y_val = pseudo_val["home_win"].to_numpy()

        pred_with = fit_predict(train_slice, pseudo_val, feature_cols_full, C)
        pred_without = fit_predict(train_slice, pseudo_val, feature_cols_without, C)

        acc_with = float((pred_with == y_val).mean())
        acc_without = float((pred_without == y_val).mean())
        delta = acc_without - acc_with

        n_flipped = int((pred_with != pred_without).sum())

        p = acc_with
        se = float(np.sqrt(p * (1 - p) / n))
        within_1se = abs(delta) <= se

        results.append({
            "pseudo_val_season": s, "train_seasons": f"2002-{s - 1}", "n_games": n,
            f"acc_with_{target_feature}": acc_with, f"acc_without_{target_feature}": acc_without,
            "delta_(ablated-full)": delta, "n_predictions_flipped": n_flipped,
            "binomial_SE": se, "within_1SE": within_1se,
        })
    return pd.DataFrame(results)


def report(tbl: pd.DataFrame, target_feature: str, C: float) -> dict:
    print("=" * 110)
    print(f"Rotating internal holdout (TRAIN seasons only, {len(tbl)} folds) -- "
          f"LR (C={C}), full 67-feature set, target={target_feature}")
    print("=" * 110)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    deltas = tbl["delta_(ablated-full)"].to_numpy()
    signs = np.where(deltas > 0, "+", np.where(deltas < 0, "-", "0"))
    n_pos = int((deltas > 0).sum())
    n_neg = int((deltas < 0).sum())
    n_zero = int((deltas == 0).sum())

    print(f"\ndeltas (ablated - full), in season order: {[round(d, 4) for d in deltas]}")
    print(f"signs: {list(signs)}")
    print(f"\nsign counts: positive={n_pos}  negative={n_neg}  exactly-zero={n_zero}")

    n_nonzero = n_pos + n_neg
    k_majority = max(n_pos, n_neg)
    p_at_least_k = sign_test_p_at_least_k(n_nonzero, k_majority) if n_nonzero else float("nan")
    print(f"non-zero folds: {n_nonzero}  (positive={n_pos}, negative={n_neg})")
    print(f"majority direction: {'positive' if n_pos >= n_neg else 'negative'}  (count={k_majority})")
    print(f"P(X >= {k_majority} out of {n_nonzero}) under Binomial(n={n_nonzero}, p=0.5): "
          f"{p_at_least_k:.4f}")

    return {"n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero,
            "n_nonzero": n_nonzero, "k_majority": k_majority, "p_value": p_at_least_k}


def main() -> int:
    frame = TW.load_train_val_frame()
    # ONLY ever touch season <= TRAIN_MAX_SEASON (2021) from here on -- the
    # season==2022 rows inside `frame` are never sliced out or used.
    train_df = frame[frame["season"] <= TW.TRAIN_MAX_SEASON].reset_index(drop=True)
    print(f"TRAIN-only frame: seasons {train_df['season'].min()}-{train_df['season'].max()}  "
          f"n={len(train_df):,}  (season 2022/2023/2024 never touched in this script)\n")

    target_feature = "away_def_epa_early_ewm"
    C = 1.0  # the UNTUNED model's actual hyperparameter (sklearn default) --
             # this is the model whose check-4 flag is under investigation.

    tbl = run_rotating_holdout(train_df, target_feature, C)
    stats = report(tbl, target_feature, C)

    return 0


if __name__ == "__main__":
    sys.exit(main())
