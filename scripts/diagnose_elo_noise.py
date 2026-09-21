"""Throwaway diagnostic (not part of the pipeline). Rotating internal holdout
restricted to TRAIN seasons (2002-2021) ONLY -- never touches the real 2022
validation season or the real 2023-2024 test seasons.

For each of 2018, 2019, 2020, 2021 in turn as a pseudo-validation season
(train on everything strictly before it): fit LogisticRegression (C=0.01,
the tuned hyperparameter) on the full 67-feature set, evaluate accuracy;
drop elo_diff, refit, re-evaluate; report the delta and its binomial
standard error at that game count. Purpose: distinguish "elo_diff's ablation
increase is validation-noise on a 269-game split" from "it's a consistent,
real effect across independent seasons."

Does not modify any file in src/, does not touch season 2022 or 2023-2024
anywhere, does not retrain or resave any real model.

Usage::

    python -m scripts.diagnose_elo_noise
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import train_winner as TW, train_winner_tuned as TWT

PSEUDO_VAL_SEASONS = (2018, 2019, 2020, 2021)
LR_C = 0.01  # the tuned hyperparameter (best_params_ from GridSearchCV, prior sessions)


def fit_eval(train_slice: pd.DataFrame, pseudo_val: pd.DataFrame, feature_cols: list[str]) -> float:
    X_train, y_train = train_slice[feature_cols], train_slice["home_win"].to_numpy()
    X_val, y_val = pseudo_val[feature_cols], pseudo_val["home_win"].to_numpy()
    pipe = TWT.make_lr_pipeline(C=LR_C, random_state=0).fit(X_train, y_train)
    pred = pipe.predict(X_val)
    return float((pred == y_val).mean())


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
    print(f"without elo_diff: {len(feature_cols_no_elo)} columns\n")

    results = []
    for s in PSEUDO_VAL_SEASONS:
        train_slice = train_df[train_df["season"] < s].reset_index(drop=True)
        pseudo_val = train_df[train_df["season"] == s].reset_index(drop=True)
        n = len(pseudo_val)

        acc_with = fit_eval(train_slice, pseudo_val, feature_cols_full)
        acc_without = fit_eval(train_slice, pseudo_val, feature_cols_no_elo)
        delta = acc_without - acc_with  # same convention as ablation_check: ablated - full

        p = acc_with  # binomial SE using the full (with-elo_diff) model's observed accuracy as p
        se = float(np.sqrt(p * (1 - p) / n))
        within_1se = abs(delta) <= se

        results.append({
            "pseudo_val_season": s, "train_seasons": f"2002-{s - 1}", "n_games": n,
            "acc_with_elo_diff": acc_with, "acc_without_elo_diff": acc_without,
            "delta_(ablated-full)": delta, "binomial_SE": se,
            "within_1SE": within_1se,
        })

    tbl = pd.DataFrame(results)
    print("=" * 100)
    print("Rotating internal holdout (TRAIN seasons only) -- LR (C=0.01), full 67-feature set")
    print("=" * 100)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    deltas = tbl["delta_(ablated-full)"].to_numpy()
    signs = ["+" if d > 0 else ("-" if d < 0 else "0") for d in deltas]
    print(f"\ndeltas (ablated - full), in season order {PSEUDO_VAL_SEASONS}: "
          f"{[round(d, 4) for d in deltas]}")
    print(f"signs: {signs}")
    all_same_sign = len(set(s for s in signs if s != "0")) <= 1
    if all_same_sign and "+" in signs:
        print("\nALL (non-zero) deltas are POSITIVE across 4 independent pseudo-validation seasons -- "
              "dropping elo_diff consistently helped. This is NOT the sign-flipping pattern expected "
              "from pure validation noise; treat as evidence against the noise explanation.")
    elif all_same_sign and "-" in signs:
        print("\nALL (non-zero) deltas are NEGATIVE across 4 independent pseudo-validation seasons -- "
              "dropping elo_diff consistently hurt here, unlike on the real 2022 validation split.")
    else:
        print("\nSigns are MIXED across the 4 independent pseudo-validation seasons -- consistent with "
              "validation noise on a single ~250-270 game split, not a stable effect.")

    n_within_1se = int(tbl["within_1SE"].sum())
    print(f"\n{n_within_1se}/4 deltas fall within +-1 binomial SE of 0 "
          f"(i.e., not distinguishable from chance at that game count).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
