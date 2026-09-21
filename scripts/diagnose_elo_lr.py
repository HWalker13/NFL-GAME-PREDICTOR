"""Throwaway diagnostic (not part of the pipeline) -- distinguish "elo_diff is
leaking" from "elo_diff is fine standalone but destabilizes LogisticRegression's
coefficients via collinearity with another retained feature."

Reuses data-loading / pipeline / grid-search functions from src/ (train_winner,
train_winner_tuned, train_winner_ensemble) instead of duplicating that logic.
Does not modify any file in src/, does not save any joblib files, does not
touch train_winner_tuned.py's save-gating logic.

Usage::

    python -m scripts.diagnose_elo_lr
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import train_winner as TW, train_winner_ensemble as TWE, train_winner_tuned as TWT


def main() -> int:
    frame = TW.load_train_val_frame()
    train_df = frame[frame["season"] <= TW.TRAIN_MAX_SEASON].reset_index(drop=True)
    val_df = frame[frame["season"] == TW.VAL_SEASON].reset_index(drop=True)
    print(f"train : seasons {train_df['season'].min()}-{train_df['season'].max()}  n={len(train_df):,}")
    print(f"val   : season  {sorted(val_df['season'].unique().tolist())}  n={len(val_df):,}\n")

    folds = TWT.build_expanding_season_folds(train_df)
    X_train = train_df[TW.FEATURE_COLS]
    y_train = train_df["home_win"].to_numpy()
    X_val = val_df[TW.FEATURE_COLS]
    y_val = val_df["home_win"].to_numpy()

    # Re-run the exact same three GridSearchCV passes as train_winner_tuned.py
    # (same grids, same folds, same refit metric) so best_params_ below are
    # the actual GridSearchCV result, not hardcoded from memory.
    lr_search = TWT.run_grid_search("LogisticRegression", TWT.make_lr_pipeline(), TWT.LR_GRID,
                                    X_train, y_train, folds)
    rf_search = TWT.run_grid_search("RandomForest", TWT.make_rf_pipeline(), TWT.RF_GRID,
                                    X_train, y_train, folds)
    hgb_search = TWT.run_grid_search("HistGradientBoosting", TWT.make_hgb_pipeline(), TWT.HGB_GRID,
                                     X_train, y_train, folds)

    # ------------------------------------------------------------------ #
    # 1. Refit LR's exact best_estimator_ (same C, same train fold) and
    #    print all coefficients sorted by |coef| descending, top 15.
    # ------------------------------------------------------------------ #
    lr_pipe = lr_search.best_estimator_
    coef_tbl = TW.coefficient_table(lr_pipe, TW.FEATURE_COLS)
    print(f"\n{'=' * 78}\n1. LogisticRegression-tuned (C={lr_search.best_params_['clf__C']}) "
          f"coefficients, top 15 by |coef|\n{'=' * 78}")
    print(coef_tbl.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    elo_diff_rank_lr = int(coef_tbl.reset_index(drop=True).index[coef_tbl["feature"] == "elo_diff"][0]) + 1
    elo_diff_coef = float(coef_tbl.loc[coef_tbl["feature"] == "elo_diff", "coef"].iloc[0])
    print(f"\nelo_diff: coef={elo_diff_coef:+.4f}  rank={elo_diff_rank_lr}/{len(coef_tbl)} by |coef|  "
          f"(football intuition: higher elo_diff = home_elo_pre - away_elo_pre favors home -> expect POSITIVE)")

    # ------------------------------------------------------------------ #
    # 2. Pairwise correlation matrix (TRAIN rows only), median-imputed the
    #    same way the pipeline imputes (impute step fit on TRAIN only) so
    #    the correlations reflect what the model actually sees.
    # ------------------------------------------------------------------ #
    imputer = lr_pipe.named_steps["impute"]
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=TW.FEATURE_COLS)
    corr = X_train_imputed.corr()

    elo_corr = corr["elo_diff"].drop("elo_diff").sort_values(key=lambda s: s.abs(), ascending=False)
    print(f"\n{'=' * 78}\n2. Top 10 |r| pairs involving elo_diff (TRAIN rows only, n={len(train_df):,}, "
          f"median-imputed)\n{'=' * 78}")
    top10 = elo_corr.head(10)
    for feat, r in top10.items():
        print(f"  elo_diff  <->  {feat:<34} r={r:+.4f}")

    # ------------------------------------------------------------------ #
    # 3. Cross-reference: elo_diff's coef next to each top-10-correlated
    #    feature's own coef, flagging same pattern (high |r|, opposite sign,
    #    large magnitude) as the signature of collinearity, not a leak.
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 78}\n3. Cross-reference: elo_diff coef vs. its top-correlated features' coefs\n{'=' * 78}")
    coef_idx = coef_tbl.set_index("feature")["coef"]
    rows = []
    for feat, r in top10.items():
        other_coef = float(coef_idx.get(feat, float("nan")))
        opposite_sign = (elo_diff_coef > 0) != (other_coef > 0)
        rows.append({
            "feature": feat, "r_with_elo_diff": r,
            "elo_diff_coef": elo_diff_coef, "feature_coef": other_coef,
            "opposite_sign": opposite_sign,
            "collinearity_signature": bool(opposite_sign and abs(r) > 0.5),
        })
    cross_tbl = pd.DataFrame(rows)
    print(cross_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    flagged_pairs = cross_tbl[cross_tbl["collinearity_signature"]]
    if len(flagged_pairs):
        print(f"\n  {len(flagged_pairs)} pair(s) show the collinearity signature "
              "(|r| > 0.5 AND opposite-signed coefficients):")
        print(flagged_pairs.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print("\n  No pair among the top-10 |r| shows |r| > 0.5 with an opposite-signed coefficient.")

    # ------------------------------------------------------------------ #
    # 4. elo_diff's importance rank in RF-tuned / HGB-tuned side by side
    #    with its LR coefficient rank.
    # ------------------------------------------------------------------ #
    rf_pipe = rf_search.best_estimator_
    hgb_pipe = hgb_search.best_estimator_

    rf_importance_fn = TWE.rf_importance
    hgb_importance_fn = TWE.make_hgb_importance_fn(X_val, y_val)

    rf_values = np.asarray(rf_importance_fn(rf_pipe), dtype=float)
    hgb_values = np.asarray(hgb_importance_fn(hgb_pipe), dtype=float)

    def _rank_of(values: np.ndarray, feature_cols: list[str], target: str, use_abs: bool = True) -> tuple[int, float]:
        s = pd.Series(np.abs(values) if use_abs else values, index=feature_cols)
        order = s.sort_values(ascending=False)
        rank = int(order.index.get_loc(target)) + 1
        return rank, float(s[target])

    rf_rank, rf_val = _rank_of(rf_values, TW.FEATURE_COLS, "elo_diff")
    hgb_rank, hgb_val = _rank_of(hgb_values, TW.FEATURE_COLS, "elo_diff")

    print(f"\n{'=' * 78}\n4. elo_diff importance rank: LR coef vs. RF-tuned vs. HGB-tuned "
          f"(out of {len(TW.FEATURE_COLS)} features)\n{'=' * 78}")
    print(f"  LogisticRegression-tuned  rank={elo_diff_rank_lr:<4} |coef|={abs(elo_diff_coef):.4f}  "
          f"(coef={elo_diff_coef:+.4f})")
    print(f"  RandomForest-tuned        rank={rf_rank:<4} feature_importances_={rf_val:.4f}")
    print(f"  HistGradientBoosting-tuned rank={hgb_rank:<4} permutation_importance={hgb_val:.4f}")

    # ------------------------------------------------------------------ #
    # 5. RandomForest-tuned's actual best_params_ vs. the untuned baseline's
    #    hardcoded params.
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 78}\n5. RandomForest-tuned best_params_ vs. untuned baseline hardcoded params\n{'=' * 78}")
    print(f"  RandomForest-tuned   best_params_ (from GridSearchCV) : {rf_search.best_params_}")
    print(f"  RandomForest-untuned hardcoded RF_PARAMS (train_winner_ensemble.py): {TWE.RF_PARAMS}")

    tuned_max_depth = rf_search.best_params_.get("clf__max_depth")
    tuned_min_leaf = rf_search.best_params_.get("clf__min_samples_leaf")
    untuned_max_depth = TWE.RF_PARAMS.get("max_depth")
    untuned_min_leaf = TWE.RF_PARAMS.get("min_samples_leaf")
    identical = (tuned_max_depth == untuned_max_depth) and (tuned_min_leaf == untuned_min_leaf)
    print(f"\n  max_depth        : tuned={tuned_max_depth}  untuned={untuned_max_depth}  "
          f"{'MATCH' if tuned_max_depth == untuned_max_depth else 'DIFFERS'}")
    print(f"  min_samples_leaf : tuned={tuned_min_leaf}  untuned={untuned_min_leaf}  "
          f"{'MATCH' if tuned_min_leaf == untuned_min_leaf else 'DIFFERS'}")
    print(f"  n_estimators     : tuned=300 (fixed, not searched -- see make_rf_pipeline docstring)  "
          f"untuned={TWE.RF_PARAMS.get('n_estimators')}  "
          f"{'MATCH' if TWE.RF_PARAMS.get('n_estimators') == 300 else 'DIFFERS'}")
    print(f"\n  searched params identical to untuned baseline: {identical} "
          f"{'-- fully explains the exact metric tie in the comparison table.' if identical else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
