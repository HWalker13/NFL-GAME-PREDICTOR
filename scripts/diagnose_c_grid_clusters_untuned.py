"""Throwaway diagnostic (not part of the pipeline). Three independent checks:

1. Widen LR's C grid on the FULL original feature set (elo levels included --
   pre the prior session's LR-only exclusion) to see whether neg_log_loss is
   still improving at the smallest C tried or has found an interior optimum.
2. Correlation-cluster analysis on the full feature set (TRAIN only):
   hierarchical clustering on 1-|r|, threshold |r|>0.7, cross-referenced
   against the SAVED RF-tuned/HGB-tuned models' importance ranks.
3. Re-verify SPEC 5.5 checks 3/4 against the CURRENTLY SAVED
   models/logreg_winner.joblib (untuned Phase 4 model, regenerated post-5A) --
   never re-verified since Phase 5A added the EWM/Elo columns.

Does not modify any file in src/, does not retrain or resave any model.
Reuses data-loading / pipeline / importance functions from src/ instead of
duplicating that logic.

Usage::

    python -m scripts.diagnose_c_grid_clusters_untuned
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import GridSearchCV

from src import evaluate, leakage_checks as LC, train_winner as TW, train_winner_ensemble as TWE, \
    train_winner_tuned as TWT


def part1_c_grid(train_df: pd.DataFrame, folds) -> None:
    print(f"\n{'=' * 78}\n1. Widened LR C grid, FULL feature set ({len(TW.FEATURE_COLS)} cols, "
          f"elo levels included)\n{'=' * 78}")
    X_train = train_df[TW.FEATURE_COLS]
    y_train = train_df["home_win"].to_numpy()

    c_grid = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    print(f"  grid: {c_grid}  ({len(c_grid)} combinations x {len(folds)} folds = "
          f"{len(c_grid) * len(folds)} fits)")

    search = GridSearchCV(
        estimator=TWT.make_lr_pipeline(), param_grid={"clf__C": c_grid},
        scoring="neg_log_loss", refit=False, cv=folds, n_jobs=-1, return_train_score=False,
    )
    search.fit(X_train, y_train)
    res = pd.DataFrame(search.cv_results_)
    by_c = res.set_index(res["param_clf__C"].astype(float))["mean_test_score"].sort_index()

    print(f"\n  {'C':>10}  {'neg_log_loss':>14}  {'log_loss':>10}")
    prev = None
    for c, score in by_c.items():
        log_loss = -score
        direction = ""
        if prev is not None:
            direction = "  (improving, lower)" if log_loss < prev else "  (worse, higher)"
        print(f"  {c:>10}  {score:>14.5f}  {log_loss:>10.5f}{direction}")
        prev = log_loss

    best_c = float(by_c.idxmax())
    is_floor = best_c == min(c_grid)
    print(f"\n  best C (lowest log_loss / highest neg_log_loss): {best_c}")
    print(f"  {'STILL IMPROVING AT THE SMALLEST C TRIED -- floor not found.' if is_floor else 'INTERIOR OPTIMUM FOUND -- log loss turns back up before the smallest C.'}")


def part2_clusters(train_df: pd.DataFrame) -> None:
    print(f"\n{'=' * 78}\n2. Correlation-cluster analysis, full feature set "
          f"(TRAIN rows only, n={len(train_df):,})\n{'=' * 78}")

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(train_df[TW.FEATURE_COLS]), columns=TW.FEATURE_COLS)
    corr = X_imputed.corr().to_numpy()
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 1.0)

    dist = 1.0 - abs_corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry against fp noise
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method="average")
    # |r| > 0.7  <=>  distance (1-|r|) < 0.3
    cluster_ids = fcluster(Z, t=0.3, criterion="distance")

    feature_cols = TW.FEATURE_COLS
    clusters: dict[int, list[str]] = {}
    for feat, cid in zip(feature_cols, cluster_ids):
        clusters.setdefault(cid, []).append(feat)

    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print(f"  clustering: scipy linkage(method='average') on 1-|r|, "
          f"fcluster(criterion='distance', t=0.3)  [|r|>0.7 membership]")
    print(f"  number of clusters: {len(clusters)}")
    print(f"  cluster sizes (largest first): {sizes}")
    n_multi = sum(1 for v in clusters.values() if len(v) >= 2)
    print(f"  clusters with 2+ members: {n_multi}")

    # RF-tuned / HGB-tuned importance ranks, pulled from the SAVED models.
    rf_saved = joblib.load(TW.MODELS_DIR / "rf_winner_tuned.joblib")
    hgb_saved = joblib.load(TW.MODELS_DIR / "hgb_winner_tuned.joblib")
    assert rf_saved["feature_cols"] == feature_cols, "rf_winner_tuned.joblib feature_cols order mismatch"
    assert hgb_saved["feature_cols"] == feature_cols, "hgb_winner_tuned.joblib feature_cols order mismatch"

    frame = TW.load_train_val_frame()
    val_df = frame[frame["season"] == TW.VAL_SEASON].reset_index(drop=True)
    X_val = val_df[feature_cols]
    y_val = val_df["home_win"].to_numpy()

    rf_values = np.asarray(TWE.rf_importance(rf_saved["pipeline"]), dtype=float)
    hgb_importance_fn = TWE.make_hgb_importance_fn(X_val, y_val)
    hgb_values = np.asarray(hgb_importance_fn(hgb_saved["pipeline"]), dtype=float)

    rf_rank = pd.Series(np.abs(rf_values), index=feature_cols).rank(ascending=False, method="min").astype(int)
    hgb_rank = pd.Series(np.abs(hgb_values), index=feature_cols).rank(ascending=False, method="min").astype(int)

    print(f"\n  Clusters with 2+ members -- features + RF-tuned / HGB-tuned importance rank "
          f"(out of {len(feature_cols)}):")
    multi_clusters = {cid: feats for cid, feats in clusters.items() if len(feats) >= 2}
    for cid, feats in sorted(multi_clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  cluster {cid}  (size={len(feats)}):")
        rows = pd.DataFrame({
            "feature": feats,
            "rf_tuned_rank": [int(rf_rank[f]) for f in feats],
            "rf_tuned_importance": [float(rf_values[feature_cols.index(f)]) for f in feats],
            "hgb_tuned_rank": [int(hgb_rank[f]) for f in feats],
            "hgb_tuned_importance": [float(hgb_values[feature_cols.index(f)]) for f in feats],
        }).sort_values("rf_tuned_rank")
        print(rows.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def part3_untuned_lr_leakage_recheck(train_df: pd.DataFrame, val_df: pd.DataFrame, schedules) -> None:
    print(f"\n{'=' * 78}\n3. SPEC 5.5 checks 3/4 -- CURRENTLY SAVED models/logreg_winner.joblib "
          f"(untuned Phase 4, post-5A)\n{'=' * 78}")
    saved = joblib.load(TW.MODELS_DIR / "logreg_winner.joblib")
    pipe = saved["pipeline"]
    feature_cols = saved["feature_cols"]
    print(f"  loaded: feature_cols={len(feature_cols)}  train_max_season={saved['train_max_season']}  "
          f"val_season={saved['val_season']}  val_metrics={saved['val_metrics']}")
    assert feature_cols == TW.FEATURE_COLS, "saved logreg_winner.joblib feature_cols no longer match TW.FEATURE_COLS"

    full_metrics = saved["val_metrics"]
    home_baseline = evaluate.home_baseline_accuracy(schedules, [TW.VAL_SEASON])

    res3 = LC.feature_importance_inspection(
        pipe, feature_cols, model_label="LogisticRegression-untuned(Phase4,post-5A)",
    )
    res4 = LC.ablation_check(
        train_df, val_df, feature_cols, res3["top_feature"], full_metrics,
        home_baseline["baseline_accuracy"], model_label="LogisticRegression-untuned(Phase4,post-5A)",
    )

    check3_verdict = "FAIL" if res3["flagged"] else "PASS"
    check4_verdict = "FAIL" if res4["suspicious"] else "PASS"
    print(f"\n  CHECK 3 (feature-importance inspection): {check3_verdict}")
    print(f"  CHECK 4 (ablation sanity check)         : {check4_verdict}")


def main() -> int:
    frame = TW.load_train_val_frame()
    train_df = frame[frame["season"] <= TW.TRAIN_MAX_SEASON].reset_index(drop=True)
    val_df = frame[frame["season"] == TW.VAL_SEASON].reset_index(drop=True)
    print(f"train : seasons {train_df['season'].min()}-{train_df['season'].max()}  n={len(train_df):,}")
    print(f"val   : season  {sorted(val_df['season'].unique().tolist())}  n={len(val_df):,}")

    folds = TWT.build_expanding_season_folds(train_df)
    schedules = evaluate.load_schedules()

    part1_c_grid(train_df, folds)
    part2_clusters(train_df)
    part3_untuned_lr_leakage_recheck(train_df, val_df, schedules)

    return 0


if __name__ == "__main__":
    sys.exit(main())
