"""Throwaway diagnostic/derivation script (not part of the pipeline).

1. Re-runs the correlation-cluster analysis at |r|>0.5 (not the 0.7 used in
   the prior diagnostic -- 0.5 matches the threshold check_lr_collinearity
   actually used to find the 72 flagged pairs).
2. Defines ONE canonical pruning rule: for every cluster with 2+ members,
   keep the single feature with the best (lowest) average of its
   RandomForest-tuned / HistGradientBoosting-tuned importance rank (pulled
   from the already-saved rf_winner_tuned.joblib / hgb_winner_tuned.joblib --
   no retraining here). Drop every other cluster member. Singletons are kept
   untouched.
3. Prints the full original feature list next to the final pruned list, with
   the cut reason (cluster id, surviving feature) for every dropped feature,
   and prints the pruned list as a literal Python list ready to paste into
   src/train_winner_tuned.py.

Does not modify any file in src/, does not retrain or resave any model.

Usage::

    python -m scripts.compute_pruned_features
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer

from src import train_winner as TW, train_winner_ensemble as TWE


def main() -> int:
    frame = TW.load_train_val_frame()
    train_df = frame[frame["season"] <= TW.TRAIN_MAX_SEASON].reset_index(drop=True)
    val_df = frame[frame["season"] == TW.VAL_SEASON].reset_index(drop=True)
    feature_cols = TW.FEATURE_COLS
    print(f"train : seasons {train_df['season'].min()}-{train_df['season'].max()}  n={len(train_df):,}")
    print(f"full feature list: {len(feature_cols)} columns\n")

    # ------------------------------------------------------------------ #
    # 1. Correlation-cluster analysis at |r| > 0.5 (distance threshold 0.5).
    # ------------------------------------------------------------------ #
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(train_df[feature_cols]), columns=feature_cols)
    corr = X_imputed.corr().to_numpy()
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 1.0)

    dist = 1.0 - abs_corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=0.5, criterion="distance")  # |r| > 0.5 <=> distance < 0.5

    clusters: dict[int, list[str]] = {}
    for feat, cid in zip(feature_cols, cluster_ids):
        clusters.setdefault(cid, []).append(feat)

    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print(f"{'=' * 78}\n1. Correlation clusters at |r| > 0.5 "
          f"(average linkage, distance=1-|r|, threshold=0.5)\n{'=' * 78}")
    print(f"  number of clusters: {len(clusters)}")
    print(f"  cluster sizes (largest first): {sizes}")
    n_multi = sum(1 for v in clusters.values() if len(v) >= 2)
    print(f"  clusters with 2+ members: {n_multi}\n")

    # ------------------------------------------------------------------ #
    # RF-tuned / HGB-tuned importance ranks, from the already-saved models.
    # ------------------------------------------------------------------ #
    rf_saved = joblib.load(TW.MODELS_DIR / "rf_winner_tuned.joblib")
    hgb_saved = joblib.load(TW.MODELS_DIR / "hgb_winner_tuned.joblib")
    assert rf_saved["feature_cols"] == feature_cols, "rf_winner_tuned.joblib feature_cols order mismatch"
    assert hgb_saved["feature_cols"] == feature_cols, "hgb_winner_tuned.joblib feature_cols order mismatch"

    X_val = val_df[feature_cols]
    y_val = val_df["home_win"].to_numpy()
    rf_values = np.asarray(TWE.rf_importance(rf_saved["pipeline"]), dtype=float)
    hgb_importance_fn = TWE.make_hgb_importance_fn(X_val, y_val)
    hgb_values = np.asarray(hgb_importance_fn(hgb_saved["pipeline"]), dtype=float)

    rf_rank = pd.Series(np.abs(rf_values), index=feature_cols).rank(ascending=False, method="min")
    hgb_rank = pd.Series(np.abs(hgb_values), index=feature_cols).rank(ascending=False, method="min")
    avg_rank = (rf_rank + hgb_rank) / 2.0

    # ------------------------------------------------------------------ #
    # 2. Pruning rule: per cluster (2+ members), keep lowest avg_rank member.
    # ------------------------------------------------------------------ #
    print(f"{'=' * 78}\n2. Pruning rule: per cluster, keep the member with best "
          f"(lowest) avg(RF-tuned rank, HGB-tuned rank)\n{'=' * 78}")
    dropped: dict[str, str] = {}  # feature -> reason string
    for cid, feats in sorted(clusters.items()):
        if len(feats) < 2:
            continue
        ranked = sorted(feats, key=lambda f: avg_rank[f])
        keep = ranked[0]
        cut = ranked[1:]
        print(f"\n  cluster {cid} (size={len(feats)}):")
        for f in ranked:
            marker = "KEEP" if f == keep else "DROP"
            print(f"    {marker:<5} {f:<34} rf_rank={int(rf_rank[f]):>3}  hgb_rank={int(hgb_rank[f]):>3}  "
                  f"avg_rank={avg_rank[f]:.1f}")
        for f in cut:
            dropped[f] = f"cluster {cid}, superseded by '{keep}' (avg_rank {avg_rank[keep]:.1f} vs {avg_rank[f]:.1f})"

    pruned_cols = [f for f in feature_cols if f not in dropped]

    print(f"\n{'=' * 78}\n3. Full list vs. pruned list ({len(pruned_cols)} of {len(feature_cols)} kept, "
          f"{len(dropped)} dropped)\n{'=' * 78}")
    for f in feature_cols:
        if f in dropped:
            print(f"  DROP  {f:<34} -- {dropped[f]}")
        else:
            print(f"  KEEP  {f}")

    print(f"\n{'=' * 78}\nPruned feature list -- paste into src/train_winner_tuned.py\n{'=' * 78}")
    print("PRUNED_FEATURE_COLS = [")
    for f in pruned_cols:
        print(f'    "{f}",')
    print("]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
