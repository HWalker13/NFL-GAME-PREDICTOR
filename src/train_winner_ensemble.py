"""Phase 5B Part 1 -- RandomForest + HistGradientBoosting BASELINES (untuned),
compared side by side against the existing LogisticRegression baseline
(SPEC Section 7.1, Section 13 Phase 5).

Explicitly OUT of scope for this module (separate, later passes):
  - hyperparameter tuning (see CLAUDE.md deferred/future-work: GridSearchCV,
    small exhaustive grid, decided ahead of time -- not run here)
  - probability calibration (SPEC 7.3, ``CalibratedClassifierCV``)
  - feature pruning (SPEC 5.5 #4 / 7.2)
  - walk-forward multi-season backtesting (SPEC Section 6 "recommended
    extension")
  - Section 12 (spread/total) -- untouched
  - seasons 2023-2024 (test) -- never loaded (see ``train_winner.load_train_val_frame``)

Same train (season <= 2021, n=5,124) / validation (season 2022, n=269) split
and the same ``features.feature_columns()`` feature set already used by
``train_winner.py``'s ``LogisticRegression`` model.

Usage::

    python -m src.train_winner_ensemble
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src import evaluate, leakage_checks as LC, train_winner as TW

MODELS_DIR = TW.MODELS_DIR

# --- RandomForest -----------------------------------------------------------
# Baseline, not tuned (real tuning is a later pass -- see CLAUDE.md deferred
# section for the GridSearchCV decision already made for that pass). Chosen
# to be "reasonable defaults", not searched:
#   n_estimators=300  -- more than sklearn's default (100) for a more stable
#                         baseline (more trees only reduces variance, it is
#                         not a bias/variance knob the way depth is, so
#                         bumping it isn't really "tuning")
#   max_depth=8        -- sklearn's default is None (unconstrained), which
#                         tends to badly overfit a ~5,124-row / 67-feature
#                         tabular dataset; capping depth is the one guard a
#                         baseline RF on this data size genuinely needs
#   min_samples_leaf=5 -- each leaf must represent >= 5 games, so a single
#                         fluky game can't drive a confident split (same
#                         small-sample-shrinkage spirit as SPEC Section 4,
#                         implemented as a tree constraint here instead)
RF_PARAMS = dict(n_estimators=300, max_depth=8, min_samples_leaf=5, n_jobs=-1)

# --- HistGradientBoosting ----------------------------------------------------
# Left at library defaults (learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
# min_samples_leaf=20, l2_regularization=0) -- this is a baseline pass, no
# tuning yet, and HGB's defaults are already reasonably regularized for a
# dataset this size (unlike RandomForest's default max_depth=None, HGB's
# default min_samples_leaf=20 / max_leaf_nodes=31 already constrain tree
# growth out of the box).
HGB_PARAMS: dict = {}

# SPEC 5.5 #3: flag the top feature if it exceeds this share (same threshold
# leakage_checks.py uses for LogisticRegression).
IMPORTANCE_FLAG_SHARE = LC.IMPORTANCE_FLAG_SHARE


def make_rf_pipeline(random_state: int = 0) -> Pipeline:
    """Median-impute (RF cannot handle NaNs, same TRAIN-fit imputer pattern
    as ``train_winner.make_pipeline``) -> ``RandomForestClassifier``.

    No ``StandardScaler`` -- tree splits are scale-invariant, unlike
    LogisticRegression's coefficient magnitudes.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(random_state=random_state, **RF_PARAMS)),
    ])


def make_hgb_pipeline(random_state: int = 0) -> Pipeline:
    """No imputer: HGB natively handles NaNs by learning, per split, which
    direction a missing value should go -- a real capability being
    demonstrated here, not just "not needed". Several features in this
    pipeline are structurally missing pre-2006 (cpoe / air_yards), i.e. NaN
    correlates with season, not at random; letting HGB treat missingness as
    its own signal preserves that information instead of median-imputing it
    away. This is a deliberate, stated choice per the task brief, not an
    oversight -- RF gets an imputer (it cannot handle NaNs at all), HGB does
    not (it doesn't need one and imputing would throw away information it
    can otherwise use).
    """
    return Pipeline([
        ("clf", HistGradientBoostingClassifier(random_state=random_state, **HGB_PARAMS)),
    ])


def fit_and_eval(make_pipeline_fn, train_df: pd.DataFrame, val_df: pd.DataFrame,
                 feature_cols: list[str] | None = None,
                 random_state: int = 0) -> tuple[Pipeline, dict]:
    """Same shape as ``train_winner.fit_and_eval``: fit on ``train_df`` only,
    evaluate on ``val_df`` only."""
    cols = feature_cols if feature_cols is not None else TW.FEATURE_COLS
    X_train, y_train = train_df[cols], train_df["home_win"].to_numpy()
    X_val, y_val = val_df[cols], val_df["home_win"].to_numpy()

    pipe = make_pipeline_fn(random_state=random_state).fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    pred = pipe.predict(X_val)
    metrics = TW.compute_metrics(y_val, pred, proba)
    return pipe, metrics


def rf_fit_and_eval(train_df, val_df, feature_cols=None, random_state=0):
    return fit_and_eval(make_rf_pipeline, train_df, val_df, feature_cols, random_state)


def hgb_fit_and_eval(train_df, val_df, feature_cols=None, random_state=0):
    return fit_and_eval(make_hgb_pipeline, train_df, val_df, feature_cols, random_state)


def rf_importance(pipe: Pipeline) -> np.ndarray:
    """RF: native ``feature_importances_`` (mean decrease in impurity)."""
    return pipe.named_steps["clf"].feature_importances_


def make_hgb_importance_fn(X_val: pd.DataFrame, y_val: np.ndarray,
                           n_repeats: int = 20, random_state: int = 0):
    """HGB has NO ``feature_importances_`` attribute in scikit-learn (unlike
    ``RandomForestClassifier``/``GradientBoostingClassifier``) -- confirmed
    against the installed sklearn 1.9.0 (``hasattr(fitted_hgb,
    'feature_importances_')`` is False). The task brief assumed
    ``feature_importances_`` would exist "for both RF and HGB"; it does not
    for HGB, so this substitutes scikit-learn's own recommended alternative,
    permutation importance (``sklearn.inspection.permutation_importance``) --
    still a $0/no-extra-dependency, scikit-learn-native method, computed on
    the validation split with accuracy as the scoring function. FLAGGED
    explicitly here and in the run report as a deviation from the literal
    task instruction, not a silent substitution.
    """
    def _fn(pipe: Pipeline) -> np.ndarray:
        r = permutation_importance(
            pipe, X_val, y_val, scoring="accuracy",
            n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
        )
        return r.importances_mean
    return _fn


def comparison_table(home_baseline: dict, lr_metrics: dict,
                     rf_metrics: dict, hgb_metrics: dict) -> pd.DataFrame:
    """SPEC Section 8 metrics side by side. ``home_baseline`` is a hard-label
    heuristic ("always predict home"), not a probabilistic model -- it has no
    predicted probability to score log_loss/brier/roc_auc against, so those
    columns are reported as NaN for that row rather than fabricating a
    probability (e.g. a constant 1.0 would blow log_loss up to a huge,
    meaningless number; any other constant would be an arbitrary choice the
    task never asked for). This is a deliberate, flagged choice, not a gap.
    """
    rows = {
        "home_baseline": {
            "accuracy": home_baseline["baseline_accuracy"],
            "log_loss": np.nan, "brier_score": np.nan, "roc_auc": np.nan,
        },
        "LogisticRegression": {
            "accuracy": lr_metrics["accuracy"], "log_loss": lr_metrics["log_loss"],
            "brier_score": lr_metrics["brier_score"], "roc_auc": lr_metrics["roc_auc"],
        },
        "RandomForest": {
            "accuracy": rf_metrics["accuracy"], "log_loss": rf_metrics["log_loss"],
            "brier_score": rf_metrics["brier_score"], "roc_auc": rf_metrics["roc_auc"],
        },
        "HistGradientBoosting": {
            "accuracy": hgb_metrics["accuracy"], "log_loss": hgb_metrics["log_loss"],
            "brier_score": hgb_metrics["brier_score"], "roc_auc": hgb_metrics["roc_auc"],
        },
    }
    return pd.DataFrame(rows).T[["accuracy", "log_loss", "brier_score", "roc_auc"]]


def main() -> int:
    print("=== Phase 5B Part 1: RandomForest + HistGradientBoosting baselines "
          "(untuned) vs. existing LogisticRegression ===\n")

    # ------------------------------------------------------------------ #
    # 0. Refresh the LogisticRegression artifact FIRST.
    #
    # FLAG: models/logreg_winner.joblib on disk (as of this run) was fit
    # against only 42 feature_cols -- it predates the Phase 5A EWM/Elo
    # features (features.feature_columns() now returns 67 columns; the
    # game_features.parquet frame has 72 total columns including keys/label).
    # The task requires "the SAME feature set... already used by
    # train_winner.py's LogisticRegression model" for all three models in the
    # comparison table -- pulling stale 42-feature metrics next to
    # 67-feature RF/HGB metrics would silently violate that. So the existing,
    # UNMODIFIED train_winner.py is re-run once here to regenerate a current
    # artifact, and its metadata is then loaded exactly as instructed
    # ("pull... from the already-saved models/logreg_winner.joblib metadata
    # rather than retraining it") -- no LogisticRegression training code is
    # duplicated in this module.
    # ------------------------------------------------------------------ #
    lr_path = MODELS_DIR / "logreg_winner.joblib"
    stale = True
    if lr_path.exists():
        saved = joblib.load(lr_path)
        stale = len(saved.get("feature_cols", [])) != len(TW.FEATURE_COLS)
    if stale:
        print(f"!! FLAG: {lr_path.name} is missing or stale relative to the current "
              f"{len(TW.FEATURE_COLS)}-column feature set -- regenerating via "
              "`python -m src.train_winner` (unmodified) before pulling its metadata.\n")
        rc = TW.main()
        if rc != 0:
            print("\ntrain_winner.py did not complete cleanly (see its own output above) "
                  "-- stopping, not proceeding to RF/HGB.")
            return rc
        print()
    saved = joblib.load(lr_path)
    lr_metrics = saved["val_metrics"]
    assert len(saved["feature_cols"]) == len(TW.FEATURE_COLS), (
        "logreg_winner.joblib feature_cols still do not match the current "
        "feature set after regeneration -- stop and investigate"
    )

    # ------------------------------------------------------------------ #
    # 1. Load the SAME train/validation split + feature set.
    # ------------------------------------------------------------------ #
    frame = TW.load_train_val_frame()
    train_df = frame[frame["season"] <= TW.TRAIN_MAX_SEASON].reset_index(drop=True)
    val_df = frame[frame["season"] == TW.VAL_SEASON].reset_index(drop=True)
    assert set(train_df["season"]).isdisjoint(TW.FORBIDDEN_TEST_SEASONS)
    assert set(val_df["season"]).isdisjoint(TW.FORBIDDEN_TEST_SEASONS)
    print(f"train : seasons {train_df['season'].min()}-{train_df['season'].max()}  n={len(train_df):,}")
    print(f"val   : season  {sorted(val_df['season'].unique().tolist())}  n={len(val_df):,}")
    print(f"features: {len(TW.FEATURE_COLS)} columns (features.feature_columns())\n")

    schedules = evaluate.load_schedules()
    home_baseline = evaluate.home_baseline_accuracy(schedules, [TW.VAL_SEASON])

    # ------------------------------------------------------------------ #
    # 1a/1b/1c. Fit RF and HGB, evaluate on validation.
    # ------------------------------------------------------------------ #
    print(f"RandomForestClassifier params: {RF_PARAMS} | random_state=0")
    rf_pipe, rf_metrics = rf_fit_and_eval(train_df, val_df, TW.FEATURE_COLS, random_state=0)
    print(f"  val accuracy={rf_metrics['accuracy']:.4f}  log_loss={rf_metrics['log_loss']:.4f}  "
          f"brier={rf_metrics['brier_score']:.4f}  roc_auc={rf_metrics['roc_auc']:.4f}")

    print(f"\nHistGradientBoostingClassifier params: library defaults {HGB_PARAMS or '{}'} "
          f"| random_state=0 | NaN handling: native (no imputer -- see make_hgb_pipeline docstring)")
    hgb_pipe, hgb_metrics = hgb_fit_and_eval(train_df, val_df, TW.FEATURE_COLS, random_state=0)
    print(f"  val accuracy={hgb_metrics['accuracy']:.4f}  log_loss={hgb_metrics['log_loss']:.4f}  "
          f"brier={hgb_metrics['brier_score']:.4f}  roc_auc={hgb_metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # 2. ONE comparison table.
    # ------------------------------------------------------------------ #
    tbl = comparison_table(home_baseline, lr_metrics, rf_metrics, hgb_metrics)
    print(f"\n=== Comparison -- validation season {TW.VAL_SEASON} (n={len(val_df)}) -- SPEC Section 8 ===")
    print(tbl.to_string(float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "n/a"))
    print("  (home_baseline log_loss/brier/roc_auc are n/a: it's a hard-label rule, not a "
          "probabilistic model -- see comparison_table() docstring for why nothing is fabricated there)")

    # ------------------------------------------------------------------ #
    # 3. Extend SPEC 5.5 checks 3 & 4 to RF and HGB (generalized functions
    #    in leakage_checks.py -- each model's OWN top feature, not assumed
    #    to match LogisticRegression's or each other's).
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 78}\nExtended SPEC 5.5 leakage checks -- RandomForest\n{'=' * 78}")
    rf_res3 = LC.feature_importance_inspection(
        rf_pipe, TW.FEATURE_COLS, importance_fn=rf_importance,
        flag_share=IMPORTANCE_FLAG_SHARE, model_label="RandomForest",
    )
    rf_res4 = LC.ablation_check(
        train_df, val_df, TW.FEATURE_COLS, rf_res3["top_feature"], rf_metrics,
        home_baseline["baseline_accuracy"], fit_and_eval_fn=rf_fit_and_eval,
        model_label="RandomForest",
    )

    print(f"\n{'=' * 78}\nExtended SPEC 5.5 leakage checks -- HistGradientBoosting\n{'=' * 78}")
    hgb_importance = make_hgb_importance_fn(val_df[TW.FEATURE_COLS], val_df["home_win"].to_numpy())
    hgb_res3 = LC.feature_importance_inspection(
        hgb_pipe, TW.FEATURE_COLS, importance_fn=hgb_importance,
        flag_share=IMPORTANCE_FLAG_SHARE, model_label="HistGradientBoosting",
    )
    hgb_res4 = LC.ablation_check(
        train_df, val_df, TW.FEATURE_COLS, hgb_res3["top_feature"], hgb_metrics,
        home_baseline["baseline_accuracy"], fit_and_eval_fn=hgb_fit_and_eval,
        model_label="HistGradientBoosting",
    )

    # ------------------------------------------------------------------ #
    # 4. SPEC 5.6/CLAUDE.md ~70% stop-and-flag rule, applied to RF and HGB.
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 78}")
    over = []
    if rf_metrics["accuracy"] > TW.SUSPICIOUS_ACCURACY:
        over.append(("RandomForest", rf_metrics["accuracy"]))
    if hgb_metrics["accuracy"] > TW.SUSPICIOUS_ACCURACY:
        over.append(("HistGradientBoosting", hgb_metrics["accuracy"]))
    if over:
        for name, acc in over:
            print(f"\n!! VALIDATION ACCURACY {acc:.4f} FOR {name} EXCEEDS {TW.SUSPICIOUS_ACCURACY} -- "
                  "SPEC 5.6/5.7 + CLAUDE.md: STOP. This is NOT being reported as a good result. "
                  "Not proceeding to model save. Flagging for manual review.")
        return 1

    print(f"\nBoth RF ({rf_metrics['accuracy']:.4f}) and HGB ({hgb_metrics['accuracy']:.4f}) "
          f"validation accuracy are <= {TW.SUSPICIOUS_ACCURACY} -- proceeding to save.")

    # ------------------------------------------------------------------ #
    # 5. Save both models.
    # ------------------------------------------------------------------ #
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rf_out = MODELS_DIR / "rf_winner.joblib"
    joblib.dump(
        {"pipeline": rf_pipe, "feature_cols": TW.FEATURE_COLS,
         "train_max_season": TW.TRAIN_MAX_SEASON, "val_season": TW.VAL_SEASON,
         "val_metrics": rf_metrics, "params": RF_PARAMS, "random_state": 0},
        rf_out,
    )
    print(f"\nSaved fitted RandomForest pipeline -> {rf_out}  ({rf_out.stat().st_size:,} bytes)")

    hgb_out = MODELS_DIR / "hgb_winner.joblib"
    joblib.dump(
        {"pipeline": hgb_pipe, "feature_cols": TW.FEATURE_COLS,
         "train_max_season": TW.TRAIN_MAX_SEASON, "val_season": TW.VAL_SEASON,
         "val_metrics": hgb_metrics, "params": HGB_PARAMS, "random_state": 0},
        hgb_out,
    )
    print(f"Saved fitted HistGradientBoosting pipeline -> {hgb_out}  ({hgb_out.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
