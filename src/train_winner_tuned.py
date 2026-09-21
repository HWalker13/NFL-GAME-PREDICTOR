"""Phase 5B Part 2 -- GridSearchCV hyperparameter tuning for LogisticRegression,
RandomForestClassifier, and HistGradientBoostingClassifier, on the single
FULL feature set (SPEC Section 7.1/7.2, Section 13 Phase 5).

Explicitly OUT of scope for this module (separate, later passes):
  - probability calibration (SPEC 7.3, ``CalibratedClassifierCV``)
  - walk-forward multi-season backtesting (SPEC Section 6 "recommended
    extension")
  - Section 12 (spread/total) -- untouched
  - seasons 2023-2024 (test) -- never loaded, never scored against, at any
    point, including inside CV (see ``build_expanding_season_folds``)

Decisions carried over from prior tasks, not re-derived here -- full story,
in order (see CLAUDE.md for the consolidated version):
  - GridSearchCV (exhaustive), not RandomizedSearchCV -- CLAUDE.md deferred
    section, recorded during Phase 5B Part 1.
  - LogisticRegression IS tuned in this pass (C only) -- motivated by the
    Part 1 diagnostic finding real-but-weak multicollinearity between
    several ``_shrunk``/``_ewm`` feature pairs.
  - LR-tuned (and, it turned out, every other model too) fails SPEC 5.5
    check 4 -- dropping ``elo_diff`` increases validation accuracy. TWO fix
    attempts were tried and BOTH REVERTED here because neither resolved it:
      1. An LR-only reduced feature set (drop ``home_elo_pre``/
         ``away_elo_pre``, keep ``elo_diff``) -- stabilized ``elo_diff``'s
         coefficient but did not clear check 4 (``scripts/diagnose_elo_lr.py``).
      2. General, model-agnostic cluster pruning at |r|>0.5 (67 -> 22
         features, applied to all three models --
         ``scripts/compute_pruned_features.py``) -- made it WORSE: all six
         retrained models (LR/RF/HGB x untuned/tuned) failed check 4 on the
         pruned set, including RF and HGB, which had PASSED cleanly on the
         full feature set.
  - THIS MODULE IS BACK to a single, uniform, FULL 67-feature set for all
    three model types (``TW.FEATURE_COLS``, elo levels included) -- no
    per-model special-casing.
  - RESOLUTION: a 12-fold TRAIN-only rotating holdout (2010-2021 pseudo-
    validation seasons, ``scripts/diagnose_elo_noise_extended.py``) found
    the ``elo_diff`` check-4 flag is NOT statistically distinguishable from
    chance -- sign test p=0.254 (6 positive / 3 negative / 3 exactly-zero
    folds). The SAME check-4 pattern was independently found on the
    UNTUNED LR baseline too, but on a DIFFERENT feature
    (``away_def_epa_early_ewm``); a parallel 12-fold holdout
    (``scripts/diagnose_feature_noise.py``) found an even weaker signal
    there (p=0.6875, 8/12 folds exactly zero). See CLAUDE.md for the full
    consolidated story, including a documentation gap: an earlier bootstrap
    diagnostic of ``away_def_epa_early_ewm`` (500 resamples of the single
    2022 val set, mean delta +0.00768, 90% positive) existed but was never
    previously recorded anywhere in this repo.
  - ``LR_CHECK4_OVERRIDES`` below applies a ONE-TIME, EXPLICITLY DOCUMENTED
    override of the save gate for LogisticRegression ONLY (both untuned and
    tuned) -- not a change to the gating LOGIC, which still blocks on check
    3 or the 70% threshold for every model, LR included. RF/HGB pass checks
    3/4 cleanly on their own merits and are never overridden.
  - Save gating is PER-MODEL, not all-or-none: each of LR/RF/HGB (untuned +
    tuned) is saved independently based on its own 70% gate + SPEC 5.5
    checks 3/4 result (or documented override), so one model failing a
    check no longer blocks saving the others.

CV strategy: expanding-window, PER-SEASON folds (NOT sklearn's generic
``TimeSeriesSplit``, which would cut mid-season or use row-count-based
splits with no season awareness). Fold i trains on all TRAIN seasons
strictly before season S and tests on season S, for S in the last 5 TRAIN
seasons (2017-2021) -- 5 folds, built as explicit (train_idx, test_idx)
integer-position arrays and passed directly as ``GridSearchCV``'s ``cv``.

Scoring: multi-metric GridSearchCV, ``refit='neg_log_loss'`` -- the final
``best_estimator_`` per model is chosen by log loss, not accuracy. Top-5 by
log loss and top-5 by accuracy are both printed so any divergence between a
log-loss-optimal and an accuracy-optimal pick is visible, not hidden.

FLAG (deviation from the task's literal scorer syntax, not a silent
substitution): the task specified
``make_scorer(brier_score_loss, needs_proba=True, greater_is_better=False)``.
``needs_proba`` does not exist on the installed scikit-learn (1.9.0)
``make_scorer`` signature -- confirmed via
``inspect.signature(make_scorer)`` returning
``(score_func, *, response_method='predict', greater_is_better=True,
**kwargs)``. Substituted the modern equivalent,
``response_method='predict_proba'`` (same sign convention: scorer output is
the NEGATIVE brier score, verified directly -- higher/less-negative is
better, consistent with ``neg_log_loss``).

Usage::

    python -m src.train_winner_tuned
"""

from __future__ import annotations

import functools
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import evaluate, leakage_checks as LC, train_winner as TW, train_winner_ensemble as TWE

MODELS_DIR = TW.MODELS_DIR

# SPEC 5.5 #3 threshold, reused from leakage_checks.py.
IMPORTANCE_FLAG_SHARE = LC.IMPORTANCE_FLAG_SHARE

# Last 5 TRAIN seasons used as expanding-window CV test folds.
CV_TEST_SEASONS = (2017, 2018, 2019, 2020, 2021)

# --------------------------------------------------------------------------- #
# ONE-TIME, EXPLICITLY DOCUMENTED override of the SPEC 5.5 check-4 save gate,
# for LogisticRegression ONLY (untuned and tuned). This does NOT change the
# gating LOGIC (still per-model, still blocks on check 3 or the 70%
# threshold) -- it is a specific, investigated exception for these two
# artifacts, citing the diagnostic evidence below. Do not extend this pattern
# to a future check-4 failure without an equivalent investigation.
# --------------------------------------------------------------------------- #
LR_CHECK4_OVERRIDES = {
    "LR-tuned": (
        "elo_diff check-4 flag investigated via 12-fold TRAIN-only rotating "
        "holdout (pseudo-validation seasons 2010-2021, scripts/"
        "diagnose_elo_noise_extended.py): sign test p=0.254 (6 positive / "
        "3 negative / 3 exactly-zero folds, out of 9 non-zero) -- not "
        "statistically distinguishable from chance at this per-season game "
        "count (~255-271 games/season)."
    ),
    "LR-untuned": (
        "away_def_epa_early_ewm check-4 flag investigated via TWO methods: "
        "(a) an EARLIER bootstrap (500 resamples of the fixed 2022 "
        "validation set: mean delta +0.00768, 90% of resamples >= 0, 95% CI "
        "[-0.00372, +0.02230]) that was never previously recorded anywhere "
        "in this repo -- documentation gap, flagged explicitly, not a new "
        "finding; (b) a 12-fold TRAIN-only rotating holdout (scripts/"
        "diagnose_feature_noise.py, C=1.0, the untuned model's actual "
        "hyperparameter): sign test p=0.6875, 8/12 folds exactly zero "
        "effect, non-zero folds split 2-positive/2-negative -- weaker "
        "signal than even elo_diff's result. The rotating holdout (12 "
        "independent seasons) is treated as the stronger, decisive test "
        "here -- across-season stability, not single-season resampling "
        "noise -- superseding the earlier bootstrap's softer read."
    ),
}


# --------------------------------------------------------------------------- #
# CV folds -- expanding-window, per-season (SPEC Section 6 discipline applied
# one level down, inside CV, not just at the outer train/val/test split).
# --------------------------------------------------------------------------- #
def build_expanding_season_folds(train_df: pd.DataFrame,
                                 test_seasons=CV_TEST_SEASONS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold i: train on all TRAIN rows with season < S, test on season == S.

    ``train_df`` MUST already be positionally 0..N-1 (``reset_index(drop=True)``)
    and restricted to TRAIN (season <= ``TW.TRAIN_MAX_SEASON``) -- the folds
    are built as raw integer positions into it, which is exactly what
    ``GridSearchCV``'s custom ``cv`` argument expects.
    """
    seasons = train_df["season"].to_numpy()
    idx_all = np.arange(len(train_df))
    folds = []
    print(f"Expanding-window per-season CV folds (from TRAIN, season <= {TW.TRAIN_MAX_SEASON}):")
    for s in test_seasons:
        train_idx = idx_all[seasons < s]
        test_idx = idx_all[seasons == s]
        tr_seasons = train_df.loc[train_idx, "season"]
        print(f"  fold test_season={s}: train seasons {int(tr_seasons.min())}-{int(tr_seasons.max())} "
              f"(n={len(train_idx):,})  |  test season={s} (n={len(test_idx):,})")
        folds.append((train_idx, test_idx))

    forbidden = {TW.VAL_SEASON, *TW.FORBIDDEN_TEST_SEASONS}
    touched = set()
    for train_idx, test_idx in folds:
        touched |= set(train_df.loc[train_idx, "season"].unique().tolist())
        touched |= set(train_df.loc[test_idx, "season"].unique().tolist())
    bad = touched & forbidden
    assert not bad, f"FLAG: forbidden seasons {bad} present inside a CV fold -- SPEC 6/8 violation"
    print(f"  ASSERTION PASSED: none of {sorted(forbidden)} (val + test seasons) appear in any "
          f"fold. Seasons touched across all folds: {sorted(touched)}")
    return folds


# --------------------------------------------------------------------------- #
# Pipeline factories, parameterized by the hyperparameters being tuned.
# --------------------------------------------------------------------------- #
def make_lr_pipeline(C: float = 1.0, random_state: int = 0) -> Pipeline:
    """Same shape as ``train_winner.make_pipeline`` (impute median -> scale ->
    LogisticRegression), with ``C`` exposed for tuning."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=random_state)),
    ])


def make_rf_pipeline(max_depth=8, min_samples_leaf=5, random_state: int = 0) -> Pipeline:
    """Same shape as ``train_winner_ensemble.make_rf_pipeline``; ``n_estimators``
    kept FIXED at 300 (not searched) -- more trees is a variance-reduction/
    compute knob, not a bias-variance tradeoff worth a grid cell, per the
    Part 1 baseline's own reasoning. ``n_jobs=1`` on the estimator itself
    (GridSearchCV parallelizes across folds/param combos instead, at
    ``n_jobs=-1``, on the search) -- avoids oversubscribing threads."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=1,
        )),
    ])


def make_hgb_pipeline(max_leaf_nodes=31, learning_rate=0.1, random_state: int = 0) -> Pipeline:
    """Same shape as ``train_winner_ensemble.make_hgb_pipeline`` (no imputer --
    HGB natively handles NaNs, same reasoning as Part 1)."""
    return Pipeline([
        ("clf", HistGradientBoostingClassifier(
            max_leaf_nodes=max_leaf_nodes, learning_rate=learning_rate,
            random_state=random_state,
        )),
    ])


# --------------------------------------------------------------------------- #
# Post-failure diagnostic: if a linear (coef_-bearing) model fails check 4,
# report which SPECIFIC feature pair(s) show the collinearity signature
# (|r|>0.5, opposite-signed coefficients), instead of guessing at a fix.
# --------------------------------------------------------------------------- #
def check_remaining_collinearity(pipe: Pipeline, feature_cols: list[str],
                                 train_df: pd.DataFrame) -> pd.DataFrame:
    """A retained feature pair with TRAIN \|r\| > 0.5 AND opposite-signed
    fitted coefficients. Returns an empty frame if none found. Only
    meaningful for a pipeline whose ``clf`` step exposes ``coef_``
    (LogisticRegression)."""
    coef_tbl = TW.coefficient_table(pipe, feature_cols)
    imputer = pipe.named_steps["impute"]
    X_imputed = pd.DataFrame(imputer.transform(train_df[feature_cols]), columns=feature_cols)
    corr = X_imputed.corr()
    coef_idx = coef_tbl.set_index("feature")["coef"]
    flags = []
    for i, f1 in enumerate(feature_cols):
        for f2 in feature_cols[i + 1:]:
            r = corr.loc[f1, f2]
            if abs(r) > 0.5 and (coef_idx[f1] > 0) != (coef_idx[f2] > 0):
                flags.append({"feature_a": f1, "feature_b": f2, "r": float(r),
                              "coef_a": float(coef_idx[f1]), "coef_b": float(coef_idx[f2])})
    return pd.DataFrame(flags)


# --------------------------------------------------------------------------- #
# Grids -- modest per the CLAUDE.md GridSearchCV decision (2-3 hyperparameters,
# 3-4 values each for RF/HGB; LR tunes a single hyperparameter).
# --------------------------------------------------------------------------- #
# LR: C only. Log-spaced as suggested in the task brief -- covers strong
# regularization (0.01) through near-unregularized (100) in factor-of-10
# steps, a standard first-pass span for C.
LR_GRID = {"clf__C": [0.01, 0.1, 1, 10, 100]}

# RF: max_depth and min_samples_leaf -- the same two knobs the Part 1 untuned
# baseline hand-picked as "the guard a baseline RF on this data size
# genuinely needs" (n_estimators=300 fixed, not searched -- see
# make_rf_pipeline docstring). Values bracket the untuned baseline's choice
# (max_depth=8, min_samples_leaf=5) on both sides.
RF_GRID = {"clf__max_depth": [4, 8, 12], "clf__min_samples_leaf": [2, 5, 10]}

# HGB: max_leaf_nodes and learning_rate -- the two first-order knobs for a
# boosting model (tree complexity and step size / overfit speed). Values
# bracket sklearn's own defaults (max_leaf_nodes=31, learning_rate=0.1) on
# both sides. min_samples_leaf left at its library default (20) to keep the
# grid at 2 hyperparameters, not 3, per the "modest" GridSearchCV decision.
HGB_GRID = {"clf__max_leaf_nodes": [15, 31, 63], "clf__learning_rate": [0.03, 0.1, 0.3]}


# --------------------------------------------------------------------------- #
# Multi-metric scoring, refit on log loss.
# --------------------------------------------------------------------------- #
def make_scoring() -> dict:
    return {
        "accuracy": "accuracy",
        "neg_log_loss": "neg_log_loss",
        "roc_auc": "roc_auc",
        "brier": make_scorer(brier_score_loss, response_method="predict_proba", greater_is_better=False),
    }


def run_grid_search(name: str, base_pipeline, param_grid: dict,
                    X_train: pd.DataFrame, y_train: np.ndarray,
                    folds: list[tuple[np.ndarray, np.ndarray]]) -> GridSearchCV:
    n_combos = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"\n--- {name}: GridSearchCV ---")
    print(f"  grid: {param_grid}  ({n_combos} combinations x {len(folds)} folds = "
          f"{n_combos * len(folds)} fits)")
    search = GridSearchCV(
        estimator=base_pipeline, param_grid=param_grid, scoring=make_scoring(),
        refit="neg_log_loss", cv=folds, n_jobs=-1, return_train_score=False,
    )
    search.fit(X_train, y_train)
    print(f"  best params (refit='neg_log_loss'): {search.best_params_}")

    res = pd.DataFrame(search.cv_results_)
    show_cols = ["params", "mean_test_accuracy", "mean_test_neg_log_loss",
                 "mean_test_roc_auc", "mean_test_brier",
                 "rank_test_accuracy", "rank_test_neg_log_loss"]
    view = res[show_cols].copy()
    view["log_loss"] = -view["mean_test_neg_log_loss"]
    view["brier"] = -view["mean_test_brier"]
    view = view.rename(columns={"mean_test_accuracy": "accuracy", "mean_test_roc_auc": "roc_auc"})
    view = view[["params", "accuracy", "log_loss", "roc_auc", "brier",
                "rank_test_accuracy", "rank_test_neg_log_loss"]]

    top5_logloss = view.sort_values("rank_test_neg_log_loss").head(5).reset_index(drop=True)
    top5_acc = view.sort_values("rank_test_accuracy").head(5).reset_index(drop=True)

    print(f"\n  top-5 by CV log loss (this ranking drives refit -> best_estimator_):")
    print(top5_logloss.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  top-5 by CV accuracy (for comparison only -- NOT what refit uses):")
    print(top5_acc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    same_top = top5_logloss.iloc[0]["params"] == top5_acc.iloc[0]["params"]
    if same_top:
        print(f"\n  log-loss-optimal and accuracy-optimal picks AGREE: {top5_logloss.iloc[0]['params']}")
    else:
        print(f"\n  log-loss-optimal and accuracy-optimal picks DIFFER:")
        print(f"    log-loss pick : {top5_logloss.iloc[0]['params']}  "
              f"(accuracy={top5_logloss.iloc[0]['accuracy']:.4f}, log_loss={top5_logloss.iloc[0]['log_loss']:.4f})")
        print(f"    accuracy pick : {top5_acc.iloc[0]['params']}  "
              f"(accuracy={top5_acc.iloc[0]['accuracy']:.4f}, log_loss={top5_acc.iloc[0]['log_loss']:.4f})")
    return search


# --------------------------------------------------------------------------- #
# Ablation-check fit_and_eval wrappers bound to a model's TUNED hyperparams
# (needed for SPEC 5.5 #4 -- must retrain with the SAME chosen hyperparams
# on a reduced feature set, not the untuned defaults).
# --------------------------------------------------------------------------- #
def make_fit_and_eval(pipeline_fn):
    def _fit_and_eval(train_df, val_df, feature_cols=None, random_state=0):
        return TWE.fit_and_eval(pipeline_fn, train_df, val_df, feature_cols, random_state)
    return _fit_and_eval


def load_untuned_metrics() -> dict:
    """Pull the Part 1 untuned val_metrics straight from the saved joblib
    files -- no retraining, per the task instruction."""
    out = {}
    for key, fname in [("LR", "logreg_winner.joblib"), ("RF", "rf_winner.joblib"),
                       ("HGB", "hgb_winner.joblib")]:
        path = MODELS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- Phase 5B Part 1 (src/train_winner_ensemble.py / "
                "src/train_winner.py) must be run first"
            )
        out[key] = joblib.load(path)["val_metrics"]
    return out


def comparison_table(home_baseline: dict, untuned: dict, tuned: dict) -> pd.DataFrame:
    rows = {
        "home_baseline": {"accuracy": home_baseline["baseline_accuracy"],
                          "log_loss": np.nan, "brier_score": np.nan, "roc_auc": np.nan},
    }
    for key, label in [("LR", "LogisticRegression"), ("RF", "RandomForest"), ("HGB", "HistGradientBoosting")]:
        rows[f"{label}-untuned"] = {k: untuned[key][k] for k in ("accuracy", "log_loss", "brier_score", "roc_auc")}
        rows[f"{label}-tuned"] = {k: tuned[key][k] for k in ("accuracy", "log_loss", "brier_score", "roc_auc")}
    order = ["home_baseline", "LogisticRegression-untuned", "LogisticRegression-tuned",
            "RandomForest-untuned", "RandomForest-tuned",
            "HistGradientBoosting-untuned", "HistGradientBoosting-tuned"]
    return pd.DataFrame(rows).T.loc[order, ["accuracy", "log_loss", "brier_score", "roc_auc"]]


def main() -> int:
    print("=== Phase 5B Part 2: GridSearchCV tuning -- LogisticRegression (C), "
          "RandomForest, HistGradientBoosting -- single full feature set ===\n")

    frame = TW.load_train_val_frame()
    train_df = frame[frame["season"] <= TW.TRAIN_MAX_SEASON].reset_index(drop=True)
    val_df = frame[frame["season"] == TW.VAL_SEASON].reset_index(drop=True)
    assert set(train_df["season"]).isdisjoint(TW.FORBIDDEN_TEST_SEASONS)
    assert set(val_df["season"]).isdisjoint(TW.FORBIDDEN_TEST_SEASONS)
    print(f"train : seasons {train_df['season'].min()}-{train_df['season'].max()}  n={len(train_df):,}")
    print(f"val   : season  {sorted(val_df['season'].unique().tolist())}  n={len(val_df):,}")
    print(f"features: {len(TW.FEATURE_COLS)} columns (features.feature_columns()) -- "
          f"ALL THREE model types use this single set, no special-casing\n")

    folds = build_expanding_season_folds(train_df)

    X_train = train_df[TW.FEATURE_COLS]
    y_train = train_df["home_win"].to_numpy()
    X_val = val_df[TW.FEATURE_COLS]
    y_val = val_df["home_win"].to_numpy()

    schedules = evaluate.load_schedules()
    home_baseline = evaluate.home_baseline_accuracy(schedules, [TW.VAL_SEASON])
    untuned = load_untuned_metrics()

    # ------------------------------------------------------------------ #
    # 0. LR-untuned: refit fresh (same architecture/hyperparameters as the
    #    saved logreg_winner.joblib -- impute->scale->LogisticRegression,
    #    C=1.0 default, random_state=0) so checks 3/4 run against a real
    #    pipeline object from THIS run, not just the loaded metrics dict.
    #    Deterministic given the same data/hyperparameters, so this should
    #    (and does, see report) reproduce the existing saved metrics exactly.
    # ------------------------------------------------------------------ #
    print(f"{'=' * 78}\n0. LogisticRegression-untuned (C=1.0, refit fresh for checks 3/4)\n{'=' * 78}")
    lr_untuned_pipe, lr_untuned_metrics = TWE.fit_and_eval(TW.make_pipeline, train_df, val_df, TW.FEATURE_COLS)
    print(f"  accuracy={lr_untuned_metrics['accuracy']:.4f}  log_loss={lr_untuned_metrics['log_loss']:.4f}  "
          f"brier={lr_untuned_metrics['brier_score']:.4f}  roc_auc={lr_untuned_metrics['roc_auc']:.4f}")
    print(f"  matches existing saved logreg_winner.joblib val_metrics: "
          f"{lr_untuned_metrics == untuned['LR']}")

    # ------------------------------------------------------------------ #
    # 1. GridSearchCV for each model, fit ONLY on TRAIN, full feature set.
    # ------------------------------------------------------------------ #
    lr_search = run_grid_search("LogisticRegression", make_lr_pipeline(), LR_GRID,
                                X_train, y_train, folds)
    rf_search = run_grid_search("RandomForest", make_rf_pipeline(), RF_GRID,
                                X_train, y_train, folds)
    hgb_search = run_grid_search("HistGradientBoosting", make_hgb_pipeline(), HGB_GRID,
                                 X_train, y_train, folds)

    searches = {"LR": lr_search, "RF": rf_search, "HGB": hgb_search}
    labels = {"LR": "LogisticRegression", "RF": "RandomForest", "HGB": "HistGradientBoosting"}

    # ------------------------------------------------------------------ #
    # 2. Evaluate each refit (log-loss-selected) best_estimator_ ONCE on
    #    the untouched 2022 validation set.
    # ------------------------------------------------------------------ #
    tuned_metrics = {}
    print(f"\n{'=' * 78}\nValidation (season {TW.VAL_SEASON}) -- tuned best_estimator_ (refit='neg_log_loss')\n{'=' * 78}")
    for key, search in searches.items():
        pipe = search.best_estimator_
        proba = pipe.predict_proba(X_val)[:, 1]
        pred = pipe.predict(X_val)
        metrics = TW.compute_metrics(y_val, pred, proba)
        tuned_metrics[key] = metrics
        print(f"  {labels[key]:<22} best_params={search.best_params_}")
        print(f"    accuracy={metrics['accuracy']:.4f}  log_loss={metrics['log_loss']:.4f}  "
              f"brier={metrics['brier_score']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # 3. SPEC 5.6/CLAUDE.md ~70% stop-and-flag rule, per tuned model.
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 78}\nSPEC 5.6/CLAUDE.md ~70% check\n{'=' * 78}")
    cleared = {}
    for key in searches:
        acc = tuned_metrics[key]["accuracy"]
        if acc > TW.SUSPICIOUS_ACCURACY:
            cleared[key] = False
            print(f"\n!! VALIDATION ACCURACY {acc:.4f} FOR {labels[key]}-tuned EXCEEDS "
                  f"{TW.SUSPICIOUS_ACCURACY} -- SPEC 5.6/5.7 + CLAUDE.md: STOP for this model. "
                  "Not proceeding to leakage checks or save for it. Flagging for manual review.")
        else:
            cleared[key] = True
            print(f"  {labels[key]:<22} accuracy={acc:.4f} <= {TW.SUSPICIOUS_ACCURACY} -- clear.")

    # ------------------------------------------------------------------ #
    # 4. Extended SPEC 5.5 checks 3/4 for each tuned model that cleared
    #    step 3 (reusing the generalized functions from Part 1).
    # ------------------------------------------------------------------ #
    importance_fns = {
        "LR": None,  # default: LC._lr_importance (coef_)
        "RF": TWE.rf_importance,
        "HGB": TWE.make_hgb_importance_fn(X_val, y_val),
    }
    tuned_pipeline_fns = {
        "LR": functools.partial(make_lr_pipeline, **{k.split("__")[1]: v
                                for k, v in lr_search.best_params_.items()}),
        "RF": functools.partial(make_rf_pipeline, **{k.split("__")[1]: v
                                for k, v in rf_search.best_params_.items()}),
        "HGB": functools.partial(make_hgb_pipeline, **{k.split("__")[1]: v
                                 for k, v in hgb_search.best_params_.items()}),
    }

    leakage_results = {}
    for key in searches:
        if not cleared[key]:
            print(f"\n(skipping SPEC 5.5 checks 3/4 for {labels[key]}-tuned -- did not clear the 70% gate)")
            continue
        print(f"\n{'=' * 78}\nExtended SPEC 5.5 leakage checks -- {labels[key]}-tuned\n{'=' * 78}")
        pipe = searches[key].best_estimator_
        res3 = LC.feature_importance_inspection(
            pipe, TW.FEATURE_COLS, importance_fn=importance_fns[key],
            flag_share=IMPORTANCE_FLAG_SHARE, model_label=f"{labels[key]}-tuned",
        )
        fit_fn = make_fit_and_eval(tuned_pipeline_fns[key])
        res4 = LC.ablation_check(
            train_df, val_df, TW.FEATURE_COLS, res3["top_feature"], tuned_metrics[key],
            home_baseline["baseline_accuracy"], fit_and_eval_fn=fit_fn,
            model_label=f"{labels[key]}-tuned",
        )
        leakage_results[key] = {
            "importance_flagged": res3["flagged"],
            "ablation_suspicious": res4["suspicious"],
            "top_feature": res3["top_feature"],
        }
        if res4["suspicious"] and hasattr(pipe.named_steps.get("clf"), "coef_"):
            flags = check_remaining_collinearity(pipe, TW.FEATURE_COLS, train_df)
            if len(flags):
                print(f"\n  !! {labels[key]}-tuned failed check 4 -- |r|>0.5 opposite-signed "
                      f"pair(s) in the feature set:")
                print(flags.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------------------ #
    # 4b. Same 70% gate + checks 3/4 for LR-untuned (not part of the
    #     GridSearchCV `searches` dict, so handled as its own unit).
    # ------------------------------------------------------------------ #
    lr_untuned_acc = lr_untuned_metrics["accuracy"]
    lr_untuned_cleared = lr_untuned_acc <= TW.SUSPICIOUS_ACCURACY
    print(f"\n{'=' * 78}\nSPEC 5.6/CLAUDE.md ~70% check -- LogisticRegression-untuned\n{'=' * 78}")
    print(f"  LogisticRegression-untuned  accuracy={lr_untuned_acc:.4f} <= {TW.SUSPICIOUS_ACCURACY} -- "
          f"{'clear.' if lr_untuned_cleared else 'EXCEEDS -- STOP.'}")

    if lr_untuned_cleared:
        print(f"\n{'=' * 78}\nExtended SPEC 5.5 leakage checks -- LogisticRegression-untuned\n{'=' * 78}")
        res3_u = LC.feature_importance_inspection(
            lr_untuned_pipe, TW.FEATURE_COLS, model_label="LogisticRegression-untuned",
        )
        res4_u = LC.ablation_check(
            train_df, val_df, TW.FEATURE_COLS, res3_u["top_feature"], lr_untuned_metrics,
            home_baseline["baseline_accuracy"], model_label="LogisticRegression-untuned",
        )
        leakage_results["LR-untuned"] = {
            "importance_flagged": res3_u["flagged"],
            "ablation_suspicious": res4_u["suspicious"],
            "top_feature": res3_u["top_feature"],
        }
        if res4_u["suspicious"]:
            flags_u = check_remaining_collinearity(lr_untuned_pipe, TW.FEATURE_COLS, train_df)
            if len(flags_u):
                print(f"\n  !! LogisticRegression-untuned failed check 4 -- |r|>0.5 opposite-signed "
                      f"pair(s) in the feature set:")
                print(flags_u.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------------------ #
    # 5. Per-model save gating: each model is saved independently based on
    #    its OWN 70% gate + SPEC 5.5 checks 3/4 result, not the group's --
    #    one model failing no longer blocks saving the others.
    #
    #    LR_CHECK4_OVERRIDES (module-level, see docstring) applies a
    #    one-time, documented exception to check 4 ONLY for
    #    LogisticRegression (both variants) -- gating LOGIC is unchanged:
    #    check 3 and the 70% gate still block normally for every model,
    #    including LR.
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 78}\nPer-model save decision\n{'=' * 78}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    grids = {"LR": LR_GRID, "RF": RF_GRID, "HGB": HGB_GRID}
    out_names = {"LR": "logreg_winner_tuned.joblib", "RF": "rf_winner_tuned.joblib",
                "HGB": "hgb_winner_tuned.joblib"}

    # gate() moved to leakage_checks.py (LC.gate) so train_winner_ensemble.py's
    # RF/HGB-untuned save path can reuse the identical logic instead of a copy.

    saved = {}
    for key, search in searches.items():
        run_key = f"{key}-tuned"
        res = leakage_results.get(key, {})
        should_save, overridden, reason = LC.gate(
            cleared[key], res.get("importance_flagged", True), res.get("ablation_suspicious", True),
            run_key=run_key, overrides=LR_CHECK4_OVERRIDES,
        )
        if not should_save:
            print(f"  {labels[key]:<22}-tuned    NOT SAVED -- {reason}")
            saved[run_key] = False
            continue
        payload = {"pipeline": search.best_estimator_, "feature_cols": TW.FEATURE_COLS,
                  "train_max_season": TW.TRAIN_MAX_SEASON, "val_season": TW.VAL_SEASON,
                  "val_metrics": tuned_metrics[key], "best_params": search.best_params_,
                  "grid": grids[key], "refit_metric": "neg_log_loss",
                  "cv_fold_test_seasons": list(CV_TEST_SEASONS),
                  "leakage_checks": {
                      "check3_top_feature": res.get("top_feature"),
                      "check3_flagged": res.get("importance_flagged"),
                      "check4_suspicious": res.get("ablation_suspicious"),
                      "override_applied": overridden,
                      "override_justification": reason if overridden else None,
                  }}
        out_path = MODELS_DIR / out_names[key]
        joblib.dump(payload, out_path)
        tag = " (OVERRIDE APPLIED)" if overridden else ""
        print(f"  {labels[key]:<22}-tuned    SAVED -> {out_path}  "
              f"({out_path.stat().st_size:,} bytes){tag}")
        saved[run_key] = True

    # LR-untuned: same LC.gate() logic, separate save path (logreg_winner.joblib,
    # untuned metadata pattern -- params/random_state, not grid/best_params).
    res_u = leakage_results.get("LR-untuned", {})
    should_save_u, overridden_u, reason_u = LC.gate(
        lr_untuned_cleared, res_u.get("importance_flagged", True), res_u.get("ablation_suspicious", True),
        run_key="LR-untuned", overrides=LR_CHECK4_OVERRIDES,
    )
    if not should_save_u:
        print(f"  LogisticRegression-untuned  NOT SAVED -- {reason_u}")
        saved["LR-untuned"] = False
    else:
        payload_u = {"pipeline": lr_untuned_pipe, "feature_cols": TW.FEATURE_COLS,
                    "train_max_season": TW.TRAIN_MAX_SEASON, "val_season": TW.VAL_SEASON,
                    "val_metrics": lr_untuned_metrics, "params": {"C": 1.0}, "random_state": 0,
                    "leakage_checks": {
                        "check3_top_feature": res_u.get("top_feature"),
                        "check3_flagged": res_u.get("importance_flagged"),
                        "check4_suspicious": res_u.get("ablation_suspicious"),
                        "override_applied": overridden_u,
                        "override_justification": reason_u if overridden_u else None,
                    }}
        out_path_u = MODELS_DIR / "logreg_winner.joblib"
        joblib.dump(payload_u, out_path_u)
        tag_u = " (OVERRIDE APPLIED)" if overridden_u else ""
        print(f"  LogisticRegression-untuned  SAVED -> {out_path_u}  "
              f"({out_path_u.stat().st_size:,} bytes){tag_u}")
        saved["LR-untuned"] = True

    # ------------------------------------------------------------------ #
    # 6. Final comparison table.
    # ------------------------------------------------------------------ #
    untuned_for_table = dict(untuned)
    untuned_for_table["LR"] = lr_untuned_metrics
    tbl = comparison_table(home_baseline, untuned_for_table, tuned_metrics)
    print(f"\n{'=' * 78}\n=== Final comparison -- validation season {TW.VAL_SEASON} "
          f"(n={len(val_df)}) -- SPEC Section 8 ===\n{'=' * 78}")
    print(tbl.to_string(float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "n/a"))
    print(f"\nsaved: {saved}")

    return 0 if any(saved.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
