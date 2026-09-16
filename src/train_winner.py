"""Phase 4 MVP model -- SPEC Section 7.1, Section 13 Phase 4.

Train/validate the baseline ``LogisticRegression`` win/loss classifier on the
season-based walk-forward split from SPEC Section 6:

    train      = seasons <= 2021
    validation = season 2022   (hyperparameter/model comparison happens here)
    test       = seasons 2023-2024 -- NEVER loaded or touched by this module.

SPEC Section 8: accuracy, log loss, Brier score and ROC-AUC are computed on
the validation split only in this phase. The held-out test set is reserved
for a later, explicitly-approved phase (SPEC Section 6/8).

Usage::

    python -m src.train_winner
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import evaluate, features as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# SPEC Section 6 walk-forward split.
TRAIN_MAX_SEASON = 2021
VAL_SEASON = 2022
# SPEC Section 6/8: untouched until a later, explicitly-approved phase.
FORBIDDEN_TEST_SEASONS = (2023, 2024)

# SPEC Section 5.6: a validation accuracy above this is a stop-and-audit
# signal, never a result to report.
SUSPICIOUS_ACCURACY = 0.70

FEATURE_COLS = F.feature_columns()

# SPEC Section 7.1 sign sanity check: (feature, expected_sign, reason).
# Target is home_win, so a feature that favors the AWAY side should carry a
# NEGATIVE coefficient even though it is itself "positive" for the away team.
SIGN_CHECKS = [
    ("home_qb_epa_per_db_shrunk", "+",
     "better home QB play -> higher home win prob"),
    ("away_qb_epa_per_db_shrunk", "-",
     "better away QB play -> lower home win prob"),
    ("home_def_epa_per_play_shrunk", "-",
     "worse home defense (higher EPA allowed) -> lower home win prob"),
    ("away_def_epa_per_play_shrunk", "+",
     "worse away defense (higher EPA allowed) -> higher home win prob"),
    ("mu_qb_epa_vs_def", "+",
     "QB EPA differential: home QB epa - away def epa allowed -> higher home win prob"),
    ("mu_def_vs_qb_epa", "-",
     "QB EPA differential: away QB epa - home def epa allowed -> lower home win prob"),
    ("mu_money_down_edge", "+",
     "QB EPA differential (money downs): home QB epa - away def epa allowed on 3rd/4th -> higher home win prob"),
]


def load_train_val_frame() -> pd.DataFrame:
    """Load ``game_features.parquet``, restricted to train+validation seasons only.

    Structural guard for the SPEC Section 6/8 rule: seasons 2023-2024 (test)
    MUST NOT be loaded, touched, or evaluated on in this phase. Restricting the
    frame immediately after reading it -- rather than filtering at each call
    site -- means a bug anywhere else in this module cannot accidentally reach
    the test rows.
    """
    frame = pd.read_parquet(PROC_DIR / "game_features.parquet")
    present_test = sorted(set(frame["season"].unique()) & set(FORBIDDEN_TEST_SEASONS))
    frame = frame[frame["season"] <= VAL_SEASON].reset_index(drop=True)
    if present_test:
        print(f"  (test seasons {present_test} exist in game_features.parquet on disk, "
              f"as expected -- dropped immediately, never loaded into the train/val frame)")
    still_present = sorted(set(frame["season"].unique()) & set(FORBIDDEN_TEST_SEASONS))
    assert not still_present, (
        f"FLAG: test seasons {still_present} leaked into the train/val frame -- "
        "SPEC Section 6/8 violation, stop and investigate"
    )
    return frame


def make_pipeline(random_state: int = 0) -> Pipeline:
    """Median-impute (handles the pre-2006 cpoe/air_yards NaNs) -> scale ->
    ``LogisticRegression``.

    Imputer/scaler statistics are fit on whatever data ``.fit()`` is called
    with. ``fit_and_eval`` below only ever calls ``.fit()`` on the train
    split, so the imputer's median is TRAIN-ONLY, consistent with the
    walk-forward discipline used throughout the feature pipeline.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """SPEC Section 8 metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "n": int(len(y_true)),
    }


def fit_and_eval(train_df: pd.DataFrame, val_df: pd.DataFrame,
                 feature_cols: list[str] | None = None,
                 random_state: int = 0) -> tuple[Pipeline, dict]:
    """Fit on ``train_df`` only, evaluate on ``val_df`` only. Returns ``(pipe, metrics)``."""
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    X_train, y_train = train_df[cols], train_df["home_win"].to_numpy()
    X_val, y_val = val_df[cols], val_df["home_win"].to_numpy()

    pipe = make_pipeline(random_state=random_state).fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    pred = pipe.predict(X_val)
    metrics = compute_metrics(y_val, pred, proba)
    return pipe, metrics


def coefficient_table(pipe: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """All fitted coefficients, sorted by |coef| descending.

    Coefficients are on the ``StandardScaler``-scaled features; scaling by a
    positive factor never flips a coefficient's sign, so sign comparisons
    against SPEC 7.1 are still valid on this scale.
    """
    coefs = pipe.named_steps["clf"].coef_[0]
    abs_total = np.abs(coefs).sum()
    tbl = pd.DataFrame({
        "feature": feature_cols,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    })
    tbl["share_of_abs_total"] = tbl["abs_coef"] / abs_total
    return tbl.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def sign_sanity_check(coef_tbl: pd.DataFrame) -> pd.DataFrame:
    """SPEC Section 7.1: QB EPA (and QB-EPA-vs-defensive-EPA-allowed
    differentials) SHOULD be positive for the home side / negative for the
    away side on home_win; defensive EPA allowed SHOULD be the mirror image."""
    idx = coef_tbl.set_index("feature")["coef"]
    rows = []
    for feat, expected, reason in SIGN_CHECKS:
        if feat not in idx.index:
            continue
        actual = float(idx[feat])
        actual_sign = "+" if actual >= 0 else "-"
        rows.append({
            "feature": feat, "coef": actual, "expected_sign": expected,
            "actual_sign": actual_sign, "matches_expected": actual_sign == expected,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def main() -> int:
    print(f"Loading data/processed/game_features.parquet, restricted to "
          f"train (<= {TRAIN_MAX_SEASON}) + validation ({VAL_SEASON}) only "
          f"-- test seasons {FORBIDDEN_TEST_SEASONS} are never loaded past this point.\n")
    frame = load_train_val_frame()

    train_df = frame[frame["season"] <= TRAIN_MAX_SEASON].reset_index(drop=True)
    val_df = frame[frame["season"] == VAL_SEASON].reset_index(drop=True)
    assert set(train_df["season"]).isdisjoint(FORBIDDEN_TEST_SEASONS)
    assert set(val_df["season"]).isdisjoint(FORBIDDEN_TEST_SEASONS)

    print(f"train : seasons {train_df['season'].min()}-{train_df['season'].max()}  n={len(train_df):,}")
    print(f"val   : season  {sorted(val_df['season'].unique().tolist())}  n={len(val_df):,}")

    n_missing_train = int(train_df[FEATURE_COLS].isna().sum().sum())
    n_missing_val = int(val_df[FEATURE_COLS].isna().sum().sum())
    print(f"\nNaN cells in train features (median imputer fit on TRAIN only): {n_missing_train:,}")
    print(f"NaN cells in val features (imputed with the TRAIN-fit median):    {n_missing_val:,}")

    pipe, metrics = fit_and_eval(train_df, val_df, FEATURE_COLS)

    schedules = evaluate.load_schedules()
    baseline = evaluate.home_baseline_accuracy(schedules, [VAL_SEASON])

    print(f"\n=== Validation (season {VAL_SEASON}) metrics -- SPEC Section 8 ===")
    print(f"  accuracy    : {metrics['accuracy']:.4f}   (n={metrics['n']})")
    print(f"  log_loss    : {metrics['log_loss']:.4f}")
    print(f"  brier_score : {metrics['brier_score']:.4f}")
    print(f"  roc_auc     : {metrics['roc_auc']:.4f}")
    print(f"\n  home baseline accuracy (season {VAL_SEASON}, per-split via "
          f"evaluate.home_baseline_accuracy): {baseline['baseline_accuracy']:.4f}"
          f"  ({baseline['home_wins']}/{baseline['n_games']} home wins)")
    delta = metrics["accuracy"] - baseline["baseline_accuracy"]
    print(f"  model vs. baseline delta: {delta:+.4f}")

    if metrics["accuracy"] > SUSPICIOUS_ACCURACY:
        print(f"\n!! VALIDATION ACCURACY {metrics['accuracy']:.4f} EXCEEDS {SUSPICIOUS_ACCURACY} -- "
              "SPEC 5.6/5.7 + CLAUDE.md: STOP. This is NOT being reported as a good result. "
              "Not proceeding to coefficient reporting, model save, or Part B checks. "
              "Flagging for manual review.")
        return 1

    coef_tbl = coefficient_table(pipe, FEATURE_COLS)
    print(f"\n=== Coefficients (StandardScaler-scaled, all {len(FEATURE_COLS)} features), "
          f"sorted by |coef| ===")
    print(coef_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    sign_tbl = sign_sanity_check(coef_tbl)
    print(f"\n=== SPEC 7.1 sign sanity check ===")
    print(sign_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    mismatches = sign_tbl[~sign_tbl["matches_expected"]]
    if len(mismatches):
        print(f"\n!! {len(mismatches)} feature(s) have an unexpected sign -- flagging, not silently accepting:")
        print(mismatches.to_string(index=False))
    else:
        print("\nAll checked features have the SPEC 7.1 expected sign.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "logreg_winner.joblib"
    joblib.dump(
        {"pipeline": pipe, "feature_cols": FEATURE_COLS,
         "train_max_season": TRAIN_MAX_SEASON, "val_season": VAL_SEASON,
         "val_metrics": metrics},
        model_path,
    )
    print(f"\nSaved fitted pipeline -> {model_path}  ({model_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
