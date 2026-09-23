"""Out-of-fold walk-forward sigmoid (Platt) calibration -- SPEC 7.3, Phase 8.

Method (project owner's decision, Phase 8):

For a target season T:
  1. For each season S in T-5 .. T-1: fit the model (frozen, saved
     hyperparameters) on all seasons < S and predict S. These are genuine
     out-of-sample predictions.
  2. Pool those OOF predictions (~1,350 games) and fit ONE sigmoid
     calibrator: an UNPENALIZED logistic regression on logit(raw prob).
  3. Fit the final model on all seasons <= T-1 and apply the calibrator to
     its season-T predictions.

Why not ``CalibratedClassifierCV``: in the installed scikit-learn (1.9.0) its
default ``cv=None`` resolves to ``StratifiedKFold(n_splits=5, shuffle=False)``
over ROWS, which mixes seasons across folds -- later seasons would inform the
calibration of earlier ones (SPEC Section 6 prohibition). The walk-forward OOF
loop above is season-ordered by construction.

Sigmoid, not isotonic: isotonic regression overfits at ~1,350 calibration rows.

Known, accepted mismatch: the final model trains on more seasons than any OOF
model, so it may be slightly sharper than the predictions the calibrator was
fit on.

Calibrator: ``LogisticRegression(C=np.inf)``. In scikit-learn 1.9.0
``penalty=None`` is deprecated (FutureWarning, removal in 1.10); ``C=np.inf``
is the supported spelling of "no regularization" and fits the identical
coefficients without a warning (verified on the installed version).

No hyperparameter or feature parameter is chosen here: models are
``sklearn.base.clone()`` of the saved pipelines, which carry their frozen
hyperparameters and ``random_state=0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src import train_winner as TW

# model label -> saved joblib (RF-untuned omitted: identical hyperparameters to
# RF-tuned, per Phase 6).
SAVED_MODELS = {
    "LR-untuned": "logreg_winner.joblib",
    "LR-tuned": "logreg_winner_tuned.joblib",
    "RandomForest": "rf_winner_tuned.joblib",
    "HGB-tuned": "hgb_winner_tuned.joblib",
}

N_OOF_SEASONS = 5
PROB_CLIP = 1e-6  # keeps logit finite if a tree model ever emits exactly 0 or 1
N_BINS = 10


def load_frozen_models() -> tuple[dict, list[str]]:
    """Unfitted clones of every saved pipeline, plus the shared feature list."""
    pipes, feature_cols = {}, None
    for name, fname in SAVED_MODELS.items():
        d = joblib.load(TW.MODELS_DIR / fname)
        cols = list(d["feature_cols"])
        if feature_cols is None:
            feature_cols = cols
        assert cols == feature_cols, f"{fname}: feature list differs from the other saved models"
        pipe = clone(d["pipeline"])
        assert pipe.steps[-1][1].get_params()["random_state"] == 0, f"{fname}: random_state != 0"
        pipes[name] = pipe
    assert feature_cols == list(TW.FEATURE_COLS), "saved feature list != current pipeline's"
    return pipes, feature_cols


def load_frame(max_season: int) -> pd.DataFrame:
    """``game_features.parquet`` restricted to seasons <= ``max_season``.

    The season filter is applied immediately after reading, so no row from a
    later season (e.g. the 2025 holdout, labels included) is ever handed to
    the caller.
    """
    frame = pd.read_parquet(TW.PROC_DIR / "game_features.parquet")
    return frame[frame["season"] <= max_season].reset_index(drop=True)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    return np.log(p / (1 - p))


@dataclass
class SigmoidCalibrator:
    """Platt scaling: calibrated = sigmoid(slope * logit(p) + intercept).

    slope < 1 -> raw model was overconfident (calibration pulls toward 0.5);
    slope > 1 -> raw model was underconfident. intercept != 0 shifts the
    probabilities toward home (> 0) or away (< 0).
    """
    slope: float
    intercept: float
    n_fit: int

    @classmethod
    def fit(cls, proba: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        lr = LogisticRegression(C=np.inf, max_iter=1000)
        lr.fit(_logit(proba).reshape(-1, 1), y)
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]), int(len(y)))

    def transform(self, proba: np.ndarray) -> np.ndarray:
        z = self.slope * _logit(proba) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def fit_predict(unfitted, train: pd.DataFrame, pred: pd.DataFrame,
                feature_cols: list[str]) -> np.ndarray:
    pipe = clone(unfitted).fit(train[feature_cols], train["home_win"].to_numpy())
    return pipe.predict_proba(pred[feature_cols])[:, 1]


def oof_predictions(unfitted, frame: pd.DataFrame, target_season: int,
                    feature_cols: list[str]) -> pd.DataFrame:
    """Walk-forward OOF predictions for seasons T-5 .. T-1 (each fit on < S)."""
    parts = []
    for s in range(target_season - N_OOF_SEASONS, target_season):
        train = frame[frame["season"] < s]
        test = frame[frame["season"] == s]
        assert len(train) and len(test), f"missing data for OOF season {s}"
        parts.append(pd.DataFrame({
            "game_id": test["game_id"].to_numpy(), "season": s,
            "home_win": test["home_win"].to_numpy(),
            "proba_raw": fit_predict(unfitted, train, test, feature_cols),
        }))
    return pd.concat(parts, ignore_index=True)


def calibrate_for_season(unfitted, frame: pd.DataFrame, target_season: int,
                         feature_cols: list[str]) -> tuple[pd.DataFrame, SigmoidCalibrator, pd.DataFrame]:
    """Full method for one model and one target season.

    ``frame`` must contain every season < ``target_season`` needed for
    training, plus the target season's rows. Returns
    ``(target_predictions, calibrator, oof_predictions)``; target predictions
    have ``proba_raw`` and ``proba_cal``.
    """
    assert frame["season"].max() == target_season, "frame must end at the target season"
    oof = oof_predictions(unfitted, frame, target_season, feature_cols)
    cal = SigmoidCalibrator.fit(oof["proba_raw"].to_numpy(), oof["home_win"].to_numpy())
    train = frame[frame["season"] < target_season]
    target = frame[frame["season"] == target_season]
    raw = fit_predict(unfitted, train, target, feature_cols)
    out = pd.DataFrame({"game_id": target["game_id"].to_numpy(), "season": target_season,
                        "week": target["week"].to_numpy(),
                        "proba_raw": raw, "proba_cal": cal.transform(raw)})
    return out, cal, oof


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    """SPEC Section 8 metrics. Picks are proba > 0.5 (home win)."""
    return {"accuracy": accuracy_score(y, (proba > 0.5).astype(int)),
            "log_loss": log_loss(y, proba), "brier_score": brier_score_loss(y, proba),
            "roc_auc": roc_auc_score(y, proba), "n": int(len(y))}


def reliability_table(y: np.ndarray, proba: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-width probability bins: count, mean predicted, observed home-win rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_pred": float(proba[m].mean()) if m.any() else np.nan,
                     "observed": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)
