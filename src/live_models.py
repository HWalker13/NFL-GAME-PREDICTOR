"""Phase 10 frozen live models: champion (LR-tuned) and calibrated challenger.

* **Champion** -- ``clone()`` of ``models/logreg_winner_tuned.joblib``
  (C=0.01, the 67 frozen features, random_state=0), trained on every played
  2002-2025 REG game, features built with the frozen ``ELO_HOME_ADV=44.73``.
* **Challenger (calibrated)** -- the frozen champion, unchanged, followed by an
  out-of-fold walk-forward sigmoid calibrator for target season T=2026, built
  with the Phase 8 method in ``src/calibration.py``: for each season S in
  T-5..T-1 = 2021..2025, a model with the champion's hyperparameters is trained
  on seasons < S and predicts S; one unpenalised logistic regression on
  logit(p) is fit to those pooled predictions. The 2021-2025 window follows
  from the standard T-5..T-1 rule (not hand-picked); it happens to exclude
  2020. The calibrator targets the LR intercept, where the documented home
  bias lives (Phase 8: negative calibrator intercepts). Never evaluated on a
  historical season; its only evaluation is live, 2026.
* **Discarded (never used live):** an earlier challenger that re-derived
  ``ELO_HOME_ADV`` from 2021-2025 (28.525). It was saved, compared label-free
  on the 240 unplayed 2026 games (0/240 picks differed, max probability
  difference 0.0037, because ``ELO_HOME_ADV`` enters only the Elo update, not
  the features), and deleted before any live prediction.
  ``derive_hfa`` is kept as the record of how 28.525 was computed.

Checks (reproduction, not evaluation -- no metric is computed):
  ``reproduce_2025``            train <=2024 reproduces Phase 8's 2025 LR-tuned
                                raw probabilities exactly (B2).
  ``reproduce_2025_calibrator`` the calibrator code for T=2025 reproduces Phase
                                8's recorded LR-tuned calibrator exactly.

Saved files are refused if they already exist and are made read-only.

Usage::

    python -m src.live_models reproduce            # both reproduction checks
    python -m src.live_models save-champion        # B2, then train + save the champion (done; refuses now)
    python -m src.live_models save-challenger      # checks, then fit + save the calibrated challenger
    python -m src.live_models show                 # print saved metadata + sha256
"""

from __future__ import annotations

import argparse
import hashlib
import math
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src import calibration as CAL, features as F, live_features as LF, train_winner as TW

SOURCE_MODEL = TW.MODELS_DIR / "logreg_winner_tuned.joblib"
CHAMPION_PATH = TW.MODELS_DIR / "live_2026_champion.joblib"
CHALLENGER_PATH = TW.MODELS_DIR / "live_2026_challenger_calibrated.joblib"
HOLDOUT_2025 = TW.PROC_DIR / "holdout_2025_predictions.parquet"
HOLDOUT_2025_RECORD = TW.PROC_DIR / "holdout_2025_run_record.json"
LIVE_TARGET_SEASON = 2026

TRAIN_MIN, TRAIN_MAX = 2002, 2025
FROZEN_ELO_HOME_ADV = 44.72586959078301
CHALLENGER_HFA_SEASONS = (2021, 2025)   # discarded Elo-HFA challenger only
EXPECTED_CLF_PARAMS = {"C": 0.01, "max_iter": 1000, "random_state": 0}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_data_sha256(train: pd.DataFrame, feature_cols: list[str]) -> str:
    """sha256 of exactly what the model is fit on: keys, features, label, in
    row order (pandas' stable per-row hash, then sha256 over those bytes)."""
    cols = ["game_id", "season"] + list(feature_cols) + ["home_win"]
    h = pd.util.hash_pandas_object(train[cols], index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


def load_source():
    """Unfitted clone of the validated LR-tuned pipeline + its feature list."""
    d = joblib.load(SOURCE_MODEL)
    cols = list(d["feature_cols"])
    assert cols == F.feature_columns(), "saved LR-tuned feature list != current feature_columns()"
    assert len(cols) == 67
    pipe = clone(d["pipeline"])
    got = {k: pipe.named_steps["clf"].get_params()[k] for k in EXPECTED_CLF_PARAMS}
    assert got == EXPECTED_CLF_PARAMS, f"LR-tuned hyperparameters changed: {got}"
    return pipe, cols


def derive_hfa(canonical: pd.DataFrame, seasons: tuple[int, int] = CHALLENGER_HFA_SEASONS) -> dict:
    """The 44.73 method: home win rate p over game_features rows (completed REG
    games, ties dropped, neutral-site games included) -> 400*log10(p/(1-p))."""
    rows = canonical[canonical["season"].between(*seasons)]
    wins, n = int(rows["home_win"].sum()), int(len(rows))
    p = wins / n
    return {"seasons": list(seasons), "home_wins": wins, "n_games": n, "p": p,
            "elo_home_adv": 400.0 * math.log10(p / (1.0 - p))}


def fit(frame: pd.DataFrame, max_season: int):
    pipe, cols = load_source()
    train = LF.model_frame(frame)
    train = train[train["season"].between(TRAIN_MIN, max_season)].reset_index(drop=True)
    pipe.fit(train[cols], train["home_win"].to_numpy())
    return pipe, cols, train


def reproduce_2025(frame: pd.DataFrame) -> dict:
    """B2: train <= 2024 with this module's code, predict the 2025 rows, and
    require bit-for-bit equality with Phase 8's saved LR-tuned raw probs."""
    pipe, cols, _ = fit(frame, 2024)
    rows = LF.model_frame(frame)
    rows = rows[rows["season"] == 2025]
    ref = pd.read_parquet(HOLDOUT_2025, columns=["game_id", "proba_raw_LR-tuned"])
    got = pd.DataFrame({"game_id": rows["game_id"].to_numpy(),
                        "p": pipe.predict_proba(rows[cols])[:, 1]})
    m = ref.merge(got, on="game_id", how="outer", validate="1:1", indicator=True)
    both = m["_merge"].eq("both")
    diff = (m.loc[both, "p"] - m.loc[both, "proba_raw_LR-tuned"]).abs()
    res = {"n_ref": len(ref), "n_got": len(got), "n_matched_ids": int(both.sum()),
           "exact_equal": bool(both.all() and np.array_equal(m["p"].to_numpy(),
                                                             m["proba_raw_LR-tuned"].to_numpy())),
           "max_abs_diff": float(diff.max()) if len(diff) else float("nan")}
    return res


def _write_readonly(obj: dict, path: Path) -> str:
    if path.exists():
        raise FileExistsError(f"{path.name} already exists -- frozen models are never overwritten")
    tmp = path.with_name(f".{path.name}.tmp")
    joblib.dump(obj, tmp)
    tmp.rename(path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return sha256_file(path)


def _metadata(role: str, pipe, cols, train, elo_adv: float, derivation: dict | None) -> dict:
    return {
        "pipeline": pipe,
        "feature_cols": cols,
        "model_name": f"live_2026_{role}",
        "role": role,
        "training_seasons": [TRAIN_MIN, int(train["season"].max())],
        "n_train": int(len(train)),
        "hyperparameters": {k: pipe.named_steps["clf"].get_params()[k]
                            for k in ("C", "max_iter", "random_state", "solver", "penalty")},
        "pipeline_steps": [name for name, _ in pipe.steps],
        "elo_home_adv": float(elo_adv),
        "elo_home_adv_derivation": derivation,
        "training_data_sha256": training_data_sha256(train, cols),
        "source_model": {"path": SOURCE_MODEL.name, "sha256": sha256_file(SOURCE_MODEL)},
        "features_py_sha256": sha256_file(F.PROJECT_ROOT / "src" / "features.py"),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "probabilities": "raw predict_proba (Phase 8: calibration not adopted)",
    }


def save_champion() -> int:
    """How live_2026_champion.joblib was produced (B1). Refuses if it exists."""
    if CHAMPION_PATH.exists():
        print(f"REFUSING: {CHAMPION_PATH.name} already exists -- frozen models are never overwritten.")
        return 2
    frame, _ = LF.build_and_verify(write_path=None)
    rep = reproduce_2025(frame)
    print(f"\nB2 reproduction (train <=2024 -> 2025 LR-tuned raw probs): {rep}")
    if not rep["exact_equal"]:
        print("STOP: B2 reproduction failed -- not saving anything.")
        return 1
    pipe, cols, train = fit(frame, TRAIN_MAX)
    canon = pd.read_parquet(LF.CANONICAL_PATH)
    assert training_data_sha256(train, cols) == training_data_sha256(canon, cols), \
        "champion training rows differ from canonical game_features.parquet"
    champ = _metadata("champion", pipe, cols, train, FROZEN_ELO_HOME_ADV,
                      {"method": "train-only (<=2021) home win rate, Phase 5A",
                       "seasons": [2002, 2021], "home_wins": 2890, "n_games": 5124})
    champ["training_data_file_sha256"] = sha256_file(LF.CANONICAL_PATH)
    print(f"\nsaved (read-only): {CHAMPION_PATH.name}  sha256 {_write_readonly(champ, CHAMPION_PATH)}")
    return 0


def fit_calibrator(frame: pd.DataFrame, target_season: int) -> tuple[CAL.SigmoidCalibrator, pd.DataFrame]:
    """Phase 8 method (src/calibration.py): OOF predictions for T-5..T-1, each
    from a model trained on seasons before it, then one sigmoid fit."""
    unfitted, cols = load_source()
    played = LF.model_frame(frame)
    played = played[played["season"] < target_season].reset_index(drop=True)
    oof = CAL.oof_predictions(unfitted, played, target_season, cols)
    return CAL.SigmoidCalibrator.fit(oof["proba_raw"].to_numpy(), oof["home_win"].to_numpy()), oof


def reproduce_2025_calibrator(frame: pd.DataFrame) -> dict:
    """The calibrator code for T=2025 must reproduce Phase 8's recorded
    LR-tuned calibrator (slope, intercept, n_fit) exactly."""
    import json
    ref = json.loads(HOLDOUT_2025_RECORD.read_text())["calibrators"]["LR-tuned"]
    cal, _ = fit_calibrator(frame, 2025)
    got = {"slope": cal.slope, "intercept": cal.intercept, "n_fit": cal.n_fit}
    return {"ref": ref, "got": got, "exact_equal": got == ref}


def calibrated_proba(d: dict, X: pd.DataFrame) -> np.ndarray:
    """P(home) for a saved live model dict: raw, or raw -> calibrator."""
    p = d["pipeline"].predict_proba(X)[:, 1]
    c = d.get("calibrator")
    return p if c is None else CAL.SigmoidCalibrator(c["slope"], c["intercept"], c["n_fit"]).transform(p)


def save_calibrated_challenger() -> int:
    if CHALLENGER_PATH.exists():
        print(f"REFUSING: {CHALLENGER_PATH.name} already exists -- frozen models are never overwritten.")
        return 2
    frame, _ = LF.build_and_verify(write_path=None)
    rep = reproduce_2025(frame)
    rc = reproduce_2025_calibrator(frame)
    print(f"\nB2 reproduction: {rep}\ncalibrator reproduction (T=2025 vs Phase 8 record): {rc}")
    if not (rep["exact_equal"] and rc["exact_equal"]):
        print("STOP: a reproduction check failed -- not saving anything.")
        return 1

    champ = joblib.load(CHAMPION_PATH)
    cal, oof = fit_calibrator(frame, LIVE_TARGET_SEASON)
    seasons = sorted(oof["season"].unique().tolist())
    print(f"\nT={LIVE_TARGET_SEASON} calibrator: slope={cal.slope:.6f} intercept={cal.intercept:+.6f} "
          f"n_fit={cal.n_fit} (OOF seasons {seasons})")
    chall = {k: v for k, v in champ.items() if k not in ("model_name", "role", "created_utc")}
    chall.update({
        "model_name": "live_2026_challenger_calibrated",
        "role": "challenger",
        "calibrator": {"slope": cal.slope, "intercept": cal.intercept, "n_fit": cal.n_fit},
        "calibration": {
            "method": "OOF walk-forward sigmoid (src/calibration.py, Phase 8): for each S in T-5..T-1 "
                      "train on seasons < S, predict S; LogisticRegression(C=inf) on logit(p)",
            "target_season": LIVE_TARGET_SEASON, "oof_seasons": seasons,
            "window_note": "2021-2025 follows from the standard T-5..T-1 rule for T=2026 (not hand-picked); "
                           "it happens to exclude 2020",
        },
        "base_model": {"path": CHAMPION_PATH.name, "sha256": sha256_file(CHAMPION_PATH)},
        "probabilities": "calibrated: sigmoid(slope * logit(raw champion p) + intercept)",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    sha = _write_readonly(chall, CHALLENGER_PATH)
    print(f"\nsaved (read-only): {CHALLENGER_PATH.name}  sha256 {sha}")
    return 0


def show() -> None:
    for p in (CHAMPION_PATH, CHALLENGER_PATH):
        d = joblib.load(p)
        meta = {k: v for k, v in d.items() if k not in ("pipeline", "feature_cols")}
        mode = oct(p.stat().st_mode & 0o777)
        print(f"\n{p.name}  sha256 {sha256_file(p)}  mode {mode}")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print(f"  feature_cols: {len(d['feature_cols'])} (== features.feature_columns(): "
              f"{list(d['feature_cols']) == F.feature_columns()})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["reproduce", "save-champion", "save-challenger", "show"])
    a = ap.parse_args().action
    if a == "reproduce":
        frame, _ = LF.build_and_verify(write_path=None)
        rep, rc = reproduce_2025(frame), reproduce_2025_calibrator(frame)
        print(f"\nB2 reproduction: {rep}\ncalibrator reproduction: {rc}")
        return 0 if rep["exact_equal"] and rc["exact_equal"] else 1
    if a == "save-champion":
        return save_champion()
    if a == "save-challenger":
        return save_calibrated_challenger()
    show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
