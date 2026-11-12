"""Audit: label-shuffle test with the SAME pipeline as the headline model.

    # agent-run (validation season only; no test season touched):
    venv/bin/python -W ignore -m audit.scripts.label_shuffle

    # OWNER-RUN ONLY (CLAUDE.md: test-set runs are executed by the owner):
    venv/bin/python -W ignore -m audit.scripts.label_shuffle --test-seasons

Pipeline: sklearn.base.clone() of models/logreg_winner_tuned.joblib
(SimpleImputer(median) -> StandardScaler -> LogisticRegression(C=0.01)),
same 67 features. Training labels are permuted; evaluation uses REAL labels.

Default mode:   train 2002-2021 (shuffled) -> evaluate 2022.
--test-seasons: the walk-forward protocol with shuffled training labels:
                train 2002-2022 (shuffled) -> 2023; 2002-2023 (shuffled) -> 2024.
                Shuffled-label models carry no information about the real
                models' test performance, but this still reads 2023-24
                labels, so per the project rule the owner runs it.

Writes audit/out/label_shuffle_<mode>.json. Expect accuracy near the share of
the majority class the shuffled model drifts toward (~50-56%), never near 64%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "out"
N_SHUFFLES = 20


def run(frame, cols, pipe0, steps, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    res = {"steps": [], "n_shuffles": N_SHUFFLES}
    pooled = []
    for _ in range(N_SHUFFLES):
        k_tot = n_tot = 0
        for train_max, test_season in steps:
            tr = frame[frame["season"] <= train_max]
            te = frame[frame["season"] == test_season]
            y_perm = rng.permutation(tr["home_win"].to_numpy())
            pipe = clone(pipe0).fit(tr[cols], y_perm)
            pred = (pipe.predict_proba(te[cols])[:, 1] > 0.5).astype(int)
            k_tot += int((pred == te["home_win"].to_numpy()).sum())
            n_tot += len(te)
        pooled.append(k_tot / n_tot)
    pooled = np.array(pooled)
    for train_max, test_season in steps:
        te = frame[frame["season"] == test_season]
        res["steps"].append({"train": f"2002-{train_max}", "eval": test_season, "n_eval": len(te),
                             "home_win_rate_eval": round(float(te["home_win"].mean()), 4)})
    res.update(mean=round(float(pooled.mean()), 4), std=round(float(pooled.std()), 4),
               min=round(float(pooled.min()), 4), max=round(float(pooled.max()), 4))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", action="store_true",
                    help="OWNER ONLY: evaluate on 2023-24 (walk-forward, shuffled training labels)")
    args = ap.parse_args()
    d = joblib.load(ROOT / "models/logreg_winner_tuned.joblib")
    cols, pipe0 = list(d["feature_cols"]), d["pipeline"]
    frame = pd.read_parquet(ROOT / "data/processed/game_features.parquet")
    if args.test_seasons:
        steps, mode = ((2022, 2023), (2023, 2024)), "test_2023_2024"
    else:
        steps, mode = ((2021, 2022),), "val_2022"
        frame = frame[frame["season"] <= 2022]          # 2023+ never loaded into memory past here
    res = run(frame, cols, pipe0, steps)
    res["pipeline"] = str(pipe0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"label_shuffle_{mode}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
