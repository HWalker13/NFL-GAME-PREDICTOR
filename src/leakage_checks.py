"""Pre-model leakage audit for the Phase 2 feature pipeline (SPEC Section 5.5).

Implements the two checks from SPEC 5.5 that do **not** require a trained model:

1. ``check_feature_timestamps`` -- for every game row, assert that every
   feature's newest underlying datum is strictly earlier than that row's kickoff.
   Raises ``AssertionError`` on any violation.
2. ``label_shuffle_test`` -- permute the training labels only, fit a plain
   ``LogisticRegression`` on the real features, evaluate on the real test
   labels. A pipeline that carries no label signal scores ~0.50. Every non-PASS
   verdict band exits non-zero.

SPEC 5.5 checks 3 (feature-importance inspection) and 4 (ablation) need a
trained model and belong to Phase 4/5 -- they are intentionally not here.

Run::

    python -m src.leakage_checks        # exit 0 = all clear, exit 1 = problem
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import evaluate, features as F

PROC_DIR = F.PROC_DIR

# SPEC Section 6 walk-forward split.
TRAIN_MAX_SEASON = 2021
TEST_SEASONS = (2023, 2024)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_frame_and_log(rebuild: bool = False):
    """Return ``(game_features, team_game_log)``, building them if missing."""
    gf_path = PROC_DIR / "game_features.parquet"
    tgl_path = PROC_DIR / "team_game_log.parquet"
    if rebuild or not gf_path.exists() or not tgl_path.exists():
        F.build_game_features()
    return pd.read_parquet(gf_path), pd.read_parquet(tgl_path)


def _sides_from_log(tgl: pd.DataFrame) -> pd.DataFrame:
    """One row per game_id with the (canonical) home/away team codes."""
    home = tgl.loc[tgl["is_home"], ["game_id", "team"]].rename(columns={"team": "home_team"})
    away = tgl.loc[~tgl["is_home"], ["game_id", "team"]].rename(columns={"team": "away_team"})
    return home.merge(away, on="game_id")


# --------------------------------------------------------------------------- #
# Check 1 -- timestamp assertion (SPEC 5.5 #1)
# --------------------------------------------------------------------------- #
def _prior_only_sd(log: pd.DataFrame, metric: str) -> pd.Series:
    """Independent oracle for ``{metric}_sd``: mean of the metric over the team's
    strictly-earlier games in the same season. Deliberately computed a different
    way from ``features.roll_left`` (shift + expanding) so a bug in one is not
    mirrored in the other."""
    g = log.groupby(["team", "season"], sort=False)[metric]
    return g.transform(lambda s: s.shift(1).expanding().mean())


def check_feature_timestamps(frame: pd.DataFrame, tgl: pd.DataFrame) -> dict:
    """Assert every feature's newest underlying datum precedes kickoff.

    * Pattern-A features (``_sd`` / in-season part of ``_shrunk`` / ``mu_*``):
      the newest possible contributor is the team's immediately-preceding game
      (``closed='left'`` leaves that as the last row in the window). Assert
      ``prev_game_kickoff < kickoff`` for every row that has one, AND
      independently recompute every ``{m}_sd`` from a strictly-prior-games-only
      oracle and assert it matches -- this is what catches a rolling window that
      silently includes the current row (``closed='right'`` / a bare
      ``.rolling().mean()``).
    * Pattern-B features (``_prior`` / ``_league``): the newest contributor is
      the last game of the matched prior season. Assert that season's max
      kickoff ``< kickoff``.
    * ``context`` features: fixed at scheduling time -- assert the underlying
      schedule fields are populated.
    * Tripwire: a rolled feature must not correlate ~1:1 with the current
      game's own raw metric. Hard-fail above 0.95.

    Raises ``AssertionError`` on any violation.
    """
    log = tgl.sort_values(["team", "kickoff", "game_id"]).copy()
    log["prev_kickoff"] = log.groupby("team")["kickoff"].shift(1)

    # --- current-row-inclusion check: {m}_sd must use PRIOR games only -------
    rolled = F.roll_features(tgl)
    rolled = rolled.sort_values(["team", "kickoff", "game_id"]).reset_index(drop=True)
    olog = rolled.merge(
        log[["game_id", "team"] + F.RAW_METRICS], on=["game_id", "team"], how="left"
    )
    sd_mismatch = {}
    for m in F.RAW_METRICS:
        oracle = _prior_only_sd(olog, m).to_numpy()
        got = olog[f"{m}_sd"].to_numpy()
        both_nan = np.isnan(oracle) & np.isnan(got)
        close = np.isclose(oracle, got, rtol=1e-6, atol=1e-9, equal_nan=True)
        bad = int((~(close | both_nan)).sum())
        if bad:
            sd_mismatch[m] = bad
    if sd_mismatch:
        raise AssertionError(
            "in-season rolling feature does not match a strictly-prior-games "
            f"oracle (current row is leaking into its own window): {sd_mismatch}"
        )

    # season max kickoff over strictly-earlier seasons, per team
    season_max = (log.groupby(["team", "season"])["kickoff"].max()
                  .rename("season_max_kick").reset_index().sort_values(["team", "season"]))
    season_max["prior_season_max_kick"] = (
        season_max.groupby("team", group_keys=False)["season_max_kick"]
        .apply(lambda s: s.shift(1).cummax())
    )
    prior_season_max = season_max[["team", "season", "prior_season_max_kick"]]

    sides = _sides_from_log(log)
    base = frame[["game_id", "season", "kickoff"]].merge(sides, on="game_id", how="left")

    violations = []
    lags = []
    for side in ("home", "away"):
        team_col = f"{side}_team"
        m = base.merge(
            log[["game_id", "team", "prev_kickoff"]].rename(columns={"team": team_col}),
            on=["game_id", team_col], how="left",
        ).merge(
            prior_season_max.rename(columns={"team": team_col}),
            on=[team_col, "season"], how="left",
        )

        has_prev = m["prev_kickoff"].notna()
        bad_prev = has_prev & ~(m["prev_kickoff"] < m["kickoff"])
        if bad_prev.any():
            violations += m.loc[bad_prev, "game_id"].tolist()
        lags.append((m.loc[has_prev, "kickoff"] - m.loc[has_prev, "prev_kickoff"])
                    .dt.total_seconds().min() / 86400.0)

        has_prior = m["prior_season_max_kick"].notna()
        bad_prior = has_prior & ~(m["prior_season_max_kick"] < m["kickoff"])
        if bad_prior.any():
            violations += m.loc[bad_prior, "game_id"].tolist()

    if violations:
        raise AssertionError(
            f"timestamp leakage: {len(violations)} feature-rows use data at/after "
            f"kickoff. Offending game_ids (first 10): {sorted(set(violations))[:10]}"
        )

    # context features must be populated (they are exempt from the time test
    # because they are fixed when the game is scheduled)
    ctx_required = ["home_rest", "away_rest", "rest_diff", "div_game",
                    "travel_diff", "tz_diff", "is_neutral"]
    ctx_missing = {c: int(frame[c].isna().sum()) for c in ctx_required
                   if frame[c].isna().any()}
    if ctx_missing:
        raise AssertionError(f"context features unexpectedly null: {ctx_missing}")

    # tripwire: rolled feature vs. this game's own raw metric
    raw = tgl[["game_id", "team", "is_home"] + F.RAW_METRICS]
    max_corr = 0.0
    worst = None
    for side in ("home", "away"):
        want_home = side == "home"
        r = raw[raw["is_home"] == want_home].drop(columns=["is_home"])
        r = r.rename(columns={m: f"cur_{m}" for m in F.RAW_METRICS}).drop(columns=["team"])
        j = frame[["game_id"] + [f"{side}_{m}_shrunk" for m in F.RAW_METRICS]].merge(
            r, on="game_id", how="inner")
        for m in F.RAW_METRICS:
            c = j[f"{side}_{m}_shrunk"].corr(j[f"cur_{m}"])
            if pd.notna(c) and abs(c) > max_corr:
                max_corr, worst = abs(c), f"{side}_{m}"
    if max_corr > 0.95:
        raise AssertionError(
            f"tripwire: {worst}_shrunk correlates {max_corr:.3f} with the "
            f"current game's own raw {worst} -- rolled feature is leaking the outcome"
        )

    result = {
        "rows": len(frame),
        "violations": 0,
        "min_prev_game_lag_days": round(float(np.nanmin(lags)), 3),
        "max_rolled_vs_current_corr": round(float(max_corr), 3),
        "worst_corr_feature": worst,
    }
    print(
        f"CHECK 1  PASS: {result['rows']:,} rows, 0 timestamp violations, "
        f"min prev-game lag = {result['min_prev_game_lag_days']} days, "
        f"max(rolled vs current-game raw) corr = {result['max_rolled_vs_current_corr']} "
        f"({worst})"
    )
    return result


# --------------------------------------------------------------------------- #
# Check 2 -- label-shuffle test (SPEC 5.5 #2)
# --------------------------------------------------------------------------- #
def _pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def _verdict(mean_acc: float) -> tuple[str, bool]:
    """Return (label, is_pass). Every non-PASS band is a hard stop."""
    if 0.47 <= mean_acc <= 0.53:
        return "PASS", True
    if 0.53 < mean_acc <= 0.55:
        return "WARN - borderline high (possible weak leak; re-run with more shuffles)", False
    if mean_acc > 0.55:
        return "FAIL - LEAK SUSPECTED (pipeline carries label signal)", False
    if 0.45 <= mean_acc < 0.47:
        return "WARN - borderline low (possible split/label bug)", False
    return "FAIL - BUG SUSPECTED (well below chance: inverted labels / broken split)", False


def label_shuffle_test(frame: pd.DataFrame, schedules: pd.DataFrame,
                       n_shuffles: int = 20, seed: int = 0) -> dict:
    cols = F.feature_columns()
    train = frame[frame["season"] <= TRAIN_MAX_SEASON]
    test = frame[frame["season"].isin(TEST_SEASONS)]

    X_train, y_train = train[cols], train["home_win"].to_numpy()
    X_test, y_test = test[cols], test["home_win"].to_numpy()

    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_shuffles):
        y_perm = rng.permutation(y_train)
        pipe = _pipeline().fit(X_train, y_perm)
        accs.append(accuracy_score(y_test, pipe.predict(X_test)))
    accs = np.array(accs)

    contrast = _pipeline().fit(X_train, y_train)
    contrast_acc = accuracy_score(y_test, contrast.predict(X_test))

    baseline = evaluate.home_baseline_accuracy(schedules, list(TEST_SEASONS))
    majority = max(y_test.mean(), 1 - y_test.mean())

    label, is_pass = _verdict(float(accs.mean()))
    res = {
        "n_shuffles": n_shuffles, "seed": seed,
        "shuffled_mean": float(accs.mean()), "shuffled_std": float(accs.std()),
        "shuffled_min": float(accs.min()), "shuffled_max": float(accs.max()),
        "majority_class_rate": float(majority),
        "home_baseline_accuracy": float(baseline["baseline_accuracy"]),
        "true_label_contrast_acc": float(contrast_acc),
        "verdict": label, "pass": is_pass,
        "n_train": len(train), "n_test": len(test),
    }

    print("\nCHECK 2  label-shuffle test (SPEC 5.5 #2)")
    print(f"  split: train season<={TRAIN_MAX_SEASON} (n={res['n_train']:,}), "
          f"test {TEST_SEASONS} (n={res['n_test']:,}); 2022 unused")
    print(f"  shuffled-label test accuracy over {n_shuffles} shuffles: "
          f"mean={res['shuffled_mean']:.4f}  std={res['shuffled_std']:.4f}  "
          f"[{res['shuffled_min']:.4f}, {res['shuffled_max']:.4f}]")
    print(f"  reference: home baseline={res['home_baseline_accuracy']:.4f}  "
          f"majority-class={res['majority_class_rate']:.4f}")
    print(f"  contrast (real labels, NOT a reportable result): {contrast_acc:.4f}")
    print(f"  VERDICT: {label}")
    if contrast_acc > 0.70:
        print("  !! contrast fit > 0.70 -- per SPEC 5.6 / CLAUDE.md, stop and "
              "re-audit before any modeling.")
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    frame, tgl = load_frame_and_log()
    schedules = evaluate.load_schedules()

    ok = True
    try:
        check_feature_timestamps(frame, tgl)
    except AssertionError as exc:
        print(f"CHECK 1  FAIL: {exc}")
        ok = False

    res2 = label_shuffle_test(frame, schedules)
    ok = ok and res2["pass"]

    print()
    if ok:
        print("ALL LEAKAGE CHECKS PASSED")
        return 0
    print("LEAKAGE CHECKS FAILED -- do not proceed to modeling")
    return 1


if __name__ == "__main__":
    sys.exit(main())
