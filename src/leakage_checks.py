"""Pre-model + post-model leakage audit for the feature pipeline (SPEC Section 5.5).

Implements all four SPEC 5.5 checks:

1. ``check_feature_timestamps`` -- for every game row, assert that every
   feature's newest underlying datum is strictly earlier than that row's kickoff.
   Raises ``AssertionError`` on any violation.
2. ``label_shuffle_test`` -- permute the training labels only, fit a plain
   ``LogisticRegression`` on the real features, evaluate on the real
   VALIDATION labels (season 2022). A pipeline that carries no label signal
   scores ~0.50. Every non-PASS verdict band exits non-zero. Evaluates against
   2022, not 2023-2024: SPEC Section 6 reserves the test seasons "untouched
   until final evaluation," and a diagnostic score computed against them is
   still information out of that split, even when it is never reported as a
   final result.
3. ``feature_importance_inspection`` -- needs a trained model (SPEC Phase 4,
   ``src/train_winner.py``). Flags the top feature if it accounts for more
   than ~40-50% of total |coefficient| weight.
4. ``ablation_check`` -- needs a trained model. Drops the top feature from
   check 3, retrains the identical pipeline on the same train split,
   re-evaluates on validation. Flags a collapse to ~baseline or an increase
   in accuracy after removal.

Checks 3-4 use ``src.train_winner``'s train (<=2021) / validation (2022)
split and never touch the 2023-2024 test seasons (SPEC Section 6/8).

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

from src import evaluate, features as F, train_winner as TW

PROC_DIR = F.PROC_DIR

# SPEC Section 6 walk-forward split -- reuse src.train_winner's constants
# directly (TW.TRAIN_MAX_SEASON=2021, TW.VAL_SEASON=2022,
# TW.FORBIDDEN_TEST_SEASONS=(2023, 2024)) so every check below shares one
# definition of "test" and one guard against loading it.


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


def label_shuffle_test(tv_frame: pd.DataFrame, schedules: pd.DataFrame,
                       n_shuffles: int = 20, seed: int = 0) -> dict:
    """SPEC 5.5 check 2. Evaluates against the VALIDATION split (season 2022),
    never against the 2023-2024 test split -- SPEC Section 6 reserves test
    "untouched until final evaluation," and a diagnostic score computed
    against it is still information out of that split, even though it is
    never reported as a final result.

    ``tv_frame`` MUST already be guarded to train+validation seasons only
    (see ``train_winner.load_train_val_frame``, same guard style reused here).
    This function re-asserts that guard itself so a caller mistake can't
    silently let 2023-2024 in.
    """
    seasons_seen = sorted(tv_frame["season"].unique().tolist())
    touched_test = sorted(set(seasons_seen) & set(TW.FORBIDDEN_TEST_SEASONS))
    assert not touched_test, (
        f"FLAG: label_shuffle_test received test seasons {touched_test} -- "
        "SPEC Section 6/8 violation, stop and investigate"
    )

    cols = F.feature_columns()
    train = tv_frame[tv_frame["season"] <= TW.TRAIN_MAX_SEASON]
    val = tv_frame[tv_frame["season"] == TW.VAL_SEASON]

    X_train, y_train = train[cols], train["home_win"].to_numpy()
    X_val, y_val = val[cols], val["home_win"].to_numpy()

    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_shuffles):
        y_perm = rng.permutation(y_train)
        pipe = _pipeline().fit(X_train, y_perm)
        accs.append(accuracy_score(y_val, pipe.predict(X_val)))
    accs = np.array(accs)

    contrast = _pipeline().fit(X_train, y_train)
    contrast_acc = accuracy_score(y_val, contrast.predict(X_val))

    baseline = evaluate.home_baseline_accuracy(schedules, [TW.VAL_SEASON])
    majority = max(y_val.mean(), 1 - y_val.mean())

    label, is_pass = _verdict(float(accs.mean()))
    res = {
        "n_shuffles": n_shuffles, "seed": seed,
        "shuffled_mean": float(accs.mean()), "shuffled_std": float(accs.std()),
        "shuffled_min": float(accs.min()), "shuffled_max": float(accs.max()),
        "majority_class_rate": float(majority),
        "home_baseline_accuracy": float(baseline["baseline_accuracy"]),
        "true_label_contrast_acc": float(contrast_acc),
        "verdict": label, "pass": is_pass,
        "n_train": len(train), "n_val": len(val),
    }

    print("\nCHECK 2  label-shuffle test (SPEC 5.5 #2)")
    print(f"  guard: seasons seen in tv_frame = {seasons_seen} "
          f"-- test seasons {TW.FORBIDDEN_TEST_SEASONS} absent, confirmed")
    print(f"  split: train season<={TW.TRAIN_MAX_SEASON} (n={res['n_train']:,}), "
          f"validation season={TW.VAL_SEASON} (n={res['n_val']:,})")
    print(f"  shuffled-label VALIDATION accuracy over {n_shuffles} shuffles: "
          f"mean={res['shuffled_mean']:.4f}  std={res['shuffled_std']:.4f}  "
          f"[{res['shuffled_min']:.4f}, {res['shuffled_max']:.4f}]")
    print(f"  reference: home baseline={res['home_baseline_accuracy']:.4f}  "
          f"majority-class={res['majority_class_rate']:.4f}")
    print(f"  contrast (real labels vs. validation, NOT a reportable final result): {contrast_acc:.4f}")
    print(f"  VERDICT: {label}")
    if contrast_acc > 0.70:
        print("  !! contrast fit > 0.70 -- per SPEC 5.6 / CLAUDE.md, stop and "
              "re-audit before any modeling.")
    return res


# --------------------------------------------------------------------------- #
# Check 3 -- feature-importance inspection (SPEC 5.5 #3)
# --------------------------------------------------------------------------- #
# SPEC's own guideline: "more than ~40-50% on its own" -- flag at the low end
# of that range so a borderline case still gets manually re-verified.
IMPORTANCE_FLAG_SHARE = 0.40


def feature_importance_inspection(pipe: Pipeline, feature_cols: list[str],
                                   flag_share: float = IMPORTANCE_FLAG_SHARE) -> dict:
    """SPEC 5.5 check 3: inspect the fitted ``LogisticRegression`` coefficient
    magnitudes. Flags the top feature for manual re-verification against SPEC
    5.4 if it accounts for more than ``flag_share`` of total |coefficient|
    weight across all features.
    """
    coefs = pipe.named_steps["clf"].coef_[0]
    abs_coefs = np.abs(coefs)
    total = abs_coefs.sum()
    shares = abs_coefs / total
    order = np.argsort(-shares)

    top_i = order[0]
    top_feature = feature_cols[top_i]
    top_share = float(shares[top_i])
    top_coef = float(coefs[top_i])
    flagged = top_share > flag_share

    ranking = [
        {"feature": feature_cols[i], "coef": float(coefs[i]), "share_of_abs_total": float(shares[i])}
        for i in order
    ]

    result = {
        "top_feature": top_feature, "top_coef": top_coef, "top_share": top_share,
        "flag_share_threshold": flag_share, "flagged": flagged,
        "ranking": ranking,
    }

    print("\nCHECK 3  feature-importance inspection (SPEC 5.5 #3)")
    print("  top-5 by share of total |coef|:")
    for r in ranking[:5]:
        print(f"    {r['feature']:<34} coef={r['coef']:+.4f}  share={r['share_of_abs_total']:.3f}")
    if flagged:
        print(f"  !! FLAGGED: '{top_feature}' accounts for {top_share:.1%} of total |coef| "
              f"(> {flag_share:.0%}) -- manually verify its computation against SPEC 5.4 "
              "before trusting this model.")
    else:
        print(f"  VERDICT: PASS -- no single feature dominates (top = '{top_feature}' at {top_share:.1%})")
    return result


# --------------------------------------------------------------------------- #
# Check 4 -- ablation sanity check (SPEC 5.5 #4)
# --------------------------------------------------------------------------- #
def ablation_check(train_df: pd.DataFrame, val_df: pd.DataFrame,
                   feature_cols: list[str], drop_feature: str,
                   full_metrics: dict, baseline_accuracy: float) -> dict:
    """SPEC 5.5 check 4: drop the single most important feature (from check 3),
    retrain the identical pipeline on the same train split, re-evaluate on
    validation.

    Accuracy SHOULD degrade gracefully. Two patterns are explicitly flagged as
    suspicious, per SPEC 5.5 #4:
      (a) collapse to ~baseline -- the model was overly reliant on one feature
      (b) an INCREASE in accuracy after removal -- the removed feature may
          have been actively harmful (a leak interacting badly with something
          else)
    """
    remaining = [c for c in feature_cols if c != drop_feature]
    _, metrics = TW.fit_and_eval(train_df, val_df, remaining)

    delta_vs_full = metrics["accuracy"] - full_metrics["accuracy"]
    collapsed_to_baseline = metrics["accuracy"] <= baseline_accuracy + 0.01
    increased = delta_vs_full > 0

    result = {
        "dropped_feature": drop_feature,
        "full_accuracy": full_metrics["accuracy"],
        "ablated_accuracy": metrics["accuracy"],
        "ablated_log_loss": metrics["log_loss"],
        "ablated_brier_score": metrics["brier_score"],
        "ablated_roc_auc": metrics["roc_auc"],
        "delta_vs_full": delta_vs_full,
        "baseline_accuracy": baseline_accuracy,
        "collapsed_to_baseline": collapsed_to_baseline,
        "increased_after_removal": increased,
        "suspicious": collapsed_to_baseline or increased,
    }

    print("\nCHECK 4  ablation sanity check (SPEC 5.5 #4)")
    print(f"  dropped feature: '{drop_feature}'")
    print(f"  full model     : accuracy={full_metrics['accuracy']:.4f}  "
          f"log_loss={full_metrics['log_loss']:.4f}  brier={full_metrics['brier_score']:.4f}  "
          f"roc_auc={full_metrics['roc_auc']:.4f}")
    print(f"  ablated model  : accuracy={metrics['accuracy']:.4f}  "
          f"log_loss={metrics['log_loss']:.4f}  brier={metrics['brier_score']:.4f}  "
          f"roc_auc={metrics['roc_auc']:.4f}")
    print(f"  delta (ablated - full): {delta_vs_full:+.4f}   home baseline: {baseline_accuracy:.4f}")
    if increased:
        print(f"  !! FLAGGED: accuracy INCREASED after removing '{drop_feature}' -- "
              "SPEC 5.5 #4 red flag: this feature may be actively harmful (a leak "
              "interacting badly with something else).")
    if collapsed_to_baseline:
        print(f"  !! FLAGGED: ablated accuracy ({metrics['accuracy']:.4f}) collapsed to "
              f"~baseline ({baseline_accuracy:.4f}) -- model may be overly reliant on "
              f"'{drop_feature}' alone.")
    if not result["suspicious"]:
        print("  VERDICT: PASS -- accuracy degraded gracefully, no red flags.")
    return result


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

    # Checks 2-4 all share one guarded train(<=2021)/validation(2022) frame --
    # TW.load_train_val_frame() applies the same SPEC 6/8 guard as
    # train_winner.py: 2023-2024 is dropped immediately on load and never
    # reaches any check below.
    tv_frame = TW.load_train_val_frame()
    train_only = tv_frame[tv_frame["season"] <= TW.TRAIN_MAX_SEASON].reset_index(drop=True)
    val_only = tv_frame[tv_frame["season"] == TW.VAL_SEASON].reset_index(drop=True)

    res2 = label_shuffle_test(tv_frame, schedules)
    ok = ok and res2["pass"]

    # Checks 3-4 need a trained model (SPEC 5.5 #3-#4) -- Phase 4 (src/train_winner.py).
    pipe, full_metrics = TW.fit_and_eval(train_only, val_only, TW.FEATURE_COLS)
    val_baseline = evaluate.home_baseline_accuracy(schedules, [TW.VAL_SEASON])["baseline_accuracy"]

    print(f"\n(model for checks 3-4: LogisticRegression trained on seasons "
          f"<= {TW.TRAIN_MAX_SEASON}, evaluated on {TW.VAL_SEASON} -- "
          f"accuracy={full_metrics['accuracy']:.4f})")

    if full_metrics["accuracy"] > TW.SUSPICIOUS_ACCURACY:
        print(f"\n!! validation accuracy {full_metrics['accuracy']:.4f} exceeds "
              f"{TW.SUSPICIOUS_ACCURACY} -- SPEC 5.6/5.7: STOP, not running checks 3-4, "
              "flagging for manual review.")
        return 1

    res3 = feature_importance_inspection(pipe, TW.FEATURE_COLS)
    res4 = ablation_check(train_only, val_only, TW.FEATURE_COLS,
                          res3["top_feature"], full_metrics, val_baseline)
    ok = ok and not res3["flagged"] and not res4["suspicious"]

    print()
    if ok:
        print("ALL LEAKAGE CHECKS PASSED")
        return 0
    print("LEAKAGE CHECKS FAILED -- do not proceed to modeling")
    return 1


if __name__ == "__main__":
    sys.exit(main())
