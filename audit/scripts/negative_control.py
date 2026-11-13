"""Audit: negative-control experiment for leakage Checks 1 and 2.

    venv/bin/python -W ignore -m audit.scripts.negative_control

Reproduces the (previously uncommitted) Phase 2 negative control: deliberately
make rolling features include the CURRENT game, then see whether
Check 1 (``leakage_checks.check_feature_timestamps``) and Check 2
(``leakage_checks.label_shuffle_test``) notice.

How the sabotage is applied -- in memory only, nothing in src/ is edited:
  ``unittest.mock.patch.object`` swaps ``src.features.roll_left`` and/or
  ``src.features.roll_ewm`` for sabotaged copies for the duration of one
  feature build + the two checks, then restores them. Both the frame builder
  (``build_game_features``) and Check 1 (which calls ``F.roll_features``
  internally) look these functions up as module globals, so one patch reaches
  both. Check 1's independent oracles (``_prior_only_sd`` and, since
  Amendment 5, ``_prior_only_ewm``) are NOT patched.

Conditions:
  clean                 real functions, no patch
  single_metric_sd      roll_left uses closed='right' for ONE metric only
                        (qb_epa_per_db -> its _sd, _sd_n, _shrunk and the
                        mu_* diffs built from it). Representative single feature.
  phase2_roll_left      roll_left uses closed='right' for EVERY call. This is
                        the Phase 2 reconstruction: at commit 16cb45a roll_left
                        was the only rolling function feeding features, and
                        its single closed="left" argument covers all 11
                        metrics + games_played_sd.
  ewm_only              roll_ewm drops its shift(1) (current row included);
                        roll_left untouched. Not part of Phase 2 (EWM did not
                        exist yet). At 07129c4 this PASSED Check 1 (no EWM
                        oracle); Amendment 5 added ``_prior_only_ewm``.
  all_rolling           phase2_roll_left + ewm_only together: every rolling
                        feature includes the current row (strongest sabotage).

Check 2 is run twice per condition: on the current 67-column feature set
(``F.feature_columns()``, as the check does today) and on the Phase 2
42-column set (current set minus *_ewm and Elo, which equals
``feature_columns()`` at 16cb45a).

Seasons: features are built from 2002-2022 only. ``evaluate.load_schedules``
is patched to open schedules_2002..2022 files only and pbp is pulled for
2002-2022 only, so no 2023+ file is ever opened. Check 2 = train <=2021,
evaluate 2022 (its current split). Phase 2's Check 2 evaluated on 2023-24;
that is NOT reproduced here (test seasons).

Integrity gates (script exits 1 if any fails):
  * the clean in-memory rebuild must equal data/processed/game_features.parquet
    restricted to <=2022 on every feature column (label column not read);
  * a second clean build at the end must equal the first (no state leaked
    between conditions via the memoized pbp);
  * Check 1 must PASS the clean condition and FAIL every sabotage condition
    (``assert_check1_catches_every_sabotage``), so a gap like the pre-Amendment-5
    EWM one cannot silently reopen. Checked after the JSON is written, so a
    failing run still leaves its evidence on disk.

Writes audit/out/negative_control.json. Deterministic: label-shuffle seed 0,
20 shuffles, lbfgs LogisticRegression; no timestamps in the output.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import data_ingest, evaluate, features as F, leakage_checks as LC  # noqa: E402

OUT = ROOT / "audit" / "out" / "negative_control.json"
MAX_SEASON = 2022                      # validation season; nothing later is opened
START_SEASON = F.DEFAULT_START         # 2002
SINGLE_METRIC = "qb_epa_per_db"
SHUFFLE_SEED = 0
N_SHUFFLES = 20
PHASE2_COMMIT = "16cb45a"

_REAL_ROLL_LEFT = F.roll_left
_REAL_ROLL_EWM = F.roll_ewm


# --------------------------------------------------------------------------- #
# Sabotaged copies of the two rolling functions (bodies copied from
# src/features.py; only the current-row exclusion is changed)
# --------------------------------------------------------------------------- #
def make_roll_left(sabotage_cols: set[str] | None):
    """Copy of ``F.roll_left``. For columns in ``sabotage_cols`` (None = every
    column) the window uses closed='right', i.e. it INCLUDES the current row.
    Every other column keeps closed='left'."""
    def roll_left(df, col, window, by, agg="mean"):
        closed = "right" if (sabotage_cols is None or col in sabotage_cols) else "left"
        grp = df.groupby(by, sort=False)[col]
        rolled = grp.rolling(window=window, min_periods=1, closed=closed).agg(agg)
        rolled = rolled.reset_index(level=list(range(len(by))), drop=True)
        return rolled.reindex(df.index)
    return roll_left


def roll_ewm_unshifted(df, col, by, halflife):
    """Copy of ``F.roll_ewm`` WITHOUT the shift(1): current row included."""
    grp = df.groupby(by, sort=False)[col]
    return grp.transform(lambda s: s.ewm(halflife=halflife, min_periods=1).mean())


CONDITIONS = {
    "clean": {
        "roll_left": None, "roll_ewm": None,
        "sabotaged": [],
        "description": "unmodified pipeline",
    },
    "single_metric_sd": {
        "roll_left": make_roll_left({SINGLE_METRIC}), "roll_ewm": None,
        "sabotaged": [f"{SINGLE_METRIC}_sd", f"{SINGLE_METRIC}_sd_n",
                      f"home_{SINGLE_METRIC}_shrunk", f"away_{SINGLE_METRIC}_shrunk",
                      "mu_qb_epa_vs_def", "mu_def_vs_qb_epa", "mu_money_down_edge"],
        "description": f"roll_left closed='right' for {SINGLE_METRIC} only",
    },
    "phase2_roll_left": {
        "roll_left": make_roll_left(None), "roll_ewm": None,
        "sabotaged": ["all 11 {m}_sd / {m}_sd_n / {side}_{m}_shrunk",
                      "home_games_played_sd", "away_games_played_sd", "all 6 mu_* diffs"],
        "description": "roll_left closed='right' for every call (Phase 2 reconstruction)",
    },
    "ewm_only": {
        "roll_left": None, "roll_ewm": roll_ewm_unshifted,
        "sabotaged": ["all 22 {side}_{m}_ewm"],
        "description": "roll_ewm without shift(1); roll_left untouched",
    },
    "all_rolling": {
        "roll_left": make_roll_left(None), "roll_ewm": roll_ewm_unshifted,
        "sabotaged": ["everything in phase2_roll_left", "all 22 {side}_{m}_ewm"],
        "description": "every rolling feature includes the current row",
    },
}


# --------------------------------------------------------------------------- #
# Data access restricted to <= MAX_SEASON
# --------------------------------------------------------------------------- #
def load_schedules_trainval(raw_dir: Path = evaluate.RAW_DIR) -> pd.DataFrame:
    paths = [raw_dir / f"schedules_{y}.parquet" for y in range(START_SEASON, MAX_SEASON + 1)]
    df = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    assert df["season"].max() <= MAX_SEASON
    return df


_PBP_CACHE: dict = {}
_REAL_GET_PBP = data_ingest.get_pbp


def get_pbp_memo(start, end, force=False):
    assert (start, end) == (START_SEASON, MAX_SEASON), (start, end)
    if "pbp" not in _PBP_CACHE:
        _PBP_CACHE["pbp"] = _REAL_GET_PBP(start, end)
    return _PBP_CACHE["pbp"]


def build(cond: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (frame, tgl) for seasons 2002-2022 with the condition's patches.
    cache=False: build_game_features writes nothing to disk."""
    captured = {}
    real_tgl = F.build_team_game_log

    def capture_tgl(pbp, game_index):
        captured["tgl"] = real_tgl(pbp, game_index)
        return captured["tgl"]

    with mock.patch.object(evaluate, "load_schedules", load_schedules_trainval), \
         mock.patch.object(F.data_ingest, "get_pbp", get_pbp_memo), \
         mock.patch.object(F, "build_team_game_log", capture_tgl), \
         mock.patch.object(F, "roll_left", cond["roll_left"] or _REAL_ROLL_LEFT), \
         mock.patch.object(F, "roll_ewm", cond["roll_ewm"] or _REAL_ROLL_EWM):
        frame = F.build_game_features(START_SEASON, MAX_SEASON, cache=False)
    assert frame["season"].max() <= MAX_SEASON
    return frame, captured["tgl"]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def phase2_feature_columns() -> list[str]:
    ewm = {f"{s}_{m}_ewm" for s in ("home", "away") for m in F.RAW_METRICS}
    return [c for c in F.feature_columns() if c not in ewm and c not in F.ELO_FEATURES]


def run_check1(frame, tgl, cond) -> dict:
    with mock.patch.object(F, "roll_left", cond["roll_left"] or _REAL_ROLL_LEFT), \
         mock.patch.object(F, "roll_ewm", cond["roll_ewm"] or _REAL_ROLL_EWM):
        try:
            res = LC.check_feature_timestamps(frame, tgl)
            return {"verdict": "PASS", "caught": False, **res}
        except AssertionError as exc:
            msg = str(exc)
            print(f"CHECK 1  FAIL: {msg}")
            out = {"verdict": "FAIL", "caught": True, "message": msg}
            m = re.search(r"(\{.*\})", msg)
            if m and "oracle" in msg:
                per = ast.literal_eval(m.group(1))
                out["failed_stage"] = ("prior-only oracle on {m}_ewm" if "EWM" in msg
                                       else "prior-only oracle on {m}_sd")
                out["n_metrics_mismatched"] = len(per)
                out["mismatched_rows_per_metric"] = per
            elif "tripwire" in msg:
                out["failed_stage"] = "tripwire"
            else:
                out["failed_stage"] = "other"
            return out


def run_check2(frame, schedules, cols_fn) -> dict:
    with mock.patch.object(F, "feature_columns", cols_fn):
        r = LC.label_shuffle_test(frame, schedules, n_shuffles=N_SHUFFLES, seed=SHUFFLE_SEED)
        n_features = len(F.feature_columns())
    keys = ["shuffled_mean", "shuffled_std", "shuffled_min", "shuffled_max",
            "majority_class_rate", "home_baseline_accuracy", "true_label_contrast_acc"]
    out = {k: round(r[k], 4) for k in keys}
    out.update(verdict=r["verdict"], pass_=r["pass"], caught=not r["pass"],
               n_features=n_features, n_train=r["n_train"], n_val=r["n_val"],
               contrast_exceeds_0_70=r["true_label_contrast_acc"] > 0.70)
    out["pass"] = out.pop("pass_")
    return out


# --------------------------------------------------------------------------- #
# Integrity gates
# --------------------------------------------------------------------------- #
def check1_gate(results: dict) -> dict:
    """Check 1 must pass the clean pipeline and catch every sabotage condition."""
    verdicts = {n: r["check1_timestamp"]["verdict"] for n, r in results.items()}
    missed = [n for n, v in verdicts.items() if n != "clean" and v != "FAIL"]
    clean_ok = verdicts.get("clean") == "PASS"
    return {"pass": clean_ok and not missed, "clean_passes": clean_ok,
            "sabotage_conditions_not_caught": missed, "verdicts": verdicts}


def assert_check1_catches_every_sabotage(gate: dict) -> None:
    assert gate["clean_passes"], f"Check 1 FAILED the clean pipeline: {gate['verdicts']}"
    assert not gate["sabotage_conditions_not_caught"], (
        "Check 1 PASSED sabotage condition(s) "
        f"{gate['sabotage_conditions_not_caught']}: a current-row leak went undetected. "
        "Add an independent oracle for the affected features before relying on Check 1."
    )
def frames_equal(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> tuple[bool, dict]:
    a = a.sort_values("game_id").reset_index(drop=True)
    b = b.sort_values("game_id").reset_index(drop=True)
    if len(a) != len(b) or not (a["game_id"].to_numpy() == b["game_id"].to_numpy()).all():
        return False, {"row_mismatch": [len(a), len(b)]}
    bad = {}
    for c in cols:
        x, y = a[c].to_numpy(dtype="float64"), b[c].to_numpy(dtype="float64")
        n = int((~np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)).sum())
        if n:
            bad[c] = n
    return not bad, bad


def canonical_trainval(cols: list[str]) -> pd.DataFrame:
    """Canonical game_features.parquet, feature columns only (home_win is never
    read), restricted to <= MAX_SEASON."""
    df = pd.read_parquet(F.PROC_DIR / "game_features.parquet",
                         columns=["game_id", "season"] + cols)
    return df[df["season"] <= MAX_SEASON]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


# --------------------------------------------------------------------------- #
def main() -> int:
    schedules = load_schedules_trainval()
    cols67 = F.feature_columns()
    cols42 = phase2_feature_columns()
    assert len(cols67) == 67 and len(cols42) == 42, (len(cols67), len(cols42))

    results, gates = {}, {}
    clean_frame = None
    for name, cond in CONDITIONS.items():
        print(f"\n{'=' * 78}\nCONDITION: {name} -- {cond['description']}\n{'=' * 78}")
        frame, tgl = build(cond)
        if name == "clean":
            clean_frame = frame
            ok, bad = frames_equal(frame, canonical_trainval(cols67), cols67)
            gates["clean_rebuild_equals_canonical_le_2022"] = {"pass": ok, "mismatches": bad,
                                                               "n_rows": len(frame)}
            print(f"\nGATE clean rebuild == canonical (<= {MAX_SEASON}): {ok} {bad or ''}")
        else:
            changed = [c for c in cols67 if not frames_equal(frame, clean_frame, [c])[0]]
        c1 = run_check1(frame, tgl, cond)
        c2_67 = run_check2(frame, schedules, lambda: list(cols67))
        c2_42 = run_check2(frame, schedules, lambda: list(cols42))
        results[name] = {
            "description": cond["description"],
            "sabotaged_features": cond["sabotaged"],
            "model_columns_changed_vs_clean": [] if name == "clean" else changed,
            "check1_timestamp": c1,
            "check2_label_shuffle_current_67": c2_67,
            "check2_label_shuffle_phase2_42": c2_42,
        }

    print(f"\n{'=' * 78}\nDETERMINISM: second clean build\n{'=' * 78}")
    frame2, _ = build(CONDITIONS["clean"])
    ok2, bad2 = frames_equal(frame2, clean_frame, cols67)
    gates["second_clean_build_identical"] = {"pass": ok2, "mismatches": bad2}
    print(f"GATE second clean build identical: {ok2} {bad2 or ''}")
    gates["check1_catches_every_sabotage"] = check1_gate(results)
    print(f"GATE Check 1 passes clean and fails every sabotage: "
          f"{gates['check1_catches_every_sabotage']['pass']}")

    out = {
        "experiment": "negative control for leakage Check 1 (timestamp/oracle) and Check 2 (label shuffle)",
        "git_head": git("rev-parse", "HEAD"),
        "src_dirty": bool(git("status", "--porcelain", "--", "src")),
        "sha256": {p: sha256(ROOT / p) for p in ("src/features.py", "src/leakage_checks.py")},
        "phase2_reference_commit": PHASE2_COMMIT,
        "sabotage_method": ("in-memory unittest.mock.patch.object of src.features.roll_left / "
                            "roll_ewm with sabotaged copies; src/ unmodified"),
        "seasons": {
            "features_built_from": [START_SEASON, MAX_SEASON],
            "check1_rows": f"all games {START_SEASON}-{MAX_SEASON}",
            "check2_train": f"{START_SEASON}-{LC.TW.TRAIN_MAX_SEASON}",
            "check2_eval": LC.TW.VAL_SEASON,
            "not_reproduced": ("Phase 2 Check 2 (16cb45a) evaluated on 2023-2024; current Check 2 and "
                               "this script use 2022 only. No 2023+ file is opened."),
        },
        "seeds": {"label_shuffle_rng": SHUFFLE_SEED, "n_shuffles": N_SHUFFLES,
                  "classifier": "LogisticRegression(max_iter=1000, class_weight='balanced'), lbfgs"},
        "single_metric": SINGLE_METRIC,
        "feature_sets": {"current_67": cols67, "phase2_42": cols42},
        "integrity_gates": gates,
        "conditions": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n")

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"{'condition':<18} {'check1':<6} {'c2(67) mean':>11} {'verdict':<8} "
          f"{'c2(42) mean':>11} {'verdict':<8} {'contrast67':>10} {'contrast42':>10}")
    for name, r in results.items():
        a, b = r["check2_label_shuffle_current_67"], r["check2_label_shuffle_phase2_42"]
        print(f"{name:<18} {r['check1_timestamp']['verdict']:<6} {a['shuffled_mean']:>11.4f} "
              f"{a['verdict'].split(' ')[0]:<8} {b['shuffled_mean']:>11.4f} "
              f"{b['verdict'].split(' ')[0]:<8} {a['true_label_contrast_acc']:>10.4f} "
              f"{b['true_label_contrast_acc']:>10.4f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    assert_check1_catches_every_sabotage(gates["check1_catches_every_sabotage"])
    return 0 if all(g["pass"] for g in gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
