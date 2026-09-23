"""Phase 9 Part B -- nflreadpy vs nfl_data_py equivalence test (the migration gate).

Stages (each writes ONLY under ``data/phase9_scratch/``; ``data/raw/`` and
``data/processed/`` are read-only here):

1. ``pull``     -- every season the pipeline uses, through nflreadpy, converted
                   to pandas at the boundary (``src.nflreadpy_boundary``), pbp
                   float64 -> float32 exactly as ``import_pbp_data(downcast=True)``.
                   pbp 2002-2025; schedules 2002-2026 (the canonical cache holds
                   a 2026 schedule file too, and ``evaluate.load_schedules`` globs
                   every ``schedules_*.parquet``, so the rebuild must see the same
                   set of seasons).
2. ``raw``      -- cached raw vs nflreadpy raw: per-season row counts, column-set
                   differences, dtypes and values of the columns the pipeline
                   reads, plus a per-column difference count for every other
                   shared column. Every difference is written to
                   ``raw_differences.csv`` (not summarized away).
3. ``rebuild``  -- runs the UNCHANGED ``src.features.build_game_features`` twice,
                   redirected to scratch by patching module-level paths only
                   (no edit to features.py):
                     control   : canonical data/raw/  -> scratch/proc_control/
                     nflreadpy : scratch raw          -> scratch/proc_nflreadpy/
4. ``features`` -- compares both rebuilds to ``data/processed/game_features.parquet``
                   (and team_game_log.parquet): rows, columns, order, values.

Tolerance: EXACT. The target is bitwise-equal values with identical NaN
positions (``np.array_equal(..., equal_nan=True)`` on float columns, ``==`` on
the rest). Rationale: the pbp source files are the same upstream release
assets for both libraries, both paths hand the pipeline float32 values, and
the pipeline is deterministic (same row order in -> same arithmetic order), so
any non-zero difference is a real difference, not float noise. The maximum
absolute difference is reported alongside so a non-exact result can be judged.

Run::

    python -m scripts.phase9_equivalence            # all stages
    python -m scripts.phase9_equivalence --stages raw features
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANON_RAW = PROJECT_ROOT / "data" / "raw"
CANON_PROC = PROJECT_ROOT / "data" / "processed"
SCRATCH = PROJECT_ROOT / "data" / "phase9_scratch"
NRP_RAW = SCRATCH / "raw_nflreadpy"
PROC_CONTROL = SCRATCH / "proc_control"
PROC_NRP = SCRATCH / "proc_nflreadpy"

PBP_SEASONS = range(2002, 2026)
SCHED_SEASONS = range(2002, 2027)

PBP_KEY = ["game_id", "play_id"]
PBP_READ = ["game_id", "posteam", "defteam", "pass", "rush", "qb_dropback", "sack",
            "qb_scramble", "epa", "qb_epa", "success", "cpoe", "interception",
            "air_yards", "down", "qb_hit"]
SCHED_READ = ["game_id", "season", "game_type", "week", "gameday", "gametime",
              "home_team", "away_team", "home_score", "away_score", "home_rest",
              "away_rest", "div_game", "roof", "location", "stadium_id",
              # benchmark-only readers (Vegas scripts), never features:
              "spread_line", "total_line", "home_moneyline", "away_moneyline", "result"]


# --------------------------------------------------------------------------- #
# 1. pull
# --------------------------------------------------------------------------- #
def stage_pull(force: bool) -> None:
    import nflreadpy as nfl
    from nflreadpy.config import update_config

    from src.nflreadpy_boundary import pbp_to_raw_contract, schedules_to_raw_contract

    update_config(cache_mode="off")
    NRP_RAW.mkdir(parents=True, exist_ok=True)
    print(f"nflreadpy {version('nflreadpy')} -> {NRP_RAW}")

    sched_all = schedules_to_raw_contract(nfl.load_schedules(True))
    for yr in SCHED_SEASONS:
        path = NRP_RAW / f"schedules_{yr}.parquet"
        s = sched_all[sched_all["season"].eq(yr)].reset_index(drop=True)
        s.to_parquet(path, index=False)
        print(f"  schedules {yr}: {len(s):>4} rows")

    for yr in PBP_SEASONS:
        path = NRP_RAW / f"pbp_{yr}.parquet"
        if path.exists() and not force:
            print(f"  pbp {yr}: exists, skipped")
            continue
        p = pbp_to_raw_contract(nfl.load_pbp([yr]))
        p.to_parquet(path, index=False)
        print(f"  pbp {yr}: {len(p):>6} rows, {p.shape[1]} cols")


# --------------------------------------------------------------------------- #
# 2. raw comparison
# --------------------------------------------------------------------------- #
def _eq_mask(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Elementwise equality with NaN/None == NaN/None. Compares values, not dtype."""
    na_a, na_b = a.isna().to_numpy(), b.isna().to_numpy()
    both_na = na_a & na_b
    try:
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(dtype="float64", na_value=np.nan)
            bv = b.to_numpy(dtype="float64", na_value=np.nan)
            eq = av == bv
        else:
            eq = (a.astype(object).to_numpy() == b.astype(object).to_numpy())
    except (TypeError, ValueError):
        eq = (a.astype(str).to_numpy() == b.astype(str).to_numpy())
    return np.asarray(eq, dtype=bool) | both_na


def _max_abs(a: pd.Series, b: pd.Series) -> float | None:
    if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
        return None
    d = np.abs(a.to_numpy("float64", na_value=np.nan) - b.to_numpy("float64", na_value=np.nan))
    return float(np.nanmax(d)) if np.isfinite(d).any() else None


def _compare_frames(kind: str, yr: int, old: pd.DataFrame, new: pd.DataFrame,
                    key: list[str], read_cols: list[str], rows: list[dict]) -> dict:
    summ = {"kind": kind, "season": yr, "rows_cached": len(old), "rows_nflreadpy": len(new)}
    only_old = sorted(set(old.columns) - set(new.columns))
    only_new = sorted(set(new.columns) - set(old.columns))
    summ["cols_only_cached"] = only_old
    summ["cols_only_nflreadpy"] = only_new
    for c in only_old:
        rows.append({"kind": kind, "season": yr, "column": c, "issue": "column only in cached",
                     "in_pipeline_read_set": c in read_cols})
    for c in only_new:
        rows.append({"kind": kind, "season": yr, "column": c, "issue": "column only in nflreadpy",
                     "in_pipeline_read_set": c in read_cols})

    dup_old = int(old.duplicated(key).sum())
    dup_new = int(new.duplicated(key).sum())
    summ["dup_keys_cached"], summ["dup_keys_nflreadpy"] = dup_old, dup_new
    if dup_old or dup_new:
        rows.append({"kind": kind, "season": yr, "column": "+".join(key),
                     "issue": f"duplicate keys cached={dup_old} nflreadpy={dup_new}",
                     "in_pipeline_read_set": True})
    o = old.drop_duplicates(key).set_index(key)
    n = new.drop_duplicates(key).set_index(key)
    ko, kn = set(o.index), set(n.index)
    summ["keys_only_cached"] = len(ko - kn)
    summ["keys_only_nflreadpy"] = len(kn - ko)
    if ko != kn:
        rows.append({"kind": kind, "season": yr, "column": "+".join(key),
                     "issue": f"row keys differ: only cached={len(ko - kn)}, only nflreadpy={len(kn - ko)}",
                     "example": str(sorted(ko ^ kn, key=str)[:5]), "in_pipeline_read_set": True})
    common = o.index.intersection(n.index)
    o, n = o.loc[common], n.loc[common]

    # row ORDER matters to the pipeline only through sort stability; record it
    order_same = list(old.drop_duplicates(key)[key].itertuples(index=False)) == \
        list(new.drop_duplicates(key)[key].itertuples(index=False))
    summ["row_order_identical"] = order_same
    if not order_same:
        rows.append({"kind": kind, "season": yr, "column": "(row order)",
                     "issue": "row order differs", "in_pipeline_read_set": True})

    shared = [c for c in old.columns if c in new.columns and c not in key]
    n_read_diff, n_other_diff = 0, 0
    for c in shared:
        in_read = c in read_cols
        dt_o, dt_n = str(o[c].dtype), str(n[c].dtype)
        if dt_o != dt_n and in_read:
            rows.append({"kind": kind, "season": yr, "column": c, "issue": "dtype",
                         "cached": dt_o, "nflreadpy": dt_n, "in_pipeline_read_set": True})
        eq = _eq_mask(o[c], n[c])
        ndiff = int((~eq).sum())
        if ndiff:
            if in_read:
                n_read_diff += 1
            else:
                n_other_diff += 1
            bad = np.flatnonzero(~eq)[:3]
            rows.append({"kind": kind, "season": yr, "column": c, "issue": "values",
                         "n_diff": ndiff, "n_common_rows": len(common),
                         "max_abs_diff": _max_abs(o[c], n[c]),
                         "cached": dt_o, "nflreadpy": dt_n,
                         "example": "; ".join(f"{common[i]}: {o[c].iloc[i]!r} vs {n[c].iloc[i]!r}"
                                              for i in bad),
                         "in_pipeline_read_set": in_read})
    summ["read_cols_with_value_diffs"] = n_read_diff
    summ["other_cols_with_value_diffs"] = n_other_diff
    return summ


def stage_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    summaries = []
    for yr in SCHED_SEASONS:
        old = pd.read_parquet(CANON_RAW / f"schedules_{yr}.parquet")
        new = pd.read_parquet(NRP_RAW / f"schedules_{yr}.parquet")
        summaries.append(_compare_frames("schedules", yr, old, new, ["game_id"], SCHED_READ, rows))
    for yr in PBP_SEASONS:
        old = pd.read_parquet(CANON_RAW / f"pbp_{yr}.parquet")
        new = pd.read_parquet(NRP_RAW / f"pbp_{yr}.parquet")
        summaries.append(_compare_frames("pbp", yr, old, new, PBP_KEY, PBP_READ + ["play_id"], rows))
        print(f"  pbp {yr} compared", file=sys.stderr)
    summ = pd.DataFrame(summaries)
    diffs = pd.DataFrame(rows)
    summ.to_csv(SCRATCH / "raw_summary.csv", index=False)
    diffs.to_csv(SCRATCH / "raw_differences.csv", index=False)
    return summ, diffs


# --------------------------------------------------------------------------- #
# 3. rebuild with the unchanged pipeline, redirected to scratch
# --------------------------------------------------------------------------- #
def _rebuild(raw_dir: Path, proc_dir: Path) -> None:
    from src import data_ingest, evaluate
    from src import features as F

    missing = [y for y in PBP_SEASONS if not (raw_dir / f"pbp_{y}.parquet").exists()]
    assert not missing, f"{raw_dir}: missing pbp {missing} -- refusing (get_pbp would download)"
    assert proc_dir.resolve() != CANON_PROC.resolve(), "refusing to write canonical processed/"
    proc_dir.mkdir(parents=True, exist_ok=True)

    orig = (data_ingest.RAW_DIR, evaluate.load_schedules, F.PROC_DIR)
    try:
        data_ingest.RAW_DIR = raw_dir
        evaluate.load_schedules = functools.partial(orig[1], raw_dir=raw_dir)
        F.PROC_DIR = proc_dir
        F.build_game_features()   # defaults: 2002..2025, k=4 -- unchanged
    finally:
        data_ingest.RAW_DIR, evaluate.load_schedules, F.PROC_DIR = orig


def stage_rebuild() -> None:
    print("control: canonical raw -> proc_control")
    _rebuild(CANON_RAW, PROC_CONTROL)
    print("nflreadpy: scratch raw -> proc_nflreadpy")
    _rebuild(NRP_RAW, PROC_NRP)


# --------------------------------------------------------------------------- #
# 4. feature comparison
# --------------------------------------------------------------------------- #
def compare_processed(ref: pd.DataFrame, new: pd.DataFrame, label: str) -> dict:
    out = {"label": label, "shape_ref": ref.shape, "shape_new": new.shape,
           "columns_identical_and_ordered": list(ref.columns) == list(new.columns)}
    if not out["columns_identical_and_ordered"]:
        out["cols_only_ref"] = sorted(set(ref.columns) - set(new.columns))
        out["cols_only_new"] = sorted(set(new.columns) - set(ref.columns))
    if ref.shape != new.shape:
        return out
    key = "game_id" if "team" not in ref.columns else None
    if key:
        out["row_keys_identical_and_ordered"] = bool((ref[key].to_numpy() == new[key].to_numpy()).all())
    col_diffs = {}
    for c in ref.columns:
        a, b = ref[c], new[c]
        if str(a.dtype) != str(b.dtype):
            col_diffs[c] = {"dtype": (str(a.dtype), str(b.dtype))}
            continue
        if pd.api.types.is_float_dtype(a):
            same = np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)
        else:
            same = bool(_eq_mask(a, b).all())
        if not same:
            col_diffs[c] = {"n_diff": int((~_eq_mask(a, b)).sum()), "max_abs": _max_abs(a, b)}
    out["columns_with_differences"] = col_diffs
    out["EXACT_MATCH"] = (not col_diffs) and out["columns_identical_and_ordered"] and \
        out.get("row_keys_identical_and_ordered", True)
    return out


def stage_features() -> list[dict]:
    results = []
    for name in ("game_features.parquet", "team_game_log.parquet"):
        ref = pd.read_parquet(CANON_PROC / name)
        for label, d in (("control", PROC_CONTROL), ("nflreadpy", PROC_NRP)):
            results.append({"file": name, **compare_processed(ref, pd.read_parquet(d / name), label)})
    man_ref = json.loads((CANON_PROC / "feature_manifest.json").read_text())
    for label, d in (("control", PROC_CONTROL), ("nflreadpy", PROC_NRP)):
        m = json.loads((d / "feature_manifest.json").read_text())
        man_ref.pop("generated_at", None)
        m.pop("generated_at", None)
        results.append({"file": "feature_manifest.json (minus generated_at)", "label": label,
                        "EXACT_MATCH": m == man_ref})
    (SCRATCH / "features_comparison.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["pull", "raw", "rebuild", "features"],
                    choices=["pull", "raw", "rebuild", "features"])
    ap.add_argument("--force", action="store_true", help="re-pull pbp already in scratch")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 120)

    if "pull" in args.stages:
        stage_pull(args.force)
    if "raw" in args.stages:
        summ, diffs = stage_raw()
        print("\n== RAW: per-season summary ==")
        print(summ.drop(columns=["cols_only_cached", "cols_only_nflreadpy"]).to_string(index=False))
        print("\n== RAW: column-set differences (per season) ==")
        for _, r in summ.iterrows():
            if r["cols_only_cached"] or r["cols_only_nflreadpy"]:
                print(f"  {r['kind']} {r['season']}: only cached={len(r['cols_only_cached'])} "
                      f"{r['cols_only_cached']} | only nflreadpy={r['cols_only_nflreadpy']}")
        if len(diffs):
            print("\n== RAW: every difference in a pipeline-read column ==")
            rd = diffs[diffs["in_pipeline_read_set"].eq(True) & ~diffs["issue"].str.startswith("column only")]
            print(rd.to_string(index=False) if len(rd) else "  (none)")
            other = diffs[diffs["in_pipeline_read_set"].eq(False) & diffs["issue"].eq("values")]
            print(f"\n== RAW: value diffs in columns the pipeline does NOT read: {len(other)} "
                  f"(column, season) pairs -- full list in raw_differences.csv ==")
    if "rebuild" in args.stages:
        stage_rebuild()
    if "features" in args.stages:
        print("\n== FEATURES ==")
        for r in stage_features():
            print(json.dumps(r, default=str))


if __name__ == "__main__":
    main()
