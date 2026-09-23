"""Phase 8, Part A: develop the OOF walk-forward sigmoid calibration on the
2022 VALIDATION season only (target T=2022, OOF seasons 2017-2021).

Reads seasons <= 2022 only (``calibration.load_frame(2022)``): 2023-2024 and
the 2025 holdout are never loaded. No tuning, no feature changes -- models
are clones of the saved pipelines.

Pre-set decision rule, applied mechanically per model:
    adopt calibration  iff  calibrated 2022 log loss <= raw 2022 log loss
Stop condition: the sigmoid is monotonic, so ROC-AUC must not move
materially (|delta| > 0.001 exits non-zero).

Consistency check: the "raw" 2022 model here is trained on <= 2021, exactly
like each saved model, so its 2022 metrics must equal the ``val_metrics``
stored in that model's joblib.

Outputs:
    docs/figures/phase8_reliability_2022.png
    data/processed/phase8_dev_2022.json  (decisions + calibrator params)

Usage::

    python -m scripts.phase8_calibration_dev
"""

from __future__ import annotations

import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import calibration as C, train_winner as TW

TARGET = 2022
AUC_TOLERANCE = 0.001
SMALL_BIN = 10
FIG_PATH = TW.PROJECT_ROOT / "docs" / "figures" / "phase8_reliability_2022.png"
OUT_JSON = TW.PROC_DIR / "phase8_dev_2022.json"

# Reference categorical palette, slots 1-2 (dataviz skill references/palette.md).
RAW_COLOR, CAL_COLOR = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(reliability: dict) -> None:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (name, tables) in zip(axes.ravel(), reliability.items()):
        ax.set_facecolor(SURFACE)
        ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1,
                label="perfect calibration")
        for kind, color, marker in (("raw", RAW_COLOR, "o"), ("calibrated", CAL_COLOR, "s")):
            t = tables[kind].dropna()
            ax.plot(t["mean_pred"], t["observed"], color=color, lw=2, marker=marker,
                    ms=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3, label=kind)
            small = t[t["n"] < SMALL_BIN]  # hollow = too few games to read
            ax.plot(small["mean_pred"], small["observed"], ls="none", marker=marker, ms=8,
                    markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=2, zorder=4)
        ax.set_title(name, color=INK, fontsize=12, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
    for ax in axes[1]:
        ax.set_xlabel("mean predicted home-win probability (bin)", color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("observed home-win rate", color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.955), labelcolor=INK)
    fig.suptitle(f"Reliability on {TARGET} validation season, raw vs OOF-sigmoid-calibrated\n"
                 f"10 equal-width bins; empty bins omitted; hollow markers = bins with "
                 f"fewer than {SMALL_BIN} games", color=INK, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_PATH, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    frame = C.load_frame(TARGET)
    assert frame["season"].max() == TARGET
    pipes, feature_cols = C.load_frozen_models()
    y = frame.loc[frame["season"] == TARGET, "home_win"].to_numpy()
    print(f"target season {TARGET}: n={len(y)}; OOF seasons "
          f"{TARGET - C.N_OOF_SEASONS}-{TARGET - 1}; frame seasons "
          f"{frame.season.min()}-{frame.season.max()}\n")

    rows, cal_params, reliability, decisions, per_game = [], {}, {}, {}, {}
    ok = True
    for name, unfitted in pipes.items():
        preds, cal, oof = C.calibrate_for_season(unfitted, frame, TARGET, feature_cols)
        assert (preds["game_id"].to_numpy() == frame.loc[frame.season == TARGET, "game_id"].to_numpy()).all()
        raw_m = C.metrics(y, preds["proba_raw"].to_numpy())
        cal_m = C.metrics(y, preds["proba_cal"].to_numpy())

        saved = joblib.load(TW.MODELS_DIR / C.SAVED_MODELS[name])["val_metrics"]
        repro = all(abs(raw_m[k] - saved[k]) < 1e-9 for k in ("accuracy", "log_loss", "roc_auc"))
        ok &= repro

        d_auc = cal_m["roc_auc"] - raw_m["roc_auc"]
        auc_ok = abs(d_auc) <= AUC_TOLERANCE
        ok &= auc_ok
        adopt = cal_m["log_loss"] <= raw_m["log_loss"]
        decisions[name] = "calibrated" if adopt else "raw"
        cal_params[name] = {"slope": cal.slope, "intercept": cal.intercept, "n_fit": cal.n_fit,
                            "oof_seasons": sorted(oof["season"].unique().tolist()),
                            "oof_raw_log_loss": C.log_loss(oof["home_win"], oof["proba_raw"])}
        per_game[name] = preds
        reliability[name] = {"raw": C.reliability_table(y, preds["proba_raw"].to_numpy()),
                             "calibrated": C.reliability_table(y, preds["proba_cal"].to_numpy())}
        for kind, m in (("raw", raw_m), ("calibrated", cal_m)):
            rows.append({"model": name, "version": kind, **m})
        print(f"{name:<13} calibrator slope={cal.slope:.4f} intercept={cal.intercept:+.4f} "
              f"(n_fit={cal.n_fit})  | raw reproduces saved val_metrics: {repro} "
              f"| dAUC={d_auc:+.6f} ok={auc_ok} | decision: {decisions[name].upper()}")

    tbl = pd.DataFrame(rows).set_index(["model", "version"])
    print("\n=== 2022 validation: raw vs calibrated ===")
    print(tbl.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nlog-loss change (calibrated - raw):")
    for name in pipes:
        dll = tbl.loc[(name, "calibrated"), "log_loss"] - tbl.loc[(name, "raw"), "log_loss"]
        dacc = tbl.loc[(name, "calibrated"), "accuracy"] - tbl.loc[(name, "raw"), "accuracy"]
        print(f"  {name:<13} dLogLoss={dll:+.4f}  dBrier="
              f"{tbl.loc[(name, 'calibrated'), 'brier_score'] - tbl.loc[(name, 'raw'), 'brier_score']:+.4f}"
              f"  dAcc={dacc:+.4f}  -> {decisions[name]}")

    print("\n=== Reliability tables (2022) ===")
    for name, t in reliability.items():
        j = t["raw"].merge(t["calibrated"], on="bin", suffixes=("_raw", "_cal"))
        print(f"\n{name}")
        print(j.to_string(index=False, float_format=lambda v: "-" if pd.isna(v) else f"{v:.3f}"))

    # Context for the pre-registered deployment rule (0.010 log-loss margin):
    # spread of paired per-game log-loss differences vs LR-tuned on 2022, using
    # each model's as-decided probabilities. Descriptive only.
    print("\n=== Context: paired per-game log-loss difference vs LR-tuned (2022, as-decided probs) ===")
    def as_decided(n):
        return per_game[n]["proba_cal" if decisions[n] == "calibrated" else "proba_raw"].to_numpy()
    def ll_vec(p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    base = ll_vec(as_decided("LR-tuned"))
    for n in pipes:
        if n == "LR-tuned":
            continue
        d = ll_vec(as_decided(n)) - base
        print(f"  {n:<13} mean diff={d.mean():+.4f}  SE={d.std(ddof=1) / np.sqrt(len(d)):.4f}")

    plot(reliability)
    OUT_JSON.write_text(json.dumps({"target_season": TARGET, "decisions": decisions,
                                    "calibrators": cal_params}, indent=2))
    print(f"\nwrote {FIG_PATH.relative_to(TW.PROJECT_ROOT)} and {OUT_JSON.relative_to(TW.PROJECT_ROOT)}")
    print(f"decisions: {decisions}")
    if not ok:
        print("\n!! STOP: a consistency check failed (saved val_metrics reproduction or AUC tolerance).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
