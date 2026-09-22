# Phase 7 Review Gate — NFL Win/Loss Predictor

**Date:** 2026-09-22
**SPEC reference:** Section 13, Phase 7 ("Compare results to Section 2 criteria; decide whether to proceed to Section 12").
**Scope of this session:** verification, one descriptive Vegas benchmark, and recording decisions. No retraining for model selection, no tuning, no feature changes, no calibration, no new data pulls, no Section 12 code.

---

## 0. Project in one paragraph (for a cold reader)

A scikit-learn classifier predicts whether the **home team wins** an NFL regular-season game, using only information available before kickoff: rolling quarterback and defensive EPA (Expected Points Added) from play-by-play data, shrunk toward prior-season rates early in a season, exponentially weighted recency versions of those stats, an Elo team rating, and context (rest, travel, divisional game, roof). All data comes free from `nfl_data_py` (nflverse). The project's main risk is **data leakage**, so every rolling feature excludes the current game (`closed='left'` / `merge_asof(allow_exact_matches=False)`), and every model passes a four-part leakage audit (SPEC 5.5). Splits are by season: train 2002–2021, validate 2022, test 2023–2024. Four models were carried to the final test: LogisticRegression untuned (C=1.0) and tuned (C=0.01), RandomForest (tuned = untuned hyperparameters), and HistGradientBoosting tuned. The test was a walk-forward backtest: train 2002–2022 → predict 2023; train 2002–2023 → predict 2024; pool the 544 predictions.

---

## 1. Part A — Verification of current state

| Check | Result |
|---|---|
| `git status` | clean, `main`, up to date with `origin/main` |
| HEAD | `ab153a6 Phase 6 complete. Look at document for pacing` (touched **only** `CLAUDE.md`) |
| Saved models | 6 files in `models/` (LR/RF/HGB × untuned/tuned), all 67 features, all `random_state=0`, all with `leakage_checks` metadata. LR × 2 carry `override_applied=True` (documented check-4 override); RF/HGB × 4 pass cleanly. |
| Phase 6 walk-forward code | **Not in the repo.** Lived only in a prior session's `/tmp` scratchpad (`walk_forward_test.py`, `mcnemar_test.py`). Console log survives at `/private/tmp/walk_forward_output.txt`. |
| Phase 6 per-game predictions | **Did not exist on disk.** Only aggregate console output + the CLAUDE.md summary. |
| Documented pooled accuracies | Match the surviving console log exactly (see below). |
| `nfl_data_py` | **0.3.3** (Python 3.11.13, pandas 1.5.3). `nflreadpy`/`polars` not installed. Note: nflverse deprecated `nfl_data_py` in Sept 2025 in favor of `nflreadpy` (polars-based, `load_*` functions). Nothing migrated this phase. |

### 1.1 Local data cache (read-only inspection)

| Dataset | Seasons cached | Notes |
|---|---|---|
| Play-by-play | 2002–2025 (24 files, none missing) | Pulled 2026-09-08 |
| Schedules | 2002–2026 (25 files) | Pulled 2026-09-08 |

**2025 is present and complete:** 272/272 REG games scored, weeks 1–18 (2025-09-04 → 2026-01-04), plus full playoffs; every completed REG game has play-by-play. `data/processed/game_features.parquet` **already contains 271 rows for 2025** (one tie dropped). No model has been fit or evaluated on 2025 (grep of `src/`, `scripts/`); the only 2025 touch is `evaluate.py`'s `__main__`, which prints the 2025 home-win rate — a label aggregate, not a model result.

**2026:** schedule only (272 REG games, 0 scored — pulled before the season opened). Caveat for Phase 9: `data_ingest.py` treats any cached file as final (`cache hit`), so `schedules_2026.parquet` will never refresh in-season without `--force`, and `--force` re-downloads **every** season.

### 1.2 Part B — Per-game predictions regenerated as a reproduction check

Because no per-game output existed, the Phase 6 protocol was re-implemented in `scripts/reproduce_walkforward_predictions.py` (models built via `sklearn.base.clone()` of each saved pipeline — identical hyperparameters, `random_state=0`, identical 67 features, identical season splits). The script refuses to write anything unless every number matches.

| Model | Regenerated | Recorded in CLAUDE.md | Match |
|---|---|---|---|
| LR-untuned | 342/544 = 0.6287 | 0.6287 | ✅ |
| LR-tuned | 350/544 = 0.6434 | 0.6434 | ✅ |
| RandomForest | 343/544 = 0.6305 | 0.6305 | ✅ |
| HGB-tuned | 342/544 = 0.6287 | 0.6287 | ✅ |
| Home baseline | 296/544 = 0.5441 | 0.544 | ✅ |

McNemar exact p-values also reproduced: LR-untuned vs LR-tuned 0.0963, LR-tuned vs RF 0.3240, LR-tuned vs HGB 0.3222; the other three pairs p = 1.0000. No pair p < 0.05. Per-season log loss / Brier / ROC-AUC (Section 3 below) also match the Phase 6 console log to 4 dp.

Minor correction to CLAUDE.md: it says "31–50 discordant games per pair"; the actual range is **18–50** (LR-untuned vs LR-tuned has 18).

Saved: `data/processed/walkforward_test_predictions.parquet` (544 rows: `game_id, season, week, home_win, train_max_season`, and `proba_*` / `pred_*` for each model).

---

## 2. SPEC Section 2 verdict (Definition of Done)

| SPEC 2 criterion | Level | Verdict | Evidence |
|---|---|---|---|
| Beats naive "always pick home" baseline by a non-trivial margin (+3–5 pts) | MUST | **MET** | Pooled 2023–24 (n=544): models 62.87–64.34% vs home baseline 54.41% → +8.5 to +9.9 pts. 95% CIs for the models (≈0.587–0.683) do not overlap the baseline's point estimate, and the margin exceeds any model-vs-model gap. |
| Evaluated only via season-based walk-forward (SPEC 7 / 6) | MUST | **MET** | Train ≤2022 → 2023; train ≤2023 → 2024. Reproduced exactly this phase. |
| Passes every SPEC 5.5 leakage check before reporting | MUST | **MET, with a documented override** | RF/HGB pass checks 3–4 cleanly. LR (both) fails check 4 (ablation increases accuracy); investigated via 12-fold train-only rotating holdout and found indistinguishable from chance (p = 0.254 and 0.6875). Override recorded in each LR model's metadata. Check 4's design is a known Phase-backlog item. |
| Predicted probabilities calibrated (SPEC 7.3) | SHOULD | **NOT MET — open** | No model has been wrapped in `CalibratedClassifierCV`. Scheduled for Phase 8. |
| Accuracy roughly mid-60s to high-60s | SHOULD | **MARGINAL — one model only** | Only LR-tuned (64.34%) arguably reaches the low edge of the range. LR-untuned 62.87%, HGB-tuned 62.87%, RF 63.05% are low-60s. Not rounded up. The only single number in the mid-to-high 60s is LR-tuned on 2024 alone (66.54%), which is not the headline. |
| **Stretch:** probabilities approach Vegas closing-moneyline implied probability | Bonus | **NOT MET** | Vegas vig-free implied prob beats every model on every metric (Section 3). The ROC-AUC gap (0.731 vs 0.670–0.683) is calibration-invariant, so calibration alone cannot close it. |

**Overall:** Phase 1's MUST criteria are met. Proceed to the next phases as recorded in Section 4.

---

## 3. Part C — Vegas benchmark, 2023–2024 (DESCRIPTIVE ONLY)

> **Use restriction.** Nothing in this section may drive any model, feature, hyperparameter, calibration, or deployment-model decision — in this phase or retroactively. It is a descriptive benchmark only. That restriction is what keeps it from being a second use of the 2023–2024 test set. Vegas lines are used strictly as a benchmark (SPEC 5.2 category 4 / 5.3), never as a feature.

Produced by `scripts/vegas_benchmark_2023_2024.py` from the saved per-game predictions and the cached schedules. Nothing was retrained.

### 3.1 What the line columns are

From the nflverse schedules data dictionary (`nflreadr/data-raw/dictionary_schedules.csv`):

- `spread_line` — "The spread line for the game. A positive number means the home team was favored by that many points… This lines up with the result column."
- `home_moneyline` / `away_moneyline` — "Odds for home [away] team to win the game."
- `total_line` — "The total line for the game."

**The documentation does not say whether these are opening, closing, or a snapshot taken at some other time**, and gives no timestamp or book. The upstream `nfldata` DATASETS.md is also silent on timing. They are commonly treated as approximately-closing consensus lines, but that is not documented, so this review calls them "the nflverse line" rather than "the closing line." (SPEC 3.1's description "Vegas opening/closing lines" overstates what the data contains — there is one line per game, not two.)

Coverage on the 544 test games: `spread_line`, both moneylines, and `total_line` are 100% non-null. Mean bookmaker overround 4.28% (range 3.64–4.90%).

### 3.2 Conventions

- **Ties:** excluded, same as every model evaluation (`evaluate.completed_regular_season`). 2023–2024 had **zero** REG ties, so all 544 games are included.
- **Vegas favorite:** side favored by `spread_line`. Pick'em fallback: lower moneyline; if still equal, unresolved and excluded. In practice **0** games had `spread_line == 0`, so the fallback never fired and nothing was excluded. (Spread favorite and moneyline favorite disagree in 1 game; 3 games have equal moneylines.)
- **Implied probability:** American odds → raw implied probability for each side, then normalized to sum to 1 (vig removed). The 3 games at exactly 0.5 count as away picks for the implied-prob accuracy row, which is why it differs from the favorite row by one game.
- **Model probabilities are UNCALIBRATED** raw `predict_proba`. Log-loss and Brier comparisons are provisional until Phase 8. ROC-AUC and accuracy are not affected by monotone calibration.

### 3.3 Side-by-side, pooled 2023+2024 (n = 544)

| | Accuracy | 95% CI (Wilson) | Log loss | Brier | ROC-AUC |
|---|---:|---|---:|---:|---:|
| Home baseline | 0.5441 | 0.502–0.586 | – | – | – |
| LR-untuned | 0.6287 | 0.587–0.668 | 0.6479 | 0.2282 | 0.6698 |
| LR-tuned | 0.6434 | 0.602–0.683 | 0.6434 | 0.2261 | 0.6774 |
| RandomForest | 0.6305 | 0.589–0.670 | 0.6380 | 0.2237 | 0.6834 |
| HGB-tuned | 0.6287 | 0.587–0.668 | 0.6418 | 0.2255 | 0.6789 |
| **Vegas favorite (spread)** | **0.6967** | 0.657–0.734 | – | – | – |
| **Vegas implied prob (vig-free ML)** | **0.6985** | 0.659–0.736 | **0.6074** | **0.2094** | **0.7306** |

Per season:

| | 2023 acc | 2023 LL | 2023 AUC | 2024 acc | 2024 LL | 2024 AUC |
|---|---:|---:|---:|---:|---:|---:|
| Home baseline | 0.5551 | – | – | 0.5331 | – | – |
| LR-untuned | 0.6176 | 0.6675 | 0.6328 | 0.6397 | 0.6282 | 0.7045 |
| LR-tuned | 0.6213 | 0.6606 | 0.6432 | 0.6654 | 0.6261 | 0.7099 |
| RandomForest | 0.6029 | 0.6574 | 0.6444 | 0.6581 | 0.6187 | 0.7233 |
| HGB-tuned | 0.6103 | 0.6639 | 0.6358 | 0.6471 | 0.6196 | 0.7222 |
| Vegas favorite | 0.6801 | – | – | 0.7132 | – | – |
| Vegas implied | 0.6838 | 0.6273 | 0.6945 | 0.7132 | 0.5875 | 0.7624 |

(Vegas favorites at 71.3% in 2024 is the market, not a model; SPEC 5.6's ~70% stop-flag applies to this project's models, none of which exceeded it. It does illustrate that a single season above 70% is possible for a well-informed predictor.)

**Reading:** the models sit ~5–7 accuracy points below the Vegas favorite in both seasons, and the gap between the models is much smaller than the gap to Vegas. This is consistent with SPEC 5.7 and 9.1 — the market is the realistic ceiling, not the bar.

### 3.4 Agreement analysis — where the model and the Vegas favorite pick different winners

When a model and the Vegas favorite disagree, exactly one of them is right, so the model's accuracy on those games is also an exact McNemar test of model vs. Vegas favorite (H0 = 50%).

| Model | Agree n | Acc. when agreeing | **Disagree n** | Model right | Vegas right | **Model acc. on disagreements** | 95% CI (Wilson) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| LR-untuned | 439 | 0.702 | 105 | 34 | 71 | **0.324** | 0.242–0.418 | 0.0004 |
| LR-tuned | 453 | 0.704 | 91 | 31 | 60 | **0.341** | 0.252–0.443 | 0.0031 |
| RandomForest | 444 | 0.700 | 100 | 32 | 68 | **0.320** | 0.237–0.417 | 0.0004 |
| HGB-tuned | 433 | 0.704 | 111 | 37 | 74 | **0.333** | 0.253–0.425 | 0.0006 |

By season (model right / disagreements): 2023 — LR-u 18/53, LR-t 14/44, RF 14/49, HGB 19/57; 2024 — LR-u 16/52, LR-t 17/47, RF 18/51, HGB 18/54.

**Plain statement of what this does and does not show.** The disagreement subsets are small (n = 91–111 pooled, 44–57 per season), so the exact figure (32–34%) is imprecise — the CIs are roughly ±9 points wide. But every CI's upper bound is below 45%, the direction is identical in both seasons for all four models, and all four McNemar tests reject "no difference" at p < 0.01. So the narrow claim *is* supported: **on 2023–2024, when these models disagreed with the market, the market was usually right.** What it does not establish is the exact size of that deficit, or anything about calibrated models, a different feature set, or other seasons. For any future betting use, the disagreement games are the only games that matter, and on this evidence they are currently the models' weakest subset, not a source of edge.

**Caveat on line timing.** The nflverse line is untimestamped and may be at or near closing (Section 3.1), so it may embed kickoff-time information — injuries, weather, late news — that the model structurally lacks. This does **not** soften the finding above. It is the reason Phase 10 must snapshot lines at prediction-log time: that gives the fairer comparison, against the line actually available when the prediction is made.

---

## 4. Part D — Decisions recorded (made by the project owner; recorded, not re-derived)

### D1. SPEC Section 2 verdict
As tabulated in Section 2 above: baseline-margin MUST **met**; walk-forward MUST **met**; leakage-audit MUST **met with documented LR override**; calibration SHOULD **not met (open, Phase 8)**; mid-to-high-60s SHOULD **marginal, LR-tuned only**; Vegas stretch goal **not met**.

### D2. Section 12 revised scope
- **Total points regression: IN SCOPE** for the next build phase (Phase 11).
- **Spread regression: DEFERRED, not dropped.** Rationale: scope control. The winner model is already a coarse margin model, and spread is the primary betting market, so spread remains a future candidate.
- **Totals kill criteria** — two tiers, evaluated on held-out seasons only:
  - **Portfolio tier:** MAE must beat a "predict the training-set mean total" baseline, with the mean computed from the training seasons (not hardcoded 44–46). **Fail → drop totals.**
  - **Betting tier:** over/under hit rate vs. the nflverse `total_line` must exceed **52.38%** (breakeven at −110). **Fail → keep totals as a portfolio artifact only, not used for betting.**

### D3. Deployment brought in scope (amends SPEC Section 14). New phases, in order:
- **Phase 8 — Calibration + 2025 holdout.** `CalibratedClassifierCV`, fit without touching test seasons; then a 2025 full-season holdout evaluation. 2025 data **is available and complete** locally (Section 1.1).
- **Phase 9 — Data-layer spike.** Test whether `nfl_data_py` can still pull 2026 in-season data. If not, migrate **only** `data_ingest.py` to `nflreadpy`, converting polars → pandas at that boundary so nothing downstream changes.
- **Phase 10 — 2026 shadow mode.** Retrain through the latest complete season, **freeze**, log predictions and model-vs-implied-probability edges **before kickoff** each week, grade after. Paper trading only for the entire 2026 season; no real-money use is justified before a full season of forward results exists (one season ≈ 272 games is too few to separate a real edge from 52.38% breakeven).
  - *Correction (this review, see §5 note 1):* the margin of error originally stated with this decision was ±3 pts. That is **1 standard error** at n = 272. The **95%** margin is **±5.9 pts**, and roughly **±13 pts** for the ~50 disagreement games per season a betting strategy would actually bet. The conclusion (paper trading only) stands and is strengthened.
- **Phase 11 — Totals model**, built on the existing leakage-safe pipeline, judged by the D2 kill criteria.

### D4. Freeze rule for deployment
The live model is locked for the season except at pre-scheduled checkpoints (default: **one midseason checkpoint**). Experimental changes happen on a separate copy and are compared against the frozen model, never swapped in ad hoc. **Rationale:** week-by-week retuning on live results is test-set peeking in slow motion.

### D5. Deployment model choice — provisional
The four models are statistically tied (no McNemar pair p < 0.05), so "highest accuracy" is not a valid justification. **Recommended default: LR-tuned, on parsimony grounds** — simplest, fastest, most interpretable, least likely to fail silently. **Provisional:** final confirmation after Phase 8 calibration and 2025 results.

### D6. Known limitations (documented, not fixed this phase)
- No injury data.
- No "starting QB changed" flag — first candidate for the midseason checkpoint.
- Regular season only.
- `nfl_data_py` deprecation risk (see Phase 9).

---

## 5. Reviewer notes on the Part D decisions

These do not change any decision above; they are points the project owner asked to have flagged.

1. **D3 margin-of-error figure is optimistic by ~2×.** At p = 0.5238 and n = 272, one standard error is ±3.0 pts; the **95%** half-width is **±5.9 pts** (±4.2 at n = 544). And a betting strategy only bets the games where it disagrees with the market, so its effective n is far smaller than 272 (≈45–55 per season here, where ±5.9 becomes ±13). This strengthens the "paper trading only" conclusion.
2. **D2 betting tier needs a significance requirement, not a point threshold.** "Hit rate > 52.38%" on one or two held-out seasons will be passed by chance often (same issue as SPEC 5.5 check 4). Also define push handling (games landing exactly on `total_line` excluded) and which seasons are the held-out ones **before** Phase 11 starts — 2023–2024 are already touched for the winner target, and 2025 is earmarked for the Phase 8 winner holdout.
3. **Phase 8 calibration must use a season-aware splitter.** `CalibratedClassifierCV`'s default `cv` is a non-shuffled `StratifiedKFold` over rows, which mixes seasons across folds and lets later seasons inform calibration of earlier ones (the SPEC 6 prohibition). Use a season-based splitter, or fit the calibrator on a single held-out season (e.g. 2022) against a pre-fit model (`FrozenEstimator`).
4. **Phase 10 must snapshot its own lines.** The nflverse line columns carry no timestamp (Section 3.1). "Model-vs-implied edge before kickoff" is only meaningful if the line is captured at the same moment the prediction is logged; comparing against whatever value the schedule file holds after the game would mix in later information.
5. **Phase 10 expectations.** Section 3.4 shows the models are right only about one-third of the time when they disagree with the market. Shadow mode is still worth running as a measurement exercise and a deployment dry-run, but the prior on finding a betting edge should be low.
6. **D5 caveat worth recording.** LR-tuned is also the model with a check-4 override. The override is well-investigated, but a parsimony argument should note it.

---

## 6. Roadmap

| Phase | Deliverable | Gate |
|---|---|---|
| 0–6 | Setup → data → features → leakage audit → models → tuning → walk-forward | ✅ Complete |
| 7 | This review gate | ✅ This document |
| 8 | Calibration (season-aware, no test seasons) + 2025 full-season holdout | Confirms D5 model choice |
| 9 | Data-layer spike: `nfl_data_py` 2026 in-season pull; migrate `data_ingest.py` to `nflreadpy` only if needed | Pipeline produces 2026 features unchanged downstream |
| 10 | 2026 shadow mode: frozen model, pre-kickoff logging with snapshotted lines, weekly grading, paper trading only | Full season of forward results before any money decision |
| 11 | Totals regression, judged by D2 kill criteria | Portfolio tier / betting tier |
| Deferred | Spread regression; starting-QB flag; per-team HFA; check-4 significance test | — |

---

## 7. Artifacts produced this phase

| Path | In git? | Purpose |
|---|---|---|
| `scripts/reproduce_walkforward_predictions.py` | new, untracked | Version-controlled Phase 6 protocol; exact reproduction gate |
| `scripts/vegas_benchmark_2023_2024.py` | new, untracked | Descriptive Vegas benchmark (Section 3) |
| `data/processed/walkforward_test_predictions.parquet` | gitignored | Per-game 2023–2024 predictions, 544 rows |
| `docs/PHASE7_REVIEW.md` | new, untracked | This document |
