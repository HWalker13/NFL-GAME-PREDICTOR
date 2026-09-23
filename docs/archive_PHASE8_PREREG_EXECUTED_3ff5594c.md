# Phase 8 Pre-Registration — 2025 Full-Season Holdout

**Status:** DRAFT — awaiting project-owner approval. `scripts/phase8_holdout_2025.py --season 2025` refuses to run until this line reads `**Status:** APPROVED`.
**Written:** 2026-09-22, before any 2025 outcome, label or score was read by any Phase 8 code.
**SPEC reference:** Section 13, Phase 8. Also serves as the SPEC 5.6 check-4 "additional untouched season."

This document fixes, in advance, exactly what will be run on 2025, which numbers will be reported, and how they will be interpreted. Items marked **[APPROVED]** filled gaps in the owner's brief and were approved by the owner on 2026-09-22; condition (b) in Section 4 was revised by the owner on the same date. Both happened before any 2025 outcome was read.

---

## 0. What is already known (and what is not)

- **Part A (development, 2022 validation only).** OOF walk-forward sigmoid calibration was fitted for T=2022 (OOF seasons 2017–2021, 1,291 calibration rows). The pre-set rule was: adopt calibration iff calibrated 2022 log loss ≤ raw 2022 log loss. Result: **no model adopts calibration**; calibration raised 2022 log loss for all four models (+0.0011 to +0.0044). Full results are in `docs/PHASE8_RESULTS.md` once written, and in `data/processed/phase8_dev_2022.json`.
- **Part B.** SPEC 5.5 check 1 passed on the 271 2025 feature rows with the `home_win` column never loaded: 0 timestamp violations, max rolled-vs-current correlation 0.306, 0 null feature cells. The only schedule game absent from the frame is `2025_04_GB_DAL` (the one tie).
- **Only prior 2025 touch.** `src/evaluate.py`'s `__main__` has printed the 2025 home-win rate, a label aggregate, at some point before Phase 8. It is not a model result and has **no bearing on any decision here**. No Phase 8 code has read 2025 labels, scores or lines.

---

## 1. Exactly what will be run

**Command (once):** `python -m scripts.phase8_holdout_2025 --season 2025`

| Item | Fixed value |
|---|---|
| Target season | T = 2025, regular season, ties excluded → **n = 271** |
| Models | LR-untuned (C=1.0), LR-tuned (C=0.01), RandomForest (n_estimators=300, max_depth=8, min_samples_leaf=5), HGB-tuned (learning_rate=0.03, max_leaf_nodes=15). All are `clone()`s of the saved pipelines, `random_state=0`, 67 features. |
| Feature pipeline | `data/processed/game_features.parquet` exactly as built. Frozen: `ELO_HOME_ADV=44.73`, `DEFAULT_K=4`, `IN_SEASON_WINDOW=22`, `EWM_HALFLIFE=5`, Elo K=20. No rebuild, no re-derivation. |
| Final models | Trained on seasons 2002–2024 (n = 5,937) |
| OOF calibration | One sigmoid per model, fit on pooled walk-forward OOF predictions for 2020–2024 (1,339 games; each season S predicted by a model trained on seasons < S). Unpenalized `LogisticRegression(C=np.inf)` on `logit(p)`. Not `CalibratedClassifierCV`. |
| As-deployed probabilities (from the Part A rule) | **All four models run RAW.** |
| Calibrated probabilities | **[APPROVED]** Still computed and reported as a clearly labelled *descriptive secondary* column. They cannot change any decision. They are useful for Phase 10 because the calibrator's intercept tracks home-field drift. |
| Leakage gate (runs first) | `check_elo_point_in_time` on seasons ≤2025. It reads 2025 final scores, so it could not run in Part B. If it fails, the run aborts before any metric is computed and the result is reported as a leakage finding. Any later run would need a new pre-registration. |

**Code frozen before approval** (SHA-256):

| File | SHA-256 |
|---|---|
| `scripts/phase8_holdout_2025.py` | `54da4088ec579e2daf3cc7e9d4fcbdfe1d798798fe677957231acda9e9284e4e` |
| `src/calibration.py` | `89c22e0a62d1e344900f654366f08b201ca1eb8efba6878c7638081e5bf3e5a9` |
| `scripts/vegas_benchmark_2023_2024.py` (helpers imported) | `11fb114e59f4b9523eee0445cfd1a82871c2a3cb82a3a3ecf90c36cb8bcdb1f5` |

The identical code path was dry-run on the 2022 validation season (`--season 2022`). It reproduced Part A and every saved model's `val_metrics` exactly (asserted in code), and the Elo point-in-time check passed.

After the owner's revision of condition (b), the dry run was repeated with the hash above. Every metric, calibrator parameter, McNemar result, band verdict and all 269 per-game predictions were identical to the previous dry run; the only change is the new (b) test.

---

## 2. Metrics that will be reported

For each model (as deployed = raw; calibrated = descriptive secondary):

1. **Accuracy (primary)** — pick = probability > 0.5, with count and 95% Wilson CI.
2. Log loss, Brier score, ROC-AUC.
3. Reliability table — 10 equal-width bins: count, mean predicted, observed home-win rate.
4. Home-team baseline accuracy on the same 271 games, and each model's margin over it.
5. **Pairwise exact McNemar** on as-deployed picks, all 6 pairs.
6. Calibrator slope and intercept per model (descriptive).
7. **Vegas benchmark + disagreement analysis, exactly as in Phase 7.**
   - Favorite from `spread_line`; moneyline fallback on a pick'em; unresolved games excluded and counted.
   - Vig-free implied probability from the moneylines.
   - Agreement / disagreement counts, model accuracy on disagreement games with a Wilson CI, and McNemar vs the Vegas favorite.
   - **Descriptive only:** no model, feature, calibration or deployment decision may use it.
   - Same caveat as Phase 7: the line is untimestamped and may be at or near closing.

Per-game output: `data/processed/holdout_2025_predictions.parquet`. It holds raw, calibrated and as-deployed probabilities, picks, and the line columns. Run record, with both SHA-256 values: `data/processed/holdout_2025_run_record.json`.

---

## 3. Interpretation bands for LR-tuned accuracy on 2025

Accuracy is a count out of 271, so bands are stated in games. The owner's three bands are kept exactly. The two **[APPROVED]** rows close gaps so that every possible result maps to a named outcome.

| LR-tuned 2025 accuracy | Correct picks (of 271) | Interpretation |
|---|---|---|
| > 70.0% | ≥ 190 | **SPEC 5.6 protocol before reporting as a result** (owner) |
| > 68.0% to 70.0% | 185–189 | **[APPROVED]** Above the 2023–24 range, below the 5.6 trigger. Report as a result with an explicit caution. The Elo check and Part B check 1 must both have passed. No action. |
| 58.0% – 68.0% | 158–184 | **Consistent with 2023–2024 (62.9–64.3%); pipeline generalizes** (owner) |
| 50.0% to < 58.0% | 136–157 | **Flag possible drift or regime change;** report honestly and investigate in a later phase. **Not** a license to modify anything now (owner) |
| < 50.0% | ≤ 135 | **[APPROVED]** Below chance: treat as a suspected pipeline bug (label or row misalignment, inverted target). Do not report it as a model result until investigated. |

**How much noise to expect.** At n = 271 and a true accuracy near 63.5%, one standard error is ±2.9 pts and the **95% CI is ±5.7 pts**, so a plausible range for an unchanged pipeline is roughly 57.8–69.2%. That range is *wider* than the 58–68% band. If nothing has changed, the probability of landing outside 58–68% by chance alone is about **9%**: ~3.4% below 58% and ~5.7% above 68% (~4.4% in 68–70%, ~1.3% above 70%). A result in an adjacent band is therefore weak evidence on its own, and will be described that way.

---

## 4. Deployment-model confirmation rule

LR-tuned remains the provisional deployment model **unless** a challenger (LR-untuned, RandomForest or HGB-tuned) meets either condition on 2025:

- **(a) Accuracy.** The challenger's 2025 accuracy is *higher* than LR-tuned's, **and** the exact McNemar test on the paired picks gives p < 0.05. A significant result in LR-tuned's favour does not qualify. **[APPROVED]**
- **(b) Log loss — both parts required.** Let *d* be the per-game log-loss difference (challenger minus LR-tuned) on the same 271 games, using *as-deployed* probabilities. Part A adopted calibration for no model, so these are raw probabilities; the original wording said "calibrated log loss". The challenger qualifies only if:
  1. the mean of *d* is below **−0.010**, i.e. its log loss is lower than LR-tuned's by more than 0.010, **and**
  2. a **one-sided paired t-test** on *d* (H₀: mean *d* = 0; H₁: mean *d* < 0; `scipy.stats.ttest_1samp(d, 0, alternative="less")`) gives **p < 0.05**.

  **[REVISED BY OWNER — 2026-09-22, before any 2025 outcome was seen]**
- **Tie-break.** If more than one challenger qualifies, the qualifier with the lowest as-deployed 2025 log loss becomes the deployment model. **[APPROVED]**

Expected outcome, given the Phase 6 ties: LR-tuned confirmed.

**Why (b) requires significance (owner's rationale, recorded).** As first drafted, (b) was a bare 0.010 margin. On 2022, the standard error of the paired per-game log-loss difference against LR-tuned was 0.0069 (RF) and 0.0080 (HGB). So 0.010 is only about 1.3 SE: a bare margin would fire on noise roughly 10% of the time per tree challenger, or about 20% for either of the two. Condition (a) already requires significance, and (b) now matches it. Moving away from the parsimony choice because of noise has a real cost. This change was made before any 2025 outcome, label, score or line had been read.

**What the revised (b) gives on 2022 (descriptive dry run; not a decision input):**

| Challenger | Mean *d* | SE | One-sided paired t p | Meets both parts of (b)? |
|---|---:|---:|---:|---|
| LR-untuned | +0.0047 | 0.0025 | 0.967 | no |
| RandomForest | −0.0042 | 0.0069 | 0.269 | no |
| HGB-tuned | −0.0029 | 0.0080 | 0.360 | no |

---

## 5. Things that will not happen

- No tuning, no feature or feature-parameter change, no retraining for model selection, no new data pull.
- The Vegas comparison does not feed any decision.
- **The 2025 result will be reported once, whatever it is. There will be no re-runs after changes.** The script refuses to run if its output already exists. Any future modification evaluated on 2025 would no longer be a clean holdout and must be labelled as such.
- 2023–2024 are used here only as *training* and OOF-calibration data for predicting 2025, which is legitimate walk-forward. They are not re-evaluated, and no choice is based on them.

---

## 6. Known caveats, recorded in advance

1. **Train/OOF mismatch.** The final models train on 2002–2024, while each OOF model trains on fewer seasons, so the final model may be slightly sharper than what the calibrator saw. This is accepted and only matters for the descriptive calibrated column.
2. **2020 sits in the 2025 OOF window.** It was the no-crowd season (home win rate 49.8%). With 2023–24 at 54.4%, the 2025 calibrator's intercept will likely shift toward the away team, as the 2022 calibrators did (intercepts −0.10 to −0.15).
3. **`ELO_HOME_ADV` is frozen at 44.73** (from 2002–2021, home win rate 56.4%). Re-derived on 2002–2024 it would be 43.35, a negligible change, so the frozen value is not wrong as a whole-history estimate. The 2020–2024 rate alone implies about 23 Elo points, however, so the whole-history value probably overstates *current* home advantage. This is flagged for Phase 10 and not changed here.
4. **Home-field advantage drift: three independent signals (descriptive, no action).** Declining home-field advantage now shows up in three separate places:
   - the 2023–2024 home win rate of 54.4%, against 56.4% over 2002–2021;
   - the negative intercepts of all four 2022 OOF calibrators (−0.10 to −0.15), meaning the models over-predicted home wins in 2017–2021;
   - the ~23-point Elo home advantage implied by 2020–2024, against the frozen `ELO_HOME_ADV` of 44.73.

   Nothing in Phase 8 acts on this. It is recorded as the **leading candidate for the Phase 10 pre-scheduled checkpoint** (SPEC 13.1). Any change would be built on a separate copy and compared against the frozen model, per the freeze rule.
