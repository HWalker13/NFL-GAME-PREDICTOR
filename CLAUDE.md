# Project rules — persistent, apply to every session

- NEVER run `git commit` or `git push`, under any circumstances, even if
  explicitly asked to "finish up" or "wrap up" a task. Staging with `git add`
  is fine. If changes are ready to commit, say so and stop — the user commits
  and pushes themselves, always.
- Follow SPEC.md literally, including its MUST/SHOULD/MAY conventions.
- Do not implement Section 12 (spread/total) under any circumstances until
  explicitly told to.
- Any rolling/aggregated feature must use closed='left' on .rolling(), or
  merge_asof with allow_exact_matches=False (Section 5.4). Flag every one you
  write, explicitly, in your report.
- If test accuracy on anything exceeds ~70%, stop and flag it — do not report
  it as a good result (Section 5.6/5.7).
- When reporting results, show actual command output (file listings, git log,
  git status, computed numbers) rather than only narrating what happened.
- Decision record: the SPEC 7.2 EWM recency-weighted features (`{m}_ewm`,
  Pattern C) are grouped by team only, with no hard reset at season
  boundaries — recency decay (halflife=5 games) is deliberately left to
  handle the season transition on its own instead of a `[team, season]`
  reset like the `_sd`/`_shrunk` features use, so this is a recorded design
  choice, not a bug to "fix" later.

## Deferred / future work (tracked, not re-derived from scratch each time)

- Starting-QB-change flag (SPEC Section 9, item 3) — a backup QB starting is
  a bigger performance swing than rolling EPA averages capture; deferred to
  a later phase once the core win/loss MVP works.
- Per-team home-field-advantage Elo adjustment — considered, deferred.
  Current `ELO_HOME_ADV` is a single global constant derived from train-only
  home win rate. A per-team version would need its own shrunk,
  point-in-time-correct feature (same rigor as Elo/EWM), since raw per-team
  home splits are too small-sample to use directly and a naive full-history
  average would leak future games into early-season ratings. Revisit only if
  Phase 5B's feature-importance check suggests home-field signal is being
  left on the table.
- Hyperparameter tuning method, decided ahead of the tuning pass: when RF/HGB
  tuning happens (after the Phase 5B Part 1 untuned baselines), use
  `GridSearchCV` (exhaustive) over a deliberately modest grid (2-3
  hyperparameters, 3-4 values each) — not `RandomizedSearchCV`. Reasoning:
  deterministic/reproducible results, and at ~5,124 train rows with
  season-based CV, exhaustive search over a small grid is fast enough that
  random sampling's efficiency advantage doesn't matter.

## Phase 5B Part 2 — LogisticRegression check-4 investigation (RESOLVED)

Two separate SPEC 5.5 check 4 (ablation) flags on LogisticRegression were
investigated end to end this phase. Both are now treated as documented,
investigated false positives — not leaks — and both models are saved with
an explicit, one-time override (`LR_CHECK4_OVERRIDES` in
`src/train_winner_tuned.py`). Full story below so neither has to be
re-derived.

**1. `elo_diff` flag (LogisticRegression-tuned, C=0.01, full 67-feature set).**
- Original finding: LR-tuned failed check 4 — dropping `elo_diff` (its top
  feature by |coef|) *increased* validation accuracy.
- Diagnosed (`scripts/diagnose_elo_lr.py`) as structural collinearity:
  `elo_diff = home_elo_pre - away_elo_pre`, TRAIN |r|=0.70-0.71 with both
  raw Elo levels, destabilizing LR's coefficients.
- Fix attempt 1: LR-only feature exclusion (drop `home_elo_pre`/
  `away_elo_pre`, keep `elo_diff`). Stabilized `elo_diff`'s coefficient
  (correctly signed, rank #1) but did **not** clear check 4.
- Wider investigation (`scripts/diagnose_c_grid_clusters_untuned.py`) found
  72 OTHER retained `_shrunk`/`_ewm` feature pairs project-wide with
  |r|>0.5 and opposite-signed coefficients — not elo-specific. A widened C
  grid (0.0001-100) ruled out regularization strength as the fix (real
  interior optimum at C=0.01, not a floor effect). The same investigation
  found the UNTUNED LR baseline independently fails check 4 too, but on a
  DIFFERENT feature (`away_def_epa_early_ewm` — see part 2 below).
- Fix attempt 2: general, model-agnostic cluster pruning at |r|>0.5 (67 ->
  22 features, `scripts/compute_pruned_features.py`, applied to all three
  model types) — made it WORSE. All six retrained models
  (LR/RF/HGB x untuned/tuned) failed check 4 on the pruned set, including
  RF and HGB, which had PASSED cleanly on the full feature set (pruning
  concentrated importance onto `elo_diff`: its share jumped to
  17.8%-39.9%, up from ~9-21%). Reverted entirely — back to the single,
  uniform, full 67-feature set for all three models, no special-casing.
- RESOLUTION: a 12-fold TRAIN-only rotating holdout
  (`scripts/diagnose_elo_noise_extended.py`, pseudo-validation seasons
  2010-2021, train = everything strictly before each, LR at C=0.01 — the
  tuned hyperparameter) found the effect is NOT statistically
  distinguishable from chance: sign test p=0.254 (6 positive / 3 negative /
  3 exactly-zero folds out of 9 non-zero), and all 12 per-fold deltas fall
  within ±1 binomial SE of zero.

**2. `away_def_epa_early_ewm` flag (LogisticRegression-untuned, C=1.0, full
67-feature set).**
- Found while re-verifying the untuned Phase 4 baseline (never previously
  re-checked since Phase 5A added the EWM/Elo columns): check 4 fails,
  dropping `away_def_epa_early_ewm` (top feature, 4.8% share) increases
  accuracy 0.6245 -> 0.6320.
- An EARLIER bootstrap diagnostic of this exact feature (500 resamples of
  the fixed 2022 validation set: mean delta +0.00768, 90% of resamples >=
  0, 95% CI [-0.00372, +0.02230]; `corr(away_def_epa_early_ewm,
  away_def_epa_early_shrunk)=0.93`) existed and found a soft positive lean
  — **but this diagnostic was never recorded anywhere in this repo (not in
  CLAUDE.md, not in any commit) until this entry. Flagging that
  documentation gap explicitly — it is not being treated as a new
  finding, but its absence from the record was a real gap.**
- A new 12-fold TRAIN-only rotating holdout (`scripts/diagnose_feature_noise.py`,
  same method as `elo_diff`'s, at C=1.0 — the untuned model's actual
  hyperparameter) found p=0.6875, 8/12 folds exactly zero effect, non-zero
  folds split evenly 2-positive/2-negative — an even weaker signal than
  `elo_diff`'s result.
- The two methods for this feature DISAGREED in strength: the bootstrap
  (resampling the single, fixed 2022 val set 500 times) suggested a mild
  lean; the rotating holdout (12 independent TRAIN seasons) found none.
  RECONCILED by treating the rotating holdout as the more decisive test —
  it measures across-season stability, which is what actually matters for
  "is this feature's signal real," where the bootstrap can only
  characterize sampling variability within one fixed 269-game draw and
  cannot speak to whether a pattern found in 2022 specifically would hold
  in a different season.

**3. Resolution applied:** both flags are treated as investigated,
documented false positives caused by known feature redundancy
(collinearity within the `_shrunk`/`_ewm` families, see the 72-pair finding
above) interacting with small per-season validation sample sizes
(~255-271 games/season — binomial SE ≈0.03, comparable in magnitude to the
observed deltas), not leakage. `LR_CHECK4_OVERRIDES` in
`src/train_winner_tuned.py` applies a one-time, explicitly-cited override
of the save gate for LogisticRegression ONLY (both untuned and tuned) — it
does **not** change the gating logic itself, which still blocks normally
on check 3 or the 70% threshold for every model, LR included, and RF/HGB
were never overridden (they passed check 3/4 cleanly on their own merits).
Per-model save gating (introduced during this investigation) means one
model's flag no longer blocks saving the others. All six models
(LR/RF/HGB x untuned/tuned) are now saved on the full, unpruned 67-feature
set:
  - `logreg_winner.joblib` / `logreg_winner_tuned.joblib` (override applied,
    justification + citation stored in each file's `leakage_checks` metadata)
  - `rf_winner.joblib` / `rf_winner_tuned.joblib` (clean pass, no override)
  - `hgb_winner.joblib` / `hgb_winner_tuned.joblib` (clean pass, no override)

**4. Phase 6 backlog item (not fixed now):** SPEC 5.5 check 4's current
threshold (any accuracy *increase* on ablation = flagged) doesn't
distinguish a real leak from small-sample noise on a feature whose true
effect is close to zero — both this phase's flags turned out to be
statistically indistinguishable from chance only after a separate,
ad hoc rotating-holdout investigation. Worth strengthening check 4 itself
with a built-in significance test (e.g. requiring the ablation delta to
exceed some multiple of the binomial SE at the validation sample size,
or running the check across multiple seasons by default rather than one
fixed validation split) so this doesn't require a manual investigation
every time. Not implemented now — tracked here for Phase 6.

**5a. Untuned RF/HGB baselines regated and re-verified (2026-09-21).** A prior
audit of this phase found that `rf_winner.joblib` and `hgb_winner.joblib`
(the untuned baselines, saved by `src/train_winner_ensemble.py`) carried NO
persisted `leakage_checks` metadata at all -- the module computed checks 3/4
for both (`LC.feature_importance_inspection` / `LC.ablation_check`) and
printed the results, but never wrote them into the saved joblib dict, and
never gated the save on their outcome (only on the SPEC 5.6 70% threshold).
Their on-disk files also predated the Phase 5B Part 2 commit, so the "RF/HGB
pass checks 3/4 cleanly, no override" claim elsewhere in this document was
unverifiable from the files themselves -- true in spirit (the console output
existed at some point), but not backed by anything reproducible.

Fixed properly, not patched around: the per-model save-gating logic
(`gate()`) that `src/train_winner_tuned.py` already used for LR/RF/HGB-tuned
was extracted to `src/leakage_checks.py` as `LC.gate()` -- one shared
definition, not a second copy -- and `train_winner_ensemble.py`'s RF/HGB
save path was rewritten to call it per-model (previously it gated both
models on a single combined 70% check and saved both unconditionally
otherwise). Both untuned baselines were then actually re-run through this
real save path (`python -m src.train_winner_ensemble`, not a diagnostic) to
get first re-verified, persisted evidence rather than trusting the
console-only Sep 17 result:

- **RandomForest-untuned:** check 3 top feature `elo_diff` (9.3% share,
  well under the 40% flag threshold) -- PASS. Check 4 (drop `elo_diff`,
  retrain): accuracy 0.6543 -> 0.6320 (delta -0.0223, degrades gracefully,
  no collapse-to-baseline, no increase) -- PASS. **Both checks PASS, no
  override needed or applied.**
- **HistGradientBoosting-untuned:** check 3 top feature `elo_diff` (12.6%
  share) -- PASS. Check 4 (drop `elo_diff`, retrain): accuracy 0.6394 ->
  0.6134 (delta -0.0260, degrades gracefully) -- PASS. **Both checks PASS,
  no override needed or applied.**

Accuracy/log-loss/brier/ROC-AUC for both matched the prior Sep 17 numbers
exactly (0.6543/0.6394 respectively), confirming the current committed
`train_winner_ensemble.py` reproduces the same result deterministically --
resolves the open provenance question from the prior audit either way, since
the files are now known-current regardless. Neither model exceeded the SPEC
5.6 70% threshold. Both models were re-saved (`models/rf_winner.joblib`,
`models/hgb_winner.joblib`, 2026-09-21 14:06) with a `leakage_checks` key of
the identical shape used by all four other saved models
(`check3_top_feature`, `check3_flagged`, `check4_suspicious`,
`override_applied`, `override_justification`). All six models now carry
consistent, persisted leakage-check metadata -- verified side by side after
the re-save. `logreg_winner.joblib` (untuned LR) was NOT touched by this run
(`train_winner_ensemble.py`'s own stale-feature-count guard did not fire --
confirmed 67 == 67 before running -- so its existing override metadata from
the check-4 investigation above was left exactly as-is).

**5. Confirmed explicitly:** season 2022 (validation) and seasons
2023-2024 (test) were NEVER touched by any diagnostic script in this
entire investigation (`scripts/diagnose_elo_lr.py`,
`scripts/diagnose_c_grid_clusters_untuned.py`,
`scripts/compute_pruned_features.py`, `scripts/diagnose_elo_noise.py`,
`scripts/diagnose_elo_noise_extended.py`,
`scripts/diagnose_feature_noise.py`) — every rotating-holdout fold used
seasons strictly within 2002-2021. Only the real, existing
`train_winner_tuned.py` pipeline (approved, pre-existing evaluation logic)
ever touches season 2022, and only to compute the reported validation
metrics — never to select features or hyperparameters based on test-set
performance. As of the entry below, this is no longer current — the
held-out 2023-2024 test set has now been touched, exactly once, for the
final Definition-of-Done evaluation.

## Phase 1 Definition of Done — Final Walk-Forward Test Result (2026-09-22)

SPEC Section 6's "recommended extension" walk-forward backtest run once,
final, against the previously-untouched 2023-2024 test seasons (confirmed
untouched beforehand: grep of src/ and scripts/ for "2023"/"2024" showed
only comments/docstrings/the FORBIDDEN_TEST_SEASONS guard). Locked-in
hyperparameters and the 67-feature set were loaded directly from each
saved model's joblib metadata -- no tuning, no feature changes, no
iteration in this session. RF-untuned skipped: confirmed identical
hyperparameters to RF-tuned (max_depth=8, min_samples_leaf=5,
n_estimators=300 -- GridSearchCV's best_params_ landed exactly on the
untuned baseline's hand-picked values).

Step A: train 2002-2022 (n=5,393) -> test 2023 (n=272).
Step B: train 2002-2023 (n=5,665) -> test 2024 (n=272).
Pooled 2023+2024 (n=544): each season predicted only by the model trained
on strictly-prior data (Step A's model predicts 2023, Step B's predicts
2024) -- not a single model retrained on 2002-2024.

| model          | pooled acc | log_loss | brier  | roc_auc |
|----------------|-----------:|---------:|-------:|--------:|
| home_baseline  |     0.5441 |        - |      - |       - |
| LR-untuned     |     0.6287 |   0.6479 | 0.2282 |  0.6698 |
| LR-tuned       |     0.6434 |   0.6434 | 0.2261 |  0.6774 |
| RandomForest   |     0.6305 |   0.6380 | 0.2237 |  0.6834 |
| HGB-tuned      |     0.6287 |   0.6418 | 0.2255 |  0.6789 |

All four models are statistically indistinguishable in pooled test
accuracy: a follow-up McNemar's exact test (paired, all 6 pairwise
comparisons among the four models on the identical 544 test games) found
no pair significant at p<0.05 (closest overall: LR-untuned vs LR-tuned,
p=0.0963; the two comparisons that matter for a "best model" claim --
LR-tuned vs RandomForest, p=0.3240, and LR-tuned vs HGB-tuned, p=0.3222 --
are both far from significant). LR-tuned is numerically highest (+1.3-1.5pt
over RF/HGB) but that gap is not distinguishable from chance at this
sample size (n=544, only 31-50 discordant games per pair). Treat the
ranking among the three non-baseline-adjacent models as a tie, not a
result.

SPEC 2 Definition of Done: MET on the baseline comparison, which IS
decisive -- every model's margin over home_baseline is well above the
+3-5pt bar (+8.5 to +9.9pts pooled) and clears it by a much wider margin
than any pairwise model-vs-model gap, so this conclusion does not depend
on the (statistically noisy) ranking among models.

Pooled accuracy (62.87%-64.34% across the four models) sits just below
SPEC 5.7's realistic mid-60s-to-high-60s SHOULD-range for three of the
four models -- LR-untuned (62.87%), HGB-tuned (62.87%), and RandomForest
(63.05%) are all low-60s, not mid-60s. Only LR-tuned's pooled 64.34%
arguably reaches the range's low edge. This is a SHOULD, not a MUST,
criterion (SPEC Section 2), so it does not change the Definition-of-Done
conclusion above, which rests on the baseline-margin MUST instead and is
cleared decisively regardless. The one number that actually lands
mid-60s-to-high-60s is a single-season figure, not the pooled headline:
LR-tuned on 2024 alone reached 66.54% (Step B) -- not representative of
the pooled result and not being substituted for it here.

Reassurance, from numbers already in hand (no new computation): each
model's shift from validation (2022) accuracy to pooled test accuracy is
within the ~3pt noise band already established elsewhere in this project
(LR-untuned +0.4pt, LR-tuned +2.6pt, RandomForest -2.4pt, HGB-tuned
-2.2pt). No model's performance collapsed going from validation to real
test data -- mild evidence against overfitting to the validation season,
though not proof of it.

Open item, not silently omitted: SPEC 7.3 probability calibration
(CalibratedClassifierCV) was NOT applied in this phase to any of the four
models -- reported log_loss/brier/roc_auc above are on raw
(uncalibrated) predict_proba output. Flagged for Phase 7 discussion,
not addressed here.

This is the final, one-time SPEC Section 2 test-set result. Seasons
2023-2024 are no longer "never touched" as of this entry -- any future
work that re-touches them (e.g. further tuning) would no longer be a
clean walk-forward test and must be flagged as such.
