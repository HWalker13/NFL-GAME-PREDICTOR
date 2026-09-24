# NFL Game Predictor

A machine-learning model that predicts which team wins an NFL game, using only information available before kickoff. The model is built and tested the way a trading desk would test it: no peeking at the future, results locked in before the games are played, and an honest comparison against the betting market.

## Summary

The model predicts the home team's chance of winning each regular-season game from free public play-by-play data. It uses rolling quarterback and defensive efficiency (EPA), an Elo team rating, and context such as rest and travel.

- **Tested on seasons it never saw:** the model picked **64.3%** of 2023–24 games correctly, against **54.4%** for always picking the home team. <!-- source: docs/PHASE7_REVIEW.md §1.2 and §3.3 (LR-tuned 350/544 = 0.6434; home baseline 296/544 = 0.5441) -->
- **Tested again, with the rules written down in advance:** on the 2025 season it picked **61.6%** correctly, against **53.9%** for the home team. That was inside the range fixed before the run. <!-- source: docs/PHASE8_RESULTS.md §5.1 and §5.2 (167/271 = 0.6162; home 146/271 = 0.5387; band 58–68%) -->
- **It does not beat Vegas.** The betting favorite won **69.7%** of 2023–24 games. When the model and the market disagreed, the market was usually right. <!-- source: docs/PHASE7_REVIEW.md §3.3 and §3.4 (Vegas favorite 0.6967) -->

Why the numbers can be trusted: every feature is checked to use only earlier games; models are always tested on later seasons than they were trained on; the one-time tests were pre-registered; and the 2026 live model is frozen, with its predictions saved as write-once files before each kickoff.

## Results

| Evaluation | Games | Model (LR-tuned) | Always pick home | Vegas favorite |
|---|---:|---:|---:|---:|
| 2023–24 walk-forward test | 544 | **64.3%** (350) | 54.4% (296) | 69.7% |
| 2025 pre-registered holdout | 271 | **61.6%** (167) | 53.9% (146) | 65.3% (177) |

<!-- source, row 1: docs/PHASE7_REVIEW.md §1.2 (LR-tuned 350/544 = 0.6434, home 296/544 = 0.5441) and §3.3 (Vegas favorite 0.6967, pooled n=544) -->
<!-- source, row 2: docs/PHASE8_RESULTS.md §5.1 (LR-tuned 167/271 = 0.6162, home 146/271 = 0.5387) and §6 (Vegas favorite 0.6531, 177/271) -->

**Against the market:**
- **Picking winners:** the Vegas favorite was more accurate than the model in 2023–24 (69.7% vs 64.3%). <!-- source: docs/PHASE7_REVIEW.md §3.3 (0.6967 vs 0.6434) -->
- **Where it matters, the model loses.** On the 91 games of 2023–24 where the model and the Vegas favorite picked different winners, the model was right 31 times (34%; McNemar p = 0.0031). All four models tested landed at 32–34% on their disagreements. In 2025, the model was right on 11 of 32 disagreements (34%). <!-- source: docs/PHASE7_REVIEW.md §3.4 (LR-tuned 31/91 = 0.341, p = 0.0031; range 0.320–0.341); docs/PHASE8_RESULTS.md §6 (LR-tuned 11/32 = 0.344) -->
- **Probabilities:** the market's probabilities are also better (2023–24 log loss 0.6074 vs 0.6434 for the model; ROC-AUC 0.7306 vs 0.6774). <!-- source: docs/PHASE7_REVIEW.md §3.3 -->

**Details:**
- The 95% interval on the 2023–24 accuracy is 60.2–68.3%. Every model cleared the home baseline by 8.5 to 9.9 points. <!-- source: docs/PHASE7_REVIEW.md §3.3 (Wilson CI 0.602–0.683) and §2 (+8.5 to +9.9 pts) -->
- Four models were carried to the final test: logistic regression (tuned and untuned), random forest and gradient boosting. They scored 62.9–64.3% and are statistically tied (no pair differs at p < 0.05). Tuned logistic regression was chosen because it is the simplest, not because it is best. <!-- source: docs/PHASE7_REVIEW.md §1.2, §2 and §4 D5 (62.87–64.34%; no McNemar pair p < 0.05; parsimony) -->
- On 2025, gradient boosting scored 65.3%. It still did not meet the pre-registered rule for replacing the chosen model (McNemar p = 0.0525, needed < 0.05), and its log loss was worse. <!-- source: docs/PHASE8_RESULTS.md §5.3 -->
- Full records: [`docs/PHASE7_REVIEW.md`](docs/PHASE7_REVIEW.md), [`docs/PHASE8_RESULTS.md`](docs/PHASE8_RESULTS.md).

## What makes it rigorous

- **Leakage audit.** The main risk in sports prediction is a model that quietly sees the result it is predicting. Every rolling feature excludes the current game (`closed='left'`, or `merge_asof(allow_exact_matches=False)`). A four-part audit runs on every rebuild:
  1. **Timestamps:** every feature is built from data older than kickoff. 0 violations in 6,208 games.
  2. **Shuffled labels:** trained on shuffled outcomes, the pipeline scores about chance (48.5% mean over 20 shuffles).
  3. **Feature importance:** no single feature dominates.
  4. **Ablation:** dropping the top feature does not improve accuracy. Two logistic-regression flags on this check were each investigated across 12 training seasons and found indistinguishable from chance (p = 0.254 and 0.6875). The one the audit still trips is now encoded as an exact-value exception, so any drift fails loudly.
  <!-- source: docs/PHASE7_REVIEW.md §0 (closed='left' / merge_asof); docs/PHASE9_DATA_LAYER.md §3.4 (CHECK 1: 6,208 rows, 0 violations; CHECK 2: mean 0.4846 over 20 shuffles); docs/PHASE7_REVIEW.md §2 (p = 0.254 and 0.6875); docs/PHASE10_PREREG.md Appendix A1 (exact-count exception 168/269 -> 170/269) -->
- **Season-based walk-forward testing.** Train on 2002–2022, predict 2023; train on 2002–2023, predict 2024. No random shuffling of games, which would let later weeks inform earlier ones. <!-- source: docs/PHASE7_REVIEW.md §0 and §2 -->
- **Pre-registration.** Before the 2025 test and before the 2026 live season, a document fixed what would be run, which numbers count, and how each outcome would be read. Every possible 2025 result mapped to a named interpretation in advance. <!-- source: docs/PHASE8_PREREG.md §3; docs/PHASE10_PREREG.md header -->
- **Frozen models.** The 2026 models are pinned by file hash (sha256), along with the feature code. The prediction script refuses to run if any of them changes. <!-- source: docs/PHASE10_PREREG.md §1 "Freeze enforcement" -->
- **Write-once live record.** Each week's predictions are published with an atomic operation that fails if the file already exists, then made read-only. Mistakes are fixed going forward, never by editing past predictions. <!-- source: docs/PHASE10_PREREG.md §8 -->
- **Reproducible.** The 2023–24 test was re-implemented from scratch and reproduced every accuracy as an exact integer count. A data-library migration was accepted only after it rebuilt the 2002–2025 feature file byte for byte (same sha256). <!-- source: docs/PHASE7_REVIEW.md §1.2; docs/PHASE9_DATA_LAYER.md §2.5 (sha256 747afbf4…) -->

## Honest findings and limitations

- **The market is better.** The model beats the naive baseline by a clear margin but trails Vegas on accuracy and on probability quality. Where the two disagree, Vegas is usually right (see Results). One caveat: nflverse's historical line is untimestamped and may be the closing line, which already includes late news such as injuries and weather. That is why the 2026 live test snapshots the line at the moment each prediction is logged. <!-- source: docs/PHASE7_REVIEW.md §3.1, §3.4 caveat, §5 note 4 -->
- **Home-field advantage is shrinking, and the model over-predicts home wins.** The evidence comes from four places:
  - home teams won 54.4% of 2023–24 games, against 56.4% over 2002–2021;
  - probability calibrators fit on recent seasons had negative intercepts for both the 2022 and 2025 targets, which means they pull predictions toward the away team;
  - recent seasons imply an Elo home advantage of about 23 points, against the frozen 44.73.
  The live challenger model is a test of correcting this. <!-- source: docs/PHASE8_PREREG.md §6.4 (54.4% vs 56.4%; Elo ~23 vs 44.73); docs/PHASE8_RESULTS.md §8 (four signals) and §3/§5.6 (negative intercepts); docs/PHASE10_PREREG.md §1 (challenger targets the intercept) -->
- **Calibration didn't help.** A leakage-safe probability calibration made 2022 log loss slightly worse for all four models (+0.0011 to +0.0044), so the deployed model uses raw probabilities. <!-- source: docs/PHASE8_RESULTS.md §3 table (+0.0011 LR-untuned ... +0.0044 RandomForest); phrase "+0.0011 to +0.0044" in docs/PHASE8_PREREG.md §0 -->
- **No injury or quarterback-change data.** A backup quarterback starting is a big swing that rolling averages miss. A "starting QB changed" flag is the first candidate for a future improvement. <!-- source: docs/PHASE7_REVIEW.md §4 D6; SPEC.md §9 item 3 and §13.1 -->
- **Regular season only.** Playoffs are not modeled. <!-- source: docs/PHASE7_REVIEW.md §4 D6 -->
- **The Phase 8 incident.** The one-time 2025 test ran before the owner had signed off. The AI coding agent triggered it while "testing" that its safety guard would refuse. The guard checked for the text `**Status:** APPROVED` anywhere in the pre-registration, and the draft's own instructions contained that text, so the guard let the run through (2026-09-22T20:54:45Z). How it was handled:
  - the executed script and pre-registration were shown by hash to be byte-identical to the versions under review, and both were archived;
  - the owner chose, after seeing the results, to treat it as the pre-registered run, and the incident is recorded in both Phase 8 documents;
  - the guard now requires an exact whole-line match, and every branch is tested without running the evaluation;
  - a standing rule: one-shot evaluations are prepared by the agent and run only by the owner.
  <!-- source: docs/PHASE8_RESULTS.md §1; docs/PHASE8_PREREG.md §7 -->

## Live 2026: shadow mode

In 2026 the frozen model runs in **shadow mode**: paper trading only, no real money. <!-- source: docs/PHASE10_PREREG.md §0 -->

- **Before each week's first kickoff,** it logs a win probability for every game next to the market's probability, taken from a timestamped odds snapshot. <!-- source: docs/PHASE10_PREREG.md §0, §3 -->
- **Paper bets:** a flat 1-unit bet is recorded wherever the model and the market differ by at least 4 percentage points. <!-- source: docs/PHASE10_PREREG.md §2 (edge >= 0.04) -->
- **Two models:** the **champion** (the model above, retrained on 2002–2025) makes the official record. A **challenger** (the same model plus a correction for the home-win bias) is tracked for comparison. <!-- source: docs/PHASE10_PREREG.md §1 (trained on 2002–2025 REG, n = 6,208) and §2 item 7 -->
- **Two decisions, fixed in advance:**
  - after week 9, the challenger replaces the champion only if it is clearly and significantly better over weeks 4–9 (88 games);
  - at season end, "evidence of an edge" is claimed only if the lines moved toward the bets on average (closing-line value > 0, one-sided p < 0.05). Profit alone never counts.
  <!-- source: docs/PHASE10_PREREG.md §4 and §5 -->
- **Expectation, stated in advance:** the rule most likely loses money and shows no edge. <!-- source: docs/PHASE10_PREREG.md §4 -->
- **The official record starts with week 3** (Amendment 1). <!-- source: docs/PHASE10_PREREG.md header and Amendments -->

Follow along:
- **Scorecard:** [`docs/live/SCORECARD_2026.md`](docs/live/SCORECARD_2026.md), cumulative results with 95% intervals. Created when week 3 is first graded.
- **Weekly picks pages:** `docs/live/week_NN_picks.md`, a plain-English table per week (the pick, the model's and Vegas's chances, the paper bet, and results once graded).
- **Raw official record:** [`data/live/2026/predictions/`](data/live/2026/predictions/).
- **Rules:** [`docs/PHASE10_PREREG.md`](docs/PHASE10_PREREG.md).

## How to run it

**Setup** (Python 3.11 is required: `nfl_data_py` pins `numpy<2` and `pandas<2`, which have no wheels for 3.12+):

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Rebuild and audit** (data is cached locally under `data/raw/`, which is not in git):

```bash
python -m src.data_ingest                 # pull 2002-present play-by-play + schedules
python -m src.features                    # build the point-in-time feature table
python -m src.leakage_checks              # the four-part leakage audit (exit 0 = pass)
python -m scripts.test_phase10_live       # live-system tests (also test_phase9_ingest, test_readable_week)
```

**Weekly live routine** (run by the project owner; the full version with git steps is in `docs/PHASE10_PREREG.md` §7):

```bash
# Wednesday: refresh odds + predict, before Thursday's kickoff
python -m src.data_ingest --refresh-season 2026
python -m scripts.live.predict_week  --season 2026 --week N --live
python -m scripts.live.readable_week --season 2026 --week N --live

# Sunday morning: refresh only (a later pre-kickoff odds snapshot)
python -m src.data_ingest --refresh-season 2026

# Tuesday: refresh, grade, scorecard
python -m src.data_ingest --refresh-season 2026
python -m scripts.live.grade_week    --season 2026 --week N --live
python -m scripts.live.scorecard     --season 2026 --live
python -m scripts.live.readable_week --season 2026 --week N --live
```

Without `--live`, every live script writes to the gitignored `data/live_scratch/` instead of the official record.

## Repository layout

```
SPEC.md                   project specification (rules, leakage policy, roadmap)
docs/                     phase reviews, pre-registrations and results
  PHASE7_REVIEW.md          2023-24 test + Vegas benchmark
  PHASE8_PREREG.md          2025 holdout pre-registration
  PHASE8_RESULTS.md         calibration + 2025 holdout (incl. incident)
  PHASE9_DATA_LAYER.md      data-library migration + in-season ingest
  PHASE10_PREREG.md         2026 shadow-mode rules
  live/                     scorecard + weekly picks pages (2026)
src/
  data_ingest.py            nflverse pulls, cache, timestamped odds snapshots
  features.py               point-in-time features (EPA, shrinkage, EWM, Elo, context)
  leakage_checks.py         the four-part audit
  train_winner*.py          model training (LR, RF, HGB; tuned)
  evaluate.py, calibration.py
  live_features.py, live_models.py   features for unplayed games; frozen 2026 models
scripts/
  live/                     predict_week, grade_week, scorecard, readable_week
  reproduce_walkforward_predictions.py, vegas_benchmark_2023_2024.py
  phase8_*, phase9_*, phase10_*        phase-specific runs and checks
  test_*.py                 tests (stdlib unittest)
models/                   frozen 2026 champion + challenger (other models are local only)
data/
  raw/snapshots/            timestamped 2026 odds snapshots (tracked; upstream overwrites lines)
  live/2026/                official write-once predictions (tracked)
```

All data and tools are free: nflverse data via `nflreadpy`, and scikit-learn models on a laptop. <!-- source: SPEC.md §3.2; docs/PHASE9_DATA_LAYER.md §2.6 -->
