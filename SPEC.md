# NFL Game Outcome Prediction — Project Specification

**Version:** 2.0
**Status:** Active build spec — Phase 1 (Win/Loss MVP)
**Audience:** Written to be read and executed by an AI coding agent (e.g. Claude Code), with human review.
**Stack:** Python, pandas, scikit-learn, nfl_data_py
**Budget:** $0 — free/public resources only

**Keyword convention used throughout this doc:** `MUST` / `MUST NOT` = hard requirement, non-negotiable. `SHOULD` / `SHOULD NOT` = strong recommendation, deviate only with a documented reason. `MAY` = optional.

---

## 0. Read This First — Constraint Summary

- **Current build target:** binary win/loss classification only. Spread and total regression are **fully specified but explicitly deferred** — see Section 12. Do **NOT** implement Section 12 until Phase 1 (Section 13) is complete and reviewed.
- **Cost constraint:** every data source, library, and compute resource used in this project **MUST** be free. See Section 3.2 for an explicit allow/deny list.
- **The single highest-risk failure mode in this project is data leakage**, not model choice. Section 5 is the longest section in this document for that reason and **MUST** be read in full before any feature-engineering code is written.
- **Accuracy expectations are capped by the problem domain, not by model quality.** A test-set accuracy above ~70% is not a win to report — it is a signal to stop and run the leakage audit in Section 5.6. See Section 5.7 for why.
- Evaluation **MUST** use season-based walk-forward splits (Section 7). Random k-fold cross-validation on individual games **MUST NOT** be used — it leaks future seasons into training.

---

## 1. Objective

### 1.1 Full project vision (reference only — do not build yet)

The end-state system predicts, for any NFL game:
1. Winner (classification)
2. Point spread (regression)
3. Total points (regression)

This full scope is preserved in Section 12. It is the target for a *later* phase, once Phase 1 below is working and validated.

### 1.2 Current build target: Phase 1 — Win/Loss Classification MVP

Build a binary classifier that predicts home-team win/loss using historical game data, rolling quarterback performance, and rolling defensive performance as the primary features. This is the only target this document asks you to implement right now.

**Why win/loss first (not the full suite):** it is one well-understood classification problem instead of three problems with different loss functions and different evaluation logic. It shares ~80% of its feature pipeline with the spread/total targets in Section 12, so building it first is not wasted effort — it's the foundation those targets will be built on later.

---

## 2. Success Criteria — Phase 1 (Definition of Done)

| Requirement | Threshold |
|---|---|
| Beats naive baseline | Test-set accuracy `MUST` exceed "always pick home team" (~57–58% historically), by a non-trivial margin (treat +3–5 points over baseline as a meaningful result, not +0.5) |
| Evaluated correctly | `MUST` be evaluated only via the season-based walk-forward split in Section 7 |
| Leakage-audited | `MUST` pass every check in Section 5.5 before results are reported as final |
| Probabilities are meaningful | Predicted win probabilities `SHOULD` be calibrated (Section 7.3) — not just the class label |
| Realistic ceiling acknowledged | Final accuracy `SHOULD` land roughly in the mid-60s to high-60s percent range; see Section 5.7 before treating a higher number as good news |

**Stretch goal (not required for Phase 1 completion):** predicted win probabilities approach the implied win probability from the Vegas closing moneyline on held-out games. This is a bonus comparison, not a pass/fail gate.

---

## 3. Data Strategy

### 3.1 Source: `nfl_data_py`

The original request referenced nflfastR — flagging this explicitly: **nflfastR is an R package.** The Python equivalent, maintained by the same "nflverse" community, is [`nfl_data_py`](https://github.com/nflverse/nfl_data_py). It reads the identical underlying play-by-play dataset (public Parquet/CSV files hosted on GitHub) that nflfastR uses. No functionality is lost by using Python instead of R.

Functions needed for Phase 1:

- `nfl_data_py.import_pbp_data(years)` — play-by-play data with **EPA (Expected Points Added)** already calculated per play. Foundation for both QB and defensive features.
- `nfl_data_py.import_schedules(years)` — one row per game: final scores, **Vegas opening/closing lines**, rest days, roof/stadium type, week, season type, game date/kickoff time.

`nfl_data_py.import_weekly_data(years)` is available as a pre-aggregated shortcut but `SHOULD NOT` be the primary source for Phase 1 — computing rolling stats directly from play-by-play data gives full control over the point-in-time correctness required in Section 5.

### 3.2 Cost & Licensing Constraints (hard requirement — $0 budget)

| Resource | Cost | Status |
|---|---|---|
| `nfl_data_py` (play-by-play, schedules) | Free, public | ✅ Approved |
| Vegas lines embedded in `import_schedules` | Free (publicly reported, bundled in the free dataset) | ✅ Approved for benchmarking only — see leakage rules in Section 5.2 before using as a feature |
| `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib` | Free, open-source | ✅ Approved |
| Local CPU (laptop/desktop) | Free (already owned) | ✅ Sufficient — dataset aggregates to roughly 5,000–6,500 game-team rows for modeling; no GPU or cloud instance is needed |
| ESPN QBR paid tiers, PFF grades/subscriptions | Paid, proprietary | ❌ Prohibited — `MUST NOT` be used |
| Paid odds APIs (paid tiers of any odds provider, Sportradar, etc.) | Paid | ❌ Prohibited — `MUST NOT` be used |
| Cloud GPU/compute instances | Costs money, also unnecessary at this data scale | ❌ `MUST NOT` be used for Phase 1 |

If a future requirement seems to need a paid resource, the correct response is to find a free substitute or scope the requirement down — not to introduce a cost.

### 3.3 Features required for Phase 1 (built on EPA, not proprietary ratings)

**QB features (per team-game, then rolled forward — see Section 5 for how):**
- EPA per dropback
- Success rate (share of plays with positive EPA)
- CPOE — Completion Percentage Over Expected
- Sack rate, interception rate
- Air yards per attempt

**Defensive features (per team-game, then rolled forward):**
- EPA allowed per play
- Success rate allowed
- Pressure rate / sack rate generated
- EPA allowed on early downs vs. money downs (3rd/4th down) — captures defenses that specifically collapse in high-leverage situations, a distinct signal from average defensive quality

**Contextual features (free, no extra data cost, meaningfully improve win/loss accuracy):**
- Home/away indicator
- Rest days since last game (short week vs. bye week)
- Divisional matchup flag
- Time zone / travel distance differential
- Roof type (dome vs. outdoor) — relevant mainly as an interaction with pace, less critical for win/loss than for total points

### 3.4 Historical range

`nfl_data_py` EPA fields are reliable from **1999 onward**; the modern 32-team era started in 2002. **Default for this doc: 2002–present.** Flag if 1999–2001 should be included.

---

## 4. Feature Engineering Overview

All features above `MUST` be computed as of a specific point in time and rolled forward using **only prior games**. The exact mechanics of "prior" are the entire subject of Section 5 — do not write feature code from this section alone.

**Handling small samples (shrinkage):** early-season ratings (1–2 games of data) are noisy. Blend the in-season sample with the player/team's prior-season rate (or league average if no prior season exists), weighted by how much current-season data is available. This prevents a single fluky game from producing an overconfident rating. A standard approach: `shrunk_estimate = (n_games * in_season_avg + k * prior_avg) / (n_games + k)`, where `k` is a tunable prior-weight constant (start with `k` ≈ 4 games worth of weight and tune it).

---

## 5. Data Leakage — Full Specification

**This section governs every line of feature-engineering code written for this project. If any instruction elsewhere in this document conflicts with this section, this section wins.**

### 5.1 Why this gets its own full section

A leaked model does not fail loudly. It trains successfully, evaluates with a suspiciously good number, and only reveals itself as broken once deployed on a real upcoming game — at which point the "future" information it was quietly relying on no longer exists. Every rule below exists to prevent that failure mode from being invisible until it's expensive.

### 5.2 Categories of leakage (know all four before writing code)

**1. Target leakage** — using a game's own outcome-derived stats to predict that same game.
Example: using this Sunday's QB EPA as a feature to predict this Sunday's winner. The QB's EPA *is* largely a byproduct of how the game went. This is the most obvious category and the easiest to avoid once named.

**2. Temporal leakage (look-ahead bias)** — using information that did not exist yet at prediction time, even if it's not literally "this game's" data.
Example: a badly written join that matches a team's stats from *any* game in the dataset instead of only games strictly before the current one. This is the category that causes real bugs, because it's a code error, not a conceptual one.

**3. Aggregation leakage** — a rolling or season-to-date average that accidentally includes the current row.
Example: `groupby('team')['epa'].expanding().mean()` in pandas includes the current row in its own average by default. If used directly as a feature for that row's game, the model partially sees the outcome it's predicting. This is the single most common leakage bug in sports-prediction codebases, precisely because the code *looks* correct.

**4. Market/line leakage** — using the Vegas **closing** line as an input feature (not as a benchmark).
The closing line has absorbed injury news, weather updates, and public betting action right up to kickoff. Using it as a feature means the model is partly just parroting the market's already-informed view rather than learning from your engineered features. The **opening** line is safer if a line-based feature is wanted at all — but for Phase 1, lines `SHOULD` be used only as an evaluation benchmark (Section 2), not as a feature.

### 5.3 Column-level audit table

Use this table as a literal checklist for every feature added to the pipeline:

| Feature | Allowed as of prediction time? | Rule |
|---|---|---|
| This game's QB EPA / defensive EPA allowed | ❌ Never | Target leakage — this is derived from the outcome being predicted |
| Rolling QB/defense EPA from **prior** games only | ✅ Yes | Core Phase 1 feature — see Section 5.4 for correct computation |
| Season-to-date average that includes the current game | ❌ Never | Aggregation leakage — must exclude current row |
| Season-to-date average that stops at the prior week | ✅ Yes | Correct pattern, see 5.4 |
| Full-season final average used mid-season | ❌ Never | Includes games not yet played relative to the prediction point |
| Vegas **opening** line | ⚠️ Benchmark only for Phase 1; `MUST NOT` be used as a training feature unless explicitly revisited later | See 5.2, category 4 |
| Vegas **closing** line | ❌ Never as a feature | Same — closing line is worse than opening for this purpose |
| Rest days, home/away, divisional flag | ✅ Yes | Known before kickoff, not outcome-derived |
| Injury report status (if added later) | ✅ Yes, if using the report available before kickoff | Must be time-stamped correctly if added |

### 5.4 Point-in-time feature computation — required implementation pattern

**Do not use plain `.expanding()` or `.rolling()` without excluding the current row.** Two correct patterns:

**Pattern A — rolling window with `closed='left'`:**
```python
# closed='left' explicitly excludes the current row from its own window —
# this is the critical argument that prevents aggregation leakage.
df = df.sort_values(['team', 'game_date'])
df['qb_epa_last8'] = (
    df.groupby('team')['qb_epa_per_dropback']
      .rolling(window=8, closed='left')
      .mean()
      .reset_index(level=0, drop=True)
)
```

**Pattern B — point-in-time join with `pd.merge_asof`:**
```python
# merge_asof matches each game to the most recent prior row of team_game_log —
# allow_exact_matches=False is required so a game cannot match itself.
schedule_with_features = pd.merge_asof(
    schedule.sort_values('game_date'),
    team_game_log.sort_values('game_date'),
    on='game_date',
    by='team',
    direction='backward',
    allow_exact_matches=False,
)
```

Either pattern `MUST` be used for every rolling/aggregated feature. `.expanding().mean()` or `.rolling().mean()` **without** `closed='left'` (or an equivalent manual shift) `MUST NOT` appear in the feature pipeline.

### 5.5 Validation protocol — run every check before trusting a result

1. **Timestamp assertion (automated, run on every pipeline execution):** for every row, assert that every feature's underlying data timestamp is strictly earlier than that row's game kickoff time. Implement as a test that fails the build if violated — this is the single most important test in the project.
2. **Label-shuffle test:** randomly permute the win/loss labels in the *training* set only, train the identical pipeline, then evaluate on the real, unshuffled test labels. Expected result: ~50% accuracy. A result meaningfully above 50% means the pipeline itself is leaking, independent of what the labels say.
3. **Feature-importance inspection:** after training, inspect feature importances (or logistic regression coefficients). If one feature accounts for a disproportionate share of importance (rough guideline: more than ~40–50% on its own), manually verify that feature's computation against Section 5.4 before trusting the model.
4. **Ablation sanity check:** remove the single most important feature and retrain. Accuracy `SHOULD` degrade gracefully. A collapse to near-baseline suggests the model was overly reliant on one feature; an *increase* in accuracy after removing a feature is a red flag that the removed feature was actively harmful (possibly a leak interacting badly with something else).

### 5.6 Suspiciously-high-accuracy protocol

If test-set accuracy exceeds **~70%**, this `MUST NOT` be reported or treated as a finished result. Before accepting it:

1. Re-run all four checks in Section 5.5.
2. Manually re-verify every rolling/aggregated feature against the patterns in Section 5.4.
3. Confirm the walk-forward split (Section 7) was actually used and no random shuffling crept into the split logic.
4. Re-run on one additional held-out season not previously touched. If accuracy drops sharply on the new season, the original number was likely inflated by leakage or overfitting to the specific test season, not a generalizable model.

Only after all four checks pass should a result above ~70% be treated as a genuine, reportable result rather than a bug.

### 5.7 Realistic accuracy ceiling — read this before chasing "high percentage"

NFL win/loss prediction has a well-known ceiling in the sports analytics community, independent of model sophistication: published work in this space, and the historical performance of betting markets, generally lands in the **mid-60s to high-60s percent** accuracy range for win/loss prediction using team/QB/defense-level features. Vegas favorites — the market's own best estimate — win somewhere in that same general range historically, not dramatically higher.

**Practical implication:** a model that reaches, say, 65–68% test accuracy on a proper walk-forward split is a strong, legitimate result for this problem — not a disappointing one. A number significantly above that `SHOULD` be treated with more suspicion than excitement until it survives Section 5.6. "High percentage rate" in this domain means *beating the baseline by a real margin while staying inside a believable range*, not approaching numbers like 85–90%, which are not achievable for single-game NFL outcomes with legitimately available pre-game information.

---

## 6. Train / Validation / Test Strategy

Random k-fold cross-validation on individual games `MUST NOT` be used — it allows the model to train on games from later in a season (or later seasons) to predict earlier ones, which is a form of temporal leakage (Section 5.2, category 2), since team performance is correlated across nearby weeks.

**Required split — season-based walk-forward:**
- Train: 2002–2021
- Validate: 2022 (hyperparameter tuning and model comparison happen here only)
- Test: 2023–2024, untouched until final evaluation

**Recommended extension (Section 5.6, check 4, also satisfies this):** true walk-forward validation — train on 2002–2021 → test 2022; train on 2002–2022 → test 2023; train on 2002–2023 → test 2024. Reveals whether performance is stable across seasons or was a one-season fluke.

---

## 7. Modeling Plan — Phase 1 (Win/Loss Only)

### 7.1 Model progression
- **Baseline:** `LogisticRegression` — interpretable, coefficients can be sanity-checked (QB EPA `SHOULD` have a positive weight, defensive EPA allowed `SHOULD` have a negative weight on opponent win probability, etc.).
- **Next:** `RandomForestClassifier`.
- **Next:** `HistGradientBoostingClassifier` (scikit-learn's built-in gradient boosting — no external library needed, satisfies the $0/no-extra-dependency constraint).

### 7.2 Legitimate accuracy-maximization techniques (use these, not leakage, to push performance)
- **Matchup differential features:** instead of only raw team stats, compute differentials directly relevant to the matchup, e.g. `home_qb_epa_last8 - away_def_epa_allowed_last8`. These interaction-style features are typically more predictive than raw independent stats.
- **Recency weighting:** an exponentially-weighted rolling average (`.ewm()` in pandas, with the same current-row-exclusion discipline as Section 5.4) `SHOULD` be tried against a simple rolling mean — recent form often matters more than a flat 8-game average.
- **Elo-style team rating as an additional feature:** a simple, free, well-established Elo rating system (update after each game based on result and margin) is a strong standalone predictor for NFL win/loss and combines well with EPA-based features. This is free to compute and `SHOULD` be added as one more engineered feature, not a replacement for the EPA features.
- **Hyperparameter tuning:** use `RandomizedSearchCV` or `GridSearchCV`, but the cross-validation splitter passed to it `MUST` respect time order (a season-based or `TimeSeriesSplit`-style splitter) — not the default random k-fold, for the same reason as Section 6.
- **Feature pruning:** drop features with negligible importance after the ablation check in Section 5.5 — fewer, cleaner features often generalize better than a large noisy set on a dataset this size (~5,000–6,500 rows).

### 7.3 Calibration
Tree-based models often produce directionally correct but poorly calibrated probabilities (clustered near 0.5–0.7 even for lopsided games). Wrap the final classifier in `CalibratedClassifierCV` if predicted probabilities themselves need to be meaningful — required if probabilities will ever be compared to Vegas implied probabilities (Section 2 stretch goal).

---

## 8. Evaluation Metrics — Phase 1

| Metric | Purpose |
|---|---|
| Accuracy | Primary metric — direct comparison to baseline (Section 2) |
| Log loss | Measures probability quality, not just the pick |
| Brier score | Same purpose as log loss, more interpretable scale |
| ROC-AUC | Overall discriminative ability independent of a specific probability threshold |

All metrics `MUST` be reported on the **held-out test set only**. Validation-set numbers are for model selection during development and `MUST NOT` be reported as final results — doing so is a soft form of overfitting to the validation data.

---

## 9. Known Risks & Hard Truths

1. **Market efficiency is real.** Sportsbooks price games professionally and adjust on injury news in real time. Beating the closing line is a stretch goal (Section 2), not the bar for success.
2. **Small season sample size.** Only 272 regular-season games per season; rosters and coaching change year to year, so a model overly tuned to 2015–2018 patterns may not generalize to a team that overhauled its scheme in 2023.
3. **Injuries are the largest unmodeled variable.** A backup QB starting is a bigger performance swing than rolling EPA averages will capture unless a "starting QB changed this week" flag is explicitly added — a reasonable Phase 1.5 addition once the core MVP works.
4. **Don't let "beat the baseline" quietly become "hit an unrealistic number."** Section 5.7 exists specifically because the instinct to chase a higher accuracy number is exactly what leads to leakage. A well-built model at 65–68% is success; the instinct to keep pushing past 70% without re-auditing is the risk.

---

## 10. Tech Stack & Environment Setup

**Run everything below in your local terminal** — whichever one is used for Python work (Mac Terminal.app, iTerm2, or VS Code's integrated terminal, opened to the project folder). These commands run on the local machine, not in this chat session.

```bash
# In your local terminal, in the folder where the project should live:
mkdir nfl-prediction
cd nfl-prediction

# Create an isolated Python environment so these packages don't
# conflict with other Python projects on the machine:
python3 -m venv venv

# Activate it (required every time a new terminal session is opened
# to work on this project):
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows equivalent, if applicable

# Install dependencies — all free, no paid packages or API keys required:
pip install pandas scikit-learn nfl_data_py jupyter matplotlib seaborn joblib
```

| Package | Role | Cost |
|---|---|---|
| `pandas` | Data loading, cleaning, feature engineering | Free |
| `scikit-learn` | Models, splitting, metrics, calibration | Free |
| `nfl_data_py` | Play-by-play and schedule data | Free |
| `jupyter` | Exploratory analysis notebooks | Free |
| `matplotlib` / `seaborn` | EDA plots | Free |
| `joblib` | Saving trained models to disk | Free |

No API keys, subscriptions, or cloud accounts are required for Phase 1.

---

## 11. Repository Structure — Phase 1

```
nfl-prediction/
├── venv/                      # (not committed to git)
├── data/
│   ├── raw/                   # untouched pulls from nfl_data_py, cached locally
│   └── processed/             # cleaned, feature-engineered datasets ready for modeling
├── notebooks/
│   └── 01_eda.ipynb           # exploratory analysis
├── src/
│   ├── data_ingest.py         # wraps nfl_data_py calls, caches to data/raw
│   ├── features.py            # rolling QB/defense feature computation (Section 5 rules enforced here)
│   ├── leakage_checks.py      # implements the Section 5.5 validation protocol as automated tests
│   ├── train_winner.py        # Phase 1 classification pipeline
│   └── evaluate.py            # metrics/reporting for Phase 1
├── models/                    # saved .joblib model files
├── requirements.txt
└── README.md
```

`train_spread.py` and `train_total.py` are intentionally **not** included yet — they belong to Section 12 and `MUST NOT` be started until Phase 1 is complete and reviewed.

---

## 12. Deferred Scope — Full Suite (Spread & Total Regression)

**Status: fully specified, but explicitly NOT part of the current build. Do not implement anything in this section until Phase 1 (Section 13) is reviewed and the decision is made to proceed.**

This section is preserved so the full original vision isn't lost — only postponed.

### 12.1 Spread (regression)
- **Baseline:** `LinearRegression` predicting home margin from the same rolling QB/defense features as Phase 1, plus a home-field indicator.
- **Next:** `RandomForestRegressor`, `HistGradientBoostingRegressor`.
- **Reality check:** point spreads have high game-to-game variance (a single turnover can swing a game by two scores independent of team quality). Expect MAE in the 9–11 point range even for well-built models — the betting market itself doesn't do dramatically better.
- **Success bar:** beat "predict the average historical home-field margin (~2.5 points) for every game."

### 12.2 Total points (regression)
- Same model progression as spread.
- Additional features to reintroduce here: pace of play (plays per game), wind speed (suppresses passing and total scoring), dome vs. outdoor.
- **Success bar:** beat "predict the league-average total (~44–46 points) for every game."

### 12.3 Why these were deferred, not cut
The feature pipeline built in Phase 1 (Sections 3–5) is almost entirely reusable here — the rolling QB/defense features don't change, only the target variable and model type do. Deferring these targets costs little rework later; attempting all three at once from the start, per Section 1.2, was the higher-risk path.

---

## 13. Project Phases / Milestones (Phase 1 Only)

| Phase | Deliverable | Notes |
|---|---|---|
| 0 — Setup | Environment + repo scaffolding | Section 10 |
| 1 — Data acquisition & EDA | Raw data pulled and cached; sanity-check EPA distributions, home win rate, score distributions against known football facts | Confirms the data is trustworthy before building on it |
| 2 — Feature pipeline | Rolling QB/defense features built using Section 5.4 patterns exclusively | Highest-risk phase — do not proceed until Section 5.5 checks pass |
| 3 — Leakage audit | Run full Section 5.5 protocol on the Phase 2 pipeline | Gate — do not proceed to modeling until this passes |
| 4 — MVP model | Baseline `LogisticRegression`, evaluated against the home-team baseline (Section 2) | First "does this work at all" checkpoint |
| 5 — Model iteration | Random Forest / HistGradientBoosting, calibration, Section 7.2 techniques | |
| 6 — Walk-forward backtesting | Multi-season validation per Section 6 | |
| 7 — Review gate | Compare results to Section 2 criteria; decide whether to proceed to Section 12 (deferred spread/total scope) | This document should be updated at this checkpoint, not before |

---

## 14. Out of Scope (Phase 1)

- Spread and total prediction (Section 12 — deferred, not cut)
- Live/in-game win probability updates
- Deep learning approaches (unnecessary for tabular data at this scale, and outside the scikit-learn-only, $0-cost constraint)
- Player prop predictions
- Betting bankroll/staking strategy
- Automated weekly deployment

---

## 15. Open Assumptions

- Historical range defaulted to **2002–present**.
- Assumed a **solo, local project** — no team collaboration tooling, no cloud deployment.
- Assumed regular-season games only; playoff games have different dynamics (extra rest, elevated stakes, tougher average opponent) and would need separate handling — not addressed above.
- Assumed no strict deadline; phases in Section 13 are sequenced by dependency, not calendar time.
- Assumed the $0-budget constraint applies indefinitely, including to the deferred scope in Section 12, not just to Phase 1.

Flag anything above that doesn't match intent — cheaper to change now than after the feature pipeline is built around it.
