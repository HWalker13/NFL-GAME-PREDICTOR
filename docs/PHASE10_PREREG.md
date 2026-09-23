# Phase 10 Pre-registration — 2026 Shadow Mode

**Status:** APPROVED

**Approved:** 2026-09-23 by the project owner, before any official (`--live`) prediction existed. `predict_week.py --live` checks for the exact line above (whole-line match).

**Date drafted:** 2026-09-22 (US/Pacific)
**SPEC reference:** Section 13, Phase 10; Section 13.1 (freeze rule); Section 14 (paper trading only).
**First official week:** Week 4 (Thursday 2026-10-01, `2026_04_PIT_CLE`, 20:15 ET).

This document fixes, before any official prediction exists, what is predicted, how paper bets are placed, how everything is graded, and the only two decisions the season can produce. Nothing here may be changed after approval except through a dated amendment at the bottom that states what changed and why, and never in a way that uses results already seen.

---

## 0. Summary for a cold reader

The project's model predicts NFL home wins from pre-game features: rolling QB/defense EPA, Elo, and context. In 2026 it runs in **shadow mode**. Each week, before kickoff, a frozen model logs its win probability for every game, next to the betting market's probability from a timestamped line snapshot. A flat 1-unit **paper** bet is recorded wherever the model and the market disagree by at least 4 points. After the games, everything is graded. No real money is involved at any point in 2026 (SPEC 14).

Two models run side by side:
- the **champion**, whose bets are the official record;
- a **challenger** (the same frozen model plus a probability calibrator), whose bets are hypothetical.

Two decisions are pre-registered:
- a single midseason checkpoint (after week 9): may the challenger replace the champion?
- a season-end verdict: is there evidence of a betting edge?

The honest prior on both is "no" (Sections 4–5).

---

## 1. Models (frozen; sha256 pinned in `scripts/live/common.py: FROZEN_MODELS`)

| | Champion | Challenger (calibrated) |
|---|---|---|
| File | `models/live_2026_champion.joblib` (read-only) | `models/live_2026_challenger_calibrated.joblib` (read-only) |
| sha256 | `f813a2b0cb58d3202c547ca30c8dc90e242a84a8a9b74d5a82f43293571c1006` | `185d565f2c1202cb693a527766f875d259b2c118140933efe776621c13daad28` |
| Estimator | `clone()` of `logreg_winner_tuned.joblib` (sha256 `11cb538e…`): median impute → StandardScaler → `LogisticRegression(C=0.01, max_iter=1000, random_state=0)` | **the champion's fitted pipeline, unchanged** (metadata records the champion's sha256), followed by a sigmoid calibrator |
| Features | the 67 frozen features (`features.feature_columns()`) | identical (same feature rows) |
| Training data | every played 2002–2025 REG game, n = 6,208 (ties dropped) | identical |
| `ELO_HOME_ADV` | **44.72586959078301** (frozen, Phase 5A: 2,890/5,124 home wins, seasons ≤2021) | identical |
| Training-data sha256 (keys + features + label, row order) | `ce45ee83…452a`; file = canonical `game_features.parquet` (`747afbf4…4898`) | identical |
| `src/features.py` sha256 the model was trained with | `6be77b8d…6de7` | same |
| Probabilities | raw `predict_proba` (Phase 8: calibration not adopted for the champion) | `sigmoid(slope·logit(p_champion) + intercept)`, **slope = 0.883211, intercept = −0.039094** (n_fit = 1,355) |

**Challenger calibrator.** It uses the existing Phase 8 method in `src/calibration.py`, with the same settings. The target season is T = 2026:
- For each season S in T−5..T−1 = **2021–2025**, a model with the champion's hyperparameters is trained on seasons < S and predicts S.
- The pooled out-of-fold predictions (1,355 games) are fit with one unpenalised logistic regression on logit(p): `LogisticRegression(C=np.inf)`.

**The 2021–2025 window follows from the standard T−5..T−1 rule. It was not hand-picked; it happens to exclude the 2020 no-crowd season.** The calibrator targets the LR **intercept**, where the documented home bias lives (Phase 8: negative calibrator intercepts in 2022 and 2025).

It uses 2021–2025 outcomes, including the already-used 2023–2025 test and holdout seasons, exactly as Phase 8's calibrators did. That is acceptable because the challenger is **never evaluated on any historical season**; its only evaluation is live 2026.

**Method reproduction check (not an evaluation).** Run for T = 2025 on data ≤2024, the same calibrator code reproduces Phase 8's recorded LR-tuned calibrator exactly: slope 0.9345212441710962, intercept −0.09909697171291454, n_fit 1,339.

Note that the T = 2026 intercept (−0.039) is smaller than T = 2025's (−0.099). Adding 2025 and dropping 2020 from the OOF window weakened the home over-prediction the calibrator corrects.

**Discarded challenger (never used live).** An earlier challenger re-derived `ELO_HOME_ADV` from 2021–2025 with the 44.73 method (733/1,355 → 28.525) and retrained on 2002–2025 (file sha256 `07ea1be6…3116f`). It was built and **discarded before any live prediction**, and the file was deleted.
- Why it was discarded: `ELO_HOME_ADV` enters only the Elo *update* (expected score), never the features themselves.
- Label-free on the 240 unplayed 2026 games: **0/240 picks differed from the champion; max probability difference 0.0037** (intercepts +0.2741 vs +0.2739).
- It could not test the home-field drift. The calibrated challenger replaces it because it acts on the intercept.
- `src/live_models.py: derive_hfa` is kept as the record of the 28.525 computation.

**Label-free comparison, calibrated challenger vs champion** (240 unplayed 2026 games, weeks 3–18, features as of 2026-09-23; `python -m scripts.phase10_challenger_compare`; not an evaluation):

| Quantity | Value |
|---|---:|
| Mean p_home: champion / challenger | 0.5482 / 0.5348 |
| Mean (challenger − champion) | −0.0134 |
| Mean \|difference\| / max \|difference\| | 0.0183 / 0.0342 |
| Picks that differ | **3 / 240** (home picks 142 → 139) |
| Paper-bet decision differs (games with lines in the latest snapshot: 32, weeks 3–4) | **5 / 32** (all 5: champion bets home, challenger no bet) |

The week-3/4 rows use features as of now. The week-4 values will change once week 3 is played.

**Freeze enforcement.** Every `predict_week` run refuses unless three things hold:
- both model files hash to the values above;
- `src/features.py` hashes to the value the models were trained with;
- the champion's `ELO_HOME_ADV` equals the frozen constant.

**Reproduction check (Phase 10 B2, not an evaluation).** The same training code, run on seasons ≤2024, reproduces Phase 8's 2025 LR-tuned raw probabilities bit for bit: 271/271 games, max |diff| = 0.0.

---

## 2. Paper-bet rule (exactly as implemented in `scripts/live/common.py: paper_bet`)

For every predicted game and each model:
1. **Vig-free implied probability:** convert both moneylines from the prediction snapshot to raw implied probabilities, then normalise them to sum to 1 (the Phase 7 convention).
2. **Edge for each side:** model probability minus vig-free implied probability. The two edges sum to zero, so at most one side can qualify.
3. **Bet** 1 unit on the side whose edge is **≥ 0.04**, at that side's **prediction-snapshot moneyline, vig included**. The comparison is made after rounding the edge to 10 decimals, so an edge that is exactly 0.04 on paper qualifies despite binary floating point.
4. **Settlement:**
   - win: profit = price/100 for positive odds, 100/|price| for negative odds;
   - loss: −1;
   - **tie: void, stake returned, 0 units.**
5. **Missing moneyline** (either side) in the prediction snapshot: no bet, logged as `missing_line`.
6. **Flat stakes only:** no staking, bankroll or Kelly logic (SPEC 14).
7. The **champion's** bets are the **official** record (`bet_status = official`). The **challenger's** bets are computed the same way and labelled `hypothetical`.

---

## 3. What is logged and how it is graded

**Prediction snapshot.** This is the most recent `data/raw/snapshots/schedules_2026_<UTC>.parquet` at run time. `predict_week` refuses unless:
- the snapshot is at least as new as the latest raw pull of 2026 pbp and schedule, and byte-identical to `schedules_2026.parquet`, the file the features are built from;
- **with `--live`, the snapshot is at most 6 hours old at run time** (`common.check_snapshot_age`), so a Monday refresh cannot pass on Wednesday;
- pbp contains every completed game before the week;
- no earlier-week game is past kickoff without a final score.

Every row records the snapshot filename and pull time, and `week_NN_run.json` records the snapshot's age at run time.

**Logged per game per model** (`week_NN_predictions.csv`, write-once, read-only):
- identifiers: game_id, kickoff (UTC), home, away;
- the model: name, file and sha256;
- the prediction: p_home and pick (home if p_home > 0.5);
- the line: snapshot file and pull time, both moneylines, spread, vig-free implied p_home;
- the bet: edge for each side, paper_bet (home / away / none), bet_status, no-bet reason, bet price, stake;
- the prediction time.

Two companion files are also written, both write-once and read-only: `week_NN_run.json` (guards, hashes, exclusions) and `week_NN_features.parquet` (the exact feature rows each model saw).

**Per-game grading** (`grade_week.py` → `week_NN_graded.csv`, regenerable; it never touches the predictions file):
- **Final scores** come from the latest snapshot, after a fresh refresh; the same freshness guard applies.
- **Accuracy, log loss, Brier.** Ties are excluded, as in every evaluation in this project. Games with no final score yet are `pending` and excluded.
- **Bet result in units.** Ties are void.
- **Closing-side line = the "last pre-kickoff snapshot":** the latest snapshot whose pull time is strictly before that game's kickoff. It is labelled that way everywhere and is **not** called the close, because snapshots exist only when the owner refreshes. If that snapshot is not later than the prediction snapshot, CLV is **excluded** for that game and counted. That is the normal case for Thursday games and for any game before the Sunday refresh (including 09:30 ET international games if the Sunday refresh is late).
- **Bet CLV:** the vig-free implied probability of the **bet side** in the last pre-kickoff snapshot minus the same in the prediction snapshot. **Positive = the line moved toward the bet.**
- **All-game line movement toward the model:** `(p_late − p_pred) × sign(p_model − p_pred)`, in home terms, for every game with a measured late line, bet or not.

**Scorecard** (`scorecard.py` → `docs/live/SCORECARD_2026.md`, regenerable). It shows cumulative and weekly tables for:
- **predictors:** champion, challenger, home baseline, and the Vegas favorite from the prediction-snapshot spread (moneyline fallback on a pick'em, otherwise unresolved and excluded);
- **metrics:** accuracy (Wilson 95% CI), log loss and Brier (t 95% CI), and the vig-free implied log loss/Brier for comparison;
- **betting:** the paper-bet record (W-L-void, pending), units, and ROI = units per unit staked on settled bets (t 95% CI);
- **line movement:** mean bet CLV (t 95% CI, n measured, n excluded, one-sided p) and mean all-game line movement toward the model (t 95% CI).

A plain-language header explains each metric and why small samples mislead.

**Edge-size breakdown (DESCRIPTIVE ONLY).** The scorecard also splits paper bets by the bet side's edge (model probability minus vig-free implied probability, at the prediction snapshot) into three buckets: **[0.04, 0.06), [0.06, 0.08) and [0.08+)**. The comparison uses the bet rule's 10-decimal rounding, so an edge of exactly 0.06 or 0.08 falls in the upper bucket. It is reported for the champion and the challenger separately. For each bucket it shows the bet count, W-L (void, pending), units, ROI (t 95% CI) and mean bet CLV (t 95% CI, n measured); buckets with fewer than 30 bets are labelled "small sample".

This breakdown is descriptive only. It cannot change the season-end verdict (Section 4), the paper-bet threshold, or the checkpoint decision. Its purpose: if the model carries real information, larger edges should perform better; a flat or non-monotonic pattern indicates edges are mostly model error. Small per-bucket samples are expected and will be labelled.

---

## 4. Season-end betting verdict (decided once, after week 18 is fully graded)

**"Evidence of an edge" ONLY IF** the official bets' mean bet CLV is > 0 **and** a one-sided one-sample t-test (H1: mean > 0) gives **p < 0.05**. The units ROI and its 95% CI are reported alongside. **ROI alone never counts as evidence**, in either direction. The rule uses rows logged `bet_status = official`, so it follows the official model through a checkpoint swap, if there is one.

**Honest expectation: the rule most likely loses money, and CLV most likely shows no edge.**
- In Phase 7 (2023–24), when a model disagreed with the Vegas favorite it was right only 32–34% of the time (n = 91–111, McNemar p < 0.01).
- In Phase 8 (2025), LR-tuned was right on 11 of 32 disagreements (34%).
- The 4-point rule bets exactly on the model-vs-market disagreements, and pays ~4.3% vig on every bet.
- A negative ROI with CLV ≈ 0 or below is the expected, uninteresting outcome. One season is far too small to establish anything else by ROI (Phase 7 §5 note 1).

---

## 5. Midseason checkpoint (once, after week 9 is fully graded)

**Rule (matches Phase 8's log-loss rule):** the challenger replaces the champion as the official model for weeks 10–18 **ONLY IF** both hold over **all shadow weeks (4–9)**:
- the mean per-game log-loss difference (challenger − champion) is **< −0.010**;
- a **one-sided paired t-test** (H1: mean < 0) gives **p < 0.05**.

The comparison uses games graded for both models, with ties and excluded games left out. Otherwise the champion continues and the comparison is recorded. `scorecard.py: checkpoint_section` implements this (`CHECKPOINT_MARGIN = 0.010`) and reports "not yet evaluable" until every game of weeks 4–9 is graded.

**Mechanics if the challenger qualifies:** a single, recorded edit sets `OFFICIAL_MODEL = "challenger"` in `scripts/live/common.py`, effective from week 10. Earlier rows keep their logged `bet_status` and are never rewritten.

**Expectation: no swap.** The shadow window holds **88 games** (weeks 4–9 = 16+15+14+14+14+15). The two models differ by 0.018 in probability on average and pick differently in about 1% of games, so a per-game log-loss gain beyond 0.010 that is also significant at n = 88 is unlikely. **The comparison's main value is informing the 2027 model:** whether the recent-era intercept correction improves live probabilities.

---

## 6. Exclusion rules

| Situation | Handling |
|---|---|
| Game kicked off at or before the run time (or already has a result) | **Never predicted.** Excluded and listed in `week_NN_run.json`. |
| Missing moneyline in the prediction snapshot | Predicted; **no bet** (`missing_line`); CLV excluded (`missing line at prediction`). |
| No snapshot after the prediction snapshot and before kickoff | CLV and line movement **excluded** for that game, and counted. |
| Missing moneyline in the last pre-kickoff snapshot | CLV excluded, counted. |
| Tie | Excluded from accuracy / log loss / Brier; bets void (0 units). |
| No final score at grading time | `pending`: excluded everywhere, counted; regrade later. |
| Earlier-week game unplayed and not yet due (postponed) | Logged; its teams' features use only completed games. |
| Earlier-week game past kickoff with no score, or completed game missing from pbp | `predict_week` **refuses**; wait and refresh. |
| Stale snapshot, or snapshot ≠ schedule file | `predict_week` / `grade_week` **refuse**; refresh. |
| Frozen model or `features.py` changed | `predict_week` **refuses**. |

---

## 7. Weekly schedule and owner commands

All commands run in your **local terminal** (Terminal.app, iTerm2, or VS Code's integrated terminal), from the **project root**:

```bash
cd "/Users/holdenanderson/Personal Projects/NFL-GAME-PREDICTOR"
source venv/bin/activate
```

**Wednesday — predict (must finish before Thursday's 20:15 ET / 17:15 PT kickoff).** Refresh and predict in the same sitting, so the logged line is current. `week_NN_run.json` records the snapshot's age.

```bash
python -m src.data_ingest --refresh-season 2026
python -m scripts.live.predict_week --season 2026 --week N --live
git add data/raw/snapshots data/raw/pull_metadata.json data/live/2026/predictions
git commit -m "Phase 10: week N predictions (official)"
git push
```

**Sunday morning — refresh only.** This gives the Sunday and Monday games a later pre-kickoff line. Do it **before 09:30 ET (06:30 PT)**, because weeks 4, 5, 6, 7, 9 and 10 each have a 09:30 ET game; otherwise before 13:00 ET (10:00 PT).

```bash
python -m src.data_ingest --refresh-season 2026
git add data/raw/snapshots data/raw/pull_metadata.json
git commit -m "Phase 10: week N Sunday line snapshot"
git push
```

**Tuesday — grade + scorecard** (after Monday night's game is final; Phase 9 saw pbp posted ~14 h after the last kickoff).

```bash
python -m src.data_ingest --refresh-season 2026
python -m scripts.live.grade_week --season 2026 --week N --live
python -m scripts.live.scorecard --season 2026 --live
git add data/raw/snapshots data/raw/pull_metadata.json data/live/2026/graded docs/live/SCORECARD_2026.md
git commit -m "Phase 10: week N graded + scorecard"
git push
```

Additional refreshes at any time (e.g. Saturday) are harmless and only add snapshots. The agent never runs any `--live` command.

---

## 8. Official predictions are never edited

`week_NN_predictions.csv`, `week_NN_run.json` and `week_NN_features.parquet` are write-once:
- they are published with an atomic hard link that fails if the name exists, then made read-only;
- `predict_week` refuses if any of them exists.

A bug found later is handled in one of two ways:
- **Grading or scorecard bug:** fixed in `grade_week.py` / `scorecard.py`, and the graded files and scorecard are regenerated. Those files are regenerable by design.
- **Prediction-time bug:** fixed going forward only, with a dated amendment below stating which weeks it affected. Past predictions stay exactly as logged.

---

## 9. Known limitations (recorded, not fixed)

1. **Retractable roofs.** `roof` is null in the schedule until game day for retractable-roof stadiums (37 games of 2026). On Wednesday, `roof_indoor` for those games is NaN and the frozen model's train-median imputer treats it as outdoors (0). There was no such null in 2002–2025 training data.
2. **Early-season shrinkage.** In 2026 the in-season weight n/(n+4) is 0 in week 1, 0.20 in week 2, 0.33 in week 3, and 0.43 in week 4 for teams with 3 games. Week-4 features therefore still lean ~57% on 2025 team rates.
3. **Lines.** A snapshot is whatever nflverse's single, overwritten line field held at pull time; its age at the source is unknown (Phase 7 §3.1, Phase 9 R4). The last pre-kickoff snapshot is typically hours before kickoff, not the close.
4. **CLV independence.** The one-sided t-test treats bets as independent. Bets in the same week share market-wide moves, so the test is somewhat optimistic.
5. **Home bias and bet volume.** The champion carries the known home-win over-prediction (Phase 8 negative calibrator intercepts). On the 32 games that currently have lines (weeks 3–4, features as of 2026-09-23, not official), the champion's rule would bet **21 of 32 games, 16 of them on the home side**. The calibrated challenger would bet 16 (11 home). Expect a high, home-heavy official bet volume; the scorecard shows the home count. The 4-point threshold is kept as agreed; it was not revised after seeing the label-free bet volume, since choosing a threshold by its bet count would be selecting a rule by looking at data.
6. **International venues.** The five 2026 venues (Melbourne, Rio, Paris, Madrid, Munich) were added to `features.STADIUM_GEO`. They are absent from every 2002–2025 schedule, and the 2002–2025 rebuild is byte-identical. Travel to Melbourne (~12,000 km) is beyond anything in training.

---

## 10. Owner decisions on the draft's open points (2026-09-23)

- **D1 — challenger replaced.** The Elo-HFA challenger was discarded before any live prediction (Section 1). The new challenger is the frozen champion plus the T = 2026 OOF walk-forward sigmoid calibrator.
- **D2 — margin added.** The checkpoint requires a mean log-loss difference < −0.010 AND one-sided paired t p < 0.05, matching Phase 8 (Section 5). A swap at n = 88 is unlikely; the comparison mainly informs 2027.
- **D3 — absolute freshness added.** `--live` refuses a prediction snapshot more than 6 hours old at run time (Section 3). The Sunday refresh deadline stays 09:30 ET (06:30 PT).

---

## Appendix A — Phase 10 build results

- **A1 (check-4 exception):** `python -m src.leakage_checks` exit 1 → **exit 0**. The documented `away_def_epa_early_ewm` false positive is accepted only on an exact match (168/269 → 170/269). Any change in those counts, a collapse, or any other feature tripping check 4 fails loudly (8 tests).
- **A2 (live features):** `data/processed/game_features_live.parquet` holds 2002–2026, with 272 rows for 2026: 32 played and 240 unplayed.
  - Gate 1: 2002–2025 rows are identical to canonical (exact values and dtypes).
  - Check 1 passes on all 272 rows of 2026, and the Elo point-in-time check passes on 12,448 team-games.
  - Unplayed rows carry no label, raw metric or Vegas column, and give identical rows under opposite placeholder results.
  - Masked-week equivalence: the unplayed path equals the played path exactly for 2026 wk 2 (16), 2025 wk 10 (14) and 2025 wk 2 (16).
  - After the `STADIUM_GEO` addition, the 2002–2025 rebuild is byte-identical (`747afbf4…` / `d09ef463…`).
- **B:** as in Section 1. The champion was saved 2026-09-23T03:46Z. The calibrated challenger was saved 2026-09-23 after both reproduction checks passed (B2 raw probabilities; T = 2025 calibrator).
- **Tests:** `python -m scripts.test_phase10_live`, `scripts.test_phase9_ingest` and `scripts.test_phase8_gate`; see the latest run in the session report.

## Appendix B — code hashes at drafting time

| File | sha256 |
|---|---|
| `scripts/live/common.py` | `010aae418f7e2bb6df665071aabdd9e59a65e485c6c8bf98b31b0e42e80a1ce2` |
| `scripts/live/predict_week.py` | `f5eee56ca1ffcec4132afc23dd0117228c048904998a3950560c8426a3bdc742` |
| `scripts/live/grade_week.py` | `4a137445bd6a653718fa19f9be3a06a669209bffca093318b7df9b19e1d8a6ac` |
| `scripts/live/scorecard.py` | `9e3726651c8d90ade948acb1d1908797f4918ad17ebf3f50a4ee4eaa784e554a` |
| `src/live_features.py` | `c4b827e8aee616c06fb2de3c168efbf42e442a800b522ad7fc06d67d66a3a486` |
| `src/live_models.py` | `83d6f7898d56cb987db24425f9886133428b7761fdfb57f69a471fd604d4a1e1` |
| `src/features.py` | `6be77b8d3547da6b9b64586bb104e61633a83221d82e249a705ed003ef156de7` |
| `src/leakage_checks.py` | `b47f91298353da7405c6621920daf0a4cbed6c211ffa89259dccc37e7b3012ae` |
| `scripts/phase10_challenger_compare.py` | `53b475cb389226c35ce5c291a8a2100ab93932b4fca68eedf37fc11cd550186b` |

Each official run also records `predict_week.py` / `common.py` hashes in its `week_NN_run.json`.

## Amendments

_None._
