# Metric audit — NFL Game Predictor (2026-11-12)

Scope: reconcile the four portfolio-site claims against the code and data. Nothing in the model,
feature or training code was changed, and no model was retrained or re-run on a test season. New
files exist only under `audit/`. Every number below comes from a script or command run in this
session (reproduction commands are in §9). Where an older file says the same thing, it is cited.

---

## 0. Reconciliation (read this first)

| # | Site claim | What it actually measures | Recomputed (this session) | Verdict | Why |
|---|---|---|---|---|---|
| 1 | "Predicts NFL winners. It beats the home-team baseline, loses to Vegas, and the write-up says so." | Straight-up winner probabilities for regular-season games, compared with always picking home and with the betting favorite | 2023-24: model 64.3%, home 54.4%, Vegas favorite 69.7%. 2025: 61.6% / 53.9% / 65.3%. Live 2026 (75 decided games): 58.7% / 46.7% / 60.0% | **MATCH** | True on both held-out sets. Caveat: on live 2026 so far the model's log loss (0.661) is *better* than the market's (0.684), but with n=75 the intervals overlap and nothing is settled. |
| 2 | "64.3% accuracy on 544 held-out games, vs. a 54.4% home-team baseline" | LR-tuned, pooled 2023 + 2024 regular season, walk-forward (train 2002–22 → 2023; 2002–23 → 2024), 0 ties | 350/544 = **64.34%** [95% CI 60.2–68.3]; home 296/544 = **54.41%** | **MATCH (number), MISLEADING (by omission)** | The number is exact. But it is the **highest of four statistically tied models** (62.9–64.3%), and LR-tuned was named the deployment model *after* these test results were known. Two later out-of-sample results are lower: the pre-registered **2025 holdout scored 61.6%**, and **live 2026 is at 58.7%** (n=75). |
| 3 | "Vegas hits 69.7%, so I report no betting edge." | Share of the 544 games won by the spread favorite in nflverse's line field (untimestamped, most likely the closing line) | 379/544 = **69.67%** [65.7–73.4] | **MATCH (number), MISLEADING (reasoning)** | How often the favorite wins is not a test of betting edge. Edge depends on price: a model can pick winners less often than the favorite and still find mispriced odds. The project never ran a betting test on 2023-24. This audit did (§4.2): the project's own bet rule, applied retroactively at the listed moneylines, loses **−54.15 units on 377 bets (ROI −14.4%)**. That, plus the market's better log loss (0.607 vs 0.643), is the real evidence. Also, the live 2026 scorecard shows **+10.98 units**, which a reader will see as contradicting "no edge" unless the site explains the pre-registered CLV rule. |
| 4 | Terminal clip of the week 7 picks with `cut -f2,4,5,7,11,12`, captioned "Week 7 picks as logged before kickoff, next to the final scores." | Fields: Game · Model picks · Model's chance (for the picked team) · **Vegas's chance for the same picked team** (not for the favorite) · Final score (away first) · **"Pick right?"** (straight-up pick correct, **not** a bet result) | Values match the write-once log exactly. Week 7: picks 8/14 right; paper bets 3–6, −3.06 units | **MISLEADING (partly)** | (a) Without the header row, "No" reads like "no bet". It means "pick wrong". The clip also drops the paper-bet columns. Example: PIT @ NO shows "Yes", but that week's official bet on that game (on NO) **lost**. (b) "Logged before kickoff" is true for the CSV record but not for the file shown: the `.md` page is regenerated after the fact. (c) The earliest server-side (GitHub-runner) timestamp after the week 7 commit is 2026-10-23T00:30:39Z, which is **15 minutes after NE @ CHI kicked off**, and NE @ CHI is the row in the clip. Pre-kickoff logging is server-corroborated for the other 13 week 7 games. For NE @ CHI it rests only on local git timestamps, which the author controls (§5). |

### Proposed corrected wording

1. **"Predicts NFL game winners. On seasons it never saw, it beats always-pick-home by 8–10 points and trails the Vegas favorite by 4–5 points. The write-up says so."**
2. **"64.3% on 544 held-out 2023–24 games (95% CI 60–68%) vs 54.4% for always picking the home team. A pre-registered 2025 test scored 61.6% (vs 53.9%). Live 2026 so far: 58.7% of 75 games."**
   Optional, for full disclosure: "best of four models that scored 62.9–64.3% and are statistically tied."
3. **"The betting market is better: the Vegas favorite won 69.7% of those games, and the market's probabilities beat mine (log loss 0.607 vs 0.643). Run against the listed moneylines, my betting rule would have lost about 14% per bet on 2023–24. I claim no betting edge, and the live test only counts closing-line value as evidence, which so far shows none."**
4. Caption: **"Week 7 picks from the write-once prediction log (committed Oct 21, before the first kickoff on Oct 22), next to final scores. Last column: was the pick right? (not a bet result)."** Show the header row. Better: use `cut -f2,4,5,7,9,11,12,13` so the paper bet and its result appear. Before keeping "before kickoff" for NE @ CHI, confirm the push time on GitHub's server-side Activity log (§5).

---

## 1. Inventory

### 1.1 Where metrics are computed or reported

| Location | Metrics | Population |
|---|---|---|
| `src/train_winner.py`, `train_winner_ensemble.py`, `train_winner_tuned.py` | val accuracy, log loss, Brier, ROC-AUC | 2022 REG, ties dropped (n=269); train ≤2021 |
| `src/evaluate.py` | home-baseline accuracy | any season list; REG, ties dropped |
| `src/leakage_checks.py` | check 2 label shuffle, check 4 ablation | train ≤2021 → 2022 only |
| `scripts/reproduce_walkforward_predictions.py` | pooled accuracy + McNemar; writes per-game `walkforward_test_predictions.parquet` | 2023 + 2024 REG (n=544) |
| `scripts/vegas_benchmark_2023_2024.py` | Vegas favorite/implied accuracy, LL, Brier, AUC; disagreement | same 544 games |
| `scripts/phase8_calibration_dev.py`, `src/calibration.py` | calibration LL deltas | OOF 2017–21 → 2022 |
| `scripts/phase8_holdout_2025.py` | 2025 holdout accuracy/LL/Brier/AUC, McNemar, Vegas | 2025 REG, 1 tie dropped (n=271) |
| `scripts/live/grade_week.py` | per game: correct, LL, Brier, home/Vegas correct, units, CLV | logged 2026 games |
| `scripts/live/scorecard.py` | cumulative accuracy (Wilson CI), LL/Brier (t CI), W-L-void, units, ROI, CLV + one-sided t, checkpoint test | all graded 2026 weeks |
| `scripts/live/readable_week.py` | per-week page (no new numbers, copies logged values) | one 2026 week |
| `README.md`, `docs/PHASE7_REVIEW.md`, `docs/PHASE8_RESULTS.md`, `docs/live/*` | reported values | as above |

**No betting/ROI metric was ever computed for 2023-24 or 2025.** The bet rule exists only in the live
2026 code (`scripts/live/common.py`). The retroactive bet numbers in §4.2 are new to this audit.

### 1.2 Bet decision logic (`scripts/live/common.py: paper_bet`, pre-registered in `docs/PHASE10_PREREG.md` §2)
- Vig-free implied P(home) = raw implied home / (raw home + raw away), from American moneylines.
- Edge(side) = model P(side) − vig-free P(side). The two edges sum to zero.
- **Bet 1 unit** on the side with edge **≥ 0.04** (edge rounded to 10 dp). Otherwise no bet. Missing line → no bet.
- **Price = the real moneyline for that side** at the prediction snapshot, vig included. **No −110 assumption.**
  Win pays price/100 (positive odds) or 100/|price| (negative); loss −1; tie void (0).
- Line used: the **prediction-time snapshot** (usually Wednesday), **not the opening or closing line**.
  CLV is measured against the last snapshot ≥ 6 h later and before kickoff, which is also not the true close.
- A bet can be on the team the model picks to *lose* (marked †). Live: 18 of 53 bets. Retroactive 2023-24: 182 of 377.
- Official record = champion bets only (`bet_status = official`). Season verdict = mean bet CLV > 0 with one-sided p < 0.05. ROI never counts.

### 1.3 Columns of `week_NN_picks.md` (champion table, `readable_week.champion_table`)
Rows start with `|`, so cut field 1 is empty and field *k* is column *k−1*.

| cut field | Column | Definition |
|---|---|---|
| **2** | Game | `AWAY @ HOME` |
| 3 | Kickoff (Pacific) | |
| **4** | Model picks | team with model P > 0.5 (P = 0.5 exactly → away) |
| **5** | Model's chance | model P(picked team wins), rounded to whole % |
| 6 | Vegas favorite | spread favorite: `spread_line > 0` → home; text "X by N" |
| **7** | Vegas's chance | **vig-free moneyline P for the *picked* team** (not the favorite). For NE @ CHI: 42% = market's P(NE). CHI was the favorite. |
| 8 | Gap (pts) | (5) − (7) before rounding |
| 9 | Paper bet | "Bet X (+g pts)", † if X is not the picked team, or "No bet" |
| 10 | If the bet wins | profit per $1 at the logged price |
| **11** | Final score | `AWAY a, HOME h` |
| **12** | Pick right? | **was the straight-up pick correct**: Yes / No / Tie / – (pending). **Not** a bet result. |
| 13 | Bet result (units) | Won +x / Lost −1.00 / Void / – |

Reproduced this session:
`grep ' @ ' docs/live/week_07_picks.md | cut -d"|" -f2,4,5,7,11,12` prints the 14 champion rows
(first: ` NE @ CHI | NE | 53% | 42% | NE 24, CHI 30 | No `) **plus 2 challenger-table rows**
(` IND @ MIN | MIN | 74% | +3.5 `, ` NYG @ HOU | HOU | 74% | -6.5 `), whose fields mean something
else (challenger pick / challenger chance / gap). Whether the site's elided `grep ...` pattern lets
those through is **unknown (ambiguous)**.

Note: `readable_week` without `--live` writes `data/live_scratch/2026/week_07_picks.md`
(gitignored). The tracked page is `docs/live/week_07_picks.md`. Both are derived views, regenerated
from the official CSV. The tracked one says "Generated 2026-11-02T20:18:49Z".

### 1.4 The held-out set
- **2023-24 walk-forward test (the site's 544):** seasons 2023 and 2024, `game_type == "REG"` only
  (no playoffs). Ties are dropped by `evaluate.completed_regular_season`. There were 0 ties, so
  272 + 272 = **544**. Step A trains on 2002–2022 and predicts 2023; Step B trains on 2002–2023 and
  predicts 2024. Hyperparameters and the 67-feature set were fixed using the 2022 validation season.
- **2025 pre-registered holdout:** 272 REG games, 1 tie dropped (`2025_04_GB_DAL`), **n = 271**,
  models trained 2002–2024, all four run on raw (uncalibrated) probabilities (verified in this session).
- **Live 2026:** official weeks 3–8 graded (week 8 = MNF only, week 9 entirely missed), 76 games
  logged, 1 tie, **75 decided**. Week 10 is logged but not yet graded.

### 1.5 How "Vegas" is defined
- **Historical (2023-24, 2025):** nflverse `schedules.spread_line`. It is a single field with no
  timestamp. Upstream overwrites it (Phase 9 saw about 1/3 of 2026 spreads change between pulls), so
  it is **most likely the closing or latest line, but this cannot be confirmed (ambiguous)**.
  - Favorite = sign of `spread_line` (> 0 = home favored). At 0 (pick'em), falls back to the lower moneyline; if still tied, "unresolved" and excluded.
  - 2023-24: 0 pick'ems, 0 unresolved, and 1 game where the spread favorite ≠ moneyline favorite.
  - 2025: 0, 0, 0.
  - The Vegas implied probability is the vig-free moneyline probability from the same row.
- **Live 2026:** the same fields, but from the **prediction-time snapshot**, not the close.
- The 2025 lines stored in `holdout_2025_predictions.parquet` still equal the current cached `schedules_2025.parquet` (0 changes).

**Sign-convention check (3 games, from `audit/out/metrics.json`).** The final score cannot show who
was favored. What confirms the sign is that `spread_line` agrees with a second, independent field:
the home moneyline.

| game | spread_line | home ML | away ML | read as | final (away–home) |
|---|---:|---:|---:|---|---|
| 2023_01_DET_KC | +4.0 | −198 | +164 | KC (home) favored | 21–20, favorite lost |
| 2023_01_ARI_WAS | +7.0 | −340 | +270 | WAS (home) favored | 16–20, favorite won |
| 2024_01_BAL_KC | +3.0 | −148 | +124 | KC (home) favored | 20–27, favorite won |

In all three, positive `spread_line` coincides with the home team holding the negative (favorite)
moneyline. Across all 544 games the two fields disagree once. From general knowledge, not
re-verified in this session: KC was favored in both openers and WAS was favored over ARI. DET @ KC
at 4.0 rather than the ~6.5 widely quoted early in that week is consistent with the field being a
late or closing line.

---

## 2. Recomputed metrics: 2023-24 held-out (n = 544)

Source: per-game probabilities in `data/processed/walkforward_test_predictions.parquet`. Labels
were re-derived in this session from `schedules_2023/2024.parquet` scores and match the stored
labels 544/544. Stored picks equal (p > 0.5) for every model. 95% Wilson CIs.

### 2.1 All games — headline model (LR-tuned)

| | 2023 (n=272) | 2024 (n=272) | Pooled (n=544) |
|---|---|---|---|
| **Model accuracy** | 169 = 62.13% [56.2, 67.7] | 181 = 66.54% [60.7, 71.9] | **350 = 64.34% [60.2, 68.3]** |
| Home baseline | 151 = 55.51% [49.6, 61.3] | 145 = 53.31% [47.4, 59.2] | **296 = 54.41% [50.2, 58.6]** |
| Vegas favorite (spread; pick'em → ML; 0 such games) | 185 = 68.01% [62.3, 73.3] | 194 = 71.32% [65.7, 76.4] | **379 = 69.67% [65.7, 73.4]** |
| Vegas vig-free implied, pick p>0.5 | 186 = 68.38% | 194 = 71.32% | 380 = 69.85% [65.9, 73.6] |
| Model Brier / log loss | 0.2339 / 0.6606 | 0.2183 / 0.6261 | **0.2261 / 0.6434** |
| Vegas implied Brier / log loss | 0.2187 / 0.6273 | 0.2002 / 0.5875 | **0.2094 / 0.6074** |

Other models, pooled (accuracy / Brier / LL): LR-untuned 342 = 62.87% / 0.2282 / 0.6479; RandomForest
343 = 63.05% / 0.2237 / 0.6380; HGB-tuned 342 = 62.87% / 0.2255 / 0.6418. **No model figure exceeds
70%** (max: LR-tuned 2024 alone, 66.54%), so the SPEC 5.6 flag does not fire. Model probabilities
are raw (uncalibrated), as in the project.

### 2.2 Disagreement with the Vegas favorite (LR-tuned)

| | n disagree | model right | rate [95% CI] | exact binomial p vs 50% |
|---|---:|---:|---|---:|
| 2023 | 44 | 14 | 31.8% [20.0, 46.6] | 0.0226 |
| 2024 | 47 | 17 | 36.2% [24.0, 50.5] | 0.0789 |
| **Pooled** | **91** | **31** | **34.1% [25.2, 44.3]** | **0.0031** |

The other three models were right on 32.0–33.4% of their pooled disagreements (n = 100–111). This
matches PHASE7_REVIEW §3.4 (31/91, p = 0.0031).

---

## 3. Recomputed metrics: 2025 holdout (n = 271) and live 2026 (n = 75)

| | 2025 holdout | Live 2026 (wks 3–8, champion) |
|---|---|---|
| Model accuracy | 167 = **61.62%** [55.7, 67.2] | 44 = **58.67%** [47.4, 69.1] |
| Home baseline | 146 = 53.87% [47.9, 59.7] | 35 = 46.67% [35.8, 57.8] |
| Vegas favorite | 177 = 65.31% [59.5, 70.7] | 45 = 60.00% [48.7, 70.3] |
| Model Brier / LL | 0.2237 / 0.6366 | 0.2353 / 0.6611 |
| Vegas implied Brier / LL | 0.2121 / 0.6094 | 0.2443 / **0.6840** |
| Disagree with Vegas: model right | 11 / 32 (34.4%) | 7 / 15 (46.7%) [24.8, 69.9] |

The live model is the 2026 champion (LR-tuned retrained on 2002–2025). The live Vegas line is the
prediction-time snapshot, not the close. Live checks run this session:
- Final scores equal the latest snapshot (`schedules_2026_20261112T210940Z.parquet`) for 76/76 games.
- An independent re-implementation of the bet rule reproduces every logged bet decision (0 mismatches).
- Re-computed units match the graded file to within 5e-11.

In 2025, HGB-tuned matched the Vegas favorite at 65.31% (19/38 disagreements right). Per the
pre-registered rule it was not adopted (PHASE8_RESULTS).

---

## 4. Bets-only metrics

### 4.1 Live 2026 official record (the only *real* paper-trading record)

| | Value |
|---|---|
| Bets | 53 of 76 logged games (69.7%); 18 on the team the model picks to lose |
| W-L-P | **28-24-1** |
| Win rate (excl. push) | 53.85% [40.5, 66.7] |
| Break-even | 52.38% at −110 (shown for reference only; **the code uses real moneylines**). Mean raw implied probability of the prices actually taken: **48.91%** |
| Units / ROI | **+10.98 units**; ROI +21.1% per settled bet; t-interval [−18.1%, +60.3%] (recomputed in this session, n = 52 settled; matches the scorecard) |
| Model accuracy on bet vs no-bet games | 29/52 = 55.8% [42.3, 68.4] vs 15/23 = 65.2% [44.9, 81.2] |
| Pre-registered verdict input (CLV) | measured on only 8 bets; mean −0.0052, one-sided p = 0.7038 (recomputed) → **no evidence of an edge** |

Week 7 alone: picks 8/14 right, bets 3–6, −3.06 units.

### 4.2 Retroactive: the same rule applied to the held-out seasons (NEW, descriptive only)

This applies `paper_bet` (threshold 0.04, real moneylines) to LR-tuned's walk-forward probabilities
at nflverse's lines. My re-implementation was asserted equal to `common.paper_bet` on every game.

**Caveats:**
- The rule was not pre-registered for these seasons.
- The lines are most likely closing lines. That is the hardest price to beat, and harder than the live Wednesday snapshot.
- Per PHASE7_REVIEW's restriction, nothing here may feed a model, threshold or deployment decision.

| | 2023 | 2024 | **2023-24 pooled** | 2025 |
|---|---|---|---|---|
| Bets (% of games) | 190 (69.9%) | 187 (68.8%) | **377 (69.3%)** | 172 (63.5%) |
| home / away bets | 116 / 74 | 116 / 71 | 232 / 145 | 81 / 91 |
| Bets against the model's own pick | 89 | 93 | 182 | 92 |
| W-L-P | 74-116-0 | 71-116-0 | **145-232-0** | 69-103-0 |
| Win rate [95% CI] | 38.9% [32.3, 46.0] | 38.0% [31.3, 45.1] | **38.5% [33.7, 43.5]** | 40.1% [33.1, 47.6] |
| Mean implied prob. of prices taken | 44.4% | 43.9% | 44.1% | 40.2% |
| Units | −19.32 | −34.83 | **−54.15** | −3.59 |
| ROI per bet | −10.2% | −18.6% | **−14.4%** | −2.1% |
| Model acc., bet games | 59.5% (113/190) | 66.3% (124/187) | 62.9% (237/377) | 57.6% (99/172) |
| Model acc., no-bet games | 68.3% (56/82) | 67.1% (57/85) | 67.7% (113/167) | 68.7% (68/99) |

The −110 break-even (52.38%) does not apply: most bets are on underdogs at plus money, so the
relevant comparator is the mean implied probability of the prices taken. Pooled 2023-24 wins
(38.5%) fall below that (44.1%); a binomial test against 44.1% gives p ≈ 0.03. That test is
approximate because prices vary per bet.

The model is *less* accurate on the games where it disagrees enough with the market to bet. That is
the pattern expected when the "edge" is mostly model error.

Integrity note: in each season the W count equals the away-bet count (e.g. 145 = 145). This is a
coincidence. A cross-tab shows home bets on home wins = away bets on home wins in both seasons (51 =
51, 48 = 48). A separate loop count confirmed 145-232.

---

## 5. Can "logged before kickoff" be proven?

All checks run this session:

| Week | `predicted_at_utc` (in CSV) | Commit (author = committer) | Local push (reflog) | First logged kickoff | Earliest GitHub-runner commit that descends from it |
|---|---|---|---|---|---|
| 3 | 09-23 21:43:26Z | 09-23 21:51:38Z | 21:51:45Z | 09-25 00:15Z | 10-23 00:30:39Z (after) |
| 4 | 10-01 19:00:58Z | 10-01 19:01:29Z | 19:01:33Z | 10-02 00:15Z | 10-23 00:30:39Z (after) |
| 5 | 10-07 18:40:56Z | 10-07 20:11:02Z | 20:11:07Z | 10-09 00:15Z | 10-23 00:30:39Z (after) |
| 6 | 10-14 17:42:05Z | 10-14 17:42:31Z | 17:42:36Z | 10-16 00:15Z | 10-23 00:30:39Z (after) |
| **7** | **10-21 21:37:57Z** | **10-21 21:38:24Z** | **21:38:30Z** | **10-23 00:15Z (NE @ CHI)** | **10-23 00:30:39Z: 15 min after NE @ CHI, before all 13 other games** |
| 8 (MNF only) | 11-02 20:14:13Z | 11-02 20:18:08Z | 20:18:14Z | 11-03 01:15Z | 11-03 03:27:53Z (after) |
| 10 | 11-12 21:09:46Z | 11-12 21:10:50Z | 21:10:54Z | 11-13 01:15Z | none yet |

**Also verified:**
- Each `week_NN_predictions.csv` is touched by exactly one commit, and its sha256 still equals the `predictions_sha256` in its `week_NN_run.json`, so the record has not been edited since it was committed.
- `predict_week` refuses any game already past kickoff at run time.

**What this proves and doesn't:**
- Local commit, `predicted_at_utc` and reflog times are all set by the owner's machine. They are consistent with logging before every kickoff, but they are not independent proof.
- Bot commits are timestamped by GitHub's runner clock, and a bot commit can only descend from a commit that was already on GitHub. They are therefore third-party corroboration, but strong rather than cryptographic: the author name could in principle be forged.
- On that basis, pre-kickoff logging is **server-corroborated only for week 7's 13 Sunday/Monday games**. For **weeks 3–6, week 8, and week 7's NE @ CHI** (the row in the site's clip) it is **not provable from the repository alone**.
- **To settle it**, check the repo's server-side push log on GitHub (Activity tab, or the repository activity API). It records push times that the author cannot set. The `gh` CLI is not installed here, and I did not query GitHub on your behalf.

---

## 6. Leakage spot-check (`audit/scripts/leakage_spotcheck.py`, `audit/out/leakage_spotcheck.json`)

1. **No Vegas line is a feature.**
   - The 67 feature names in `features.feature_columns()`, in the saved LR-tuned model and in the frozen 2026 champion are identical.
   - None contains spread / moneyline / total / odds / line / result / score / vegas / implied, and neither does any column of `game_features.parquet`.
   - `features.build_game_features` also asserts that no `_VEGAS_COLS` name reaches the frame.
   - Lines are used **only** as a benchmark and for the live paper-bet rule, after the model's probability is computed.
2. **Rolling features exclude the current game.** Perturbation tests use the exact functions `features.py` uses. Every week-10-2023 team-game outcome was changed by +5 EPA, and the result was measured:
   - `roll_ewm` (shift(1) → ewm, halflife 5): same-week feature change **0.0**, next week 0.647 → **PASS**
   - `roll_left` (rolling(22, closed='left')): same-week **0.0**, next week 0.556 → **PASS**
   - Contrast: the same EWM *without* `shift(1)` moves 0.647 in the same week. The test detects a leak when one exists.
   - Elo (`build_elo_table`): flipping every week-10-2023 result changes that week's pre-game Elo by **0.0** and the next week's by up to 76.9 → **PASS**
   - Independent plain-loop EWM over strictly earlier kickoffs, 80 team-games from 2023-24, `def_epa_per_play_ewm`: max |diff| 6.9e-17 → **PASS**
   - **Not perturbation-tested here:** the `_prior` / `_league` season-level features (`merge_asof(..., allow_exact_matches=False)` on season). I checked them by reading the code only. The project's own check 1 covers them.
3. **Label shuffle** (`audit/scripts/label_shuffle.py`): a clone of the saved LR-tuned pipeline, 20 permutations of the training labels, scored on real labels.
   - **Agent-run (2002–21 → 2022):** mean **55.52%** (sd 1.69, range 50.6–57.6), against a 2022 home-win rate of 56.13%.
   - Why not ~50%: shuffling keeps the class balance, so an unweighted LR learns "home wins about 57%" and predicts mostly home. The correct null is the home baseline, and the shuffled model sits at or below it. **No leak signal.**
   - The project's own check 2 reports ~48.5% because it uses `class_weight="balanced"` (`src/leakage_checks.py:_pipeline`). That is a different but equally valid setup. I did not re-run it.
   - **Owner-run (not executed, per CLAUDE.md, because it reads 2023-24 labels):** `venv/bin/python -W ignore -m audit.scripts.label_shuffle --test-seasons` writes `audit/out/label_shuffle_test_2023_2024.json`. Expected result is near the 2023-24 home rate (~54%), not 64%.

**Bugs found in metric/model code: none.** Known, documented limitations remain:
- Historical lines are untimestamped.
- `roof_indoor` is NaN for retractable roofs on Wednesdays.
- The CLV t-test assumes independent bets.

---

## 7. Old outputs vs this recomputation

| Figure | Existing source | This session | Agree? |
|---|---|---|---|
| LR-tuned 2023-24 | README / PHASE7: 350/544 = 64.34%, CI 60.2–68.3 | 350/544, [60.23, 68.25] | yes |
| Home 2023-24 | 296/544 = 54.41% | same | yes |
| Vegas favorite 2023-24 | 69.67% | 379/544 = 69.67% | yes |
| Vegas LL / model LL 2023-24 | 0.6074 / 0.6434 | same | yes |
| Disagreement 2023-24 | 31/91, p 0.0031 | same | yes |
| 2025 LR-tuned / home / Vegas | 167, 146, 177 of 271 | same | yes |
| Live scorecard | 44/75, 28-24-1, +10.98 | same | yes |
| Label shuffle | 48.5% (balanced plain LR, 2022) | 55.5% (LR-tuned pipeline, unbalanced, 2022) | different setups, both consistent with no leak |

**Provenance gap, stated plainly:** the 2023-24 per-game probabilities come from a saved parquet. It
is gitignored and was written 2026-09-22 by `scripts/reproduce_walkforward_predictions.py`, which
refuses to write unless it reproduces the Phase 6 counts. I did **not** regenerate it, because that
refits models and predicts 2023-24, which is an owner-run step. To regenerate, run
`python -m scripts.reproduce_walkforward_predictions`.

---

## 8. Proposed `site_data.json` (not built)

```jsonc
{
  "schema_version": "1.0.0",
  "generated_at": "2026-11-12T00:00:00Z",          // ISO-8601 UTC, set by the exporter
  "source": { "git_commit": "5ef6895", "exporter": "scripts/export_site_data.py" },
  "definitions": {
    "accuracy": "Share of decided games where the team the model gave >50% won. Ties excluded.",
    "home_baseline": "Always pick the home team.",
    "vegas_favorite": "Team favored by the point spread (nflverse spread_line > 0 = home). Pick'em falls back to the lower moneyline. Historical lines are untimestamped and most likely closing lines; live lines are the snapshot taken when the prediction was logged.",
    "vegas_implied": "Moneyline win probability with the bookmaker's margin removed (normalised to sum to 1).",
    "brier": "Mean squared error of the win probability. Lower is better; 0.25 = coin flip.",
    "log_loss": "Mean negative log-likelihood of the outcome. Lower is better; 0.693 = coin flip.",
    "ci95": "95% Wilson score interval.",
    "paper_bet": "1 unit on the side whose model probability exceeds the vig-free market probability by >= 4 points, at that side's real moneyline. Ties void.",
    "roi": "Units won / units staked on settled bets.",
    "clv": "Change in the bet side's vig-free market probability from the prediction snapshot to the last pre-kickoff snapshot (not the true close). The only pre-registered evidence of an edge.",
    "retroactive": "Rule applied after the fact to seasons it was not pre-registered for; descriptive only."
  },
  "evaluations": [
    {
      "id": "walkforward_2023_2024",
      "label": "2023-24 walk-forward test",
      "kind": "held_out",                // held_out | preregistered_holdout | live
      "seasons": [2023, 2024],
      "game_types": ["REG"],
      "train_rule": "train on all seasons before the test season",
      "model": "LR-tuned",
      "pooled": { "$ref": "#/$defs/metric_block" },
      "per_season": { "2023": { "$ref": "#/$defs/metric_block" }, "2024": { "$ref": "#/$defs/metric_block" } },
      "other_models": [ { "model": "RandomForest", "accuracy": { "$ref": "#/$defs/rate" } } ]
    }
    // + "holdout_2025" (preregistered_holdout), + "live_2026" (live)
  ],
  "picks": [
    {
      "evaluation_id": "live_2026",
      "game_id": "2026_07_NE_CHI",
      "season": 2026, "week": 7,
      "kickoff_utc": "2026-10-23T00:15:00Z",
      "away_team": "NE", "home_team": "CHI",
      "model_p_home": 0.467269,                  // from the log (full precision in the CSV)
      "vegas_p_home": 0.582993,                  // vig-free; null if no line
      "spread_line": 3.0,                        // nflverse sign convention
      "pick": "away",                            // model pick: home | away
      "vegas_favorite": "home",                  // home | away | unresolved
      "bet": { "side": "away", "price": 130, "edge": 0.116, "retroactive": false },  // null if no bet
      "result": { "status": "final", "home_score": 30, "away_score": 24, "winner": "home" },  // status: final | tie | pending
      "pick_correct": false,
      "bet_outcome": "loss",                     // win | loss | void | pending | null
      "units": -1.0,
      "provenance": {                            // live only
        "predicted_at_utc": "2026-10-21T21:37:57Z",
        "prediction_commit": "3ba226d",
        "earliest_server_timestamp_utc": "2026-10-23T00:30:39Z",
        "logged_before_kickoff_server_proven": false
      }
    }
  ],
  "$defs": {
    "rate": { "k": "int", "n": "int", "value": "float", "ci95": ["float", "float"] },
    "metric_block": {
      "n_games": "int", "ties_excluded": "int",
      "model": { "accuracy": "#rate", "brier": "float", "log_loss": "float" },
      "home_baseline": { "accuracy": "#rate" },
      "vegas_favorite": { "accuracy": "#rate", "unresolved_excluded": "int", "line_timing": "closing_likely | prediction_snapshot" },
      "vegas_implied": { "brier": "float", "log_loss": "float" },
      "disagreement": { "n": "int", "model_right": "#rate", "binom_p": "float" },
      "bets": {
        "retroactive": "bool", "n_bets": "int", "pct_of_games": "float",
        "w": "int", "l": "int", "p": "int", "win_rate": "#rate",
        "breakeven_mean_implied": "float", "units": "float", "roi": "float",
        "model_accuracy_bet_games": "#rate", "model_accuracy_no_bet_games": "#rate",
        "clv": { "n": "int", "mean": "float", "p_one_sided": "float" }
      }
    }
  }
}
```

Design notes:
- Every rate carries k, n and its CI, so the site can never show a percentage without its sample size.
- `retroactive` and `line_timing` flags force the page to label hypothetical numbers.
- `pick_correct` and `bet_outcome` are separate fields, which removes the Yes/No ambiguity in the clip.
- `provenance.logged_before_kickoff_server_proven` lets the caption claim only what is proven.
- The example pick values come from the week 7 log, but the exporter should read them from the CSV at full precision rather than these rounded figures.

---

## 9. Reproduce

```bash
venv/bin/python -W ignore -m audit.scripts.recompute_metrics     # §2-4 -> audit/out/metrics.json
venv/bin/python -W ignore -m audit.scripts.leakage_spotcheck     # §6.1-6.2 -> audit/out/leakage_spotcheck.json
venv/bin/python -W ignore -m audit.scripts.label_shuffle         # §6.3, 2022 only
# OWNER ONLY (reads 2023-24 labels):
venv/bin/python -W ignore -m audit.scripts.label_shuffle --test-seasons
# §1.3 clip:
grep ' @ ' docs/live/week_07_picks.md | cut -d"|" -f2,4,5,7,11,12
# §5 timing:
git log --diff-filter=A --format='%h %aI %cI %s' -- data/live/2026/predictions/week_07_predictions.csv
git reflog show --date=iso-strict refs/remotes/origin/main | grep 3ba226d
git log --reverse --ancestry-path --format='%h %an %aI' 3ba226d..HEAD | grep github-actions | head -1
```

Rolling/aggregation code written for this audit (CLAUDE.md flag requirement):
- `leakage_spotcheck.independent_ewm` uses strictly earlier kickoffs (`kickoff < this kickoff`).
- `perturb_rolling` calls the project's own `roll_ewm` / `roll_left` (`shift(1)` / `closed='left'`).
- `perturb_rolling` also builds one **deliberately leaky** EWM without `shift(1)`, only as a contrast. It is not a feature and is never used for modeling.
