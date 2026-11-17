# `site_data.json` schema (v1.4.0)

Written by `python -m scripts.export_site_data` and tested by `python -m scripts.test_export_site_data`.
This is an export layer only. Every number is recomputed from per-game data with the project's own
functions and then checked against `audit/out/metrics.json`, `docs/live/SCORECARD_2026.md`,
`docs/PHASE7_REVIEW.md` (2023-24 McNemar) and `data/processed/holdout_2025_run_record.json` (2025
McNemar). If any of them disagree, the exporter writes nothing and exits 2.

The `negative_control` block is the exception: it is copied, not recomputed, from
`audit/out/negative_control.json`, and gated instead (see [`negative_control`](#negative_control)).

**Changelog.** 1.4.0: Amendment 7's three display tiers replace Amendment 6's two. Per-game `value_tier` is now the string `"lean"`, `"one_star"` or `"two_star"` (was the int 1 or 2); `live_2026.value_woe` gains a `lean` key, and `one_star` / `two_star` cover the new edge ranges; `live_2026.value_tier_two_star_edge` is now 0.16 (was 0.08); added `live_2026.value_tier_one_star_edge` (0.11). The `value_tier` type change is not backward compatible for a reader that expects 1 / 2. No other field changed. 1.3.0: added the Amendment 6 display-only fields: `live_2026.value_tier_two_star_edge`, `live_2026.toss_up_threshold`, `live_2026.value_woe`, `live_2026.prediction_quality_by_confidence`, `live_2026.amendment_6_note`, and per-game `value_tier` / `toss_up` on live games (additive; no existing field changed). 1.2.0: added the top-level `schedule_2026` block and `live_2026.edge_threshold`
(additive; no existing field changed). 1.1.0: added the top-level `negative_control` block (additive; no existing field changed).
1.0.0: first release.

## Conventions

- **Floats** are rounded to 4 decimal places. A missing value or NaN is `null`.
- **Rates** have the shape `{ "k": int, "n": int, "value": float, "ci95": [lo, hi] }`. `value` = k/n, and
  `ci95` is a 95% Wilson interval. The site should never show a rate without `n`.
- **Means** (live log loss, Brier) have the shape `{ "mean": float, "ci95": [lo, hi] }` with a 95%
  t-interval. `ci95` is `[null, null]` when n < 2.
- **Sides** are always the strings `"home"` / `"away"`. Team abbreviations are in `home` / `away`.
- **Timestamps** are UTC ISO-8601 with a `Z` suffix.
- Keys are sorted alphabetically. `games` is ordered by (`season`, `week`, `kickoff_utc`, `game_id`).
- **Determinism:** the same inputs give byte-identical output. The live "as of" time is the latest
  2026 schedule snapshot, not the wall clock. `generated_at` is the only wall-clock value. It is kept
  from the existing file when nothing else changed, or set with `SOURCE_DATE_EPOCH` / `--generated-at`.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Semver of this schema. Currently `"1.4.0"`. |
| `generated_at` | string | UTC time the export content last changed. |
| `source_commit` | object | Git state at export time: `hash` (full HEAD hash), `dirty` (true if anything outside `exports/` was uncommitted or untracked), `dirty_paths` (those paths). A `dirty: true` export was built from code or data not in `hash`. |
| `definitions` | object | Plain-English one-liners for non-technical readers. Keys: `accuracy`, `log_loss`, `brier`, `home_baseline`, `vegas_favorite`, `vegas_implied`, `units`, `roi`, `clv`, `bet_rule`, `ci95`, `mcnemar_p`, `retroactive`, `pre_registered`, `prediction_committed_at`. |
| `held_out_2023_24` | object | 2023-24 walk-forward test. See [Evaluation block](#evaluation-block-held_out_2023_24-prereg_2025). |
| `prereg_2025` | object | 2025 pre-registered holdout. Same structure, without `per_season`. |
| `live_2026` | object | 2026 live shadow mode. See [`live_2026`](#live_2026). |
| `negative_control` | object | Leakage-check negative control: does Check 1 catch deliberately leaked features? See [`negative_control`](#negative_control). |
| `schedule_2026` | array | Every 2026 regular-season game, played or not. See [`schedule_2026`](#schedule_2026). |
| `games` | array | One entry per game across all three evaluations. See [`games`](#games). |

## Evaluation block (`held_out_2023_24`, `prereg_2025`)

| Field | Type | Meaning |
|---|---|---|
| `label` | string | Display name. |
| `seasons` | int[] | Seasons covered. |
| `game_types` | string[] | Always `["REG"]` (regular season only). |
| `n` | int | Decided games scored (ties excluded): 544 for 2023-24, 271 for 2025. |
| `ties_excluded` | int | Tied games dropped (0 for 2023-24; 1 for 2025: `2025_04_GB_DAL`). |
| `train_rule` | string | How each season was kept out of training. |
| `line_timing` | string | Which betting line was used: one untimestamped nflverse line, most likely the closing line. |
| `probabilities` | string | Always raw (uncalibrated) model probabilities. |
| `models` | object | Keyed by `LR-untuned`, `LR-tuned`, `RandomForest`, `HGB-tuned`. Each has `accuracy` (rate), `log_loss` (float) and `brier` (float). |
| `home_baseline` | object | `accuracy` (rate) for always picking home. |
| `vegas_favorite` | object | `accuracy` (rate) for always picking the spread favorite (pick'em falls back to the lower moneyline), plus `unresolved_excluded` (int). |
| `vegas_implied` | object | Vig-free moneyline probability: `accuracy` (rate, picking the side with p > 0.5), `log_loss`, `brier`. |
| `selected_model` | string | `"LR-tuned"`, the model used in `games` and for bets. |
| `selection_note` | string | Plain-language disclosure of when and why LR-tuned was chosen, the statistical tie, and the McNemar p-values. For 2025 it also discloses the Phase 8 early-run incident. |
| `mcnemar` | array | All 6 model pairs: `model_a`, `model_b`, `only_a_right`, `only_b_right`, `p` (exact two-sided McNemar). |
| `retroactive_bets` | object | The 2026 bet rule applied after the fact to the selected model. See [Bet record](#bet-record). Always `retroactive: true`. |
| `retroactive_bets_note` | string | Caveat text to show next to `retroactive_bets`. |
| `per_season` | object | 2023-24 only. Keyed `"2023"` / `"2024"`, each with `n`, `models`, `home_baseline`, `vegas_favorite`, `vegas_implied` and `retroactive_bets`. |

### Bet record

| Field | Type | Meaning |
|---|---|---|
| `bets` | int | Bets placed. |
| `home_bets` / `away_bets` | int | Bets by side. `away_bets` appears in historical blocks only. |
| `wins` / `losses` / `voids` / `pending` | int | Outcome counts. A void is a tie with the stake returned. |
| `record` | string | `"W-L-Void"`. |
| `units` | float | Net profit in units on settled bets. |
| `roi` | float | Mean profit per settled bet, i.e. units ÷ settled stakes. |
| `roi_ci95` | [float, float] | 95% t-interval on `roi`. |
| `retroactive` | bool | Historical blocks only: always `true`. |
| `bets_against_model_pick` | int | Historical blocks only: bets on the team the model picked to lose. |
| `clv` | object | Live only: `n`, `mean`, `ci95`, `p_one_sided` (H1: mean > 0), `n_excluded`. |

## `live_2026`

| Field | Type | Meaning |
|---|---|---|
| `label` | string | Display name. |
| `as_of_utc` | string | Pull time of the latest 2026 schedule snapshot. Week statuses are judged at this time. |
| `as_of_source` | string | That snapshot's filename. |
| `first_official_week` | int | 3 (Amendment 1). Weeks 1-2 came before shadow mode began and are not part of the record. |
| `edge_threshold` | float | The bet rule's threshold (`scripts/live/common.py` `EDGE_THRESHOLD`, 0.04): a side qualifies when model P(win) minus vig-free market P(win) is at least this. Compare at full precision; the 4 dp probabilities in `games` can round a borderline edge either way, so `bet` is authoritative for live games. |
| `value_tier_one_star_edge` | float | 0.11 (`scripts/live/common.py` `VALUE_TIER_1_EDGE`, Amendment 7): a value call with edge >= this is at least one-star (★); below it (down to `edge_threshold`) it is a Lean. |
| `value_tier_two_star_edge` | float | 0.16 (`VALUE_TIER_2_EDGE`, Amendment 7): a value call with edge >= this is two-star (★★). Tiers: Lean = [`edge_threshold`, `value_tier_one_star_edge`), one-star = [`value_tier_one_star_edge`, `value_tier_two_star_edge`), two-star = `value_tier_two_star_edge` and above, compared at 10 decimals like the bet rule. |
| `toss_up_threshold` | float | 0.55 (`TOSS_UP_MAX_FAV`, Amendment 6): a game is a toss-up when the champion's favorite has probability <= this (inclusive), at full precision. |
| `value_woe` | object | Wins over expected for the official value calls (Amendments 6 and 7, descriptive only). Keys `all`, `lean`, `one_star`, `two_star`; see [`value_woe`](#value_woe). Computed with the scorecard's own `value_woe_rows` and checked against its value-tier table. |
| `prediction_quality_by_confidence` | object | Champion pick accuracy split by the toss-up flag. Keys `confident`, `toss_up`, `all`, each `{ "games_logged": int, "accuracy": rate }`. `games_logged` counts the prediction files (pending included); `accuracy` follows the usual convention (ties and pending excluded). `confident` + `toss_up` = `all`, and `all` equals `prediction_quality.champion.accuracy`. |
| `amendment_6_note` | string | Caveat to show with the tiers, WOE and toss-ups (Amendments 6 and 7): descriptive only; the verdict rests on CLV. |
| `games_logged` | int | Champion predictions logged, including pending games. |
| `games_graded` / `games_decided` / `ties` / `games_pending` | int | Graded = decided + ties. Pending = logged but not yet graded or without a final score. |
| `line_timing` | string | Live lines are the snapshot saved when the prediction was logged, not the close. |
| `prediction_quality` | object | Cumulative, from the graded files. `champion`, `challenger`: `n`, `accuracy` (rate), `log_loss` and `brier` (means). `home_baseline`, `vegas_favorite`: `n`, `accuracy`. `vegas_implied`: `n`, `log_loss`, `brier`. |
| `official_bets` | object | The official paper-bet record (`bet_status = official`). See [Bet record](#bet-record), including `clv`. |
| `challenger_bets_hypothetical` | object | The challenger's hypothetical bets, same shape. Not the official record. |
| `clv_verdict` | object | The pre-registered season-end verdict. `verdict` is the bold phrase, `statement` is the scorecard's sentence without markdown, and `scorecard_line` is the scorecard's line verbatim. Also `final` (false until week 18 is graded), `n`, `mean`, `ci95`, `p_one_sided`, `n_excluded`, `rule`, `clv_window_rule`. |
| `weeks` | array | One entry for **every** scheduled week from `first_official_week` through the latest week that has kicked off or been predicted. See [`weeks[]`](#weeks). |
| `checkpoint` | object | Midseason checkpoint: `window_weeks`, `evaluable`, `planned_games`, `paired_games`, `champion_mean_log_loss`, `challenger_mean_log_loss`, `mean_diff_challenger_minus_champion`, `p_one_sided`, `decision`, `rule`, `week_status` (the scorecard's per-week status text), and `governing_amendment` (`id`, `date`, `source`, `summary`, `disclosure`). |
| `provenance_note` | string | Caveat on `prediction_committed_at`. |

### `value_woe`

One object per key (`all`, `lean`, `one_star`, `two_star`). The three tiers always add up to `all`
(counts exactly; `vegas_expected`, `model_expected` and `woe` exactly at full precision), and `all`'s
`wins` / `losses` / `pushes` / `calls_graded` equal `official_bets` `wins` / `losses` / `voids` / `bets`.
Sums are over **graded calls with pushes excluded** (`n` of them). Floats are 4 dp, so the sum of the
three exported tier values can differ from `all` by up to 0.0002 after rounding.

| Field | Type | Meaning |
|---|---|---|
| `calls_logged` | int | Official value calls in the prediction files, pending ones included. |
| `calls_graded` | int | Official value calls in the graded files (wins + losses + pushes + pending). |
| `wins` / `losses` / `pushes` | int | Graded outcomes. A push is a tied game. |
| `n` | int | wins + losses: the calls the sums below are over. |
| `won` | int | Calls whose side won (= `wins`). |
| `vegas_expected` | float | Sum of the vig-free implied probability of each call's side: the wins the market expected. |
| `model_expected` | float | Sum of the calls' edges (model minus market probability for the side): the wins over expected the model's own probabilities predicted. Not shown on the site. |
| `woe` | float | `won` − `vegas_expected`. Positive = more wins than the market expected. |
| `woe_per_call` | float \| null | `woe` / `n`. Null when `n` is 0. |
| `model_expected_per_call` | float \| null | `model_expected` / `n`. Null when `n` is 0. |

### `weeks[]`

| Field | Type | Meaning |
|---|---|---|
| `week` | int | Week number. |
| `status` | string | `graded`: every scheduled game was predicted and graded. `partial`: some scheduled games were never predicted, every kickoff has passed, and all predicted games are graded. `missed`: no prediction file and every kickoff has passed. `pending`: predicted games still await grading or a final score, or no prediction yet but kickoffs remain. |
| `scheduled` | int | Regular-season games on the 2026 schedule that week. |
| `predicted` | int | Games in the week's prediction file (champion). |
| `graded` | int | Predicted games graded with a final result (wins, losses and ties). |
| `decided` / `ties` | int | Of those graded: games with a winner, and ties. |
| `pending` | int | Predicted games not yet graded or without a final score. |
| `first_kickoff_utc` | string | Earliest scheduled kickoff that week. |
| `prediction_commit` | string \| null | Full hash of the commit that added `week_NN_predictions.csv`. Null if none. |
| `prediction_committed_at` | string \| null | That commit's local author time (UTC). It is not proof of push time. |

Note: the scorecard's checkpoint table labels week 8 "graded" because all of its predicted games
are graded. Here week 8 is `partial`, because only 1 of its 14 scheduled games was predicted.
Both labels describe the same fact.

## `negative_control`

Copied from `audit/out/negative_control.json` (written by `python -m audit.scripts.negative_control`).
Values are the source's, rounded to 4 dp (the source already holds 4 dp). The exporter writes nothing
and exits 2 if the source has `src_dirty` missing or not `false`, the `clean` condition fails Check 1
or has a contrast accuracy of 0.70 or more, **any sabotage condition passes Check 1**, or the
source's own `integrity_gates` verdicts disagree with its per-condition verdicts.

| Field | Type | Meaning |
|---|---|---|
| `label` | string | Display name. |
| `source` | string | `"audit/out/negative_control.json"`. |
| `source_git_head` | string | Full HEAD hash recorded in the source JSON when the control was run. This is the code the control tested, which can differ from `source_commit.hash`. |
| `src_dirty` | bool | Always `false` in a written export (`src/` had no uncommitted changes when the control ran). |
| `seasons` | object | Copied verbatim from the source: `features_built_from` ([first, last]), `check1_rows`, `check2_train`, `check2_eval` (int), `not_reproduced` (what the control did not re-run). No 2023+ season is used. |
| `conditions` | array | One entry per condition, in the source's order (`clean` first). See [`conditions[]`](#conditions). |
| `note` | string | Caveat text to show with the block: sabotage contrast accuracies are a leak's signature, not model results. |

### `conditions[]`

| Field | Type | Meaning |
|---|---|---|
| `condition` | string | Source condition id: `clean`, `single_metric_sd`, `phase2_roll_left`, `ewm_only`, `all_rolling`. |
| `role` | string | `control` for `clean`, `sabotage` for every other condition. |
| `description` | string | What the sabotage changed, in the source's words. |
| `columns_changed` | string[] | Model input columns whose values differ from the clean build. Empty for `clean`. |
| `n_columns_changed` | int | Length of `columns_changed`. |
| `check1` | object | Check 1 (timestamp / prior-games oracle): `verdict` (`PASS` / `FAIL`; always `FAIL` for a sabotage) and `failing_oracle` (the stage that failed, e.g. `prior-only oracle on {m}_ewm`; null on `PASS`). |
| `check2` | object | Check 2 (label shuffle), keyed `current_67` (the current 67-feature set) and `phase2_42` (the Phase 2 42-feature set). Each has `n_features`, `shuffled_mean` (mean accuracy over 20 shuffled-label fits), `contrast_accuracy` (same model fit on the true labels, scored on 2022) and `verdict`. **`contrast_accuracy` above 0.70 on a sabotage is the leak showing, never a result to report.** |

## `schedule_2026`

Every regular-season game in the latest 2026 schedule snapshot (the file the live pipeline reads), with
the same kickoff definition predict/grade use. Ordered by (`week`, `kickoff_utc`, `game_id`). The
exporter writes nothing and exits 2 if any `live_2026` game is missing here or disagrees on week or teams.

| Field | Type | Meaning |
|---|---|---|
| `game_id` | string | nflverse id. |
| `week` | int | 1-18. |
| `kickoff_utc` | string | Scheduled kickoff (UTC). |
| `away`, `home` | string | Team abbreviations. |
| `away_score`, `home_score` | int \| null | Final score as of the snapshot. Both null until the game has a final score. |

## `games`

| Field | Type | Meaning |
|---|---|---|
| `evaluation` | string | `held_out_2023_24`, `prereg_2025` or `live_2026`. |
| `season`, `week` | int | |
| `game_id` | string | nflverse id, e.g. `2026_07_NE_CHI`. |
| `away`, `home` | string | Team abbreviations. |
| `kickoff_utc` | string | Scheduled kickoff (UTC). |
| `model` | string | Which model's probability this is: `LR-tuned` for history, `champion (LR-tuned, frozen 2026)` for live. |
| `model_home_prob` | float | Model P(home win). |
| `vegas_implied_home_prob` | float \| null | Vig-free market P(home win). Null if no moneyline. |
| `model_pick` | string | `home` if `model_home_prob` > 0.5, else `away`. |
| `vegas_favorite` | string | `home` / `away` / `unresolved`. |
| `bet` | bool | Whether the bet rule placed a bet. |
| `bet_side` | string \| null | `home` / `away`. Null when `bet` is false. This can be the team the model picked to lose. |
| `bet_odds` | float \| null | American odds taken, e.g. `130.0` or `-155.0`. Null when `bet` is false. |
| `bet_retroactive` | bool | True for 2023-24 and 2025: the rule was applied after the fact. |
| `bet_status` | string \| null | Live only: `official` when `bet` is true. |
| `result` | string | `home` / `away` / `tie` / `pending`. |
| `home_score`, `away_score` | int \| null | Final score. Null while pending. |
| `pick_correct` | bool \| null | Whether the straight-up pick was right. **Not a bet result.** Null for ties and pending games. |
| `bet_units` | float \| null | Units won or lost on this game's bet (0 for a void tie). Null when there is no bet or the bet is pending. Never non-null unless `bet` is true. |
| `predicted_at_utc` | string | Live only: the run time recorded in the prediction CSV (set by the owner's machine). |
| `value_tier` | string \| null | Live only: `"lean"` (edge in [0.04, 0.11)), `"one_star"` (★, [0.11, 0.16)) or `"two_star"` (★★, 0.16 and above) for an official value call, decided at full precision with the bet rule's 10-decimal rounding; null when `bet` is false. A Lean is still an official value call. Descriptive only (Amendment 7). |
| `toss_up` | bool | Live only: true when the champion's favorite has probability <= `live_2026.toss_up_threshold`, at full precision. The pick is still made and graded. |
| `prediction_commit` | string | Live only: full hash of the commit that added the week's prediction file. |
| `prediction_committed_at` | string | Live only: that commit's **local author timestamp** (UTC). The owner's computer sets it, so it is not proof of push time. |
| `prediction_committed_at_kind` | string | Live only: says the above in words, so a per-game display carries the caveat. |

Omitted on purpose: the 2025 tie (`2025_04_GB_DAL`) has no stored model prediction (the holdout dropped
ties before scoring), so it is not in `games`. It is counted in `prereg_2025.ties_excluded`.
