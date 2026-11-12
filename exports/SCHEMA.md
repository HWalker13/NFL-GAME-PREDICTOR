# `site_data.json` schema (v1.0.0)

Written by `python -m scripts.export_site_data` and tested by `python -m scripts.test_export_site_data`.
This is an export layer only. Every number is recomputed from per-game data with the project's own
functions and then checked against `audit/out/metrics.json`, `docs/live/SCORECARD_2026.md`,
`docs/PHASE7_REVIEW.md` (2023-24 McNemar) and `data/processed/holdout_2025_run_record.json` (2025
McNemar). If any of them disagree, the exporter writes nothing and exits 2.

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
| `schema_version` | string | Semver of this schema. Starts at `"1.0.0"`. |
| `generated_at` | string | UTC time the export content last changed. |
| `source_commit` | object | Git state at export time: `hash` (full HEAD hash), `dirty` (true if anything outside `exports/` was uncommitted or untracked), `dirty_paths` (those paths). A `dirty: true` export was built from code or data not in `hash`. |
| `definitions` | object | Plain-English one-liners for non-technical readers. Keys: `accuracy`, `log_loss`, `brier`, `home_baseline`, `vegas_favorite`, `vegas_implied`, `units`, `roi`, `clv`, `bet_rule`, `ci95`, `mcnemar_p`, `retroactive`, `pre_registered`, `prediction_committed_at`. |
| `held_out_2023_24` | object | 2023-24 walk-forward test. See [Evaluation block](#evaluation-block-held_out_2023_24-prereg_2025). |
| `prereg_2025` | object | 2025 pre-registered holdout. Same structure, without `per_season`. |
| `live_2026` | object | 2026 live shadow mode. See [`live_2026`](#live_2026). |
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
| `prediction_commit` | string | Live only: full hash of the commit that added the week's prediction file. |
| `prediction_committed_at` | string | Live only: that commit's **local author timestamp** (UTC). The owner's computer sets it, so it is not proof of push time. |
| `prediction_committed_at_kind` | string | Live only: says the above in words, so a per-game display carries the caveat. |

Omitted on purpose: the 2025 tie (`2025_04_GB_DAL`) has no stored model prediction (the holdout dropped
ties before scoring), so it is not in `games`. It is counted in `prereg_2025.ties_excluded`.
