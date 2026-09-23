# Phase 9 — Data-Layer Spike + In-Season Ingest

**Date:** 2026-09-22 (US/Pacific; pulls were at 2026-09-23 ~01:40–02:10 UTC)
**SPEC reference:** Section 13, Phase 9.
**Scope:** data pulls allowed (repeatable). No model training, no feature-logic or feature-parameter changes, no Phase 10 work. Only `src/data_ingest.py` changed in `src/`, plus the new helper `src/nflreadpy_boundary.py`. `features.py` and everything downstream are untouched.

---

## 0. Summary

| Question | Answer |
|---|---|
| Can `nfl_data_py` 0.3.3 pull 2026? | **Yes.** Schedules and pbp both work today (272 games; pbp complete through week 2). |
| Can `nflreadpy` 0.1.5 pull 2026? | **Yes.** Same counts, and the 2026 schedule values are identical to nfl_data_py's. |
| Equivalence test (2002–2025) | **PASS, byte-identical.** The rebuilt `game_features.parquet` and `team_game_log.parquet` have the **same sha256** as the canonical files. |
| Decision (owner's pre-set rule) | **MIGRATE** `data_ingest.py` to nflreadpy. |
| In-season ingest | `python -m src.data_ingest --refresh-season 2026` re-pulls only the in-progress season and writes an immutable, timestamped schedule snapshot. Every file written gets pull metadata. 14/14 tests pass. |
| Canonical data | All 52 files in `data/raw/*.parquet` + `data/processed/{game_features,team_game_log}.parquet` + `feature_manifest.json` are **unchanged** (sha256 before = after). Afterwards, at owner request, two files were **added** (nothing modified): the Sep 8 schedule snapshot and `pull_metadata.json` (§1.3). |

One judgement call, **confirmed by the owner** (§2.4): a naive swap **crashes** the pipeline on an int32/int64 dtype mismatch. The boundary now restores the old dtype contract, and no values change.

---

## 1. Part A — Can each library get 2026 data?

Script: `scripts/phase9_probe_2026.py`. Outputs go to `data/phase9_scratch/part_a/` (gitignored), including `part_a_report.json`.

### 1.1 Results (pulled 2026-09-23 01:42 UTC)

| | nfl_data_py 0.3.3 | nflreadpy 0.1.5 |
|---|---|---|
| API used | `import_schedules([2026])`, `import_pbp_data([2026], downcast=True)` | `load_schedules([2026])`, `load_pbp([2026])` (verified from the installed package: `load_pbp(seasons: int\|list[int]\|bool\|None) -> polars.DataFrame`, `load_schedules(seasons=True) -> polars.DataFrame`) |
| Schedules 2026 | OK: 272 rows, all REG | OK: 272 rows, all REG |
| Scored REG games | 32 (weeks 1–2; latest gameday 2026-09-21) | 32 (same) |
| pbp 2026 | OK: 5,489 plays, 32 games, 372 columns | OK: 5,489 plays, 32 games, 372 columns |
| Latest week in pbp | 2 (complete: 16/16 and 16/16 scored games have pbp) | 2 (same) |
| Upcoming week (3) lines | `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`: 16/16 each | 16/16 each |
| Week 4 lines | 16/16 each | 16/16 each |
| Returned type | pandas | **polars**, converted at the ingest boundary |
| `get_current_season()` | n/a | 2026 (date rule: the season starts the Thursday after Labor Day) |

**Differences between the two, for 2026:** none in schedule values (every column compared, all 272 games) and none in pbp shape or column set. nfl_data_py's `pbp_participation` merge adds nothing for 2026 because no 2026 participation file exists yet.

### 1.2 Where each library actually reads from (verified in the installed source)

| | nfl_data_py 0.3.3 | nflreadpy 0.1.5 |
|---|---|---|
| pbp | `github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_<yr>.parquet`, then **left-merges** `pbp_participation_<yr>.parquet` | the **same** pbp release asset, no participation merge |
| schedules | `http://www.habitatring.com/games.csv` (Lee Sharpe's CSV), read with `read_csv` | `nflverse-data/releases/download/schedules/games.parquet`; unrecognised `roof` values are set to null |
| float handling | `downcast=True` casts float64 → float32 | float64 kept |
| caching | none (`cache=False`) | in-memory, 24 h by default. The ingest sets `cache_mode="off"` so a refresh is always a real download. |

### 1.3 Freshness

| Source | Last updated (UTC) | Context |
|---|---|---|
| `pbp/play_by_play_2026.parquet` release asset | 2026-09-22 14:02:53 | Week 2's last game (MNF, NYG @ LA) kicked off 2026-09-22 00:15 UTC. So the asset was rebuilt **about 14 h after kickoff, about 10–11 h after the game ended**. This is a single observation, not a measured SLA. |
| `schedules/games.parquet` release asset | 2026-09-23 01:36:35 | Checked at 01:42, so about 6 minutes old. Schedules and lines update frequently in-season. |
| `habitatring.com/games.csv` | Last-Modified 2026-09-23 01:40:09 | Also minutes old. The two schedule sources stay in sync. |

Practical reading: the previous week's pbp should be complete by Tuesday. Week 3 opens Thursday 2026-09-24 20:15 ET (`2026_03_ATL_GB`), so a Wednesday refresh has complete week 1–2 pbp and current lines.

**Lines move in place.** Against the Sep 8 cached 2026 schedule, `spread_line` now differs on 98–99 of 272 games and the moneylines on 111 (see §2.3). The count changed between two pulls about 30 minutes apart. nflverse keeps **one** line per game and overwrites it, so the snapshots in Part C are the only record of what the line was at a given time.

> **Upstream lines mutate in place (~1/3 of 2026 spreads changed Sep 8 → Sep 23), which strengthens the Phase 7 caveat: historical line columns are most likely latest/closing values. Snapshots are the only record of time-specific lines and are irreplaceable.**

The Sep 8 pre-season file was therefore preserved (owner request) before any refresh. It was copied byte for byte, and the original was left untouched, to `data/raw/snapshots/schedules_2026_20260908T182732Z.parquet` (sha256 `613ead13…`, read-only) by `scripts/phase9_preserve_sep8_snapshot.py`. Its pull time is the file mtime, recorded in `pull_metadata.json` as `"pulled_at_precision": "approximate (mtime)"`, library nfl_data_py 0.3.3. Snapshots are tracked in git (`.gitignore` re-includes `data/raw/snapshots/`).

---

## 2. Part B — Equivalence test (the migration gate)

Script: `scripts/phase9_equivalence.py` (stages `pull`, `raw`, `rebuild`, `features`). Everything is written under `data/phase9_scratch/` (gitignored). `data/raw/` and `data/processed/` are only read.

### 2.1 Confound ruled out first: data vintage

Both libraries read the same pbp release assets. So if nflverse had rebuilt historical pbp since the 2026-09-08 cache pull, any mismatch would reflect the data version, not the library. GitHub release metadata shows every `play_by_play_2002..2025.parquet` asset was last updated **before** 2026-09-08 (latest: 2020 on 2026-08-26; 2002/2024/2025 on 2026-08-13). The schedules asset updates continuously, so historical schedule rows *could* have changed. §2.3 shows they did not change in any column the pipeline reads.

### 2.2 Method

1. Pull pbp 2002–2025 and schedules 2002–2026 through nflreadpy, converted with `src/nflreadpy_boundary.py`. Schedules include 2026 because the canonical cache holds `schedules_2026.parquet` and `evaluate.load_schedules` globs every `schedules_*.parquet`, so the rebuild must see the same set of seasons.
2. Compare cached vs nflreadpy raw per season, keyed on `game_id` (schedules) or `(game_id, play_id)` (pbp). The comparison covers row counts, duplicate keys, key sets, row order, column sets, dtypes of pipeline-read columns, and values of **every** shared column. Output: `raw_summary.csv` and `raw_differences.csv` (one row per column × season × issue).
3. Run the **unchanged** `features.build_game_features()` (defaults: 2002–2025, k=4) twice. Module-level paths (`data_ingest.RAW_DIR`, `evaluate.load_schedules`'s `raw_dir`, `features.PROC_DIR`) are redirected to scratch at runtime; no source edit.
   - **control:** canonical `data/raw/` → `proc_control/`, which proves the current code still reproduces the Sep 17 canonical file;
   - **nflreadpy:** scratch raw → `proc_nflreadpy/`.
4. Compare both rebuilds to `data/processed/`.

**Pipeline-read columns** (from a grep of every reader in `src/` and `scripts/`):
- pbp (only `features.build_team_game_log`): `game_id, posteam, defteam, pass, rush, qb_dropback, sack, qb_scramble, epa, qb_epa, success, cpoe, interception, air_yards, down, qb_hit`.
- schedules: `game_id, season, game_type, week, gameday, gametime, home_team, away_team, home_score, away_score, home_rest, away_rest, div_game, roof, location, stadium_id`. The Vegas benchmark scripts also read `spread_line, total_line, home_moneyline, away_moneyline, result`, which are never features.

**Float tolerance: exact (zero).** Floats are compared with `np.array_equal(..., equal_nan=True)`, so NaN positions must match too. Why exact:
- the pbp source files are the same upstream assets;
- both paths hand the pipeline float32 values;
- the pipeline is deterministic.

So any non-zero difference would be a real difference, not float noise. The max absolute difference was also computed so a non-exact result could be judged. It was not needed.

### 2.3 Raw comparison — every difference

**Row counts:** identical for every season, both datasets. pbp: 47,355 / 46,811 / 46,705 / 46,823 / 46,299 / 46,266 / 45,917 / 46,519 / 46,892 / 47,448 / 47,834 / 48,158 / 47,629 / 48,122 / 47,651 / 47,245 / 47,109 / 47,260 / 47,705 / 49,922 / 49,434 / 49,665 / 49,492 / 48,771 for 2002…2025. Schedules: 267 per season 2002–2019, 269 (2020), 285 (2021), 284 (2022), 285 (2023–2025), 272 (2026). No duplicate keys and no key-set differences in any season.

**pbp values:** **zero** differences in **any** shared column (all ~371 columns, not just the 16 read), in every season 2002–2025. Row order is identical in every season.

**pbp column sets** (none read by the pipeline):

| Only in cached (nfl_data_py `pbp_participation` merge) | Seasons |
|---|---|
| `defenders_in_box, defense_coverage_type, defense_man_zone_type, defense_personnel, defense_players, n_defense, n_offense, nflverse_game_id, ngs_air_yards, number_of_pass_rushers, offense_formation, offense_personnel, offense_players, old_game_id_x, old_game_id_y, players_on_play, possession_team, route, time_to_throw, was_pressure` | 2016–2025 |
| `defense_names, defense_numbers, defense_positions, offense_names, offense_numbers, offense_positions` | 2023–2025 |
| **Only in nflreadpy:** `old_game_id` (nfl_data_py's merge renamed it to `_x`/`_y`) | 2016–2025 |

**pbp dtypes:** `season` int64 (cached; nfl_data_py overwrote it with the Python int year) vs int32 (nflreadpy). Not read by the pipeline. All floats are float32 on both sides after the boundary's downcast.

**Schedule dtypes** (before the boundary contract, §2.4): `season, week, home_rest, away_rest, div_game` were int64 in the cache and int32 from nflreadpy, all 25 seasons. `old_game_id, espn` were int64 in the cache and strings from nflreadpy (same digits), all 25 seasons.

**Schedule row order:** 2025 only. 43 positions differ, all reshuffles among games with the same `gameday` and `gametime` (e.g. week 14, 2025-12-07 13:00). The pipeline sorts by `(kickoff, game_id)`.

**Schedule values** (n rows differing / rows):

| Column | Read by pipeline? | Seasons | Nature |
|---|---|---|---|
| `old_game_id` | no | every season 2002–2026, all rows | int vs string, same digits |
| `espn` | no | every season 2002–2026, all rows | int vs string, same digits |
| `surface` | no | 2022 (6/284), 2023 (35/285), 2024 (2/285), 2025 (1/285), 2026 (2/272) | null vs `''` |
| `ftn` | no | 2024 (46/285), 2025 (7/285), 2026 (272/272) | upstream ID renumbering / newly filled |
| `home_score`, `away_score`, `result` | **yes** | **2026 only** (32/272 each) | games played since the Sep 8 cache |
| `total`, `overtime`, `gsis`, `pff`, `referee` | no | 2026 only (32/272 each) | same |
| `spread_line` | **yes** (benchmark only) | 2026 only (98/272; 99 in the first pull ~30 min earlier) | lines moved since Sep 8 |
| `total_line` | **yes** (benchmark only) | 2026 only (97/272) | same |
| `home_moneyline`, `away_moneyline` | **yes** (benchmark only) | 2026 only (111/272 each) | same |
| `home_spread_odds`, `away_spread_odds` | no | 2026 only (100/272 each) | same |
| `over_odds`, `under_odds` | no | 2026 only (85/272 each) | same |
| `roof` | **yes** | 2026 only (6/272): `2026_01_BUF_HOU, 2026_01_BAL_IND, 2026_02_CAR_ATL, 2026_02_CIN_HOU, 2026_02_SEA_ARI, 2026_02_WAS_DAL` changed from null to `closed` | retractable roofs filled in on game day |
| `temp`, `wind` | no | 2026 only (20/272 each) | game-day weather |
| `away_qb_id`/`away_qb_name` | no | 2026 only (19/272) | starters filled in |
| `home_qb_id`/`home_qb_name` | no | 2026 only (16/272) | same |

**Net, seasons 2002–2025, pipeline-read columns:** zero value differences. The only differences are the int32/int64 widths and the 2025 row order.

### 2.4 First rebuild attempt crashed — and the judgement call

The first nflreadpy rebuild, with the boundary converting polars → pandas naively, **crashed** inside the unchanged `features.prior_season_mean`:

```
pandas.errors.MergeError: incompatible merge keys [1] dtype('int32') and dtype('int64'), must be the same type
```

(log: `data/phase9_scratch/rebuild_naive_int32_CRASH.log`). The control rebuild succeeded in the same run.

**What I did:** `src/nflreadpy_boundary.py` now restores the raw-file dtype contract nfl_data_py produced:
- `schedules_to_raw_contract`: every integer column → int64, which is `read_csv`'s default and what nfl_data_py gave;
- `pbp_to_raw_contract`: float64 → float32 (reproduces `downcast=True`) and `season` → int64 (reproduces nfl_data_py's `raw['season'] = year`). Other integer columns keep their parquet width, as they did before.

No value changes, and `features.py` is not touched. The equivalence test's `pull` stage calls these same functions, so the test exercises the production code path.

**Owner decision: CONFIRMED.** The gate is on the feature output, which is byte-identical. `features.py` is untouched, and int32 → int64 widening is lossless. This is the boundary conversion SPEC Phase 9 calls for.

### 2.5 Feature comparison (with the boundary contract)

| File | control vs canonical | nflreadpy vs canonical |
|---|---|---|
| `game_features.parquet` | 6,208 × 72, same columns in order, same `game_id` order, 0 columns with any difference: **EXACT** | **EXACT** (same) |
| `team_game_log.parquet` | 12,416 × 20: **EXACT** | **EXACT** |
| `feature_manifest.json` (minus `generated_at`) | identical | identical |

sha256 (file bytes):

```
747afbf463980ec1932ec71f2cf4d44fe4bb7c23063b77bd86c34a3e73f74898  data/processed/game_features.parquet
747afbf463980ec1932ec71f2cf4d44fe4bb7c23063b77bd86c34a3e73f74898  data/phase9_scratch/proc_control/game_features.parquet
747afbf463980ec1932ec71f2cf4d44fe4bb7c23063b77bd86c34a3e73f74898  data/phase9_scratch/proc_nflreadpy/game_features.parquet
d09ef4634d1b813b2dc823437a5aee7efbd352e669ba408ee724a2601b63effa  data/processed/team_game_log.parquet
d09ef4634d1b813b2dc823437a5aee7efbd352e669ba408ee724a2601b63effa  data/phase9_scratch/proc_nflreadpy/team_game_log.parquet
```

The nflreadpy rebuild ran with the **refreshed** 2026 schedule (32 scored games, moved lines, 6 changed `roof` values) in place of the Sep 8 pre-season file. It still produced byte-identical 2002–2025 features. That is also empirical evidence that refreshing `schedules_2026.parquet` does not perturb historical features, even though `evaluate.load_schedules` and `features.team_home_stadium` read every season.

### 2.6 Decision

**Pre-set rule: rebuilt features match within tolerance → MIGRATE.** Applied. `src/data_ingest.py` now uses nflreadpy, and `requirements.txt` adds `nflreadpy==0.1.5` and `polars==1.44.2`.

---

## 3. Part C — In-season ingest (`src/data_ingest.py`, nflreadpy)

### 3.1 Behaviour

| Rule | Implementation |
|---|---|
| Completed seasons are never re-pulled unless forced | A season earlier than `nflreadpy.get_current_season()` with a cached file is a cache hit. Only `--force` re-downloads it. |
| In-progress season refreshable on demand, nothing else re-downloaded | `--refresh-season <season>` re-pulls that season's pbp and schedule only. It refuses any season other than the in-progress one, both in the CLI and in `_pull_year`. |
| Timestamped, immutable schedule snapshots | Every pull of the in-progress season's schedule writes `data/raw/snapshots/schedules_<season>_<YYYYMMDDTHHMMSSZ>.parquet`. The file is written to a temp name, then published with `os.link`, which is atomic and **fails if the target exists**, then chmod'ed read-only. There is no code path that overwrites a snapshot; a same-second collision raises `FileExistsError`. |
| Snapshots don't leak into the pipeline | They live in a subdirectory; `evaluate.load_schedules` globs `schedules_*.parquet` non-recursively (tested). |
| Pull metadata | `data/raw/pull_metadata.json` records, per written file (snapshots included): `pulled_at_utc`, `library`, `library_version`, `rows`, `sha256`. Snapshots also record `snapshot_of`. Written atomically. |
| No stale library cache | `nflreadpy` `cache_mode="off"` is set at import. |
| No torn files | Season files are written to a temp name, then `os.replace`d. |
| `END_YEAR` | Now `nflreadpy.get_current_season()` (the date rule), not the calendar year, so a January run doesn't reach for a season that hasn't started. |

The public API features.py uses (`get_pbp(start, end, force)`, `get_schedules(...)`, `RAW_DIR`) is unchanged.

### 3.2 How to run the in-season refresh

In your **local terminal** (Terminal.app, iTerm2, or VS Code's integrated terminal), from the **project root**:

```bash
cd "/Users/holdenanderson/Personal Projects/NFL-GAME-PREDICTOR"
source venv/bin/activate
python -m src.data_ingest --refresh-season 2026
python -m src.data_ingest --status        # confirm: new snapshot listed, metadata present
```

The first refresh will:
- **overwrite `data/raw/schedules_2026.parquet`**, the Sep 8 pre-season copy with 0 scores and stale lines (already preserved byte for byte as `snapshots/schedules_2026_20260908T182732Z.parquet`);
- create `data/raw/pbp_2026.parquet`;
- write the first snapshot and create `pull_metadata.json`.

It touches no 2002–2025 file. I did **not** run it against `data/raw/`, because this phase's hard rule kept `data/raw/` read-only. It is a repeatable pull, not a one-shot evaluation, so either of us can run it from now on.

Other commands: `python -m src.data_ingest` fills any missing season file and would now download `pbp_2026` because it is missing. `python -m src.data_ingest --start Y --end Y --force` re-downloads completed seasons; see risk R2 before using it.

### 3.3 Tests — `python -m scripts.test_phase9_ingest`

These tests use stdlib `unittest` because pytest isn't installed and wasn't added. There is no network access: `_fetch` is stubbed, and `RAW_DIR` is a fresh temp directory per test.

```
test_pbp_contract ... ok
test_schedules_contract ... ok
test_cache_hit_does_not_rewrite_metadata ... ok
test_every_written_file_has_metadata ... ok
test_game_features_identical_2002_2025 ... ok
test_completed_season_is_a_cache_hit_without_force ... ok
test_force_redownloads_completed_season ... ok
test_missing_file_is_pulled ... ok
test_refresh_does_not_touch_completed_season_files ... ok
test_refresh_of_completed_season_is_refused ... ok
test_every_in_progress_schedule_pull_writes_a_new_snapshot ... ok
test_snapshot_is_never_overwritten ... ok
test_snapshot_is_read_only ... ok
test_snapshots_are_invisible_to_load_schedules ... ok
Ran 14 tests — OK
```

What the key tests check:
- **"refresh does not touch completed-season files"** compares sha256 **and** mtime_ns of the completed-season files before and after a refresh, and asserts the fetcher was called only for the in-progress season.
- **"never overwritten"** covers a direct same-timestamp write and a full refresh at the same timestamp. Both raise, the original bytes are unchanged, and no temp file is left behind.
- **`test_game_features_identical_2002_2025`** is the migration test. It asserts EXACT equality of the nflreadpy rebuild against the canonical `game_features` and `team_game_log`. It skips if Part B's outputs are absent.

### 3.4 Existing leakage checks — `python -m src.leakage_checks`

Run after the migration, through the new import path. Checks are unchanged from the documented state:

```
CHECK 1  PASS: 6,208 rows, 0 timestamp violations, min prev-game lag = 3.674 days, max(rolled vs current-game raw) corr = 0.378 (home_def_pressure_rate)
Elo point-in-time: team-games checked 12,384, mismatches: 0 -- VERDICT: PASS
CHECK 2  label-shuffle: mean=0.4846 std=0.0307 over 20 shuffles -- VERDICT: PASS
CHECK 3  PASS -- top = 'away_def_epa_early_ewm' at 4.8%
CHECK 4  FLAGGED: accuracy 0.6245 -> 0.6320 after dropping 'away_def_epa_early_ewm'
LEAKAGE CHECKS FAILED -- do not proceed to modeling      (exit 1)
```

Check 4's flag is the **pre-existing, investigated false positive** recorded in CLAUDE.md (Phase 5B Part 2, item 2), with identical numbers (0.6245 → 0.6320). It is overridden in the LR models' metadata via `LR_CHECK4_OVERRIDES`. `leakage_checks.main` does not apply that override, so it exits 1 exactly as it did before Phase 9. Because the features are byte-identical, every check result is necessarily identical too.

---

## 4. Known risks

- **R1. nflreadpy is pre-1.0 (0.1.5).** The API could change. It is pinned in `requirements.txt`. `load_pbp` raises `ValueError` for seasons after `get_current_season()`; the ingest treats that as "no data yet".
- **R2. Upstream history is mutable.** nflverse re-releases historical pbp (2002, 2024 and 2025 were rebuilt on 2026-08-13), and schedule IDs/fields get edited (`ftn`, `surface`). The cache under `data/raw/` is the frozen copy that Phases 6–8 depend on. **Do not `--force` completed seasons** without first re-running `python -m scripts.phase9_equivalence` against the would-be replacement. A re-pull today would reproduce, but not necessarily next month.
- **R3. The schedules source changed.** It moved from Lee Sharpe's `habitatring.com/games.csv` to nflverse's `games.parquet`. The two are value-identical today in every column the pipeline reads, but they are separate publishing paths.
- **R4. Line semantics are unchanged by this phase.** A snapshot records what nflverse's single, untimestamped line field held **at pull time**; the line's age at the source is unknown (Phase 7 §3.1). Phase 10 has to refresh immediately before logging predictions and store the snapshot filename with each prediction. Optionally it could also record the `games.parquet` release `updated_at`.
- **R5. Snapshot cadence is manual.** A snapshot exists only when a refresh runs. Two refreshes in the same second raise `FileExistsError`. The main file and its metadata will already have been written, so wait a second and rerun.
- **R6. Custom polars → pandas converter.** It exists to avoid adding pyarrow. **Do not install pyarrow** into this venv: pandas' `engine='auto'` would switch every parquet read and write from fastparquet to pyarrow. fastparquet previously came in only transitively through nfl_data_py; it is now pinned explicitly.
- **R7. Participation columns are no longer in raw pbp** (2016+). Nothing reads them. If a future feature needs them (e.g. `was_pressure`, `time_to_throw`), load them with `nflreadpy.load_participation`, which would need its own point-in-time review.
- **R8. `evaluate.load_schedules` includes every season.** After the first refresh, 2026 scored games appear in anything that calls `home_baseline_accuracy(..., seasons=None)`, e.g. the "all available" line in `evaluate.py __main__`. §2.5 shows 2002–2025 features are unaffected.
- **R9. nfl_data_py stays installed and listed through the 2026 season as the rollback path** (owner decision). It still pulls 2026 (Part A). If nflreadpy breaks mid-season, the rollback is `git checkout <pre-Phase-9 commit> -- src/data_ingest.py`. Snapshots and metadata would then have to be re-added by hand, since the old module has neither. Its `numpy<2`/`pandas<2` pins stay with it. Removal is a post-season decision.
- **R10. Season rollover.** `get_current_season()` flips on the Thursday after Labor Day. Until 2027-09-09, 2026 stays "in progress", so it stays refreshable (playoffs, stat corrections) and 2027 cannot be pulled. After that, 2026 becomes a completed season, and its final pull should happen before the flip.

---

## 5. Artifacts

| Path | In git? | Purpose |
|---|---|---|
| `src/data_ingest.py` | modified | nflreadpy ingest, `--refresh-season`, snapshots, metadata, `--status` |
| `src/nflreadpy_boundary.py` | new | pyarrow-free polars → pandas, raw-file dtype contract |
| `scripts/phase9_probe_2026.py` | new | Part A |
| `scripts/phase9_equivalence.py` | new | Part B (pull / raw / rebuild / features) |
| `scripts/test_phase9_ingest.py` | new | Part C tests |
| `scripts/phase9_preserve_sep8_snapshot.py` | new | one-time copy of the Sep 8 schedule into `snapshots/` |
| `data/raw/snapshots/schedules_2026_20260908T182732Z.parquet` | **tracked** (re-included) | preserved early-September 2026 lines |
| `data/raw/pull_metadata.json` | gitignored (under `data/raw/*`) | pull provenance |
| `requirements.txt` | modified | + `nflreadpy==0.1.5`, `polars==1.44.2`, `fastparquet==2026.5.0` |
| `.gitignore` | modified | + `data/phase9_scratch/` |
| `data/phase9_scratch/` | gitignored | scratch pulls, `raw_differences.csv`, `raw_summary.csv`, `features_comparison.json`, rebuilds, logs, protected-file hashes before/after, pip freeze before/after |

venv change: `pip install nflreadpy` added only `nflreadpy 0.1.5, polars 1.44.2, polars-runtime-32 1.44.2, pydantic 2.13.5, pydantic_core 2.46.5, pydantic-settings 2.15.0, python-dotenv 1.2.3, tqdm 4.70.1, annotated-types 0.8.0, typing-inspection 0.4.4`. No existing package changed (pip freeze diff is additions only).
