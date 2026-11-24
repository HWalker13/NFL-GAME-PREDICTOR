# Test fixture: the 2026-11-12 site export

`site_data_2026-11-12.json` (sha256 `3077a13b…7816b2`, 676,274 bytes, schema 1.5.0) is a frozen
copy of the site export as of the 2026-11-12T21:09:40Z schedule snapshot: current week 10, scores
through week 9, week 10 picks logged but not played, week 11 onward in the future.

Every exact-value assertion in `site/tests/` (records like "Picks 44–31 59%", banners, week 10
as the current week, NE at DET's 49.6%, the snapshot time, the ★ / ★★ tier rows, the paper-bet record) reads
this file through `loadFixture()` / `buildOnce('fixture')` in `tests/exports.ts`. Tests against
the real `exports/site_data.json` check only invariants that hold every week. Never edit or
regenerate this file to make a test pass; it is the fixed input those assertions were written
against. Regenerate it only when the schema changes, by the recipe below, and check the diff.

## Recipe (regenerated 2026-11-24 for schema 1.5.0, Amendment 8)

The game data is the repo at commit `c9f2354` (Amendment 6), the last commit whose `data/live/`
is the 2026-11-12 state (weeks 3–9 graded, week 10 predicted). The overlaid 1.5.0 code reads that
state unchanged: the graded files of that date have no `actual_kickoff_utc` column, and the
exporter skips that one check for them by design.

1. `git worktree add --detach <scratch>/wt c9f2354` (a throwaway worktree outside the repo).
2. Copy the schema-1.5.0 versions of `scripts/live/common.py`, `scripts/live/scorecard.py`,
   `scripts/export_site_data.py` and `exports/SCHEMA.md` from the Amendment 8 Stage 2 working
   tree into the worktree.
3. Delete the two later schedule snapshots
   (`schedules_2026_20261115T145029Z.parquet`, `schedules_2026_20261115T202714Z.parquet`) from
   the worktree, so the 2026-11-12 snapshot is the latest input.
4. Copy the gitignored inputs the exporter reads from the main checkout into the worktree:
   everything under `data/processed/`, and `data/raw/schedules_{2023,2024,2025}.parquet`.
5. In the worktree, render the scorecard from the official graded files at `c9f2354` with the
   1.5.0 `scorecard` functions (`load_graded` / `build_markdown` on `data/live`) into a scratch
   file, then run the exporter with `SCORECARD_PATH` pointed at that scratch file and
   `--generated-at 2026-11-12T21:09:40Z` (the snapshot time), writing to scratch. The committed
   scorecard at `c9f2354` has Amendment 6's two tier rows, so it can't be the cross-check
   target. Every exporter cross-check passed, including the one against `audit/out/metrics.json`.
6. Copy the output here and remove the worktree.

Nothing under `data/live/`, `docs/live/` or `exports/` in the main checkout was written.

`source_commit` records this honestly: `hash` = `c9f2354…`, `dirty: true`, and `dirty_paths` =
the two deleted snapshots plus the three overlaid code files. That is why the fixture build
runs with `NFL_ALLOW_DIRTY=1`.

## Check

Compared with the previous (schema 1.4.0) fixture, sha256 `2cda464a…40d1b1`, every field the two
share is identical except `schema_version` and `live_2026.amendment_6_note`. The rest is:
- added: `live_2026.value_calls`, `live_2026.paper_bets`, `live_2026.picks_vs_vegas`,
  `definitions.paper_bet`, `definitions.value_call`, and (from 1.4.1) `games[].kickoff_logged_utc` /
  `games[].kickoff_moved` on the 90 live games (no game had moved);
- removed: `live_2026.value_woe`.

The official record (28–24–1, 5 pending) is unchanged. Value calls (★ 12, ★★ 8 logged) are 11–7;
non-starred paper bets (38 logged) are 17–17–1.
