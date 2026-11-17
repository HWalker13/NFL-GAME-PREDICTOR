# Test fixture: the 2026-11-12 site export

`site_data_2026-11-12.json` (sha256 `b63cce3b…73dbf`, 666,001 bytes) is a frozen copy of
the site export as of the 2026-11-12T21:09:40Z schedule snapshot: current week 10, scores
through week 9, week 10 picks logged but not played, week 11 onward in the future.

Every exact-value assertion in `site/tests/` (records like "Picks 44–31 59%", banners, week 10
as the current week, NE at DET's 49.6%, the snapshot time) reads this file through
`loadFixture()` / `buildOnce('fixture')` in `tests/exports.ts`. Tests against the real
`exports/site_data.json` check only invariants that hold every week. Never edit or regenerate
this file to make a test pass; it is the fixed input those assertions were written against.

## Why it is regenerated, not copied from git

The only committed export as of 2026-11-12 is `1a3fac7:exports/site_data.json`, which is
schema 1.2.0. The site now requires 1.3.0 (Amendment 6: `value_tier`, `toss_up`,
`value_woe`, …), so that file no longer parses, and the tests' tier and toss-up values were
written against a 1.3.0 export that was never committed.

This file was produced by the schema-1.3.0 exporter at commit `c9f2354` (Amendment 6), run in a
throwaway git worktree with the two later schedule snapshots
(`schedules_2026_20261115T145029Z.parquet`, `schedules_2026_20261115T202714Z.parquet`)
removed, so the 2026-11-12 snapshot is the latest input. `source_commit` records this
honestly (`dirty: true`, `dirty_paths` = those two files), which is why the fixture build
runs with `NFL_ALLOW_DIRTY=1`. `generated_at` was pinned to the snapshot time.

The exporter's scorecard cross-check target was regenerated from the official graded files at
`c9f2354` into a scratch file (the committed scorecard at that commit predates its Amendment
6 sections). Nothing under `data/live/`, `docs/live/` or `exports/` was written. Every other
exporter cross-check passed unchanged.

Check: compared with `1a3fac7:exports/site_data.json`, every field the two share is identical
except `schema_version`, `generated_at` and `source_commit`. The only other differences are the
1.3.0 additions (`games[].toss_up`, `games[].value_tier`, `live_2026.amendment_6_note`,
`prediction_quality_by_confidence`, `toss_up_threshold`, `value_tier_two_star_edge`,
`value_woe`).
