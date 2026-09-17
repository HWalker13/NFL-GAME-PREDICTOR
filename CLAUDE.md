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
