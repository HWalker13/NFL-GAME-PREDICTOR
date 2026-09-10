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
