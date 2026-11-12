"""Phase 10: kickoff-aware gate for the hourly line-snapshot workflow.

    python -m scripts.live.snapshot_gate --season 2026 --dry-run   # print decision + reason only
    python -m scripts.live.snapshot_gate --season 2026             # refresh if the gate fires

Takes a snapshot ONLY if both hold:
  (a) some unplayed REG game kicks off within the next LOOKAHEAD_HOURS
      (now < kickoff <= now + 4 h), and
  (b) no snapshot in data/raw/snapshots/ was pulled in the last
      DEDUPE_MINUTES (age < 90 min; a snapshot exactly 90 min old does not block).
Otherwise it exits 0 and does nothing.

Kickoffs come from the latest committed snapshot (the raw schedules_<season>.parquet
is gitignored, so it does not exist on the Actions runner), through
``live_features.season_schedule_with_kickoff`` -- the same ET->UTC kickoff
definition predict_week and grade_week use. All comparisons are in UTC, so DST,
Saturdays, holidays and international games need no special handling.

When it fires (and not --dry-run) it runs exactly the existing refresh,
``python -m src.data_ingest --refresh-season <season>``, which writes only under
data/raw/. It never writes under data/live/ or docs/live/.

If $GITHUB_OUTPUT is set, ``fire=true|false`` is appended to it for the workflow.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.live import common as C

LOOKAHEAD_HOURS = 4.0
DEDUPE_MINUTES = 90.0


def decide(schedule: pd.DataFrame, snapshot_times: list[pd.Timestamp], now: pd.Timestamp,
           lookahead_hours: float = LOOKAHEAD_HOURS,
           dedupe_minutes: float = DEDUPE_MINUTES) -> tuple[bool, str]:
    """(fire, reason). ``schedule`` needs ``game_id`` and naive-UTC ``kickoff``
    (``home_score`` optional: a game with a score is played). ``now`` naive UTC."""
    s = schedule
    unplayed = s["home_score"].isna() if "home_score" in s else pd.Series(True, index=s.index)
    future = s[unplayed & (s["kickoff"] > now)].sort_values(["kickoff", "game_id"])
    horizon = now + pd.Timedelta(hours=lookahead_hours)
    soon = future[future["kickoff"] <= horizon]
    if soon.empty:
        if future.empty:
            return False, "no unplayed game left this season"
        nxt = future.iloc[0]
        return False, (f"no unplayed game kicks off within {lookahead_hours:g} h; next is {nxt['game_id']} "
                       f"at {C.iso(nxt['kickoff'])} (in {_dur(nxt['kickoff'] - now)})")
    first = soon.iloc[0]
    game_txt = (f"{first['game_id']} kicks off at {C.iso(first['kickoff'])} (in {_dur(first['kickoff'] - now)})"
                + (f", +{len(soon) - 1} more within {lookahead_hours:g} h" if len(soon) > 1 else ""))
    recent = [t for t in snapshot_times if now - t < pd.Timedelta(minutes=dedupe_minutes)]
    if recent:
        newest = max(recent)
        return False, (f"{game_txt}, but snapshot {C.iso(newest)} is only {_dur(now - newest)} old "
                       f"(< {dedupe_minutes:g} min)")
    newest_txt = (f"newest snapshot {C.iso(max(snapshot_times))} is {_dur(now - max(snapshot_times))} old"
                  if snapshot_times else "no snapshot yet")
    return True, f"{game_txt}; {newest_txt}"


def _dur(td: pd.Timedelta) -> str:
    mins = int(td.total_seconds() // 60)
    sign, mins = ("-" if mins < 0 else ""), abs(mins)
    return f"{sign}{mins // 60}h{mins % 60:02d}m"


def load_inputs(season: int, snap_dir: Path = C.SNAPSHOT_DIR) -> tuple[pd.DataFrame, list[pd.Timestamp], Path]:
    """(schedule with kickoff, all snapshot times, the snapshot the schedule came from)."""
    from src import live_features as LF   # heavy import (features -> data_ingest); kept local
    snaps = C.list_snapshots(season, snap_dir)
    if not snaps:
        raise C.GuardError(f"no schedule snapshot for {season} in {snap_dir}")
    sched = LF.season_schedule_with_kickoff(pd.read_parquet(snaps[-1]), season)
    return sched[["game_id", "week", "kickoff", "home_score"]], [C.snapshot_time(p) for p in snaps], snaps[-1]


def refresh(season: int) -> None:
    subprocess.run([sys.executable, "-m", "src.data_ingest", "--refresh-season", str(season)],
                   check=True, cwd=C.PROJECT_ROOT)


def _github_output(fire: bool) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"fire={'true' if fire else 'false'}\n")


def main(argv=None, refresh_fn=refresh) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the decision and reason; never refresh")
    ap.add_argument("--force", action="store_true", help="fire regardless of (a)/(b) (manual workflow runs)")
    ap.add_argument("--now", help="override the current time (ISO, UTC) -- for testing and inspection")
    args = ap.parse_args(argv)

    now = C.naive_utc(pd.Timestamp(args.now)) if args.now else C.naive_utc(C.utc_now())
    sched, snap_times, source = load_inputs(args.season)
    fire, reason = decide(sched, snap_times, now)
    if args.force:
        fire, reason = True, f"forced (--force); gate alone would say: {reason}"
    print(f"now (UTC):      {C.iso(now)}")
    print(f"schedule from:  {source.name}")
    print(f"decision:       {'SNAPSHOT' if fire else 'SKIP'}{' (dry run, no refresh)' if args.dry_run else ''}")
    print(f"reason:         {reason}")
    _github_output(fire and not args.dry_run)
    if fire and not args.dry_run:
        refresh_fn(args.season)
    return 0


if __name__ == "__main__":
    sys.exit(main())
