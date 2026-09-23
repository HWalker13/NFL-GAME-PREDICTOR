"""Phase 9 Part A -- can each library pull 2026 in-season data? (scratch only)

Pulls 2026 schedules + play-by-play through BOTH ``nfl_data_py`` (0.3.3) and
``nflreadpy``, writes whatever comes back to ``data/phase9_scratch/part_a/``,
and reports, per library:

* success/failure and row counts
* the latest 2026 week present in pbp, and whether that week is complete
  (every scored REG game in that week has pbp)
* whether the line columns are populated for the upcoming (first unscored) week
* upstream freshness: the GitHub release-asset ``updated_at`` for the nflverse
  files, and the HTTP ``Last-Modified`` of the CSV nfl_data_py's schedules use

Never writes to ``data/raw/`` or ``data/processed/``. Repeatable (a data pull,
not a one-shot evaluation).

Run::

    python -m scripts.phase9_probe_2026
"""

from __future__ import annotations

import datetime as _dt
import json
import traceback
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "phase9_scratch" / "part_a"
SEASON = 2026
LINE_COLS = ["spread_line", "total_line", "home_moneyline", "away_moneyline"]


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _week_report(sched: pd.DataFrame, pbp: pd.DataFrame | None) -> dict:
    reg = sched[sched["game_type"].eq("REG")]
    scored = reg.dropna(subset=["home_score", "away_score"])
    out: dict = {
        "sched_rows": int(len(sched)),
        "reg_games": int(len(reg)),
        "reg_scored": int(len(scored)),
        "max_scored_week": int(scored["week"].max()) if len(scored) else None,
        "max_scored_gameday": str(scored["gameday"].max()) if len(scored) else None,
    }
    unscored = reg[reg["home_score"].isna()]
    if len(unscored):
        wk = int(unscored["week"].min())
        g = reg[reg["week"].eq(wk)]
        out["upcoming_week"] = wk
        out["upcoming_week_games"] = int(len(g))
        out["upcoming_week_line_nonnull"] = {c: int(g[c].notna().sum()) for c in LINE_COLS}
        nxt = reg[reg["week"].eq(wk + 1)]
        out["following_week_line_nonnull"] = {c: int(nxt[c].notna().sum()) for c in LINE_COLS}
    if pbp is not None and len(pbp):
        pg = pbp.groupby("game_id").size()
        out["pbp_rows"] = int(len(pbp))
        out["pbp_games"] = int(pg.size)
        out["pbp_max_week"] = int(pbp["week"].max())
        per_week = {}
        for wk, g in scored.groupby("week"):
            have = int(g["game_id"].isin(pg.index).sum())
            per_week[int(wk)] = f"{have}/{len(g)}"
        out["pbp_games_per_scored_week"] = per_week
        complete = [w for w, s in per_week.items() if s.split("/")[0] == s.split("/")[1]]
        out["latest_complete_week_in_pbp"] = max(complete) if complete else None
    return out


def probe_nfl_data_py() -> dict:
    import nfl_data_py as nfl
    res: dict = {"library": "nfl_data_py", "version": "0.3.3", "pulled_at": _utcnow()}
    sched = pbp = None
    try:
        sched = nfl.import_schedules([SEASON])
        res["schedules"] = "ok" if len(sched) else "empty"
        sched.to_parquet(OUT / f"ndp_schedules_{SEASON}.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        res["schedules"] = f"FAIL {type(exc).__name__}: {exc}"
    try:
        pbp = nfl.import_pbp_data([SEASON], downcast=True, cache=False)
        res["pbp"] = "ok" if len(pbp) else "empty"
        if len(pbp):
            pbp.to_parquet(OUT / f"ndp_pbp_{SEASON}.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        res["pbp"] = f"FAIL {type(exc).__name__}: {exc}"
        res["pbp_traceback"] = traceback.format_exc(limit=3)
    if sched is not None and len(sched):
        res.update(_week_report(sched, pbp))
    return res


def probe_nflreadpy() -> dict:
    import nflreadpy as nfl
    from nflreadpy.config import update_config
    from src.nflreadpy_boundary import polars_to_pandas

    update_config(cache_mode="off")
    res: dict = {"library": "nflreadpy", "version": nfl.version if isinstance(nfl.version, str)
                 else __import__("importlib.metadata").metadata.version("nflreadpy"),
                 "pulled_at": _utcnow(),
                 "get_current_season": nfl.get_current_season()}
    sched = pbp = None
    try:
        sched = polars_to_pandas(nfl.load_schedules([SEASON]))
        res["schedules"] = "ok" if len(sched) else "empty"
        sched.to_parquet(OUT / f"nrp_schedules_{SEASON}.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        res["schedules"] = f"FAIL {type(exc).__name__}: {exc}"
    try:
        pbp = polars_to_pandas(nfl.load_pbp([SEASON]))
        res["pbp"] = "ok" if len(pbp) else "empty"
        if len(pbp):
            pbp.to_parquet(OUT / f"nrp_pbp_{SEASON}.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        res["pbp"] = f"FAIL {type(exc).__name__}: {exc}"
    if sched is not None and len(sched):
        res.update(_week_report(sched, pbp))
    return res


def upstream_freshness() -> dict:
    out: dict = {"checked_at": _utcnow()}
    for tag, name in [("pbp", f"play_by_play_{SEASON}.parquet"),
                      ("schedules", "games.parquet")]:
        try:
            r = requests.get(f"https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}",
                             timeout=30)
            r.raise_for_status()
            assets = {a["name"]: a for a in r.json()["assets"]}
            a = assets.get(name)
            out[f"nflverse-data/{tag}/{name}"] = (
                {"updated_at": a["updated_at"], "size": a["size"]} if a else "asset not found")
        except Exception as exc:  # noqa: BLE001
            out[f"nflverse-data/{tag}/{name}"] = f"FAIL {type(exc).__name__}: {exc}"
    try:
        r = requests.head("http://www.habitatring.com/games.csv", timeout=30, allow_redirects=True)
        out["habitatring games.csv Last-Modified"] = r.headers.get("Last-Modified")
    except Exception as exc:  # noqa: BLE001
        out["habitatring games.csv Last-Modified"] = f"FAIL {type(exc).__name__}: {exc}"
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"nfl_data_py": probe_nfl_data_py(),
              "nflreadpy": probe_nflreadpy(),
              "freshness": upstream_freshness()}
    (OUT / "part_a_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
