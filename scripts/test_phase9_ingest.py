"""Phase 9 Part C tests -- in-season ingest, snapshots, migration equivalence.

No network: ``data_ingest._fetch`` is replaced by a stub, and ``RAW_DIR`` is
pointed at a fresh temporary directory for every test, so ``data/raw/`` is
never read or written by the ingest tests.

The migration-equivalence test reads the Part B rebuild outputs
(``data/phase9_scratch/proc_nflreadpy/``) and the canonical
``data/processed/`` files, read-only; it is skipped if Part B has not been run.

Usage (project root)::

    python -m scripts.test_phase9_ingest
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl

from src import data_ingest as DI
from src import evaluate
from src.nflreadpy_boundary import pbp_to_raw_contract, schedules_to_raw_contract

IN_PROGRESS = 2026
COMPLETED = 2024
T0 = dt.datetime(2026, 9, 23, 1, 0, 0, tzinfo=dt.timezone.utc)
T1 = dt.datetime(2026, 9, 23, 13, 30, 0, tzinfo=dt.timezone.utc)


def _frame(kind: str, year: int, tag: float) -> pd.DataFrame:
    """Tiny stand-in for one season. ``tag`` lets a test tell pulls apart."""
    if kind == "schedules":
        return pd.DataFrame({"game_id": [f"{year}_01_A_B", f"{year}_01_C_D"],
                             "season": [year, year], "game_type": ["REG", "REG"],
                             "spread_line": [tag, -tag]})
    return pd.DataFrame({"game_id": [f"{year}_01_A_B"] * 3, "play_id": [1, 2, 3],
                         "season": [year] * 3, "epa": np.float32([tag, 0.0, -tag])})


def _digest(p: Path) -> tuple[str, int]:
    return hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns


class IngestTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name)
        self.calls: list[tuple[str, int]] = []
        self.tag = 1.0

        def fake_fetch(kind: str, year: int):
            self.calls.append((kind, year))
            return _frame(kind, year, self.tag)

        self.clock = [T0]
        patches = [
            mock.patch.object(DI, "RAW_DIR", self.raw),
            mock.patch.object(DI, "_fetch", fake_fetch),
            mock.patch.object(DI, "current_season", lambda: IN_PROGRESS),
            mock.patch.object(DI, "_utcnow", lambda: self.clock[0]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        snaps = self.raw / DI.SNAPSHOT_SUBDIR
        if snaps.exists():   # read-only files: make removable
            for f in snaps.iterdir():
                f.chmod(stat.S_IRUSR | stat.S_IWUSR)
        self._tmp.cleanup()

    def seed(self, kind: str, year: int, tag: float = 9.0) -> Path:
        p = self.raw / f"{kind}_{year}.parquet"
        _frame(kind, year, tag).to_parquet(p, index=False)
        return p

    def snapshots(self) -> list[Path]:
        d = self.raw / DI.SNAPSHOT_SUBDIR
        return sorted(d.glob("*.parquet")) if d.exists() else []


class TestRefresh(IngestTestBase):
    def test_refresh_does_not_touch_completed_season_files(self):
        done = [self.seed("pbp", COMPLETED), self.seed("schedules", COMPLETED)]
        live = [self.seed("pbp", IN_PROGRESS, 9.0), self.seed("schedules", IN_PROGRESS, 9.0)]
        before = {p: _digest(p) for p in done}

        self.tag = 2.0
        DI.refresh_season(IN_PROGRESS)

        self.assertEqual(sorted(self.calls), [("pbp", IN_PROGRESS), ("schedules", IN_PROGRESS)])
        for p in done:
            self.assertEqual(_digest(p), before[p], f"{p.name} was modified by an in-season refresh")
        for p in live:   # the in-progress files really were refreshed
            self.assertIn(2.0, pd.read_parquet(p).filter(regex="spread_line|epa").abs().max().tolist())

    def test_refresh_of_completed_season_is_refused(self):
        self.seed("schedules", COMPLETED)
        with self.assertRaises(ValueError):
            DI.refresh_season(COMPLETED)
        self.assertEqual(self.calls, [])

    def test_completed_season_is_a_cache_hit_without_force(self):
        p = self.seed("pbp", COMPLETED)
        before = _digest(p)
        DI.get_pbp(COMPLETED, COMPLETED)
        self.assertEqual(self.calls, [])
        self.assertEqual(_digest(p), before)

    def test_force_redownloads_completed_season(self):
        self.seed("pbp", COMPLETED)
        DI.get_pbp(COMPLETED, COMPLETED, force=True)
        self.assertEqual(self.calls, [("pbp", COMPLETED)])

    def test_missing_file_is_pulled(self):
        DI.get_schedules(COMPLETED, COMPLETED)
        self.assertEqual(self.calls, [("schedules", COMPLETED)])
        self.assertTrue((self.raw / f"schedules_{COMPLETED}.parquet").exists())
        self.assertEqual(self.snapshots(), [], "completed seasons are not snapshotted")


class TestSnapshots(IngestTestBase):
    def test_every_in_progress_schedule_pull_writes_a_new_snapshot(self):
        self.tag = 1.0
        DI.refresh_season(IN_PROGRESS)
        first = self.snapshots()
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].name, f"schedules_{IN_PROGRESS}_20260923T010000Z.parquet")
        first_digest = _digest(first[0])

        self.clock[0], self.tag = T1, 2.0
        DI.refresh_season(IN_PROGRESS)
        snaps = self.snapshots()
        self.assertEqual(len(snaps), 2)
        self.assertEqual(_digest(first[0]), first_digest, "earlier snapshot changed")
        self.assertEqual(pd.read_parquet(snaps[0])["spread_line"].iloc[0], 1.0)
        self.assertEqual(pd.read_parquet(snaps[1])["spread_line"].iloc[0], 2.0)

    def test_snapshot_is_never_overwritten(self):
        DI.refresh_season(IN_PROGRESS)
        snap = self.snapshots()[0]
        before = _digest(snap)
        self.tag = 5.0   # same timestamp, different content
        with self.assertRaises(FileExistsError):
            DI.write_schedule_snapshot(_frame("schedules", IN_PROGRESS, 5.0), IN_PROGRESS, T0)
        with self.assertRaises(FileExistsError):
            DI.refresh_season(IN_PROGRESS)
        self.assertEqual(_digest(snap), before)
        self.assertEqual(len(self.snapshots()), 1)
        self.assertEqual([p.name for p in (self.raw / DI.SNAPSHOT_SUBDIR).iterdir()], [snap.name],
                         "a temp file was left behind")

    def test_snapshot_is_read_only(self):
        DI.refresh_season(IN_PROGRESS)
        mode = self.snapshots()[0].stat().st_mode
        self.assertFalse(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    def test_snapshots_are_invisible_to_load_schedules(self):
        self.seed("schedules", COMPLETED)
        DI.refresh_season(IN_PROGRESS)
        self.clock[0] = T1
        DI.refresh_season(IN_PROGRESS)
        self.assertEqual(len(self.snapshots()), 2)
        sched = evaluate.load_schedules(raw_dir=self.raw)
        self.assertEqual(len(sched), 4, "snapshots leaked into evaluate.load_schedules")


class TestMetadata(IngestTestBase):
    def test_every_written_file_has_metadata(self):
        DI.get_pbp(COMPLETED, COMPLETED)
        DI.refresh_season(IN_PROGRESS)
        meta = DI.load_metadata()
        names = {f"pbp_{COMPLETED}.parquet", f"pbp_{IN_PROGRESS}.parquet",
                 f"schedules_{IN_PROGRESS}.parquet",
                 f"{DI.SNAPSHOT_SUBDIR}/{self.snapshots()[0].name}"}
        self.assertEqual(set(meta), names)
        for name, m in meta.items():
            self.assertEqual(m["library"], "nflreadpy")
            self.assertEqual(m["library_version"], DI.LIBRARY_VERSION)
            self.assertEqual(m["pulled_at_utc"], "2026-09-23T01:00:00Z")
            self.assertEqual(m["sha256"], hashlib.sha256((self.raw / name).read_bytes()).hexdigest())

    def test_cache_hit_does_not_rewrite_metadata(self):
        self.seed("pbp", COMPLETED)
        DI.get_pbp(COMPLETED, COMPLETED)
        self.assertEqual(DI.load_metadata(), {})


class TestBoundaryContract(unittest.TestCase):
    def test_pbp_contract(self):
        df = pl.DataFrame({"season": pl.Series([2025, 2025], dtype=pl.Int32),
                           "week": pl.Series([1, 2], dtype=pl.Int32),
                           "epa": pl.Series([0.5, None], dtype=pl.Float64),
                           "posteam": ["KC", None]})
        out = pbp_to_raw_contract(df)
        self.assertEqual(str(out["season"].dtype), "int64")
        self.assertEqual(str(out["week"].dtype), "int32")   # parquet width kept, as before
        self.assertEqual(str(out["epa"].dtype), "float32")
        self.assertTrue(np.isnan(out["epa"].iloc[1]))
        self.assertIsNone(out["posteam"].iloc[1])

    def test_schedules_contract(self):
        df = pl.DataFrame({"season": pl.Series([2025, 2025], dtype=pl.Int32),
                           "home_score": pl.Series([21, None], dtype=pl.Int32),
                           "spread_line": [3.5, None]})
        out = schedules_to_raw_contract(df)
        self.assertEqual(str(out["season"].dtype), "int64")
        self.assertEqual(str(out["home_score"].dtype), "float64")   # int with nulls -> float64
        self.assertEqual(str(out["spread_line"].dtype), "float64")


SCRATCH = DI.PROJECT_ROOT / "data" / "phase9_scratch" / "proc_nflreadpy"
CANON = DI.PROJECT_ROOT / "data" / "processed"


@unittest.skipUnless((SCRATCH / "game_features.parquet").exists(),
                     "run `python -m scripts.phase9_equivalence` first")
class TestMigrationEquivalence(unittest.TestCase):
    def test_game_features_identical_2002_2025(self):
        from scripts.phase9_equivalence import compare_processed
        for name in ("game_features.parquet", "team_game_log.parquet"):
            res = compare_processed(pd.read_parquet(CANON / name),
                                    pd.read_parquet(SCRATCH / name), "nflreadpy")
            self.assertTrue(res["EXACT_MATCH"], f"{name}: {res}")
        gf = pd.read_parquet(SCRATCH / "game_features.parquet")
        self.assertEqual((int(gf["season"].min()), int(gf["season"].max())), (2002, 2025))


if __name__ == "__main__":
    unittest.main(verbosity=2)
