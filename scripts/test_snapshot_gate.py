"""Tests for the kickoff-aware snapshot gate (scripts/live/snapshot_gate.py).

Every test uses a fixed fake "now". Nothing here refreshes, touches the
network, or writes under data/ (the integration tests only READ the latest
committed snapshot for its real kickoff times).

Usage::

    python -m scripts.test_snapshot_gate
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.live import snapshot_gate as G

T = pd.Timestamp
KICK = T("2026-11-15 18:00:00")            # a 1 PM ET (EST) Sunday kickoff, naive UTC


def _sched(*kickoffs, scores=None):
    return pd.DataFrame({"game_id": [f"g{i}" for i in range(len(kickoffs))],
                         "kickoff": [T(k) for k in kickoffs],
                         "home_score": scores if scores is not None else [np.nan] * len(kickoffs)})


class TestDecide(unittest.TestCase):
    def test_fires_3h_before_kickoff(self):
        fire, why = G.decide(_sched(KICK), [], KICK - pd.Timedelta(hours=3))
        self.assertTrue(fire, why)
        self.assertIn("g0 kicks off at 2026-11-15T18:00:00Z (in 3h00m)", why)

    def test_does_not_fire_5h_before(self):
        fire, why = G.decide(_sched(KICK), [], KICK - pd.Timedelta(hours=5))
        self.assertFalse(fire)
        self.assertIn("next is g0", why)

    def test_lookahead_boundaries(self):
        self.assertTrue(G.decide(_sched(KICK), [], KICK - pd.Timedelta(hours=4))[0])        # exactly 4 h: in
        self.assertFalse(G.decide(_sched(KICK), [], KICK - pd.Timedelta(hours=4, seconds=1))[0])
        self.assertFalse(G.decide(_sched(KICK), [], KICK)[0])                                 # at kickoff: out

    def test_does_not_fire_if_snapshot_30_min_ago(self):
        now = KICK - pd.Timedelta(hours=3)
        fire, why = G.decide(_sched(KICK), [now - pd.Timedelta(minutes=30)], now)
        self.assertFalse(fire)
        self.assertIn("only 0h30m old (< 90 min)", why)

    def test_fires_if_newest_snapshot_2h_old(self):
        now = KICK - pd.Timedelta(hours=3)
        snaps = [now - pd.Timedelta(hours=30), now - pd.Timedelta(hours=2)]
        fire, why = G.decide(_sched(KICK), snaps, now)
        self.assertTrue(fire, why)
        self.assertIn("is 2h00m old", why)

    def test_dedupe_boundary_and_future_dated_snapshot(self):
        now = KICK - pd.Timedelta(hours=3)
        self.assertTrue(G.decide(_sched(KICK), [now - pd.Timedelta(minutes=90)], now)[0])    # exactly 90: allowed
        self.assertFalse(G.decide(_sched(KICK), [now - pd.Timedelta(minutes=89)], now)[0])
        self.assertFalse(G.decide(_sched(KICK), [now + pd.Timedelta(minutes=5)], now)[0])    # clock skew: block

    def test_played_game_does_not_count(self):
        # a game with a final score is not "unplayed", even if its listed kickoff is ahead
        self.assertFalse(G.decide(_sched(KICK, scores=[24.0]), [], KICK - pd.Timedelta(hours=1))[0])

    def test_saturday_game_synthetic(self):
        sat = T("2026-12-19 22:00:00")       # Sat 17:00 EST
        self.assertTrue(G.decide(_sched(sat), [], sat - pd.Timedelta(hours=2))[0])

    def test_1430z_sunday_game_synthetic(self):
        intl = T("2026-11-15 14:30:00")      # Sun 09:30 EST international game
        # 11:00Z run fires for it; the 1 PM slate (18:00Z) alone would not until 14:00Z
        fire, why = G.decide(_sched(intl, KICK), [], T("2026-11-15 11:00:00"))
        self.assertTrue(fire, why)
        self.assertIn("g0 kicks off at 2026-11-15T14:30:00Z", why)

    def test_never_fires_after_last_game(self):
        last = T("2027-01-10 21:25:00")
        s = _sched("2027-01-10 18:00:00", last)
        for now in (last, last + pd.Timedelta(minutes=1), last + pd.Timedelta(days=30)):
            fire, why = G.decide(s, [], now)
            self.assertFalse(fire)
            self.assertEqual(why, "no unplayed game left this season")


class TestRealSchedule(unittest.TestCase):
    """Real kickoff times from the latest committed 2026 snapshot (read only),
    so the ET -> UTC conversion (incl. DST) is the one predict/grade use."""

    @classmethod
    def setUpClass(cls):
        cls.sched, _, cls.source = G.load_inputs(2026)

    def _kick(self, game_id):
        return self.sched.loc[self.sched["game_id"] == game_id, "kickoff"].iloc[0]

    def test_saturday_game(self):
        sat = self.sched[self.sched["kickoff"].dt.dayofweek == 5]
        self.assertGreater(len(sat), 0, "no Saturday game in the schedule")
        g = sat.sort_values("kickoff").iloc[0]
        # strip it down to that game alone, unplayed, so only it can trigger
        one = g.to_frame().T.assign(home_score=np.nan)
        one["kickoff"] = pd.to_datetime(one["kickoff"])
        self.assertTrue(G.decide(one, [], g["kickoff"] - pd.Timedelta(hours=3))[0])
        self.assertFalse(G.decide(one, [], g["kickoff"] - pd.Timedelta(hours=5))[0])

    def test_1430z_international_game(self):
        k = self._kick("2026_10_NE_DET")                      # Sun Nov 15, 09:30 ET (EST)
        self.assertEqual(k, T("2026-11-15 14:30:00"))
        one = _sched(k)
        self.assertTrue(G.decide(one, [], k - pd.Timedelta(hours=3))[0])

    def test_dst_tnf_kickoffs(self):
        self.assertEqual(self._kick("2026_08_CAR_GB"), T("2026-10-30 00:15:00"))    # EDT
        self.assertEqual(self._kick("2026_09_JAX_BAL"), T("2026-11-06 01:15:00"))   # EST

    def test_never_fires_after_last_game(self):
        last = self.sched["kickoff"].max()
        s = self.sched.assign(home_score=np.nan)
        self.assertFalse(G.decide(s, [], last)[0])
        self.assertFalse(G.decide(s, [], last + pd.Timedelta(days=60))[0])


class TestMain(unittest.TestCase):
    def test_dry_run_never_refreshes_and_writes_no_fire(self):
        calls = []
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        out = Path(d.name) / "gh_out"
        os.environ["GITHUB_OUTPUT"] = str(out)
        self.addCleanup(os.environ.pop, "GITHUB_OUTPUT", None)
        # --force makes the gate fire; --dry-run must still not refresh
        G.main(["--season", "2026", "--dry-run", "--force"], refresh_fn=calls.append)
        self.assertEqual(calls, [])
        self.assertEqual(out.read_text(), "fire=false\n")

    def test_fires_calls_refresh_once(self):
        calls = []
        os.environ.pop("GITHUB_OUTPUT", None)
        G.main(["--season", "2026", "--force"], refresh_fn=calls.append)
        self.assertEqual(calls, [2026])


if __name__ == "__main__":
    prog = unittest.main(exit=False, verbosity=2)
    sys.exit(0 if prog.result.wasSuccessful() else 1)
