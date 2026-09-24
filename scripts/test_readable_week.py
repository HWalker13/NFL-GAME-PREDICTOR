"""Tests for scripts/live/readable_week.py (the derived weekly picks page).

Same rules as scripts/test_phase10_live.py: everything runs on synthetic rows
and temporary files. Nothing here runs ``main()``, passes ``--live``, or
touches data/live/, data/live_scratch/ or docs/live/.

Usage::

    python -m scripts.test_readable_week
"""

from __future__ import annotations

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts.live import common as C, readable_week as R


def _tmpdir(tc: unittest.TestCase) -> Path:
    d = tempfile.TemporaryDirectory()
    tc.addCleanup(d.cleanup)
    return Path(d.name)


def _row(game_id, home, away, p_home, home_ml, away_ml, spread, model="champion", kickoff="2026-09-27T17:00:00Z"):
    """One logged prediction row, built with the real bet rule."""
    b = C.paper_bet(p_home, home_ml, away_ml)
    return {"season": 2026, "week": 3, "game_id": game_id, "kickoff_utc": kickoff,
            "home_team": home, "away_team": away, "model": model, "p_home": p_home,
            "pick": "home" if p_home > 0.5 else "away", "snapshot_file": "schedules_2026_20260923T214318Z.parquet",
            "snapshot_pulled_at_utc": "2026-09-23T21:43:18Z", "home_moneyline": home_ml,
            "away_moneyline": away_ml, "spread_line": spread, "implied_p_home": b["implied_p_home"],
            "edge_home": b["edge_home"], "edge_away": b["edge_away"], "paper_bet": b["paper_bet"],
            "bet_status": "official" if model == "champion" else "hypothetical",
            "no_bet_reason": b["no_bet_reason"], "bet_price": b["bet_price"], "stake": b["stake"],
            "predicted_at_utc": "2026-09-23T21:43:26Z"}


def _week() -> pd.DataFrame:
    # Values copied from the week-3 official file's champion/challenger rows.
    return pd.DataFrame([
        _row("2026_03_HOU_IND", "IND", "HOU", 0.380342, 120, -142, -2.5),              # away pick, away bet
        _row("2026_03_KC_MIA", "MIA", "KC", 0.393141, 490, -675, -11.5),               # away pick, HOME bet
        _row("2026_03_LAC_BUF", "BUF", "LAC", 0.785740, -340, 270, 7.0),               # home pick, home bet
        _row("2026_03_HOU_IND", "IND", "HOU", 0.384572, 120, -142, -2.5, "challenger"),
        _row("2026_03_KC_MIA", "MIA", "KC", 0.395913, 490, -675, -11.5, "challenger"),
        _row("2026_03_LAC_BUF", "BUF", "LAC", 0.751864, -340, 270, 7.0, "challenger"),  # no bet: differs
    ])


def _graded(pred: pd.DataFrame, scores: dict) -> pd.DataFrame:
    g = pred.copy()
    for i, r in g.iterrows():
        hs, as_ = scores.get(r["game_id"], (np.nan, np.nan))
        g.loc[i, ["home_score", "away_score"]] = [hs, as_]
        if pd.isna(hs):
            res, correct = "pending", np.nan
        elif hs == as_:
            res, correct = "tie", np.nan
        else:
            res = "home" if hs > as_ else "away"
            correct = float(res == r["pick"])
        g.loc[i, ["result", "correct"]] = [res, correct]
        out, units = C.bet_units(r["paper_bet"], r["bet_price"], hs, as_)
        g.loc[i, ["bet_outcome", "units"]] = [out, units]
    return g


class TestSpreadConvention(unittest.TestCase):
    """nflverse: positive spread_line = HOME favored; negative = AWAY favored."""

    def test_positive_is_home_favored(self):
        self.assertEqual(R.vegas_favorite_text(5.5, "GB", "ATL"), "GB by 5.5")

    def test_negative_is_away_favored(self):
        self.assertEqual(R.vegas_favorite_text(-3.5, "PIT", "CIN"), "CIN by 3.5")

    def test_whole_number_and_pickem_and_missing(self):
        self.assertEqual(R.vegas_favorite_text(7.0, "BUF", "LAC"), "BUF by 7")
        self.assertEqual(R.vegas_favorite_text(0.0, "BUF", "LAC"), "Pick'em")
        self.assertEqual(R.vegas_favorite_text(np.nan, "BUF", "LAC"), "–")

    def test_agrees_with_common_favorite_rule(self):
        for s in (5.5, -3.5, 1.0, -11.5):
            fav = C.vegas_favorite(s, np.nan, np.nan)
            self.assertTrue(R.vegas_favorite_text(s, "H", "A").startswith("H" if fav == "home" else "A"))


class TestPayout(unittest.TestCase):
    def test_plus_odds(self):
        self.assertEqual(R.payout_text(154), "+$1.54 per $1 risked")
        self.assertEqual(R.payout_text(490), "+$4.90 per $1 risked")
        self.assertEqual(R.payout_text(100), "+$1.00 per $1 risked")

    def test_minus_odds(self):
        self.assertEqual(R.payout_text(-250), "+$0.40 per $1 risked")
        self.assertEqual(R.payout_text(-110), "+$0.91 per $1 risked")
        self.assertEqual(R.payout_text(-142), "+$0.70 per $1 risked")


class TestPickedTeamPerspective(unittest.TestCase):
    def test_away_pick(self):
        r = pd.Series(_row("2026_03_HOU_IND", "IND", "HOU", 0.38, 120, -142, -2.5))
        v = R.picked_team_view(r)
        self.assertEqual(v["team"], "HOU")
        self.assertAlmostEqual(v["p_model"], 0.62)
        self.assertAlmostEqual(v["p_vegas"], 1 - r["implied_p_home"])
        self.assertAlmostEqual(v["gap"], r["edge_away"])
        self.assertEqual(v["bet"], "Bet HOU (+5.7 pts)")          # bet team = picked team: same gap
        self.assertEqual(R.gap_text(v["gap"]), "+5.7")
        self.assertEqual(v["payout"], "+$0.70 per $1 risked")

    def test_home_pick(self):
        r = pd.Series(_row("2026_03_LAC_BUF", "BUF", "LAC", 0.78574, -340, 270, 7.0))
        v = R.picked_team_view(r)
        self.assertEqual((v["team"], v["bet"]), ("BUF", "Bet BUF (+4.5 pts)"))
        self.assertAlmostEqual(v["gap"], r["edge_home"])

    def test_bet_against_the_pick_is_marked(self):
        r = pd.Series(_row("2026_03_KC_MIA", "MIA", "KC", 0.393141, 490, -675, -11.5))
        v = R.picked_team_view(r)
        self.assertEqual(v["team"], "KC")
        self.assertEqual(v["bet"], "Bet MIA † (+23.0 pts)")       # the BET team's gap, not the pick's
        self.assertEqual(R.gap_text(v["gap"]), "-23.0")
        self.assertAlmostEqual(r["edge_home"], -v["gap"])
        self.assertLessEqual(v["gap"], -0.04)
        self.assertEqual(v["payout"], "+$4.90 per $1 risked")

    def test_no_bet_and_missing_line(self):
        self.assertEqual(R.picked_team_view(pd.Series(_row("g", "H", "A", 0.55, -120, 100, 1.5)))["bet"], "No bet")

    def test_bet_gap_is_always_at_least_threshold(self):
        for args in [("g", "IND", "HOU", 0.380342, 120, -142, -2.5), ("g", "MIA", "KC", 0.393141, 490, -675, -11.5),
                     ("g", "CLE", "CAR", 0.469194, 130, -155, -2.5), ("g", "BUF", "LAC", 0.78574, -340, 270, 7.0)]:
            r = pd.Series(_row(*args))
            gap = float(R.picked_team_view(r)["bet"].split("(")[1].split(" pts")[0])
            self.assertGreaterEqual(gap, 4.0)
        v = R.picked_team_view(pd.Series(_row("g", "H", "A", 0.55, np.nan, np.nan, np.nan)))
        self.assertEqual((v["bet"], v["payout"]), ("No bet (no line)", "–"))
        self.assertEqual(R.pct(v["p_vegas"]), "–")


class TestFormatting(unittest.TestCase):
    def test_kickoff_pacific(self):
        self.assertEqual(R.kickoff_pacific("2026-09-27T17:00:00Z"), "Sun 9/27 10:00 AM")
        self.assertEqual(R.kickoff_pacific("2026-09-25T00:15:00Z"), "Thu 9/24 5:15 PM")
        self.assertEqual(R.kickoff_pacific("2026-11-08T18:00:00Z"), "Sun 11/8 10:00 AM")   # PST

    def test_rounding(self):
        self.assertEqual(R.pct(0.7667), "77%")
        self.assertEqual(R.gap_text(0.0813074), "+8.1")
        self.assertEqual(R.gap_text(-0.2302), "-23.0")


class TestOfficialNeverWritten(unittest.TestCase):
    def _official(self, root: Path) -> tuple[Path, Path, Path]:
        pred = C.predictions_path(root, 2026, 3)
        pred.parent.mkdir(parents=True)
        _week().to_csv(pred, index=False)
        pred.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        run = pred.with_name("week_03_run.json")
        run.write_text(json.dumps({"predictions_sha256": C.sha256_file(pred)}))
        return pred, C.graded_path(root, 2026, 3), run

    def test_output_paths_never_official(self):
        for live in (False, True):
            out = R.output_path(live, 2026, 3)
            self.assertFalse(out.resolve().is_relative_to(C.OFFICIAL_ROOT.resolve()))
            self.assertEqual(out.suffix, ".md")
        self.assertTrue(R.output_path(False, 2026, 3).is_relative_to(C.SCRATCH_ROOT))
        self.assertEqual(R.output_path(True, 2026, 3), C.PROJECT_ROOT / "docs" / "live" / "week_03_picks.md")
        self.assertFalse(R.parse_args(["--season", "2026", "--week", "3"]).live)

    def test_full_render_leaves_official_bytes_untouched(self):
        tmp = _tmpdir(self)
        official = tmp / "live"
        pred, graded, run = self._official(official)
        before = (C.sha256_file(pred), pred.stat().st_mtime_ns)
        with mock.patch.object(C, "OFFICIAL_ROOT", official):
            df, g = R.load_inputs(pred, graded, run)
            out = tmp / "scratch" / "week_03_picks.md"
            R.write_page(R.build_markdown(df, g, 2026, 3, "x.csv", "t"), out, [pred, graded, run])
        self.assertTrue(out.exists())
        self.assertEqual((C.sha256_file(pred), pred.stat().st_mtime_ns), before)
        self.assertEqual(sorted(p.name for p in pred.parent.iterdir()), ["week_03_predictions.csv", "week_03_run.json"])

    def test_write_into_official_dir_refused(self):
        tmp = _tmpdir(self)
        official = tmp / "live"
        pred, graded, run = self._official(official)
        before = C.sha256_file(pred)
        with mock.patch.object(C, "OFFICIAL_ROOT", official):
            for target in (pred, official / "2026" / "week_03_picks.md"):
                with self.assertRaises(C.GuardError):
                    R.write_page("x", target, [pred, graded, run])
        self.assertEqual(C.sha256_file(pred), before)
        self.assertFalse((official / "2026" / "week_03_picks.md").exists())

    def test_input_file_as_output_refused(self):
        tmp = _tmpdir(self)
        pred, graded, run = self._official(tmp / "elsewhere")
        with self.assertRaises(C.GuardError):
            R.write_page("x", pred, [pred, graded, run])

    def test_tampered_predictions_refused(self):
        tmp = _tmpdir(self)
        pred, graded, run = self._official(tmp / "live")
        run.write_text(json.dumps({"predictions_sha256": "0" * 64}))
        with self.assertRaises(C.GuardError):
            R.load_inputs(pred, graded, run)


class TestGradedView(unittest.TestCase):
    def _files(self, graded_df: pd.DataFrame) -> tuple[Path, Path]:
        tmp = _tmpdir(self)
        pred, graded = tmp / "p.csv", tmp / "g.csv"
        _week().to_csv(pred, index=False)
        graded_df.to_csv(graded, index=False)
        return pred, graded

    def test_results_columns_and_summary(self):
        scores = {"2026_03_HOU_IND": (17, 24), "2026_03_KC_MIA": (27, 20)}   # LAC_BUF pending
        pred, graded = self._files(_graded(_week(), scores))
        df, g = R.load_inputs(pred, graded, None)
        self.assertTrue(g)
        md = R.build_markdown(df, g, 2026, 3, "x.csv", "t")
        self.assertIn("| Final score | Pick right? | Bet result (units) |", md)
        self.assertIn("HOU 24, IND 17 | Yes | Won: +0.70", md)          # away pick, away bet, won
        self.assertIn("KC 20, MIA 27 | No | Won: +4.90", md)            # pick wrong, bet on MIA won
        self.assertIn("Not played yet | – | Pending", md)
        self.assertIn("**1 of 2** decided games", md)
        self.assertIn("net **+5.60 units**", md)

    def test_graded_not_matching_predictions_refused(self):
        g = _graded(_week(), {})
        g.loc[0, "p_home"] = 0.5
        pred, graded = self._files(g)
        with self.assertRaises(C.GuardError):
            R.load_inputs(pred, graded, None)

    def test_no_graded_file_means_no_result_columns(self):
        tmp = _tmpdir(self)
        pred = tmp / "p.csv"
        _week().to_csv(pred, index=False)
        df, g = R.load_inputs(pred, tmp / "missing.csv", None)
        md = R.build_markdown(df, g, 2026, 3, "x.csv", "t")
        self.assertFalse(g)
        self.assertNotIn("Final score", md)
        self.assertIn("before kickoff", md)


class TestChallengerSection(unittest.TestCase):
    def test_only_differing_games_listed(self):
        md = R.challenger_section(_week(), False)
        self.assertIn("agreed with the champion on 2 of 3 games", md)
        self.assertIn("LAC @ BUF | BUF / Bet BUF (+4.5 pts) | BUF |", md)
        self.assertNotIn("HOU @ IND", md)
        self.assertNotIn("KC @ MIA", md)

    def test_no_differences(self):
        w = _week()
        w = pd.concat([w[w.model == "champion"], w[w.model == "champion"].assign(model="challenger")])
        self.assertTrue(R.challenger_section(w, False).endswith("on 3 of 3 games."))


if __name__ == "__main__":
    prog = unittest.main(exit=False, verbosity=2)
    sys.exit(0 if prog.result.wasSuccessful() else 1)
