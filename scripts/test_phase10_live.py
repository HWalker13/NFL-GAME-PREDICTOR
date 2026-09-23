"""Phase 10 tests: guards, bet rule, grading conventions, check-4 exception.

Every guard is tested by calling its function directly on temporary files.
Nothing here runs a script's ``main()``, passes ``--live``, or touches
data/live/, data/live_scratch/, models/ or data/raw/ (the one data-reading
test, the masked-week equivalence, only READS the raw cache).

Usage::

    python -m scripts.test_phase10_live
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.live import common as C, grade_week as G, predict_week as P, scorecard as S
from src import leakage_checks as LC


def _tmpdir(tc: unittest.TestCase) -> Path:
    d = tempfile.TemporaryDirectory()
    tc.addCleanup(d.cleanup)
    return Path(d.name)


# --------------------------------------------------------------------------- #
# A1 -- documented check-4 exception
# --------------------------------------------------------------------------- #
def _res4(feature="away_def_epa_early_ewm", full=168, ablated=170, n=269, collapsed=False,
          model="LogisticRegression"):
    return {"model": model, "dropped_feature": feature, "full_accuracy": full / n,
            "ablated_accuracy": ablated / n, "increased_after_removal": ablated > full,
            "collapsed_to_baseline": collapsed, "suspicious": collapsed or ablated > full}


class TestCheck4Exception(unittest.TestCase):
    def test_exact_match_passes(self):
        ok, msg = LC.check4_known_exception(_res4(), 269)
        self.assertTrue(ok, msg)
        self.assertIn("known exception", msg)

    def test_changed_ablated_value_fails(self):
        ok, msg = LC.check4_known_exception(_res4(ablated=171), 269)
        self.assertFalse(ok)
        self.assertIn("LOUD", msg)

    def test_changed_full_value_fails(self):
        self.assertFalse(LC.check4_known_exception(_res4(full=167), 269)[0])

    def test_values_drifting_to_a_plain_pass_still_fail(self):
        # no longer an increase -> would pass SPEC 5.5 #4 alone, but the numbers moved
        self.assertFalse(LC.check4_known_exception(_res4(ablated=160), 269)[0])

    def test_changed_validation_size_fails(self):
        self.assertFalse(LC.check4_known_exception(_res4(n=270), 270)[0])

    def test_collapse_fails(self):
        self.assertFalse(LC.check4_known_exception(_res4(collapsed=True), 269)[0])

    def test_other_feature_tripping_fails(self):
        ok, msg = LC.check4_known_exception(_res4(feature="elo_diff"), 269)
        self.assertFalse(ok)
        self.assertIn("NO documented exception", msg)

    def test_other_feature_clean_passes(self):
        self.assertTrue(LC.check4_known_exception(_res4(feature="elo_diff", ablated=160), 269)[0])


# --------------------------------------------------------------------------- #
# Write-once / overwrite refusal
# --------------------------------------------------------------------------- #
class TestWriteOnce(unittest.TestCase):
    def test_second_write_refused_and_file_unchanged(self):
        p = _tmpdir(self) / "2026" / "predictions" / "week_04_predictions.csv"
        C.write_once_readonly(lambda t: t.write_text("first\n"), p)
        self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o444)
        with self.assertRaises(C.GuardError):
            C.write_once_readonly(lambda t: t.write_text("second\n"), p)
        self.assertEqual(p.read_text(), "first\n")
        self.assertEqual([x.name for x in p.parent.iterdir()], [p.name], "temp file left behind")

    def test_existing_file_refused_before_writer_runs(self):
        p = _tmpdir(self) / "x.csv"
        p.write_text("keep")
        called = []
        with self.assertRaises(C.GuardError):
            C.write_once_readonly(lambda t: called.append(1), p)
        self.assertEqual(called, [])

    def test_graded_file_is_regenerable(self):
        p = _tmpdir(self) / "week_04_graded.csv"
        G.write_graded(pd.DataFrame({"a": [1]}), p)
        G.write_graded(pd.DataFrame({"a": [2]}), p)
        self.assertEqual(pd.read_csv(p)["a"].tolist(), [2])


# --------------------------------------------------------------------------- #
# --live is never the default
# --------------------------------------------------------------------------- #
class TestLiveNeverDefault(unittest.TestCase):
    def test_predict(self):
        a = P.parse_args(["--season", "2026", "--week", "4"])
        self.assertFalse(a.live)
        self.assertEqual(C.root_dir(a.live), C.SCRATCH_ROOT)
        self.assertTrue(P.parse_args(["--season", "2026", "--week", "4", "--live"]).live)

    def test_grade(self):
        self.assertFalse(G.parse_args(["--season", "2026", "--week", "4"]).live)

    def test_scorecard(self):
        a = S.parse_args(["--season", "2026"])
        self.assertFalse(a.live)
        self.assertTrue(str(S.output_path(a.live, 2026)).startswith(str(C.SCRATCH_ROOT)))
        self.assertEqual(S.output_path(True, 2026), C.PROJECT_ROOT / "docs" / "live" / "SCORECARD_2026.md")

    def test_scratch_is_gitignored_and_distinct(self):
        self.assertNotEqual(C.SCRATCH_ROOT, C.OFFICIAL_ROOT)
        self.assertIn("data/live_scratch/", (C.PROJECT_ROOT / ".gitignore").read_text().splitlines())


# --------------------------------------------------------------------------- #
# Kicked-off exclusion
# --------------------------------------------------------------------------- #
class TestKickedOff(unittest.TestCase):
    def test_split(self):
        now = pd.Timestamp("2026-10-04 17:00:00")
        rows = pd.DataFrame({
            "game_id": ["thu", "at_now", "sun_late", "mon", "scored_future"],
            "kickoff": pd.to_datetime(["2026-10-02 00:15", "2026-10-04 17:00", "2026-10-04 20:25",
                                       "2026-10-06 00:15", "2026-10-06 00:20"]),
            "is_played": [True, False, False, False, True],
        })
        keep, gone = P.split_kicked_off(rows, now)
        self.assertEqual(keep["game_id"].tolist(), ["sun_late", "mon"])
        self.assertEqual(sorted(gone["game_id"]), ["at_now", "scored_future", "thu"])


# --------------------------------------------------------------------------- #
# Freshness + pbp completeness + frozen files
# --------------------------------------------------------------------------- #
class TestFreshness(unittest.TestCase):
    def setUp(self):
        self.raw = _tmpdir(self)
        self.snaps = self.raw / "snapshots"
        self.snaps.mkdir()
        self.meta = self.raw / "pull_metadata.json"
        pd.DataFrame({"game_id": ["g1"], "x": [1]}).to_parquet(self.raw / "schedules_2026.parquet", index=False)
        pd.DataFrame({"game_id": ["g1"]}).to_parquet(self.raw / "pbp_2026.parquet", index=False)

    def _meta(self, pbp_at, sched_at):
        self.meta.write_text(json.dumps({
            "pbp_2026.parquet": {"pulled_at_utc": pbp_at},
            "schedules_2026.parquet": {"pulled_at_utc": sched_at}}))

    def _snap(self, ts, same=True):
        p = self.snaps / f"schedules_2026_{ts}.parquet"
        if same:
            p.write_bytes((self.raw / "schedules_2026.parquet").read_bytes())
        else:
            pd.DataFrame({"game_id": ["g1"], "x": [2]}).to_parquet(p, index=False)
        return p

    def check(self):
        return C.check_freshness(2026, self.raw, self.snaps, self.meta)

    def test_fresh_identical_snapshot_passes(self):
        self._meta("2026-09-30T15:00:00Z", "2026-09-30T15:00:01Z")
        self._snap("20260929T120000Z", same=False)
        s = self._snap("20260930T150001Z")
        self.assertEqual(self.check(), s)

    def test_snapshot_older_than_raw_pull_refused(self):
        self._meta("2026-09-30T15:00:00Z", "2026-09-30T15:00:01Z")
        self._snap("20260930T150000Z")          # one second older than the schedule pull
        with self.assertRaisesRegex(C.GuardError, "older than the latest raw pull"):
            self.check()

    def test_snapshot_not_identical_refused(self):
        self._meta("2026-09-30T15:00:00Z", "2026-09-30T15:00:01Z")
        self._snap("20260930T150001Z", same=False)
        with self.assertRaisesRegex(C.GuardError, "not byte-identical"):
            self.check()

    def test_no_snapshot_refused(self):
        self._meta("2026-09-30T15:00:00Z", "2026-09-30T15:00:01Z")
        with self.assertRaisesRegex(C.GuardError, "no schedule snapshot"):
            self.check()

    def test_missing_metadata_refused(self):
        self._snap("20260930T150001Z")
        with self.assertRaisesRegex(C.GuardError, "no pull metadata"):
            self.check()


class TestSnapshotAge(unittest.TestCase):
    def test_six_hour_limit(self):
        snap = Path("schedules_2026_20260930T120000Z.parquet")
        self.assertAlmostEqual(C.check_snapshot_age(snap, pd.Timestamp("2026-09-30 18:00:00")), 6.0)
        with self.assertRaisesRegex(C.GuardError, "h old"):
            C.check_snapshot_age(snap, pd.Timestamp("2026-09-30 18:00:01"))
        with self.assertRaisesRegex(C.GuardError, "h old"):   # Monday refresh, Wednesday predict
            C.check_snapshot_age(snap, pd.Timestamp("2026-10-02 12:00:00"))


class TestPbpComplete(unittest.TestCase):
    now = pd.Timestamp("2026-09-30 18:00")

    def sched(self, rows):
        return pd.DataFrame(rows, columns=["game_id", "season", "game_type", "week",
                                           "home_score", "away_score", "kickoff"])

    def test_complete_passes(self):
        s = self.sched([["a", 2026, "REG", 3, 20, 17, pd.Timestamp("2026-09-27 17:00")],
                        ["t", 2026, "REG", 3, 20, 20, pd.Timestamp("2026-09-27 17:00")],
                        ["n", 2026, "REG", 4, np.nan, np.nan, pd.Timestamp("2026-10-04 17:00")]])
        self.assertEqual(C.check_pbp_complete(s, {"a", "t"}, 2026, 4, self.now), [])

    def test_missing_pbp_refused(self):
        s = self.sched([["a", 2026, "REG", 3, 20, 17, pd.Timestamp("2026-09-27 17:00")],
                        ["t", 2026, "REG", 3, 20, 20, pd.Timestamp("2026-09-27 17:00")]])
        with self.assertRaisesRegex(C.GuardError, "pbp missing"):
            C.check_pbp_complete(s, {"a"}, 2026, 4, self.now)   # the tie has no pbp

    def test_past_kickoff_without_score_refused(self):
        s = self.sched([["mnf", 2026, "REG", 3, np.nan, np.nan, pd.Timestamp("2026-09-29 00:15")]])
        with self.assertRaisesRegex(C.GuardError, "no final score"):
            C.check_pbp_complete(s, set(), 2026, 4, self.now)

    def test_postponed_future_game_logged_not_fatal(self):
        s = self.sched([["pp", 2026, "REG", 3, np.nan, np.nan, pd.Timestamp("2026-12-01 18:00")]])
        self.assertEqual(C.check_pbp_complete(s, set(), 2026, 4, self.now), ["pp"])


class TestPreregGate(unittest.TestCase):
    def test_exact_line_only(self):
        d = _tmpdir(self)
        cases = [
            (None, False),
            ("**Status:** DRAFT -- `--live` refuses until this line reads `**Status:** APPROVED`.\n", False),
            ("# x\n**Status:** APPROVED\n", True),
            ("  **Status:** APPROVED  \n", True),
            ("**Status:** APPROVED (conditional)\n", False),
        ]
        for i, (text, want) in enumerate(cases):
            p = d / f"p{i}.md"
            if text is not None:
                p.write_text(text)
            self.assertEqual(C.prereg_approved(p), want, text)


class TestCalibratedChallenger(unittest.TestCase):
    def test_challenger_is_champion_plus_calibrator(self):
        import joblib
        from src import live_models as LM
        champ = joblib.load(C.PROJECT_ROOT / C.FROZEN_MODELS["champion"][0])
        chall = joblib.load(C.PROJECT_ROOT / C.FROZEN_MODELS["challenger"][0])
        self.assertEqual(chall["base_model"]["sha256"], C.FROZEN_MODELS["champion"][1])
        self.assertEqual(chall["calibration"]["oof_seasons"], [2021, 2022, 2023, 2024, 2025])
        self.assertEqual(chall["elo_home_adv"], champ["elo_home_adv"])
        X = pd.DataFrame(np.zeros((3, len(champ["feature_cols"]))), columns=champ["feature_cols"])
        np.testing.assert_array_equal(chall["pipeline"].predict_proba(X), champ["pipeline"].predict_proba(X))
        raw = champ["pipeline"].predict_proba(X)[:, 1]
        c = chall["calibrator"]
        z = c["slope"] * np.log(raw / (1 - raw)) + c["intercept"]
        np.testing.assert_allclose(LM.calibrated_proba(chall, X), 1 / (1 + np.exp(-z)))
        np.testing.assert_array_equal(LM.calibrated_proba(champ, X), raw)


class TestFrozenFiles(unittest.TestCase):
    def test_model_sha_mismatch_refused(self):
        root = _tmpdir(self)
        (root / "m.joblib").write_bytes(b"model")
        good = C.sha256_file(root / "m.joblib")
        self.assertEqual(C.check_frozen_models(root, {"champion": ("m.joblib", good)})["champion"][1], good)
        (root / "m.joblib").write_bytes(b"model, retrained")
        with self.assertRaisesRegex(C.GuardError, "frozen model changed"):
            C.check_frozen_models(root, {"champion": ("m.joblib", good)})

    def test_features_py_change_refused(self):
        p = _tmpdir(self) / "features.py"
        p.write_text("x = 1\n")
        C.check_features_py(C.sha256_file(p), p)
        with self.assertRaisesRegex(C.GuardError, "feature code changed"):
            C.check_features_py("0" * 64, p)

    def test_pinned_models_match_disk(self):
        C.check_frozen_models()   # read-only: hashes the two saved files


# --------------------------------------------------------------------------- #
# Bet rule
# --------------------------------------------------------------------------- #
class TestBetRule(unittest.TestCase):
    def test_exactly_threshold_bets(self):
        self.assertTrue(C.edge_qualifies(0.57 - 0.53))        # 0.03999999999999992 in floats
        b = C.paper_bet(0.54, -100, -100)                       # implied 0.5, edge 0.04
        self.assertEqual((b["paper_bet"], b["bet_price"], b["stake"]), ("home", -100.0, 1.0))

    def test_just_below_threshold_no_bet(self):
        self.assertFalse(C.edge_qualifies(0.0399))
        b = C.paper_bet(0.5399, -100, -100)
        self.assertEqual((b["paper_bet"], b["no_bet_reason"], b["stake"]), ("none", "edge_below_threshold", 0.0))

    def test_away_side(self):
        b = C.paper_bet(0.45, -100, -100)
        self.assertEqual((b["paper_bet"], b["bet_price"]), ("away", -100.0))
        self.assertAlmostEqual(b["edge_away"], 0.05)
        self.assertAlmostEqual(b["edge_home"], -0.05)

    def test_vig_is_removed_for_the_edge_but_kept_in_the_price(self):
        b = C.paper_bet(0.70, -150, 130)
        imp = (150 / 250) / (150 / 250 + 100 / 230)
        self.assertAlmostEqual(b["implied_p_home"], imp)
        self.assertEqual(b["paper_bet"], "home")
        self.assertEqual(b["bet_price"], -150.0)

    def test_missing_lines_no_bet(self):
        for h, a in ((np.nan, 120), (-140, np.nan), (np.nan, np.nan)):
            b = C.paper_bet(0.9, h, a)
            self.assertEqual((b["paper_bet"], b["no_bet_reason"], b["stake"]), ("none", "missing_line", 0.0))
            self.assertTrue(np.isnan(b["implied_p_home"]))

    def test_settlement(self):
        self.assertEqual(C.bet_units("home", 150, 24, 17), ("win", 1.5))
        self.assertEqual(C.bet_units("away", -200, 17, 24), ("win", 0.5))
        self.assertEqual(C.bet_units("home", -200, 17, 24), ("loss", -1.0))
        self.assertEqual(C.bet_units("away", 110, 20, 20), ("void", 0.0))     # tie: stake returned
        self.assertEqual(C.bet_units("none", np.nan, 20, 10), ("none", 0.0))
        o, u = C.bet_units("home", 110, np.nan, np.nan)
        self.assertEqual(o, "pending")
        self.assertTrue(np.isnan(u))


# --------------------------------------------------------------------------- #
# CLV sign convention + grading
# --------------------------------------------------------------------------- #
class TestCLV(unittest.TestCase):
    def test_bet_clv_sign(self):
        self.assertAlmostEqual(C.bet_clv("home", 0.50, 0.55), 0.05)    # moved toward home bet
        self.assertAlmostEqual(C.bet_clv("away", 0.50, 0.55), -0.05)   # moved away from away bet
        self.assertAlmostEqual(C.bet_clv("away", 0.50, 0.45), 0.05)
        self.assertTrue(np.isnan(C.bet_clv("none", 0.5, 0.6)))

    def test_line_move_toward_model(self):
        self.assertAlmostEqual(C.line_move_toward_model(0.60, 0.50, 0.55), 0.05)
        self.assertAlmostEqual(C.line_move_toward_model(0.40, 0.50, 0.55), -0.05)
        self.assertAlmostEqual(C.line_move_toward_model(0.40, 0.50, 0.45), 0.05)
        self.assertEqual(C.line_move_toward_model(0.50, 0.50, 0.55), 0.0)


def _snap(d: Path, ts: str, rows: list[dict]) -> Path:
    p = d / f"schedules_2026_{ts}.parquet"
    pd.DataFrame(rows, columns=["game_id", "home_moneyline", "away_moneyline",
                                "home_score", "away_score"]).to_parquet(p, index=False)
    return p


class TestGrade(unittest.TestCase):
    def test_grading_conventions(self):
        d = _tmpdir(self)
        pred_snap = _snap(d, "20260930T150000Z", [])
        mid = _snap(d, "20261003T160000Z", [
            {"game_id": "sun", "home_moneyline": -150, "away_moneyline": 130},
            {"game_id": "thu", "home_moneyline": -300, "away_moneyline": 250}])
        after = _snap(d, "20261006T120000Z", [   # after every kickoff: scores only, never a "late line"
            {"game_id": "sun", "home_moneyline": -1000, "away_moneyline": 700, "home_score": 20, "away_score": 20},
            {"game_id": "thu", "home_moneyline": -1000, "away_moneyline": 700, "home_score": 27, "away_score": 10},
            {"game_id": "mon", "home_moneyline": -1000, "away_moneyline": 700,
             "home_score": np.nan, "away_score": np.nan}])
        base = dict(season=2026, week=4, model="champion", snapshot_file=pred_snap.name,
                    spread_line=3.0, bet_status="official", stake=1.0)
        pred = pd.DataFrame([
            dict(base, game_id="thu", kickoff_utc="2026-10-02T00:15:00Z", p_home=0.60, pick="home",
                 home_moneyline=-120, away_moneyline=100, implied_p_home=0.52, paper_bet="home", bet_price=-120.0),
            dict(base, game_id="sun", kickoff_utc="2026-10-04T17:00:00Z", p_home=0.40, pick="away",
                 home_moneyline=-120, away_moneyline=100, implied_p_home=0.52, paper_bet="away", bet_price=100.0),
            dict(base, game_id="mon", kickoff_utc="2026-10-06T00:15:00Z", p_home=0.55, pick="home",
                 home_moneyline=np.nan, away_moneyline=np.nan, implied_p_home=np.nan, paper_bet="none",
                 bet_price=np.nan, stake=0.0),
        ])
        g = G.grade(pred, [pred_snap, mid, after], after, pd.Timestamp("2026-10-06 12:00")).set_index("game_id")

        # Thursday: no snapshot between prediction and kickoff -> CLV excluded; win settled at -120
        self.assertEqual(g.at["thu", "clv_status"], G.CLV_NO_LATER)
        self.assertEqual((g.at["thu", "bet_outcome"], g.at["thu", "correct"]), ("win", 1.0))
        self.assertAlmostEqual(g.at["thu", "units"], 100 / 120)

        # Sunday: last pre-kickoff snapshot is `mid`, never `after`; tie -> void, 0 units, no accuracy
        self.assertEqual(g.at["sun", "last_prekickoff_snapshot_file"], mid.name)
        p_late = (150 / 250) / (150 / 250 + 100 / 230)
        self.assertAlmostEqual(g.at["sun", "p_late_home"], p_late)
        self.assertAlmostEqual(g.at["sun", "bet_clv"], (1 - p_late) - (1 - 0.52))       # away bet, line moved home
        self.assertLess(g.at["sun", "bet_clv"], 0)
        self.assertAlmostEqual(g.at["sun", "line_move_toward_model"], -(p_late - 0.52))
        self.assertEqual((g.at["sun", "result"], g.at["sun", "bet_outcome"], g.at["sun", "units"]),
                         ("tie", "void", 0.0))
        self.assertTrue(np.isnan(g.at["sun", "correct"]) and np.isnan(g.at["sun", "log_loss"]))

        # Monday: no score yet -> pending; no line at prediction -> CLV excluded, no bet
        self.assertEqual((g.at["mon", "result"], g.at["mon", "bet_outcome"]), ("pending", "none"))
        self.assertEqual(g.at["mon", "clv_status"], G.CLV_NO_LINE_PRED)

    def test_log_loss_and_brier(self):
        d = _tmpdir(self)
        s = _snap(d, "20260930T150000Z", [{"game_id": "a", "home_moneyline": -110, "away_moneyline": -110,
                                           "home_score": 10, "away_score": 3}])
        pred = pd.DataFrame([dict(season=2026, week=4, model="champion", game_id="a", snapshot_file=s.name,
                                  kickoff_utc="2026-10-04T17:00:00Z", p_home=0.8, pick="home", spread_line=1.0,
                                  home_moneyline=-110, away_moneyline=-110, implied_p_home=0.5,
                                  paper_bet="home", bet_price=-110.0, stake=1.0)])
        g = G.grade(pred, [s], s, pd.Timestamp("2026-10-05")).iloc[0]
        self.assertAlmostEqual(g["log_loss"], -np.log(0.8))
        self.assertAlmostEqual(g["brier"], 0.04)
        self.assertAlmostEqual(g["vegas_log_loss"], -np.log(0.5))
        self.assertEqual((g["vegas_fav"], g["vegas_correct"], g["home_correct"]), ("home", 1.0, 1.0))


# --------------------------------------------------------------------------- #
# Scorecard decision rules (synthetic graded rows)
# --------------------------------------------------------------------------- #
def _graded(weeks, ll_champ, ll_chall, clv=0.0, pending_week=None):
    rows = []
    rng = np.random.default_rng(0)
    for w in weeks:
        for i in range(15):
            gid = f"{w}_{i}"
            for model, ll in (("champion", ll_champ), ("challenger", ll_chall)):
                rows.append(dict(week=w, game_id=gid, model=model, result="home", correct=1.0,
                                 log_loss=ll + rng.normal(0, 0.01), brier=0.2, home_correct=1.0,
                                 vegas_correct=1.0, vegas_log_loss=0.6, vegas_brier=0.2,
                                 paper_bet="home", edge_home=0.05, edge_away=-0.05,
                                 bet_outcome="win", units=0.9,
                                 bet_status="official" if model == "champion" else "hypothetical",
                                 clv_status="measured", bet_clv=clv + rng.normal(0, 0.01),
                                 line_move_toward_model=0.0))
    df = pd.DataFrame(rows)
    if pending_week is not None:
        df.loc[df["week"] == pending_week, "result"] = "pending"
    return df


class TestScorecard(unittest.TestCase):
    def test_checkpoint_not_evaluable_before_week9_complete(self):
        self.assertIn("Not yet evaluable", S.checkpoint_section(_graded(range(4, 9), 0.65, 0.60)))
        self.assertIn("Not yet evaluable", S.checkpoint_section(_graded(range(4, 10), 0.65, 0.60, pending_week=9)))

    def test_checkpoint_swap_only_when_lower_and_significant(self):
        self.assertIn("Decision: SWAP", S.checkpoint_section(_graded(range(4, 10), 0.65, 0.60)))
        self.assertIn("NO SWAP", S.checkpoint_section(_graded(range(4, 10), 0.60, 0.65)))
        self.assertIn("NO SWAP", S.checkpoint_section(_graded(range(4, 10), 0.65, 0.65)))

    def test_checkpoint_margin_required(self):
        # challenger better by 0.005 per game, highly significant -> still NO SWAP (margin 0.010)
        self.assertIn("NO SWAP", S.checkpoint_section(_graded(range(4, 10), 0.650, 0.645)))
        self.assertIn("Decision: SWAP", S.checkpoint_section(_graded(range(4, 10), 0.650, 0.635)))

    def test_checkpoint_ignores_weeks_after_9(self):
        df = pd.concat([_graded(range(4, 10), 0.60, 0.65), _graded(range(10, 14), 0.70, 0.40)])
        self.assertIn("NO SWAP", S.checkpoint_section(df))

    def test_verdict_needs_significant_positive_clv(self):
        self.assertIn("**Evidence of an edge**", S.verdict_section(_graded(range(4, 6), 0.6, 0.6, clv=0.02), True))
        self.assertIn("**No evidence of an edge**", S.verdict_section(_graded(range(4, 6), 0.6, 0.6, clv=-0.02), True))

    def test_verdict_uses_logged_bet_status(self):
        # champion bets have negative CLV, challenger positive; only rows logged
        # as official count -- here the challenger's, as after a checkpoint swap
        df = _graded(range(4, 6), 0.6, 0.6, clv=0.02)
        df.loc[df["model"] == "champion", "bet_clv"] = -0.02
        df["bet_status"] = np.where(df["model"] == "challenger", "official", "hypothetical")
        self.assertIn("**Evidence of an edge**", S.verdict_section(df, True))

    def test_edge_bucket_boundaries(self):
        self.assertIsNone(S.edge_bucket(0.0399))
        self.assertEqual(S.edge_bucket(0.04), "[0.04, 0.06)")
        self.assertEqual(S.edge_bucket(0.0599), "[0.04, 0.06)")
        self.assertEqual(S.edge_bucket(0.06), "[0.06, 0.08)")
        self.assertEqual(S.edge_bucket(0.36 - 0.30), "[0.06, 0.08)")     # 0.059999999999999942 in floats
        self.assertEqual(S.edge_bucket(0.0799), "[0.06, 0.08)")
        self.assertEqual(S.edge_bucket(0.08), "[0.08+)")
        self.assertEqual(S.edge_bucket(0.58 - 0.50), "[0.08+)")          # 0.07999999999999996 in floats
        self.assertEqual(S.edge_bucket(0.25), "[0.08+)")
        self.assertIsNone(S.edge_bucket(np.nan))

    def test_edge_breakdown_uses_bet_side_edge(self):
        df = pd.DataFrame([
            dict(model="champion", paper_bet="home", edge_home=0.07, edge_away=-0.07, bet_outcome="win",
                 units=1.0, clv_status="measured", bet_clv=0.01),
            dict(model="champion", paper_bet="away", edge_home=-0.09, edge_away=0.09, bet_outcome="loss",
                 units=-1.0, clv_status="measured", bet_clv=-0.02),
            dict(model="champion", paper_bet="none", edge_home=0.01, edge_away=-0.01, bet_outcome="none",
                 units=0.0, clv_status="measured", bet_clv=np.nan),
        ])
        t = S.edge_breakdown_table(df).splitlines()
        row = {ln.split("|")[2].strip(): ln for ln in t if ln.startswith("| champion")}
        self.assertIn("| 0 | 0-0 (0, 0) |", row["[0.04, 0.06)"])
        self.assertIn("| 1 | 1-0 (0, 0) | +1.00 |", row["[0.06, 0.08)"])
        self.assertIn("| 1 | 0-1 (0, 0) | -1.00 |", row["[0.08+)"])
        self.assertIn("small sample (n<30)", row["[0.08+)"])

    def test_markdown_renders(self):
        md = S.build_markdown(_graded(range(4, 6), 0.6, 0.62), 2026, 2, "2026-10-13T00:00:00Z", False)
        self.assertIn("## Cumulative -- prediction quality", md)
        self.assertIn("PROVISIONAL", md)
        self.assertIn("No graded weeks yet", S.build_markdown(pd.DataFrame(), 2026, 0, "t", False))


# --------------------------------------------------------------------------- #
# Live features: unplayed path == played path (reads the raw cache only)
# --------------------------------------------------------------------------- #
class TestLiveFeatures(unittest.TestCase):
    def test_masked_week_equivalence_2026_week2(self):
        from src import live_features as LF
        schedules, pbp = LF.load_raw(2026)
        frame, _ = LF.build_live_frame(schedules, pbp, 2026)
        self.assertEqual(LF.masked_week_equivalence(schedules, pbp, frame, 2026, 2), 16)


if __name__ == "__main__":
    prog = unittest.main(exit=False, verbosity=2)
    sys.exit(0 if prog.result.wasSuccessful() else 1)
