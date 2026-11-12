"""Tests for scripts/export_site_data.py (the portfolio-site JSON export).

Read-only against the project's data: the export under test is the committed
exports/site_data.json, compared against the verified sources it must agree
with. Builds for the determinism / discrepancy tests go to temporary files.
Nothing here runs a model, predicts, grades, passes --live, or writes under
data/, docs/ or exports/.

Usage::

    python -m scripts.test_export_site_data
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import export_site_data as E
from scripts.live import common as C
from scripts.live import scorecard as SC

MODELS = ("LR-untuned", "LR-tuned", "RandomForest", "HGB-tuned")


def _load() -> dict:
    return json.loads(E.OUT_PATH.read_text())


def _r4(x):
    return None if x is None else round(float(x), 4)


class HeadlineMatchesAuditMetrics(unittest.TestCase):
    """Held-out headline numbers == audit/out/metrics.json."""

    @classmethod
    def setUpClass(cls):
        cls.d = _load()
        cls.m = json.loads(E.METRICS_PATH.read_text())

    def _check(self, block: dict, ref: dict):
        self.assertEqual(block["n"], ref["n"])
        for m in MODELS:
            a = block["models"][m]["accuracy"]
            self.assertEqual((a["k"], a["n"]), (ref[m]["k"], ref[m]["n"]), m)
            self.assertEqual(_r4(a["value"]), ref[m]["acc"], m)
            self.assertEqual([_r4(x) for x in a["ci95"]], ref[m]["wilson95"], m)
            self.assertEqual(_r4(block["models"][m]["log_loss"]), ref[m]["log_loss"], m)
            self.assertEqual(_r4(block["models"][m]["brier"]), ref[m]["brier"], m)
        for key, rk in (("home_baseline", "home_baseline"), ("vegas_favorite", "vegas_favorite")):
            a = block[key]["accuracy"]
            self.assertEqual((a["k"], a["n"], _r4(a["value"])), (ref[rk]["k"], ref[rk]["n"], ref[rk]["acc"]), key)
            self.assertEqual([_r4(x) for x in a["ci95"]], ref[rk]["wilson95"], key)
        vi = ref["vegas_novig_implied"]
        self.assertEqual(_r4(block["vegas_implied"]["log_loss"]), vi["log_loss"])
        self.assertEqual(_r4(block["vegas_implied"]["brier"]), vi["brier"])
        b, rb = block["retroactive_bets"], ref["bets_LR-tuned_RETROACTIVE"]
        self.assertTrue(b["retroactive"])
        self.assertEqual((b["bets"], b["wins"], b["losses"], b["voids"]), (rb["n_bets"], rb["W"], rb["L"], rb["P"]))
        self.assertEqual(round(b["units"], 2), rb["units"])
        self.assertEqual(_r4(b["roi"]), rb["roi"])

    def test_2023_24_pooled(self):
        self._check(self.d["held_out_2023_24"], self.m["heldout_2023_2024"]["pooled"])

    def test_2023_24_per_season(self):
        for s in ("2023", "2024"):
            with self.subTest(season=s):
                self._check(self.d["held_out_2023_24"]["per_season"][s], self.m["heldout_2023_2024"]["per_season"][s])

    def test_2025(self):
        self._check(self.d["prereg_2025"], self.m["holdout_2025"]["per_season"]["2025"])

    def test_headline_values(self):
        # the site's headline claims, spelled out
        h = self.d["held_out_2023_24"]
        self.assertEqual(h["selected_model"], "LR-tuned")
        self.assertEqual((h["models"]["LR-tuned"]["accuracy"]["k"], h["n"]), (350, 544))
        self.assertEqual(self.d["prereg_2025"]["models"]["LR-tuned"]["accuracy"]["k"], 167)
        self.assertIn("AFTER these 2023-24 results were known", h["selection_note"])
        self.assertIn("statistically tied", h["selection_note"])
        self.assertTrue(all(r["p"] >= 0.05 for r in h["mcnemar"]))
        self.assertEqual(len(h["mcnemar"]), 6)

    def test_no_model_over_70pct(self):
        # CLAUDE.md / SPEC 5.6: any model accuracy > ~70% is a stop-and-flag, never a headline.
        for key in ("held_out_2023_24", "prereg_2025"):
            for m in MODELS:
                self.assertLess(self.d[key]["models"][m]["accuracy"]["value"], 0.70, (key, m))


class LiveMatchesScorecard(unittest.TestCase):
    """Live cumulative numbers == docs/live/SCORECARD_2026.md (parsed independently here)."""

    @classmethod
    def setUpClass(cls):
        cls.d = _load()["live_2026"]
        cls.md = E.SCORECARD_PATH.read_text()

    def _row(self, label: str) -> list[str]:
        for line in self.md.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] == label:
                return cells
        self.fail(f"row {label!r} not in scorecard")

    def _close(self, shown: str, exported: float, msg):
        """The scorecard rounds full precision to its displayed decimals; the JSON holds 4 dp.
        Equal = within half a unit of the scorecard's last digit plus half a unit of 4 dp."""
        decimals = len(shown.split(".")[1]) if "." in shown else 0
        self.assertLessEqual(abs(float(shown) - exported), 0.5 * 10 ** -decimals + 0.5e-4 + 1e-12, msg)

    def _nums(self, cell: str) -> list[str]:
        return re.findall(r"[+-]?\d+\.\d+", cell)

    def test_prediction_quality(self):
        labels = {"champion": "Champion", "challenger": "Challenger", "home_baseline": "Home baseline",
                  "vegas_favorite": "Vegas favorite (prediction snapshot)"}
        for key, label in labels.items():
            a = self.d["prediction_quality"][key]["accuracy"]
            cell = self._row(label)[2]
            self.assertTrue(cell.startswith(f"{a['k']}/{a['n']} = "), (key, cell))
            for shown, mine in zip(self._nums(cell), [a["value"], *a["ci95"]]):
                self._close(shown, mine, (key, "accuracy"))
        for key, label in (("champion", "Champion"), ("challenger", "Challenger"),
                           ("vegas_implied", "Vegas implied (prediction snapshot)")):
            q = self.d["prediction_quality"][key]
            for col, met in ((3, "log_loss"), (4, "brier")):
                shown = self._nums(self._row(label)[col])
                self.assertEqual(len(shown), 3, (key, met))
                for s, mine in zip(shown, [q[met]["mean"], *q[met]["ci95"]]):
                    self._close(s, mine, (key, met))

    def test_official_bets(self):
        b = self.d["official_bets"]
        cells = self._row("Official record (bet_status = official)")
        self.assertEqual(cells[1], f"{b['bets']} ({b['home_bets']})")
        self.assertEqual(cells[2], f"{b['wins']}-{b['losses']}-{b['voids']} ({b['pending']})")
        self._close(cells[3], b["units"], "units")
        for s, mine in zip(self._nums(cells[4]), [b["roi"], *b["roi_ci95"]]):
            self._close(s, mine, "roi")
        c = b["clv"]
        self.assertIn(f"(n={c['n']}, excl {c['n_excluded']})", cells[5])
        for s, mine in zip(self._nums(cells[5]), [c["mean"], *c["ci95"]]):
            self._close(s, mine, "clv")
        self._close(cells[6], c["p_one_sided"], "clv p")

    def test_clv_verdict_verbatim(self):
        v = self.d["clv_verdict"]
        self.assertIn("\n" + v["scorecard_line"] + "\n", self.md)
        m = re.search(r"Official bets with measured CLV: n = (\d+); mean CLV = (-?\d\.\d{4}) "
                      r"\[(-?\d\.\d{4}), (-?\d\.\d{4})\]; one-sided p = (\d\.\d{4})", self.md)
        self.assertEqual(m.groups(), (str(v["n"]), f"{v['mean']:.4f}", f"{v['ci95'][0]:.4f}",
                                      f"{v['ci95'][1]:.4f}", f"{v['p_one_sided']:.4f}"))

    def test_checkpoint(self):
        cp = self.d["checkpoint"]
        self.assertIn(f"**Decision: {cp['decision']}.**", self.md)
        m = re.search(r"champion (\d\.\d+), challenger (\d\.\d+); mean difference \(challenger - champion\) "
                      r"= ([+-]\d\.\d+)", self.md)
        self.assertEqual([round(float(x), 4) for x in m.groups()],
                         [cp["champion_mean_log_loss"], cp["challenger_mean_log_loss"],
                          cp["mean_diff_challenger_minus_champion"]])
        self.assertIn(f"one-sided paired t p = {cp['p_one_sided']:.4f}", self.md)
        self.assertIn(f"paired games actually graded for both models: n = {cp['paired_games']}.", self.md)
        self.assertEqual(cp["governing_amendment"]["id"], "Amendment 4")

    def test_games_logged_line(self):
        self.assertIn(f"Games logged: {self.d['games_graded']} (decided {self.d['games_decided']}, "
                      f"tie {self.d['ties']}, pending 0).", self.md)


class WeeksComplete(unittest.TestCase):
    """Every scheduled 2026 week from the first official week to date is present; missed weeks explicit."""

    def test_every_week_present(self):
        d = _load()["live_2026"]
        sched = SC.load_schedule_kickoffs(E.LIVE_SEASON)
        as_of = C.snapshot_time(C.latest_snapshot(E.LIVE_SEASON))
        predicted = SC.load_predicted(E.LIVE_ROOT, E.LIVE_SEASON)
        last = max(max(predicted), int(sched.loc[sched["kickoff"] <= as_of, "week"].max()))
        got = [w["week"] for w in d["weeks"]]
        self.assertEqual(got, list(range(E.FIRST_OFFICIAL_WEEK, last + 1)))
        for w in d["weeks"]:
            self.assertIn(w["status"], {"graded", "partial", "missed", "pending"})
            self.assertEqual(w["scheduled"], int((sched["week"] == w["week"]).sum()), w["week"])
            self.assertEqual(w["predicted"], len(predicted.get(w["week"], ())), w["week"])
        by = {w["week"]: w for w in d["weeks"]}
        self.assertEqual(by[9]["status"], "missed")       # commit cdd5123
        self.assertEqual(by[9]["predicted"], 0)
        self.assertEqual(by[8]["status"], "partial")      # MNF only
        self.assertEqual(sum(w["graded"] for w in d["weeks"]), d["games_graded"])

    def test_week_status_rules(self):
        s = E.week_status
        self.assertEqual(s(15, 0, 0, 0, True), "missed")
        self.assertEqual(s(15, 0, 0, 0, False), "pending")
        self.assertEqual(s(14, 1, 1, 0, True), "partial")
        self.assertEqual(s(14, 14, 14, 0, True), "graded")
        self.assertEqual(s(14, 14, 0, 14, False), "pending")
        self.assertEqual(s(14, 14, 13, 1, True), "pending")


class GameInvariants(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.games = _load()["games"]

    def test_no_units_without_bet(self):
        bad = [g["game_id"] for g in self.games if g["bet_units"] is not None and g["bet"] is not True]
        self.assertEqual(bad, [])

    def test_bet_fields_consistent(self):
        for g in self.games:
            if g["bet"]:
                self.assertIn(g["bet_side"], ("home", "away"), g["game_id"])
                self.assertIsNotNone(g["bet_odds"], g["game_id"])
            else:
                self.assertIsNone(g["bet_side"], g["game_id"])
                self.assertIsNone(g["bet_odds"], g["game_id"])

    def test_counts_and_order(self):
        by = {}
        for g in self.games:
            by[g["evaluation"]] = by.get(g["evaluation"], 0) + 1
        self.assertEqual(by["held_out_2023_24"], 544)
        self.assertEqual(by["prereg_2025"], 271)
        keys = [(g["season"], g["week"], g["kickoff_utc"], g["game_id"]) for g in self.games]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len({g["game_id"] for g in self.games}), len(self.games))

    def test_live_provenance(self):
        for g in self.games:
            if g["evaluation"] == "live_2026":
                self.assertRegex(g["prediction_commit"], r"^[0-9a-f]{40}$")
                self.assertRegex(g["prediction_committed_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
                self.assertIn("not proof of push time", g["prediction_committed_at_kind"])

    def test_live_bets_sum_to_official_units(self):
        live = [g for g in self.games if g["evaluation"] == "live_2026" and g["bet_units"] is not None]
        self.assertAlmostEqual(sum(g["bet_units"] for g in live), _load()["live_2026"]["official_bets"]["units"], 2)


class DeterminismAndGuards(unittest.TestCase):

    def test_byte_identical_rebuild(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a.json", Path(t) / "b.json"
            self.assertEqual(E.main(["--out", str(a), "--generated-at", "2026-01-01T00:00:00Z"]), 0)
            self.assertEqual(E.main(["--out", str(b), "--generated-at", "2026-01-01T00:00:00Z"]), 0)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            # re-run onto an existing file with unchanged inputs keeps its generated_at -> same bytes
            before = a.read_bytes()
            self.assertEqual(E.main(["--out", str(a)]), 0)
            self.assertEqual(a.read_bytes(), before)

    def test_committed_export_is_current(self):
        # source_commit is excluded: committing the export itself moves HEAD.
        on_disk = _load()
        fresh = E.clean(E.build_payload(generated_at=on_disk["generated_at"]))
        for d in (on_disk, fresh):
            d.pop("source_commit")
        self.assertEqual(fresh, on_disk, "exports/site_data.json is stale: re-run the exporter")

    def test_discrepancy_writes_nothing(self):
        m = json.loads(E.METRICS_PATH.read_text())
        m["heldout_2023_2024"]["pooled"]["LR-tuned"]["k"] += 1
        with tempfile.TemporaryDirectory() as t:
            bad, out = Path(t) / "metrics.json", Path(t) / "out.json"
            bad.write_text(json.dumps(m))
            with mock.patch.object(E, "METRICS_PATH", bad), mock.patch("sys.stderr"):
                self.assertEqual(E.main(["--out", str(out)]), 2)
            self.assertFalse(out.exists())

    def test_schema_documents_every_field(self):
        schema = (E.EXPORT_DIR / "SCHEMA.md").read_text()
        d = _load()
        keys = set(d) | set(d["games"][0]) | {k for g in d["games"] for k in g} | set(d["live_2026"]) \
            | set(d["live_2026"]["weeks"][0]) | set(d["held_out_2023_24"]) | set(d["definitions"])
        missing = sorted(k for k in keys if f"`{k}`" not in schema)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    prog = unittest.main(exit=False, verbosity=2)
    sys.exit(0 if prog.result.wasSuccessful() else 1)
