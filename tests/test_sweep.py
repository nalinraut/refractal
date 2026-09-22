"""Assembling a curve from several comparisons.

A sweep is six treatments over one base, joined rather than sharing an
identity. Tested on synthetic rows: the question here is what the assembler
refuses and how it places a point, not whether a simulator works.
"""

import unittest

from refractal.sweep import Sweep, SweepError, assemble


def row(level, checkpoint, success, *, base="sha256:b0", fired=True, infra=False):
    return {
        "base_scenario_hash": base,
        "checkpoint_id": checkpoint,
        "success": success,
        "is_infra_failure": infra,
        "perturbation_level": level,
        "perturbation_count": 0 if level is None else 1,
        "perturbations_fired": None if level is None else [{
            "effect": "scale_actuator", "target": "g", "specified_step": 0,
            "fired_step": 0 if fired else None,
            "reason": None if fired else "ended early",
            "before": [20.0], "after": [20.0 * level],
        }],
    }


def comparison(plan_id, level, *, a_wins, b_wins, n=5, **kw):
    rows = []
    for i in range(n):
        rows.append(row(level, "pi0", i < a_wins, **kw))
        rows.append(row(level, "pi05", i < b_wins, **kw))
    return (plan_id, rows)


CKPT = ["pi0", "pi05"]


class TestItRefusesWhatCannotBeACurve(unittest.TestCase):
    def test_plans_that_do_not_share_a_base(self):
        """Six runs over DIFFERENT scenarios plotted on one axis look exactly
        like a result. compare refuses a mismatched scene_hash for the same
        reason; nothing else checks this one."""
        good = comparison("sha256:p1", 1.0, a_wins=5, b_wins=5)
        other = ("sha256:p2", [row(0.1, "pi0", True, base="sha256:ELSEWHERE")])
        with self.assertRaises(SweepError) as ctx:
            assemble([good, other], checkpoints=CKPT)
        self.assertIn("do not share a base", str(ctx.exception))

    def test_a_comparison_holding_two_levels_is_not_a_point(self):
        mixed = ("sha256:p1", [row(1.0, "pi0", True), row(0.1, "pi0", True)])
        with self.assertRaises(SweepError) as ctx:
            assemble([mixed], checkpoints=CKPT)
        self.assertIn("is not a point", str(ctx.exception))

    def test_an_empty_comparison(self):
        with self.assertRaises(SweepError):
            assemble([("sha256:p1", [])], checkpoints=CKPT)


class TestWhatLandsOnTheCurve(unittest.TestCase):
    def test_points_are_ordered_high_to_low(self):
        sweep = assemble([
            comparison("sha256:p2", 0.1, a_wins=1, b_wins=2),
            comparison("sha256:p1", 1.0, a_wins=5, b_wins=5),
        ], checkpoints=CKPT)
        self.assertEqual([p.level for p in sweep.points], [1.0, 0.1])

    def test_an_episode_that_never_experienced_its_level_is_excluded(self):
        """Not merely mislabelled -- it succeeded unperturbed, and counting it
        at the assigned level credits robustness never demonstrated."""
        sweep = assemble([
            comparison("sha256:p1", 0.1, a_wins=5, b_wins=5, fired=False),
        ], checkpoints=CKPT)
        point = sweep.points[0]
        self.assertEqual(point.episodes, 0)
        self.assertEqual(point.excluded_unfired, 10)
        self.assertIsNone(point.rate("pi0"))
        self.assertTrue(any("never experienced" in n for n in sweep.notes))

    def test_infra_failures_are_not_policy_failures(self):
        sweep = assemble([
            comparison("sha256:p1", 1.0, a_wins=0, b_wins=0, infra=True),
        ], checkpoints=CKPT)
        self.assertEqual(sweep.points[0].episodes, 0)


class TestTheBandIsTheFinding(unittest.TestCase):
    """Reported before any verdict. If every level gives the same answer there
    is no band, and that is the result rather than a failed run."""

    def test_the_band_spans_levels_where_behaviour_moves(self):
        sweep = assemble([
            comparison("sha256:p1", 1.0, a_wins=5, b_wins=5),
            comparison("sha256:p2", 0.1, a_wins=3, b_wins=4),
            comparison("sha256:p3", 0.01, a_wins=1, b_wins=0),
            comparison("sha256:p4", 0.001, a_wins=0, b_wins=0),
        ], checkpoints=CKPT)
        self.assertEqual(sweep.band, (0.01, 0.1))
        self.assertIsNone(sweep.degenerate)

    def test_all_at_ceiling_is_a_finding_not_a_band(self):
        sweep = assemble([
            comparison("sha256:p1", 1.0, a_wins=5, b_wins=5),
            comparison("sha256:p2", 0.5, a_wins=5, b_wins=5),
        ], checkpoints=CKPT)
        self.assertEqual(sweep.band, (None, None))
        self.assertIn("above where it matters", sweep.degenerate)

    def test_all_at_floor_measures_the_objects_weight(self):
        sweep = assemble([
            comparison("sha256:p1", 0.001, a_wins=0, b_wins=0),
            comparison("sha256:p2", 0.0005, a_wins=0, b_wins=0),
        ], checkpoints=CKPT)
        self.assertEqual(sweep.band, (None, None))
        self.assertIn("impossible", sweep.degenerate)

    def test_a_sweep_where_nothing_fired_says_so(self):
        sweep = assemble([
            comparison("sha256:p1", 0.1, a_wins=5, b_wins=5, fired=False),
        ], checkpoints=CKPT)
        self.assertIn("not a single episode", sweep.degenerate.replace(
            "no level had a single episode", "not a single episode"))


if __name__ == "__main__":
    unittest.main()
