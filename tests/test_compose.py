"""The producer: a verdict's rows read distributionally.

Everything the distributional layer builds was reachable only from tests
before this -- analysis_hash, the grain on every line, the per-arm scalars
with their parameters, the install provenance. A report type with no producer
is the same defect as a check that does not print when it finds nothing.
"""

import unittest

from refractal.compare import build_units, evaluate
from refractal.compare.compose import compose_report
from refractal.compare.verdict import ArmContext


def rows(*, with_steps=True, scenes=3, seeds=3, budget=300,
         bases=(("baseline", 60), ("candidate", 110)), infra_in=None,
         candidate_fails_scene=None):
    out = []
    for arm, base in bases:
        for s in range(scenes):
            for seed in range(seeds):
                if candidate_fails_scene is not None:
                    ok = not (arm == "candidate" and s == candidate_fails_scene)
                else:
                    ok = not (s == scenes - 1 and seed == 0)
                row = {
                    "scene_id": f"sc{s}", "scene_hash": "h", "task_id": "t",
                    "task_hash": "th", "scenario_hash": f"sh{s}",
                    "checkpoint_id": arm, "seed": seed, "session_id": "x",
                    "harness_version": "v", "harness_surface": "v",
                    "is_infra_failure": bool(infra_in and arm == infra_in and s == 0),
                    "success": ok, "termination_map_digest": "sha256:a",
                }
                if with_steps:
                    row.update({
                        "outcome": "SUCCESS" if ok else "BUDGET_EXHAUSTED",
                        "step_to_success": base + s * 4 + seed if ok else None,
                        "step_budget": budget,
                    })
                out.append(row)
    return out


def report(data, **kw):
    e = build_units(data, checkpoints=["baseline", "candidate"])
    v = evaluate(e, ["baseline", "candidate"], resamples=200, seed=1)
    return compose_report(v, e, plan_id="sha256:abc", tier="full", **kw)


class TheNumbersArriveWithWhatTheyNeed(unittest.TestCase):
    def test_the_medians_separate_where_the_rate_cannot(self):
        """The case the whole phase exists for: identical success sets, and
        one arm takes 50 steps longer per episode."""
        r = report(rows())
        by_arm = {a.arm: {s.name: s.value for s in a.scalars} for a in r.arms}
        self.assertAlmostEqual(by_arm["baseline"]["success@budget"],
                               by_arm["candidate"]["success@budget"])
        self.assertLess(by_arm["baseline"]["median time-to-success"],
                        by_arm["candidate"]["median time-to-success"] - 40)

    def test_every_scalar_states_how_many_cells_it_averaged(self):
        """A median over 3 of 8 cells is a different claim from one over 8,
        and the macro-average hides that unless it is printed."""
        r = report(rows())
        for arm in r.arms:
            for scalar in arm.scalars:
                self.assertIn("cells", dict(scalar.parameters))

    def test_the_restricted_mean_refuses_rather_than_defaulting(self):
        """No penalty was declared, so there is no measurement to fall back
        on. The scalar is undefined and says why."""
        r = report(rows())
        rmst = [s for s in r.arms[0].scalars if s.name == "restricted mean"][0]
        self.assertIsNone(rmst.value)
        self.assertIn("ghost_penalty", rmst.undefined_reason)
        self.assertIn("convention", rmst.undefined_reason)


class TheGateAndTheScreenAreNotConflated(unittest.TestCase):
    def test_an_adjusted_p_is_never_below_the_p_it_adjusts(self):
        """Regression, found on a real run rather than here.

        `adjusted_p` belongs to the bootstrap family; pairing it with
        McNemar's raw p printed `p=1 holm=0.2136`. Holm can only raise a
        p-value, so an adjustment below its own p is arithmetically impossible
        -- and it read as the tighter of the two results.
        """
        # The fixture matters as much as the assertion. On a fixture where
        # both values are 1.0 this passes under the bug as well as without
        # it -- measured, after writing exactly that test first. Six scenes
        # with the candidate failing one of them outright gives McNemar
        # 1.0000 against an adjusted 0.0299, which is the real run's shape
        # and the documented power gap: a uniform shift is what McNemar is
        # weakest against.
        r = report(rows(scenes=6, candidate_fails_scene=0))
        pairs = [(line.p_value, line.adjusted_p)
                 for _, lines in r.pairwise for line in lines
                 if line.adjusted_p is not None]
        self.assertTrue(pairs)
        for p_value, adjusted in pairs:
            self.assertGreaterEqual(adjusted, p_value)

        # And the two statistics genuinely disagree on the scene that moved,
        # so a mispairing is caught rather than hidden by a tie. Keyed by
        # name alone this collapsed across scenes and kept the last one,
        # where both are 1.0 -- the assertion then failed on correct code.
        moved = {line.name: line.p_value
                 for _, lines in r.pairwise for line in lines
                 if dict(line.detail).get("scene") == "sc0"}
        self.assertNotAlmostEqual(moved["McNemar (exact)"],
                                  moved["clustered bootstrap (gates)"])

    def test_mcnemar_carries_no_adjusted_value_at_all(self):
        """It is reported and never gates, so it is not in the family. A
        number there would imply it had been corrected with the others."""
        r = report(rows())
        lines = {line.name: line for _, ls in r.pairwise for line in ls}
        self.assertIsNone(lines["McNemar (exact)"].adjusted_p)
        self.assertIsNotNone(lines["clustered bootstrap (gates)"].adjusted_p)

    def test_every_line_names_its_scene(self):
        """Eight scenes printed eight identical `baseline vs candidate`
        blocks before this: a reader could not tell which line was which, or
        that they were different lines."""
        r = report(rows(scenes=4))
        seen = [dict(l.detail).get("scene") for _, ls in r.pairwise for l in ls]
        self.assertEqual(len(set(seen)), 4)

    def test_grain_is_on_every_line(self):
        r = report(rows())
        for _, lines in r.pairwise:
            for line in lines:
                self.assertTrue(line.grain)


class WhatItDeclinesRatherThanOmits(unittest.TestCase):
    def test_pre_v2_rows_decline_instead_of_reporting_zeros(self):
        """Rows with no step data cannot produce a distribution, and the
        scalars say that rather than reading as measurements."""
        r = report(rows(with_steps=False))
        self.assertTrue(any("step_to_success" in d for d in r.declined))
        for arm in r.arms:
            for scalar in arm.scalars:
                self.assertIsNone(scalar.value)
                self.assertIsNotNone(scalar.undefined_reason)

    def test_the_per_phase_gap_is_named_not_hidden(self):
        """Built and unreachable is worse than absent, because the output
        gives a reader no way to know it exists."""
        r = report(rows())
        self.assertTrue(any("per-phase" in d for d in r.declined))

    def test_apparatus_is_per_arm_in_the_report(self):
        r = report(rows(infra_in="candidate"))
        by_arm = {a.arm: (a.apparatus, a.episodes) for a in r.arms}
        self.assertEqual(by_arm["baseline"][0], 0)
        self.assertGreater(by_arm["candidate"][0], 0)

    def test_installs_unchecked_is_not_installs_clean(self):
        unchecked = report(rows())
        self.assertIn("not checked", unchecked.reproducible.render())
        flagged = report(rows(), context={
            a: ArmContext(editable=("libero",), installs_checked=True)
            for a in ("baseline", "candidate")})
        self.assertIn("EDITABLE", flagged.reproducible.render())

    def test_the_analysis_is_identified(self):
        """Two readings of one plan under different choices are not
        comparable, so the choices get an id of their own."""
        r = report(rows())
        self.assertTrue(r.analysis_hash.startswith("sha256:"))
        self.assertNotEqual(r.analysis_hash, r.plan_id)


class TheDeclineMarkersStayInSync(unittest.TestCase):
    def test_a_reworded_decline_would_be_caught(self):
        """`compose_report` selects declines by phrase, because by the time
        they reach it they are strings. This pins the phrases so a rewording
        in `evaluate` fails here rather than silently emptying the section.
        """
        import re

        from refractal.compare.compose import _DECLINE_MARKERS

        source = (__import__("pathlib").Path("src/refractal/compare/verdict.py")
                  .read_text())
        for marker in _DECLINE_MARKERS:
            self.assertIn(marker, source,
                          f"compose selects on {marker!r} and verdict no longer "
                          "produces it")


if __name__ == "__main__":
    unittest.main()
