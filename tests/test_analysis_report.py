"""Steps 5 and 6: the declared family, analysis identity, and the report."""

import unittest

from refractal.compare.analysis import AnalysisSpec, Family, FamilyError
from refractal.compare.report import ArmReport, ComparisonReport, Scalar, TestLine


class TestFamily(unittest.TestCase):
    def test_m_is_the_pairwise_count(self):
        self.assertEqual(Family(arms=("A", "B")).m, 1)
        self.assertEqual(Family(arms=("A", "B", "C")).m, 3)
        self.assertEqual(Family(arms=tuple("ABCDEF")).m, 15)

    def test_a_duplicate_arm_inflates_m_until_pruned(self):
        # Five distinct arms deserve m = 10. Six with one duplicate give 15,
        # so the smallest p faces alpha/15 rather than alpha/10 -- a 50%
        # tighter bar on the strongest result, for nothing.
        f = Family(arms=tuple("ABCDEF"))
        self.assertEqual(f.m, 15)
        notes = f.prune_duplicates([("A", "F")])
        self.assertEqual(f.m, 10)
        self.assertIn("m reduced 15 -> 10", notes[0])

    def test_adding_after_sealing_is_refused(self):
        f = Family(arms=("A", "B"))
        f.seal()
        with self.assertRaises(FamilyError) as c:
            f.add("A", "C")
        self.assertIn("already run against m", str(c.exception))

    def test_the_sweep_and_stability_cannot_enter_the_family(self):
        # Structural: the family is built from the arm list, so anything that
        # is not a contrast between two declared arms has no way in.
        f = Family(arms=("A", "B", "C"))
        self.assertTrue(all(len(c) == 2 for c in f.contrasts))
        self.assertTrue(all(set(c) <= {"A", "B", "C"} for c in f.contrasts))


class TestAnalysisIdentity(unittest.TestCase):
    def base(self, **kw):
        return AnalysisSpec(dichotomy="majority", budget=300, tau=300,
                            ghost_penalty=300.0, family=(("A", "B"),), **kw)

    def test_every_compare_time_choice_moves_the_hash(self):
        # plan_id says what ran; this says how it was read. Each of these
        # changes the conclusion from identical episodes.
        base = self.base()
        for field, value in (("dichotomy", "all"), ("budget", 200),
                             ("tau", 250), ("ghost_penalty", 250.0),
                             ("alpha", 0.01),
                             ("permutation_statistic", "rmst"),
                             ("permutation_seed", 9)):
            other = AnalysisSpec(**{**base.__dict__, field: value})
            self.assertNotEqual(base.analysis_hash(), other.analysis_hash(),
                                f"{field} must move analysis_hash")

    def test_the_family_size_is_inside_it(self):
        two = self.base()
        three = AnalysisSpec(**{**two.__dict__,
                                "family": (("A", "B"), ("A", "C"), ("B", "C"))})
        self.assertNotEqual(two.analysis_hash(), three.analysis_hash())

    def test_the_hash_is_order_independent(self):
        a = AnalysisSpec(family=(("A", "B"), ("C", "D")))
        b = AnalysisSpec(family=(("D", "C"), ("B", "A")))
        self.assertEqual(a.analysis_hash(), b.analysis_hash())


class TestReport(unittest.TestCase):
    def test_a_scalar_states_its_parameters_inline(self):
        s = Scalar("rmst", 169.2,
                   parameters=(("tau", "300"), ("ghost_penalty", "300.0")))
        self.assertIn("tau=300", s.render())
        self.assertIn("ghost_penalty=300.0", s.render())

    def test_an_undefined_median_says_why(self):
        s = Scalar("median", None, undefined_reason="fewer than half succeeded")
        self.assertIn("undefined", s.render())
        self.assertIn("fewer than half", s.render())
        self.assertNotIn("0", s.render().split("undefined")[0])

    def test_every_test_line_carries_its_grain(self):
        # The two tests aggregate differently, and both p-values are correct
        # for what they test. Without the grain a reader concludes the
        # permutation test is simply more powerful, when part of the gap is
        # that McNemar discarded five sixths of its input before looking.
        line = TestLine("mcnemar", 1.0, "scenario, seeds collapsed")
        self.assertIn("grain:", line.render())

    def test_an_adjusted_p_always_shows_m(self):
        line = TestLine("mcnemar", 0.04, "scenario", adjusted_p=0.04, m=15)
        self.assertIn("m=15", line.render(),
                      "0.04 at m=2 and 0.04 at m=15 are different claims")

    def test_arm_dependent_apparatus_is_flagged(self):
        rep = ComparisonReport(
            plan_id="p", analysis_hash="h", tier="full", m=1,
            arms=(ArmReport("A", {}, (), apparatus=1, episodes=100),
                  ArmReport("B", {}, (), apparatus=20, episodes=100)))
        self.assertTrue(rep.apparatus_differs())
        self.assertIn("different subsets", rep.render())

    def test_stability_records_what_it_swept(self):
        rep = ComparisonReport(
            plan_id="p", analysis_hash="h", tier="full", m=1, arms=(),
            stability="ranking invariant", stability_swept=(("tau", "100..300"),))
        self.assertIn("spans many analysis hashes", rep.render())


if __name__ == "__main__":
    unittest.main()


class TestAuthorshipAndReproducibility(unittest.TestCase):
    """Two fields because they fail independently.

    A catalog run missing engine_version is unreproducible though every line
    of it was written here. A customer adapter with complete provenance
    re-runs exactly while remaining conditional on code nobody here read.
    One enum over both would lose the distinction.
    """

    def test_they_are_independent(self):
        from refractal.compare.report import Authorship, Reproducible

        catalog_unreproducible = (Authorship(), Reproducible(False, ("no engine_version",)))
        customer_reproducible = (Authorship(adapter="customer"), Reproducible(True))
        self.assertTrue(catalog_unreproducible[0].all_catalog)
        self.assertFalse(catalog_unreproducible[1].complete)
        self.assertFalse(customer_reproducible[0].all_catalog)
        self.assertTrue(customer_reproducible[1].complete)

    def test_a_customer_arm_is_still_internally_valid(self):
        from refractal.compare.report import Authorship

        said = Authorship(scene="customer").render()
        self.assertIn("internally valid", said)
        self.assertIn("same apparatus", said)
        self.assertIn("not comparable against the catalog", said)
        self.assertIn("scene", said.split("customer-supplied")[0].split("catalog:")[-1])

    def test_incomplete_names_what_is_missing(self):
        from refractal.compare.report import Reproducible

        self.assertIn("engine_version",
                      Reproducible(False, ("scene s has no engine_version",)).render())


class TestDeclinedAlwaysPrints(unittest.TestCase):
    def test_an_empty_declined_section_still_appears(self):
        # A section that vanishes when empty leaves a reader unable to tell a
        # check that ran and passed from one that never ran -- which is the
        # reason the variance detector's declines were surfaced at all.
        rep = ComparisonReport(plan_id="p", analysis_hash="h", tier="full", m=1)
        self.assertIn("declined: none", rep.render())

    def test_a_populated_one_lists_them(self):
        rep = ComparisonReport(plan_id="p", analysis_hash="h", tier="full", m=1,
                               declined=("the identical-checkpoints check did not run",))
        self.assertIn("identical-checkpoints", rep.render())
        self.assertNotIn("declined: none", rep.render())
