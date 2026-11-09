"""Phase 5: per-phase conditional rates, and what chaining costs."""

import hashlib
import struct
import unittest

from refractal.compare.phases import (
    PhaseError,
    PhaseRate,
    compounding,
    conditional_rates,
    read_compounding,
    require_canonical_states,
)
from refractal.compare.report import ComparisonReport


def _u(i, k):
    d = hashlib.sha256(f"{i}|{k}".encode()).digest()
    return struct.unpack(">Q", d[:8])[0] / float(1 << 64)


def chained_arm(per_phase, n=20000):
    """Episodes where a phase is attempted only if every earlier one passed."""
    eps = []
    for i in range(n):
        recs, alive = [], True
        for k, p in enumerate(per_phase):
            ok = alive and _u(i, k) < p
            recs.append({"phase_id": f"p{k+1}", "attempted": alive,
                         "outcome": "SUCCESS" if ok
                                    else (None if not alive else "BUDGET_EXHAUSTED")})
            alive = ok
        eps.append({"phase_records": recs})
    return eps


def episode_rate(eps):
    return sum(1 for e in eps
               if all(r["outcome"] == "SUCCESS" for r in e["phase_records"])) / len(eps)


class TestTheCaseForPerPhaseRecords(unittest.TestCase):
    """Two policies, one episode rate, opposite problems."""

    def test_identical_at_episode_level_and_nothing_alike_per_phase(self):
        a = chained_arm([0.80, 0.80, 0.80, 0.80])
        b = chained_arm([1.0, 1.0, 1.0, 0.41])
        self.assertAlmostEqual(episode_rate(a), episode_rate(b), places=2)

        ra = {r.phase_id: r.rate for r in conditional_rates(a)}
        rb = {r.phase_id: r.rate for r in conditional_rates(b)}
        # A is uniformly mediocre; B is perfect until one skill it lacks.
        self.assertTrue(all(0.75 < v < 0.85 for v in ra.values()))
        self.assertTrue(all(v > 0.98 for k, v in rb.items() if k != "p4"))
        self.assertLess(rb["p4"], 0.45)

    def test_the_censoring_is_visible_in_the_denominator(self):
        # A phase after a failure is never attempted, and the reason it is
        # missing is the policy's own failure. The shrinking n is that fact.
        a = conditional_rates(chained_arm([0.80, 0.80, 0.80, 0.80]))
        b = conditional_rates(chained_arm([1.0, 1.0, 1.0, 0.41]))
        self.assertEqual(b[3].attempted, 20000, "B reaches the last phase always")
        self.assertLess(a[3].attempted, 11000, "A has lost half its n by then")

    def test_rates_are_conditional_not_marginal(self):
        eps = [{"phase_records": [
            {"phase_id": "p1", "attempted": True, "outcome": "SUCCESS"},
            {"phase_id": "p2", "attempted": i < 8,
             "outcome": "SUCCESS" if i < 6 else ("BUDGET_EXHAUSTED" if i < 8 else None)},
        ]} for i in range(10)]
        p2 = conditional_rates(eps)[1]
        self.assertEqual((p2.succeeded, p2.attempted), (6, 8))
        self.assertAlmostEqual(p2.rate, 0.75,
                               msg="6/8 attempted, not the marginal 6/10")


class TestPairingOnPhaseId(unittest.TestCase):
    def test_phases_are_keyed_by_id_not_position(self):
        # Under a free order "phase 2" is a different phase in each arm.
        # Linear phases make this free to get right now.
        swapped = [{"phase_records": [
            {"phase_id": "grasp", "attempted": True, "outcome": "SUCCESS"},
            {"phase_id": "reach", "attempted": True, "outcome": "SUCCESS"},
        ]}]
        rates = {r.phase_id: r.rate for r in conditional_rates(swapped)}
        self.assertEqual(set(rates), {"reach", "grasp"})


class TestCompounding(unittest.TestCase):
    def iso(self):
        return (PhaseRate("p1", 100, 90), PhaseRate("p2", 100, 90),
                PhaseRate("p3", 100, 90))

    def test_positive_means_chaining_costs(self):
        self.assertAlmostEqual(compounding(self.iso(), 0.600), 0.129, places=3)

    def test_near_zero_means_the_phases_are_independent(self):
        self.assertAlmostEqual(compounding(self.iso(), 0.729), 0.0, places=6)

    def test_negative_is_surfaced_not_clamped(self):
        # The policy's own end state suits it better than the canonical one,
        # which says the canonical state is not neutral.
        self.assertLess(compounding(self.iso(), 0.80), 0)

    def test_a_phase_with_no_isolated_rate_refuses(self):
        with self.assertRaises(PhaseError) as c:
            compounding(self.iso() + (PhaseRate("p4", 0, 0),), 0.5)
        self.assertIn("product over a different task", str(c.exception))


class TestIsolatedModeNeedsCanonicalStates(unittest.TestCase):
    def test_it_refuses_rather_than_synthesising(self):
        with self.assertRaises(PhaseError) as c:
            require_canonical_states("scene-1", ["reach", "grasp"],
                                     {("scene-1", "reach"): object()})
        self.assertIn("not synthesised", str(c.exception))
        self.assertIn("grasp", str(c.exception))

    def test_complete_states_pass(self):
        require_canonical_states("scene-1", ["a"], {("scene-1", "a"): object()})

    def test_a_state_is_not_inherited_from_another_scene(self):
        """Per (scene, phase), and this is the assertion rather than the
        assumption.

        One canonical state shared across scenes erases scene difficulty in
        the isolated condition while leaving it in the chained one, so
        compounding absorbs the difference and reports a fact about the
        scenes as though it were about the policy.
        """
        with self.assertRaises(PhaseError) as c:
            require_canonical_states("scene-2", ["reach"],
                                     {("scene-1", "reach"): object()})
        self.assertIn("scene-2", str(c.exception))
        self.assertIn("not inherited across scenes", str(c.exception))


class CrossArmReadingTests(unittest.TestCase):
    """The sign is a fact about an arm; the PATTERN is a fact about the scene."""

    def test_one_arm_negative_is_about_that_arm(self):
        r = read_compounding("map_1", {"idm": 0.12, "aggressive": -0.04},
                             method="replay")
        self.assertFalse(r.suspects_catalog)
        self.assertEqual(r.negative_arms, ("aggressive",))
        self.assertEqual(r.as_finding().about, "arm")

    def test_most_arms_negative_is_about_the_catalog(self):
        """The reading that would otherwise be left to a reader to notice.

        Three arms each slightly negative on the same scene looks, arm by
        arm, like three unremarkable results. It is the agreement that is the
        finding: a canonical state no policy's own end state is worse than.
        """
        r = read_compounding("map_3", {"idm": -0.08, "aggressive": -0.11,
                                       "timid": -0.05}, method="hand")
        self.assertTrue(r.suspects_catalog)
        self.assertEqual(r.as_finding().about, "catalog")
        self.assertIn("mis-authored", r.render())

    def test_a_minority_negative_is_not_a_catalog_finding(self):
        """Strictly more than half, so a tie is not promoted.

        Two of four arms negative is the knife edge, and on a scene where the
        policies genuinely differ it is the expected result rather than
        evidence about the scene.
        """
        r = read_compounding("map_2", {"a": -0.02, "b": -0.03, "c": 0.05,
                                       "d": 0.07})
        self.assertEqual(len(r.negative_arms), 2)
        self.assertFalse(r.suspects_catalog)

    def test_a_single_arm_cannot_implicate_the_catalog(self):
        """One arm is one arm, however negative. There is no pattern to read."""
        r = read_compounding("map_5", {"only": -0.40})
        self.assertFalse(r.suspects_catalog)
        self.assertEqual(r.as_finding().about, "arm")

    def test_the_number_says_what_it_is_measured_against(self):
        """And says so when it cannot.

        Compounding is a comparison with a canonical state. The same arm on
        the same scene gives a different number against a hand-authored state
        and against one replayed from a successful episode, so the method
        belongs with the number and not only in the scene hash.
        """
        self.assertIn("authored by solver",
                      read_compounding("s", {"a": 0.1, "b": 0.1},
                                       method="solver").render())
        unknown = read_compounding("s", {"a": 0.1, "b": 0.1}).render()
        self.assertIn("method is unrecorded", unknown)

    def test_a_catalog_finding_is_marked_in_the_report(self):
        report = ComparisonReport(
            plan_id="p", analysis_hash="h", tier="full", m=1,
            findings=(read_compounding("map_3", {"a": -0.1, "b": -0.2},
                                       method="hand").as_finding(),),
        )
        self.assertIn("CATALOG:", report.render())


if __name__ == "__main__":
    unittest.main()
