"""Phase 3: the inferential layer, and what it refuses to answer."""

import dataclasses
import unittest

from refractal.compare.distribution import estimate, median, success_at
from refractal.compare.pairing import EpisodeOutcome, Unit, UnitKey
from refractal.compare.permutation import (
    BudgetCurve,
    MIN_SCENES_FOR_ALPHA_05,
    PermutationError,
    PermutationResult,
    budget_curve,
    contingency_at,
    permutation_test,
    stability,
)
from refractal.compare.distribution import DistributionError
from refractal.compare.stats import mcnemar_exact
from refractal.schema.outcome import Outcome

S, B_ = Outcome.SUCCESS, Outcome.BUDGET_EXHAUSTED
BUDGET = 300


def tied(n_scenes=8, misses=2, reps=1):
    """Arms tied on success-at-budget, far apart in median."""
    units = []
    for i in range(n_scenes):
        off = i * 3

        def side(times):
            d = {}
            for j, t in enumerate(times):
                d[j] = EpisodeOutcome(outcome=S, step_to_success=t,
                                      step_budget=BUDGET)
            for k in range(misses):
                d[len(times) + k] = EpisodeOutcome(
                    outcome=B_, step_to_success=None, step_budget=BUDGET)
            return d

        units.append(Unit(
            key=UnitKey(scene_id=f"scene-{i}", task_id="t", task_hash="th",
                        scenario_hash=f"sh-{i}"),
            outcomes={"A": side([65 + off, 70 + off, 75 + off, 80 + off] * reps),
                      "B": side([115 + off, 120 + off, 125 + off, 130 + off] * reps)},
            seeds_kept=[], seeds_dropped=[]))
    return units


class TestTheArgumentForThisPhase(unittest.TestCase):
    """One fixture, two instruments, opposite answers.

    The arms succeed on exactly the same number of episodes within the
    budget, so every binary statistic is blind by construction rather than
    underpowered. No N fixes it. The median separates them at p < 0.01 on
    the same episodes.
    """

    def test_the_success_sets_are_identical_only_the_timing_differs(self):
        """The construction the argument depends on.

        If the arms succeeded on DIFFERENT seeds, the discordance would be
        nonzero by accident and "the gate sees nothing" would be luck rather
        than structure. They succeed on the same seeds, at different times.
        """
        cell = tied()[0]
        ok = {arm: sorted(k for k, o in cell.outcomes[arm].items() if o.succeeded)
              for arm in ("A", "B")}
        self.assertEqual(ok["A"], ok["B"])
        times = {arm: [cell.outcomes[arm][k].step_to_success for k in ok[arm]]
                 for arm in ("A", "B")}
        self.assertNotEqual(times["A"], times["B"])

    def test_the_binary_view_finds_nothing(self):
        u = tied()
        from refractal.compare.stats import contingency

        c = contingency(u, "A", "B", "majority")
        # A NON-DEGENERATE table. At 3 successes of 6 the majority rule is a
        # knife edge -- strictly greater-than, so a 50% cell fails for both
        # arms and every unit lands in both_fail, which is McNemar having
        # almost no data rather than McNemar looking and finding nothing.
        # Four of six puts every unit in both_pass, where zero discordance
        # means the success sets genuinely agree.
        self.assertEqual(c.both_pass, len(u))
        self.assertEqual((c.only_a_passes, c.only_b_passes), (0, 0))
        self.assertEqual(c.both_fail, 0)
        self.assertEqual(mcnemar_exact(c).p_value, 1.0)
        r = permutation_test(u, "A", "B", statistic="success_at",
                             ghost_penalty=float(BUDGET), b=BUDGET)
        self.assertEqual(r.p_value, 1.0)
        self.assertEqual(r.observed, 0.0, "tied exactly, not nearly")

    def test_the_distribution_separates_them(self):
        r = permutation_test(tied(), "A", "B", statistic="median",
                             ghost_penalty=float(BUDGET))
        self.assertLess(r.p_value, 0.01)
        self.assertAlmostEqual(r.observed, -50.0)
        self.assertEqual(r.path, "exact")


class TestTheSceneFloor(unittest.TestCase):
    def test_below_six_scenes_it_refuses(self):
        for n in (4, 5):
            with self.assertRaises(PermutationError) as c:
                permutation_test(tied(n), "A", "B", statistic="median",
                                 ghost_penalty=float(BUDGET))
            said = str(c.exception)
            self.assertIn(f"{n} scene(s)", said)
            self.assertIn("2**(1-", said, "names the floor arithmetic")
            self.assertIn("add scenes rather than episodes", said,
                          "says what would actually help")

    def test_six_scenes_is_the_first_that_works(self):
        r = permutation_test(tied(MIN_SCENES_FOR_ALPHA_05), "A", "B",
                             statistic="median", ghost_penalty=float(BUDGET))
        self.assertLess(r.p_value, 0.05)
        self.assertAlmostEqual(r.floor, 0.03125)

    def test_more_episodes_do_not_raise_the_floor(self):
        """The permutation unit is the scene, so seeds within one buy nothing.

        Ten times the episodes, same six scenes, same floor -- and the same
        p, because every added seed repeats a difference the scene already
        showed. This is the sentence the refusal above is trying to save
        someone from discovering the expensive way.
        """
        few = permutation_test(tied(6, misses=3, reps=1), "A", "B",
                               statistic="median", ghost_penalty=float(BUDGET))
        many = permutation_test(tied(6, misses=30, reps=10), "A", "B",
                                statistic="median", ghost_penalty=float(BUDGET))
        self.assertEqual(few.floor, many.floor)
        self.assertEqual(few.p_value, many.p_value)


class TestTheSampledPath(unittest.TestCase):
    def wide(self, n=22):
        return tied(n)

    def test_sampling_without_a_seed_is_refused(self):
        with self.assertRaises(PermutationError) as c:
            permutation_test(self.wide(), "A", "B", statistic="median",
                             ghost_penalty=float(BUDGET))
        self.assertIn("seed", str(c.exception))

    def test_the_path_and_its_parameters_are_recorded(self):
        r = permutation_test(self.wide(), "A", "B", statistic="median",
                             ghost_penalty=float(BUDGET), seed=7, samples=2000)
        self.assertEqual(r.path, "sampled")
        self.assertEqual(r.seed, 7)
        self.assertEqual(r.samples, 2000)

    def test_the_plus_one_correction_keeps_p_above_zero(self):
        # B draws bound the tail from above; they cannot establish zero.
        r = permutation_test(self.wide(), "A", "B", statistic="median",
                             ghost_penalty=float(BUDGET), seed=1, samples=500)
        self.assertGreater(r.p_value, 0.0)
        self.assertAlmostEqual(r.p_value, 1 / 501, places=6)


class TestBudgetParameterisedCallsRefuseOnOldRows(unittest.TestCase):
    def old_rows(self):
        seeds = {i: EpisodeOutcome(outcome=S if i < 2 else B_,
                                   step_to_success=None, step_budget=None)
                 for i in range(4)}
        return [Unit(key=UnitKey(scene_id="s", task_id="t", task_hash="th",
                                 scenario_hash="sh"),
                     outcomes={"A": seeds, "B": dict(seeds)},
                     seeds_kept=[], seeds_dropped=[])]

    def test_contingency_at_refuses(self):
        with self.assertRaises(PermutationError) as c:
            contingency_at(self.old_rows(), "A", "B", "majority", 100)
        self.assertIn("schema version 2", str(c.exception))


class TestTheDescriptiveTypesCarryNoPValue(unittest.TestCase):
    def test_the_budget_curve_has_no_field_for_one(self):
        # Structural, not conventional: adding one means changing the type.
        names = {f.name for f in dataclasses.fields(BudgetCurve)}
        self.assertEqual(names, {"arm", "points", "cells"})

    def test_the_curve_shows_shape_the_endpoint_hides(self):
        u = tied()
        a = budget_curve(u, "A", [60, 100, 300], ghost_penalty=float(BUDGET))
        b = budget_curve(u, "B", [60, 100, 300], ghost_penalty=float(BUDGET))
        self.assertEqual(a.at(300), b.at(300), "identical where the gate looks")
        self.assertGreater(a.at(100), b.at(100), "and far apart before it")

    def test_stability_reports_invariance(self):
        st = stability(tied(), "A", "B", taus=list(range(100, 301, 20)),
                       penalties={"tau": "tau", "fixed": 300.0})
        self.assertTrue(st.invariant)
        self.assertIn("invariant", st.describe())


class TheDefaultsNothingEverOmitted(unittest.TestCase):
    """Measured, not guessed: instrumenting the suite showed `ghost_penalty`
    was supplied at all 15 `permutation_test` calls and both `budget_curve`
    calls, so the None branch -- which is a refusal -- had never run through
    either. An argument the tests always supply is an argument whose default
    is untested, and the default is what production uses.
    """

    def _with_a_terminal(self):
        units = tied(n_scenes=6, misses=0)
        first = units[0]
        first.outcomes["A"][0] = EpisodeOutcome(
            outcome=Outcome.TERMINAL, step_to_success=None, step_budget=BUDGET)
        return units

    def test_the_refusal_survives_the_resampling_loop(self):
        """The case the nesting makes non-obvious.

        The refusal is raised per unit, inside a loop that runs it hundreds of
        times under permuted labels. If anything in that loop swallowed it,
        the function would return a p-value computed over episodes whose
        contribution nobody chose -- a number that looks exactly like a
        result.
        """
        with self.assertRaises(DistributionError) as c:
            permutation_test(self._with_a_terminal(), "A", "B",
                             statistic="median", samples=64, seed=0)
        self.assertIn("ghost_penalty", str(c.exception))

    def test_budget_curve_refuses_the_same_way(self):
        with self.assertRaises(DistributionError):
            budget_curve(self._with_a_terminal(), "A", [100, 200])

    def test_no_penalty_is_fine_when_nothing_needs_one(self):
        """The default is not broken, only untested: with no TERMINAL episode
        there is no convention to choose, and omitting it must work."""
        got = permutation_test(tied(n_scenes=6, misses=0), "A", "B",
                               statistic="median", samples=64, seed=0)
        self.assertGreater(got.p_value, 0.0)


if __name__ == "__main__":
    unittest.main()


class TestPartialAgreement(unittest.TestCase):
    """The case that exercises the counting rather than the plumbing.

    In the tied fixture every cell's difference is exactly -50, so under
    sign flipping only the all-negative and all-positive assignments reach
    |T| >= 50 and the count is 2 of 256 -- the floor. That proves the
    machinery runs; it never exercises the comparison, the tie handling or
    the >= boundary, because the answer is forced.

    Partial agreement with VARYING magnitudes does, and it is the only way
    to get a p strictly between the floor and 1.
    """

    def mixed(self, gaps, budget=300):
        units = []
        for i, gap in enumerate(gaps):
            def side(mid):
                d = {j: EpisodeOutcome(outcome=S, step_to_success=t,
                                       step_budget=budget)
                     for j, t in enumerate((mid - 5, mid, mid + 5, mid + 10))}
                for k in range(2):
                    d[4 + k] = EpisodeOutcome(outcome=B_, step_to_success=None,
                                              step_budget=budget)
                return d
            units.append(Unit(
                key=UnitKey(scene_id=f"scene-{i}", task_id="t", task_hash="th",
                            scenario_hash=f"sh-{i}"),
                outcomes={"A": side(100), "B": side(100 - gap)},
                seeds_kept=[], seeds_dropped=[]))
        return units

    def test_the_uniform_case_matches_a_hand_count(self):
        """Checked against an independent derivation, not against itself.

        With uniform +-50 and six scenes favouring A, the assignments
        reaching |T| >= 25 are those where at least six of eight signs agree:
        C(8,0..2) + C(8,6..8) = 1+8+28 + 28+8+1 = 74.
        """
        r = permutation_test(self.mixed([-50] * 6 + [50] * 2), "A", "B",
                             statistic="median", ghost_penalty=300.0)
        self.assertAlmostEqual(r.p_value, 74 / 256, places=9)

    def test_varying_magnitudes_are_not_a_function_of_the_sign_count(self):
        # Same 6-vs-2 split, different magnitudes, different answer. If the
        # statistic depended only on how many scenes agreed, these would
        # match the uniform case.
        uniform = permutation_test(self.mixed([-50] * 6 + [50] * 2), "A", "B",
                                   statistic="median", ghost_penalty=300.0)
        varied = permutation_test(
            self.mixed([-60, -45, -30, -55, -20, -40, 35, 25]), "A", "B",
            statistic="median", ghost_penalty=300.0)
        self.assertNotAlmostEqual(uniform.p_value, varied.p_value, places=3)

    def test_the_p_lands_strictly_between_the_floor_and_one(self):
        r = permutation_test(
            self.mixed([-60, -45, -30, -55, -20, -40, 35, 25]), "A", "B",
            statistic="median", ghost_penalty=300.0)
        self.assertGreater(r.p_value, r.floor,
                           "a floor value proves only that the test ran")
        self.assertLess(r.p_value, 1.0)

    def test_the_exact_path_takes_no_plus_one(self):
        # The observed assignment is one of the 2**c enumerated, so >=
        # already counts it; adding one would double-count it. The
        # correction belongs only where draws are a sample of the space.
        r = permutation_test(self.mixed([-50] * 8), "A", "B",
                             statistic="median", ghost_penalty=300.0)
        self.assertEqual(r.path, "exact")
        self.assertAlmostEqual(r.p_value, 2 / 256, places=9)
