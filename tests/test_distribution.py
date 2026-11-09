"""Phase 2: the distribution, the functionals, and what they refuse.

Every guard is tested by what it REJECTS. A guard whose test only feeds it
valid input is a guard nobody has seen fire.
"""

import unittest

from refractal.compare.distribution import (
    DistributionError,
    EXCLUDE,
    across_cells,
    ecdf,
    estimate,
    median,
    quantile,
    rmst,
    success_at,
)
from refractal.compare.pairing import EpisodeOutcome, Unit, UnitKey
from refractal.schema.outcome import Outcome, TerminalBasis

KEY = UnitKey(scene_id="s", task_id="t", task_hash="th", scenario_hash="sh")


def unit(*outcomes, budget=100):
    """One cell, from (outcome, step) pairs."""
    seeds = {}
    for i, (o, step) in enumerate(outcomes):
        seeds[i] = EpisodeOutcome(outcome=o, step_to_success=step,
                                  step_budget=budget)
    return Unit(key=KEY, outcomes={"A": seeds},
                seeds_kept=list(seeds), seeds_dropped=[])


S, B, T, X = (Outcome.SUCCESS, Outcome.BUDGET_EXHAUSTED,
              Outcome.TERMINAL, Outcome.APPARATUS)


class TestEstimate(unittest.TestCase):
    def test_apparatus_never_enters_the_distribution(self):
        u = unit((S, 10), (X, None), (B, None))
        est = estimate(u, "A", ghost_penalty=100.0)
        self.assertEqual(est.n, 2, "the apparatus row is out")
        self.assertEqual(est.apparatus, 1, "and is counted, not silently dropped")

    def test_a_cell_of_only_apparatus_is_refused(self):
        u = unit((X, None), (X, None))
        with self.assertRaises(DistributionError) as c:
            estimate(u, "A", ghost_penalty=100.0)
        self.assertIn("no non-APPARATUS episode", str(c.exception))

    def test_a_missing_step_budget_is_refused_not_reconstructed(self):
        # Where all 16,671 pre-v2 rows land. The plan beside the rows almost
        # certainly has the budget, and "almost certainly" is not a basis for
        # the bound every functional is evaluated under.
        seeds = {0: EpisodeOutcome(outcome=S, step_to_success=10, step_budget=None)}
        u = Unit(key=KEY, outcomes={"A": seeds}, seeds_kept=[0], seeds_dropped=[])
        with self.assertRaises(DistributionError) as c:
            estimate(u, "A", ghost_penalty=100.0)
        self.assertIn("step_budget", str(c.exception))
        self.assertIn("schema version 2", str(c.exception))


class TestFunctionals(unittest.TestCase):
    def setUp(self):
        self.est = estimate(unit((S, 10), (S, 20), (S, 80), (B, None)),
                            "A", ghost_penalty=100.0)

    def test_success_at(self):
        self.assertAlmostEqual(success_at(self.est, 20), 0.5)
        self.assertAlmostEqual(success_at(self.est, 100), 0.75)

    def test_tau_past_the_budget_is_refused(self):
        with self.assertRaises(DistributionError) as c:
            success_at(self.est, 101)
        self.assertIn("exceeds the smallest step_budget", str(c.exception))

    def test_ecdf_is_the_curve(self):
        self.assertEqual(ecdf(self.est), ((10, 0.25), (20, 0.5), (80, 0.75)))

    def test_median_needs_no_penalty(self):
        # Three of four succeeded, so the median is fixed by observations and
        # no ghost convention can move it.
        a = estimate(unit((S, 10), (S, 20), (S, 80), (B, None)), "A",
                     ghost_penalty=100.0)
        b = estimate(unit((S, 10), (S, 20), (S, 80), (B, None)), "A",
                     ghost_penalty=1e9)
        self.assertEqual(median(a), median(b))

    def test_median_is_undefined_rather_than_coerced(self):
        # Fewer than half succeeded. Coercing to the budget would make this
        # read the same as a cell where 49% succeeded, which is the exact
        # distinction a caller came for.
        est = estimate(unit((S, 10), (B, None), (B, None)), "A",
                       ghost_penalty=100.0)
        self.assertIsNone(median(est))

    def test_rmst_requires_a_penalty_the_estimate_does_not_have(self):
        # A cell with no TERMINAL needs no convention, so it can be estimated
        # without one -- and rmst then has nothing to read and must refuse
        # rather than assume.
        est = estimate(unit((S, 10), (B, None)), "A")
        self.assertIsNone(est.ghost_penalty)
        with self.assertRaises(DistributionError) as c:
            rmst(est, tau=100)
        self.assertIn("records none", str(c.exception))

    def test_rmst_reads_the_penalty_from_the_estimate(self):
        # No forwarding, no signature inspection: the value lives on the
        # estimate, which is also where Phase 6 reads it to refuse two arms
        # compared under different conventions.
        est = estimate(unit((S, 10), (T, None)), "A", ghost_penalty=100.0)
        self.assertEqual(est.ghost_penalty, 100.0)
        self.assertEqual(rmst(est, tau=100), rmst(est, tau=100, ghost_penalty=100.0))

    def test_rmst_refuses_a_penalty_the_ghosts_were_not_built_with(self):
        # The ghosts are already fixed, so a different value here would
        # describe a distribution that was never estimated.
        est = estimate(unit((S, 10), (T, None)), "A", ghost_penalty=100.0)
        with self.assertRaises(DistributionError) as c:
            rmst(est, tau=100, ghost_penalty=20.0)
        self.assertIn("already fixed", str(c.exception))

    def test_a_terminal_without_a_penalty_is_refused_at_estimate_time(self):
        with self.assertRaises(DistributionError) as c:
            estimate(unit((S, 10), (T, None)), "A")
        self.assertIn("convention", str(c.exception))

    def test_rmst_moves_with_the_penalty(self):
        cheap = estimate(unit((S, 10), (T, None)), "A", ghost_penalty=20.0)
        dear = estimate(unit((S, 10), (T, None)), "A", ghost_penalty=100.0)
        self.assertLess(rmst(cheap, tau=100, ghost_penalty=20.0),
                        rmst(dear, tau=100, ghost_penalty=100.0))


class TestGhostBasis(unittest.TestCase):
    def cell(self):
        return unit((S, 10), (T, None), (T, None))

    def test_a_partial_basis_mapping_is_refused(self):
        with self.assertRaises(DistributionError) as c:
            estimate(self.cell(), "A",
                     ghost_penalty={TerminalBasis.TASK: 100.0},
                     bases={1: TerminalBasis.TASK, 2: TerminalBasis.SIMULATOR})
        self.assertIn("partial mapping", str(c.exception))

    def test_a_terminal_without_a_basis_is_refused_under_a_mapping(self):
        with self.assertRaises(DistributionError) as c:
            estimate(self.cell(), "A",
                     ghost_penalty={TerminalBasis.TASK: 100.0,
                                    TerminalBasis.SIMULATOR: 100.0},
                     bases={1: TerminalBasis.TASK})
        self.assertIn("no terminal_basis", str(c.exception))

    def test_exclude_drops_and_counts(self):
        est = estimate(self.cell(), "A",
                       ghost_penalty={TerminalBasis.TASK: 100.0,
                                      TerminalBasis.SIMULATOR: EXCLUDE},
                       bases={1: TerminalBasis.TASK, 2: TerminalBasis.SIMULATOR})
        self.assertEqual(est.n, 2, "the excluded row is out of the denominator")
        self.assertEqual(est.excluded, 1)
        self.assertEqual(est.excluded_bases, ("SIMULATOR",),
                         "a silently smaller denominator is the failure mode")


if __name__ == "__main__":
    unittest.main()


class TestTheCasesTheFunctionalsExistFor(unittest.TestCase):
    """Two fixtures that are the argument for this whole layer.

    Built as tests rather than asserted in a write-up, because both are
    claims about what the binary gate cannot see and a claim like that should
    fail loudly if the plumbing stops producing it.
    """

    def arms(self, *, a_times, a_terminals, b_times, b_terminals, budget):
        """Two arms as explicit distributions, one cell each."""
        def side(times, terminals):
            seeds, i = {}, 0
            for t in times:
                seeds[i] = EpisodeOutcome(outcome=S, step_to_success=t,
                                          step_budget=budget); i += 1
            for _ in range(terminals):
                seeds[i] = EpisodeOutcome(outcome=T, step_to_success=None,
                                          step_budget=budget); i += 1
            return seeds
        return Unit(key=KEY,
                    outcomes={"A": side(a_times, a_terminals),
                              "B": side(b_times, b_terminals)},
                    seeds_kept=[], seeds_dropped=[])

    def test_two_arms_tied_on_success_at_budget_differ_in_median(self):
        """The case no N fixes.

        Both arms succeed on exactly the same number of episodes within the
        budget, so success-at-budget is IDENTICAL and no amount of data makes
        a binary test see a difference. The median separates them two to one.
        """
        u = self.arms(a_times=[60, 65, 70, 75, 80], a_terminals=3,
                      b_times=[110, 115, 120, 125, 130], b_terminals=3,
                      budget=300)
        ea = estimate(u, "A", ghost_penalty=300.0)
        eb = estimate(u, "B", ghost_penalty=300.0)
        self.assertEqual(success_at(ea, 300), success_at(eb, 300),
                         "tied exactly on the binary statistic")
        # n = 8 per arm (5 successes + 3 terminals), so the median is the
        # smallest observed time at which the ECDF reaches 0.5 -- the fourth
        # success, not the third.
        self.assertEqual(median(ea), 75.0)
        self.assertEqual(median(eb), 125.0)
        self.assertGreater(median(eb) / median(ea), 1.6,
                           "and they are far apart while the gate sees nothing")

    def test_the_ranking_inverts_with_tau(self):
        """The sentence the stability report exists to produce.

        FAST finishes early but ends irrecoverably more often; STEADY is
        slower and almost always finishes. Under ghost_penalty = tau each
        terminal contributes tau, so an arm's RMST climbs with tau at a rate
        set by its ghost SHARE: FAST starts lower and climbs faster. Where
        the lines cross, the ranking inverts, and which arm is 'better'
        depends entirely on a horizon the caller chose.

        A report that only ever agrees with itself has not been shown to
        detect this.
        """
        u = self.arms(a_times=[20, 25, 30], a_terminals=7,
                      b_times=[150, 160, 170, 180, 190, 200, 210, 220], b_terminals=2,
                      budget=400)

        def rmst_at(cp, tau):
            est = estimate(u, cp, ghost_penalty=float(tau))
            return rmst(est, tau=tau)

        low, high = 60, 400
        self.assertLess(rmst_at("A", low), rmst_at("B", low),
                        "at a short horizon the fast arm wins")
        self.assertGreater(rmst_at("A", high), rmst_at("B", high),
                           "at a long horizon the reliable arm wins")

        crossing = next(t for t in range(low, high + 1)
                        if rmst_at("A", t) >= rmst_at("B", t))
        self.assertTrue(low < crossing < high)
        # Not a fixed number: it moves with the fixture, and pinning it would
        # make this test about arithmetic rather than about the inversion.
        self.assertLess(rmst_at("A", crossing - 1), rmst_at("B", crossing - 1))


class TestCrossings(unittest.TestCase):
    """All sign changes, not the first one.

    The gap between two arms under `ghost_penalty = tau` is not monotone in
    tau: an arm still accumulating slow successes grows faster than its
    asymptotic ghost-share slope and flattens once they are exhausted. A
    quantity that falls and then rises can cross zero more than once, and a
    finder that stopped at the first would name an arbitrary member of a set
    while calling it "the" crossing.
    """

    def sweep(self, a_times, a_ghosts, b_times, b_ghosts, budget=400, step=2):
        from refractal.compare.distribution import crossings

        def side(times, ghosts):
            d, i = {}, 0
            for t in times:
                d[i] = EpisodeOutcome(outcome=S, step_to_success=t,
                                      step_budget=budget); i += 1
            for _ in range(ghosts):
                d[i] = EpisodeOutcome(outcome=T, step_to_success=None,
                                      step_budget=budget); i += 1
            return d

        u = Unit(key=KEY,
                 outcomes={"A": side(a_times, a_ghosts),
                           "B": side(b_times, b_ghosts)},
                 seeds_kept=[], seeds_dropped=[])
        taus = range(10, budget + 1, step)
        va = {t: rmst(estimate(u, "A", ghost_penalty=float(t)), tau=t) for t in taus}
        vb = {t: rmst(estimate(u, "B", ghost_penalty=float(t)), tau=t) for t in taus}
        return crossings(va, vb)

    def test_an_invariant_ranking_says_so(self):
        c = self.sweep([20, 30, 40], 0, [200, 210, 220], 0)
        self.assertTrue(c.invariant)
        self.assertIn("invariant", c.describe("A", "B"))

    def test_one_crossing_is_named(self):
        c = self.sweep([20, 30, 260, 270], 4, [120, 130, 140, 150], 4, budget=300)
        self.assertEqual(len(c.at), 1)
        self.assertIn("inverts at tau =", c.describe("A", "B"))

    def test_two_crossings_are_both_named(self):
        """Neither arm dominates, which is the most informative of the three.

        A succeeds once almost immediately and then not again until very
        late; B clusters in the middle and carries more ghosts. The gap
        leaves zero, returns, and leaves again, so quoting a single crossing
        tau would be quoting one of two.
        """
        c = self.sweep([15, 300, 320, 340, 360], 0, [90, 100, 110], 4, budget=400)
        self.assertEqual(len(c.at), 2, f"expected two crossings, got {c.at}")
        said = c.describe("A", "B")
        self.assertIn("and again at", said)
        self.assertIn("neither arm dominates", said.lower())
        self.assertNotIn("the crossing", said.lower(),
                         "must not imply a single well-defined tau")

    def test_the_lead_is_not_read_from_the_first_gap(self):
        """At a small tau the gap is reliably zero, so gaps[0] names nobody.

        Every episode contributes exactly tau there: successes have not
        landed and ghosts are charged the horizon. Reading the lead from
        index 0 reported whichever arm the tie-break favoured, which the
        end-to-end witness caught by naming the wrong winner on a sweep whose
        numbers said the opposite.
        """
        c = self.sweep([20, 30, 260, 270], 4, [120, 130, 140, 150], 4, budget=300)
        self.assertEqual(c.gaps[0], 0.0, "the fixture reproduces the zero gap")
        first = c.at[0]
        before = [g for t, g in zip(c.taus, c.gaps) if t < first and g]
        said = c.describe("A", "B")
        winner_below = "A" if before[-1] < 0 else "B"
        self.assertIn(f"Below it {winner_below} wins", said)
