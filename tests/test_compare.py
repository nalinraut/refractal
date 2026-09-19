"""Tests for pairing and the statistics.

The interesting ones are at the bottom. A fixture whose only cases are 58.8%
against 82.6% and 51.1% against 53.2% cannot tell a correct test from a broken
one -- any test calls the first significant and the second noise. The cases that
discriminate are pinned here, with the salts that produce them.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from refractal.compare import (
    holm,
    measure_variance,
    modelled_variance,
    evaluate,
    render,
    build_units,
    clustered_bootstrap,
    contingency,
    mcnemar_exact,
    mcnemar_unclustered,
    two_proportion_z,
)
from refractal.execute import FakeBenchmark
from refractal.resolve import resolve

SRC = Path(__file__).resolve().parents[1] / "examples" / "catalog"
A, B = "ckpt-46", "ckpt-47"
_PLAN = None
_TMP = None


def setUpModule():
    global _PLAN, _TMP
    _TMP = tempfile.TemporaryDirectory(prefix="refractal-compare-")
    root = Path(_TMP.name) / "catalog"
    shutil.copytree(SRC, root)
    doc = yaml.safe_load((root / "run.yaml").read_text(encoding="utf-8"))
    doc["run"]["tier"] = "full"
    doc["run"]["seeds"] = 5
    (root / "run.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    _PLAN = resolve(root, hardware_profile="rtx5090")


def tearDownModule():
    _TMP.cleanup()


def rows_for(task_id="vial-slot-7", **kwargs):
    """Episode rows straight from the fixture, without the storage round trip."""
    benchmark = FakeBenchmark(**kwargs)
    out = []
    for scene in _PLAN.scenes:
        for worker in scene.workers:
            for episode in worker.episodes:
                if task_id and episode.task_id != task_id:
                    continue
                outcome = benchmark.run(episode)
                out.append(
                    {
                        "scene_id": scene.scene_id,
                        "scene_hash": scene.scene_hash,
                        "task_id": episode.task_id,
                        "task_hash": episode.task_hash,
                        "scenario_hash": episode.scenario_hash,
                        "checkpoint_id": episode.checkpoint_id,
                        "seed": episode.seed,
                        "session_id": "session-a",
                        "harness_version": "none/local-backend",
                        "success": outcome["success"],
                        "is_infra_failure": outcome["is_infra_failure"],
                    }
                )
    return out


def synthetic(spec, *, seeds=(0, 1, 2), infra=()):
    """Hand-built rows: ``spec[checkpoint][scenario] = [bool per seed]``."""
    out = []
    for checkpoint, scenarios in spec.items():
        for scenario, successes in scenarios.items():
            for seed, ok in zip(seeds, successes):
                out.append(
                    {
                        "scene_id": "s",
                        "scene_hash": "sha256:aa",
                        "task_id": "t",
                        "task_hash": "sha256:t",
                        "scenario_hash": scenario,
                        "checkpoint_id": checkpoint,
                        "seed": seed,
                        "session_id": "session-a",
                        "harness_version": "v",
                        "success": ok,
                        "is_infra_failure": (checkpoint, scenario, seed) in infra,
                    }
                )
    return out


class TestPairingRule(unittest.TestCase):
    """One infra failure drops that seed from *every* checkpoint."""

    def test_seed_dropped_across_all_checkpoints(self):
        rows = synthetic(
            {
                A: {"x": [True, True, False]},
                B: {"x": [True, False, False]},
            },
            infra=[(A, "x", 1)],  # only A crashed at seed 1
        )
        eligibility = build_units(rows, checkpoints=[A, B], min_seeds=2)
        unit = eligibility.units[0]
        # B's seed-1 episode was clean, and it goes anyway: McNemar needs both
        # members of a pair, so a half-pair is not usable data.
        self.assertEqual(unit.seeds_kept, [0, 2])
        self.assertEqual(unit.seeds_dropped, [1])
        self.assertNotIn(1, unit.outcomes[B])
        self.assertEqual(len(unit.outcomes[A]), len(unit.outcomes[B]))

    def test_unit_refused_below_the_seed_floor(self):
        rows = synthetic(
            {A: {"x": [True, True, True]}, B: {"x": [True, True, True]}},
            infra=[(A, "x", 0), (A, "x", 1)],
        )
        eligibility = build_units(rows, checkpoints=[A, B], min_seeds=2)
        self.assertEqual(eligibility.units, [])
        self.assertEqual(eligibility.units_dropped_below_floor, 1)
        # Visible, not silent -- the same principle as the overlap counts.
        self.assertTrue(
            any("fewer than 2 usable seeds" in line for line in eligibility.summary_lines())
        )

    def test_floor_defaults_to_half_the_planned_seeds(self):
        rows = synthetic({A: {"x": [True] * 5}, B: {"x": [True] * 5}}, seeds=(0, 1, 2, 3, 4))
        self.assertEqual(build_units(rows, checkpoints=[A, B]).min_seeds, 3)

    def test_unequal_seed_counts_are_reported(self):
        rows = synthetic(
            {A: {"x": [True] * 3, "y": [True] * 3}, B: {"x": [True] * 3, "y": [True] * 3}},
            infra=[(B, "y", 2)],
        )
        eligibility = build_units(rows, checkpoints=[A, B], min_seeds=2)
        self.assertEqual(eligibility.seed_counts, {3: 1, 2: 1})
        self.assertTrue(
            any("unequal seed counts" in line for line in eligibility.summary_lines())
        )

    def test_scenario_missing_from_one_checkpoint_is_not_a_unit(self):
        rows = synthetic({A: {"x": [True] * 3, "y": [True] * 3}, B: {"x": [True] * 3}})
        eligibility = build_units(rows, checkpoints=[A, B], min_seeds=2)
        self.assertEqual(len(eligibility.units), 1)
        self.assertEqual(eligibility.only_in[A], 1)
        self.assertIn("Only in ckpt-46: 1", eligibility.summary_lines())


class TestOverlapAndProvenance(unittest.TestCase):
    def test_overlap_is_the_first_line(self):
        eligibility = build_units(rows_for(), checkpoints=[A, B])
        self.assertTrue(eligibility.summary_lines()[0].startswith("Scenarios in all"))

    def test_scene_hash_disagreement_is_surfaced(self):
        """Editing a mesh between runs must be loud, not an empty join."""
        rows = synthetic({A: {"x": [True] * 3}, B: {"x": [True] * 3}})
        for row in rows:
            if row["checkpoint_id"] == B:
                row["scene_hash"] = "sha256:bb"
        eligibility = build_units(rows, checkpoints=[A, B], min_seeds=2)
        self.assertIn("s", eligibility.scene_hash_conflicts)
        self.assertEqual(len(eligibility.scene_hash_conflicts["s"]), 2)

    def test_sessions_are_tracked(self):
        rows = synthetic({A: {"x": [True] * 3}, B: {"x": [True] * 3}})
        rows[-1]["session_id"] = "session-b"
        eligibility = build_units(rows, checkpoints=[A, B], min_seeds=2)
        self.assertEqual(eligibility.sessions, {"session-a", "session-b"})


class TestContingency(unittest.TestCase):
    def test_all_four_cells(self):
        rows = synthetic(
            {
                A: {"w": [True] * 3, "x": [True] * 3, "y": [False] * 3, "z": [False] * 3},
                B: {"w": [True] * 3, "x": [False] * 3, "y": [True] * 3, "z": [False] * 3},
            }
        )
        units = build_units(rows, checkpoints=[A, B], min_seeds=2).units
        cells = contingency(units, A, B, "majority")
        self.assertEqual(
            (cells.both_pass, cells.only_a_passes, cells.only_b_passes, cells.both_fail),
            (1, 1, 1, 1),
        )
        # The diagonal is the actionable half and is never discarded, even
        # though the test reads only the off-diagonal.
        self.assertEqual(cells.total, 4)
        self.assertEqual(cells.discordant, 2)

    def test_dichotomisation_rule_is_a_choice(self):
        rows = synthetic({A: {"x": [True, True, False]}, B: {"x": [True, True, False]}})
        units = build_units(rows, checkpoints=[A, B], min_seeds=2).units
        self.assertTrue(units[0].passed(A, "majority"))
        self.assertFalse(units[0].passed(A, "all"))


class TestClusteringMatters(unittest.TestCase):
    """The measured condition under which ignoring clustering goes wrong.

    Measured over 300 trials at a true difference of zero, alpha 0.05:

        scenario  interaction | McNemar  bootstrap | seeds-as-pairs  2-prop z
            0.00         0.00 |   3.7%      8.0%   |      5.3%         7.3%
            0.45         0.00 |   5.0%      8.3%   |      6.0%         2.7%
            0.00         0.35 |   4.7%      4.7%   |     13.7%        14.0%
            0.30         0.35 |   5.0%      6.3%   |     14.0%        13.7%

    A scenario effect *shared* by both checkpoints cancels in a paired
    difference, so ignoring clustering costs nothing -- the unpaired test even
    becomes conservative. What breaks the naive tests is a checkpoint x scenario
    interaction: a scenario that is hard for one checkpoint and easy for the
    other, correlated across that scenario's seeds. That is what a real
    regression looks like, and there the false-positive rate roughly triples.
    """

    #: Searched for, not pinned.
    #:
    #: A hardcoded salt names one pseudo-random draw, and every episode_id in
    #: that draw derives from task_hash -- so ANY change to what task identity
    #: covers reshuffles it. That happened: adding `provider_ref` to
    #: `task_identity` moved every hash and this test failed at p=0.0657 against
    #: a 0.05 threshold, with nothing wrong in the statistics.
    #:
    #: A demonstration that "the wrong test declares a finding here" is
    #: inherently about a particular draw, so the fixture must FIND one and say
    #: so if it cannot -- rather than assert that a remembered one still works.
    #: Failing with "no draw exhibits this" is a real finding; failing with
    #: "0.0657 is not less than 0.05" is a maintenance chore wearing its clothes.
    #: 150, not 40, and the number is a measurement rather than a guess.
    #:
    #: The four conditions hold together on about 2% of draws -- measured: the
    #: unclustered tests declare a false finding on 8%, the z-test on 7%, and the
    #: conjunction with both correct tests staying null on 3 of 150. That rarity
    #: IS the property being demonstrated: anti-conservatism at roughly the
    #: inflated false-positive rate. A search sized for a common event would be
    #: the wrong size for a test about an uncommon one.
    #:
    #: 40 was too few, and found out the day a predicate string changed in the
    #: example catalog: `predicate` is in `task_hash`, which is in `episode_id`,
    #: which seeds every synthetic outcome. The first qualifying salt is n40.
    CANDIDATE_SALTS = tuple(f"n{i}" for i in range(150))

    def setUp(self):
        self.units = None
        self.salt = None
        for salt in self.CANDIDATE_SALTS:
            rows = rows_for(
                success_rate=0.5,
                infra_failure_rate=0.03,
                scenario_spread=0.30,
                interaction_spread=0.35,
                salt=salt,
                # No injected difference: any "finding" here is a false positive.
            )
            units = build_units(rows, checkpoints=[A, B]).units
            if (
                mcnemar_unclustered(units, A, B).p_value < 0.05
                and two_proportion_z(units, A, B).p_value < 0.05
                and mcnemar_exact(contingency(units, A, B, "majority")).p_value > 0.05
                # The bootstrap too. Searching for three of the four conditions
                # this class asserts found a draw that satisfied three and broke
                # the fourth the next time any identity field moved -- a
                # searching fixture has to search for everything its tests check,
                # or it is a pinned salt with extra steps.
                and not clustered_bootstrap(units, A, B, resamples=2000, seed=7).excludes_zero
            ):
                self.units, self.salt = units, salt
                break
        if self.units is None:
            self.fail(
                f"no draw among {len(self.CANDIDATE_SALTS)} exhibits the anti-conservatism "
                "this class exists to demonstrate: no salt produced a false positive from "
                "both unclustered tests while the clustered ones stayed null. Either the "
                "generator stopped producing the interaction, or the wrong tests stopped "
                "being wrong -- both are findings, and neither is a flaky test."
            )

    def test_correct_tests_find_nothing(self):
        self.assertGreater(mcnemar_exact(contingency(self.units, A, B, "majority")).p_value, 0.05)
        self.assertFalse(clustered_bootstrap(self.units, A, B, resamples=2000, seed=7).excludes_zero)

    def test_both_wrong_tests_declare_a_finding(self):
        self.assertLess(mcnemar_unclustered(self.units, A, B).p_value, 0.05)
        self.assertLess(two_proportion_z(self.units, A, B).p_value, 0.05)

    def test_unclustered_inflates_n_by_the_seed_count(self):
        clustered = mcnemar_exact(contingency(self.units, A, B, "majority")).n
        unclustered = mcnemar_unclustered(self.units, A, B).n
        self.assertGreater(unclustered, 3 * clustered)


class TestBorderlineRealDifference(unittest.TestCase):
    """A real difference small enough that the two *correct* methods disagree.

    Seven points, injected. McNemar on majority-vote does not reject; the
    clustered bootstrap excludes zero. Both are right -- they answer different
    questions. McNemar asks whether the set of scenarios a checkpoint solves
    changed; the bootstrap asks whether the rate did, and dichotomising to a
    majority throws away the within-scenario proportion that the rate keeps.

    This is why D6 reports both rather than picking one, and why leading with a
    single p-value would be a misrepresentation in either direction.
    """

    def setUp(self):
        # SEARCHED, not pinned. This class demonstrates a disagreement between
        # two correct methods, which is a property of a dataset -- so the fixture
        # has to find one and say so if it cannot, rather than remember a salt
        # that worked once. `salt="b0"` broke the day a predicate string changed
        # in the example catalog, because `predicate` is in `task_hash`, which is
        # in `episode_id`, which seeds every synthetic outcome.
        self.units = None
        for salt in (f"b{i}" for i in range(40)):
            rows = rows_for(
                success_rate=0.5,
                infra_failure_rate=0.03,
                scenario_spread=0.30,
                salt=salt,
                per_task={(B, "vial-slot-7"): 0.57},
            )
            units = build_units(rows, checkpoints=[A, B]).units
            if (
                mcnemar_exact(contingency(units, A, B, "majority")).p_value > 0.05
                and clustered_bootstrap(units, A, B, resamples=2000, seed=7).excludes_zero
            ):
                self.units = units
                break
        if self.units is None:
            self.fail(
                "no draw among 40 produced the disagreement this class exists to "
                "demonstrate: McNemar retaining its null while the clustered "
                "bootstrap excludes zero. If that cannot happen the two methods are "
                "not answering different questions, which is the claim."
            )

    def test_mcnemar_does_not_reject(self):
        self.assertGreater(mcnemar_exact(contingency(self.units, A, B, "majority")).p_value, 0.05)

    def test_clustered_bootstrap_excludes_zero(self):
        result = clustered_bootstrap(self.units, A, B, resamples=2000, seed=7)
        self.assertTrue(result.excludes_zero)
        self.assertGreater(result.difference, 0)

    def test_the_difference_is_real_by_construction(self):
        rate_a = sum(u.rate(A) for u in self.units) / len(self.units)
        rate_b = sum(u.rate(B) for u in self.units) / len(self.units)
        self.assertGreater(rate_b, rate_a)


class TestACeilingIsNamed(unittest.TestCase):
    """An arm pinned at 0% or 100% on every episode.

    Added after a real run put pi0.5 at 60/60 on both tasks. `compare` printed
    "within-cell ICC 0.000, between-scenario Var 0.00000" -- the receipt -- and
    said nothing about what it implies: a further improvement to that arm cannot
    be observed, and the interval's bound on that side is arithmetic rather than
    evidence.

    Not a block. The contrast is real and the interval is honest about what it
    measured. It is a limit on what the measurement could have shown, which is
    the kind of thing a reader takes from the absence of a note.
    """

    def _rows(self, baseline_rate, candidate_rate, scenarios=8, seeds=3):
        rows = []
        for scenario in range(scenarios):
            for seed in range(seeds):
                for checkpoint, rate in (("base", baseline_rate), ("cand", candidate_rate)):
                    # Deterministic, so the fixture's rates are exactly the ones
                    # asked for rather than approximately.
                    index = scenario * seeds + seed
                    success = index < round(rate * scenarios * seeds)
                    rows.append({
                        "episode_id": f"{checkpoint}-{scenario}-{seed}",
                        "scenario_hash": f"sha256:s{scenario}",
                        "scene_id": "scene", "scene_hash": "sha256:scene",
                        "task_id": "task", "task_hash": "sha256:task",
                        "checkpoint_id": checkpoint, "seed": seed,
                        "session_id": "s", "worker_id": "w",
                        "execution_mode": "interleaved",
                        "harness_version": "v", "harness_surface": "d",
                        "success": success, "phase_outcomes": [],
                        "terminal_phase": None,
                        "failure_reason": None if success else "policy_failure",
                        "is_infra_failure": False, "steps": 10, "elapsed_sec": 1.0,
                        "started_at": None, "ended_at": None, "artifact_uri": None,
                    })
        return rows

    def _notes(self, baseline_rate, candidate_rate):
        eligibility = build_units(
            self._rows(baseline_rate, candidate_rate), checkpoints=["base", "cand"]
        )
        verdict = evaluate(eligibility, ["base", "cand"], baseline="base", resamples=200)
        return " | ".join(
            n for task in verdict.tasks for c in task.contrasts for n in c.notes
        )

    def test_an_arm_at_the_ceiling_is_named(self):
        notes = self._notes(0.5, 1.0)
        self.assertIn("at 100% on every episode", notes)
        self.assertIn("no room to move", notes)

    def test_an_arm_at_the_floor_is_named(self):
        """0% is the same problem pointing down: a change that made it worse
        could not be measured."""
        notes = self._notes(0.0, 0.5)
        self.assertIn("at 0% on every episode", notes)
        self.assertIn("worsened", notes)

    def test_two_arms_in_the_interior_get_no_ceiling_note(self):
        """The break test: if this also produced the note, the note would be
        decoration rather than a finding."""
        notes = self._notes(0.375, 0.75)
        self.assertNotIn("no room to move", notes)


if __name__ == "__main__":
    unittest.main()


class TestGate(unittest.TestCase):
    """The gate is one bit and the choice is deliberate: the bootstrap decides.

    Measured power at 120 scenarios x 5 seeds with an interaction present, 300
    trials, at the same false-positive rate for both tests:

        true delta   McNemar   bootstrap
              0.00      4.0%        4.3%
              0.05     13.0%       29.3%
              0.07     24.7%       47.3%
              0.10     46.0%       75.3%

    Gating on McNemar would miss a real 7-point regression three times in four.
    """

    def _verdict(self, delta, **kw):
        rows = rows_for(
            success_rate=0.5,
            infra_failure_rate=0.03,
            scenario_spread=0.30,
            salt=kw.pop("salt", "g0"),
            per_task={(B, "vial-slot-7"): 0.5 + delta},
            **kw,
        )
        return evaluate(
            build_units(rows, checkpoints=[A, B]), [A, B], resamples=2000, seed=7
        )

    def test_a_real_regression_goes_red(self):
        verdict = self._verdict(-0.12)
        self.assertTrue(verdict.regressed)
        self.assertEqual(verdict.exit_code, 1)

    def test_no_difference_stays_green(self):
        verdict = self._verdict(0.0)
        self.assertFalse(verdict.regressed)
        self.assertEqual(verdict.exit_code, 0)

    def test_an_improvement_is_not_a_regression(self):
        verdict = self._verdict(+0.12)
        self.assertEqual([c.gate for t in verdict.tasks for c in t.contrasts], ["improved"])
        self.assertEqual(verdict.exit_code, 0)

    def test_bootstrap_gates_even_when_mcnemar_retains_its_null(self):
        """The disagreement case, resolved in the bootstrap's favour.

        The qualifying draw is SEARCHED FOR, not pinned. `salt="r3"` named one
        particular pseudo-random dataset, and every episode_id in it derives from
        task_hash -- so adding a field to `task_identity` reshuffled the draw and
        this failed with the two tests no longer disagreeing. Nothing was wrong.

        A disagreement between two tests is a property of a dataset, so the
        fixture has to produce one. "No draw disagrees" is a genuine finding --
        it would mean the two tests had stopped being distinguishable, which is
        the whole reason the gate chooses between them.
        """
        for salt in (f"r{i}" for i in range(40)):
            verdict = self._verdict(-0.06, salt=salt)
            contrast = verdict.tasks[0].contrasts[0]
            if contrast.mcnemar.p_value > 0.05 and contrast.bootstrap.excludes_zero:
                break
        else:
            self.fail(
                "no draw among 40 produced a disagreement: McNemar never retained its "
                "null while the bootstrap excluded zero. The gate exists to resolve that "
                "disagreement, so if it cannot occur the gate has nothing to decide."
            )
        self.assertGreater(contrast.mcnemar.p_value, 0.05)      # McNemar sees nothing
        self.assertTrue(contrast.bootstrap.excludes_zero)        # the rate moved
        self.assertTrue(verdict.regressed)
        # ...and the report says why, so a red is interpretable rather than obeyed.
        self.assertTrue(any("flipped their majority" in n for n in contrast.notes))

    def test_small_scenario_count_is_flagged_as_optimistic(self):
        verdict = self._verdict(0.0)
        self.assertTrue(
            any("nominal 5%" in n for t in verdict.tasks for c in t.contrasts for n in c.notes)
        )

    def test_changed_geometry_blocks_rather_than_reporting_a_regression(self):
        rows = synthetic({A: {"x": [True] * 3}, B: {"x": [False] * 3}})
        for row in rows:
            if row["checkpoint_id"] == B:
                row["scene_hash"] = "sha256:changed"
        verdict = evaluate(
            build_units(rows, checkpoints=[A, B], min_seeds=2), [A, B], resamples=200, seed=1
        )
        # "the geometry changed" is a different sentence from "the policy got
        # worse", and exit code 2 keeps them different.
        self.assertEqual(verdict.exit_code, 2)
        self.assertFalse(verdict.regressed)
        self.assertEqual(verdict.tasks, [])
        self.assertIn("not comparable", verdict.blocking[0])

    def test_session_spanning_is_noted(self):
        rows = rows_for(success_rate=0.6, salt="s0")
        for row in rows[: len(rows) // 2]:
            row["session_id"] = "session-b"
        verdict = evaluate(
            build_units(rows, checkpoints=[A, B]), [A, B], resamples=200, seed=1
        )
        self.assertTrue(any("spans 2 sessions" in n for n in verdict.notes))

    def test_report_leads_with_overlap(self):
        text = render(self._verdict(0.0))
        self.assertTrue(text.strip().startswith("Scenarios in all"))
        self.assertIn("McNemar p=", text)
        self.assertIn("holm=", text)


C = "ckpt-48"


class TestMultipleCheckpoints(unittest.TestCase):
    """Two is the common case, not a special one. D7."""

    def _rows(self, rates):
        """rates: {checkpoint: success_rate} for one task.

        The salt varies per checkpoint, which is not cosmetic. Sharing one salt
        made every arm draw the same outcomes, so two checkpoints at the same rate
        were byte-identical -- a self-comparison wearing two names. The fixture
        looked right and every test passed, until the degenerate-arms guard
        started asking. Independent draws at the same rate is what a true null
        actually is.
        """
        out = []
        for checkpoint, rate in rates.items():
            for row in rows_for(
                success_rate=0.5,
                scenario_spread=0.30,
                salt=f"m0-{checkpoint}",
                per_task={(B, "vial-slot-7"): rate},
            ):
                if row["checkpoint_id"] != B:
                    continue
                out.append({**row, "checkpoint_id": checkpoint})
        return out

    def test_three_checkpoints_compare_against_one_baseline(self):
        rows = self._rows({A: 0.50, B: 0.50, C: 0.75})
        verdict = evaluate(
            build_units(rows, checkpoints=[A, B, C]), [A, B, C],
            baseline=A, resamples=2000, seed=3,
        )
        contrasts = {c.candidate: c for c in verdict.tasks[0].contrasts}
        # (k-1) contrasts against the baseline, not k(k-1)/2 pairs: the question
        # is "did my change help", which has a reference point.
        self.assertEqual(sorted(contrasts), [B, C])
        self.assertEqual(contrasts[C].gate, "improved")
        self.assertEqual(contrasts[B].gate, "ok")

    def test_outcome_vectors_show_what_pairwise_cannot(self):
        """"Scenarios only ckpt-48 solves" is not recoverable from pairwise tables."""
        rows = self._rows({A: 0.50, B: 0.50, C: 0.75})
        task = evaluate(
            build_units(rows, checkpoints=[A, B, C]), [A, B, C], baseline=A,
            resamples=200, seed=3,
        ).tasks[0]
        self.assertEqual(len(next(iter(task.patterns))), 3)
        only_c = task.patterns.get((False, False, True), 0)
        self.assertGreater(only_c, 0)
        self.assertEqual(sum(task.patterns.values()), task.units)

    def test_cochran_q_screens_for_any_difference(self):
        same = self._rows({A: 0.50, B: 0.50, C: 0.50})
        differ = self._rows({A: 0.50, B: 0.50, C: 0.85})
        q_same = evaluate(build_units(same, checkpoints=[A, B, C]), [A, B, C],
                          resamples=200, seed=3).tasks[0].cochran
        q_differ = evaluate(build_units(differ, checkpoints=[A, B, C]), [A, B, C],
                            resamples=200, seed=3).tasks[0].cochran
        self.assertGreater(q_same.p_value, 0.05)
        self.assertLess(q_differ.p_value, 0.05)

    def test_a_single_checkpoint_is_refused(self):
        rows = self._rows({A: 0.5})
        with self.assertRaises(ValueError):
            evaluate(build_units(rows, checkpoints=[A]), [A])


class TestMultiplicityCorrection(unittest.TestCase):
    """The gate fires if ANY contrast trips, so all of them are one family."""

    def test_family_spans_tasks_and_checkpoints(self):
        rows = rows_for(task_id=None, success_rate=0.55, salt="f0")
        verdict = evaluate(
            build_units(rows, checkpoints=[A, B]), [A, B], resamples=400, seed=5
        )
        # three tasks in the example catalog, one candidate against the baseline
        self.assertEqual(verdict.family_size, 3 * 1)
        self.assertTrue(any("Holm-adjusted" in n for n in verdict.notes))

    def test_correction_makes_the_gate_stricter(self):
        rows = rows_for(task_id=None, success_rate=0.55, salt="f1")
        kwargs = dict(resamples=2000, seed=5)
        corrected = evaluate(
            build_units(rows, checkpoints=[A, B]), [A, B], correct=True, **kwargs
        )
        raw = evaluate(
            build_units(rows, checkpoints=[A, B]), [A, B], correct=False, **kwargs
        )
        for task_c, task_r in zip(corrected.tasks, raw.tasks):
            for c, r in zip(task_c.contrasts, task_r.contrasts):
                self.assertGreaterEqual(c.adjusted_p, r.adjusted_p)

    def test_holm_is_monotone_and_bounded(self):
        adjusted = holm({"a": 0.001, "b": 0.04, "c": 0.30, "d": 0.9})
        values = [adjusted[k] for k in ("a", "b", "c", "d")]
        self.assertEqual(values, sorted(values))
        self.assertLessEqual(max(values), 1.0)
        self.assertAlmostEqual(adjusted["a"], 0.004)

    def test_uncorrected_family_wise_rate_is_stated(self):
        rows = rows_for(task_id=None, success_rate=0.55, salt="f0")
        verdict = evaluate(
            build_units(rows, checkpoints=[A, B]), [A, B], resamples=400, seed=5
        )
        note = next(n for n in verdict.notes if "Holm-adjusted" in n)
        self.assertIn("14%", note)   # 1 - 0.95^3


class TestUntestedPairsAreDisclosed(unittest.TestCase):
    """Baseline choice is a modelling decision, so say what it left out."""

    def _verdict(self, rates, baseline):
        rows = []
        for checkpoint, rate in rates.items():
            for row in rows_for(
                success_rate=0.5, scenario_spread=0.30, salt=f"u0-{checkpoint}",
                per_task={(B, "vial-slot-7"): rate},
            ):
                if row["checkpoint_id"] == B:
                    rows.append({**row, "checkpoint_id": checkpoint})
        names = list(rates)
        return evaluate(
            build_units(rows, checkpoints=names), names,
            baseline=baseline, resamples=400, seed=3,
        )

    def test_two_checkpoints_leave_nothing_untested(self):
        verdict = self._verdict({A: 0.5, B: 0.6}, A)
        self.assertEqual(verdict.untested_pairs, [])

    def test_three_checkpoints_leave_one_pair_untested(self):
        verdict = self._verdict({A: 0.5, B: 0.6, C: 0.7}, A)
        self.assertEqual(verdict.untested_pairs, [(B, C)])
        self.assertTrue(any("NOT tested" in n for n in verdict.notes))

    def test_it_warns_when_the_best_checkpoint_is_in_an_untested_pair(self):
        # The case that actually bites: the reader wants to ship the leader, and
        # no contrast covers it against the runner-up.
        verdict = self._verdict({A: 0.50, B: 0.80, C: 0.75}, A)
        note = next(n for n in verdict.notes if "NOT tested" in n)
        self.assertIn("highest mean rate", note)
        self.assertIn("nobody ran", note)


class TestHarnessSurfaceIsAPrecondition(unittest.TestCase):
    """Two columns, one of each kind, and the distinction is the whole point.

    `harness_version` is provenance: it moves on every release and on most
    commits, so gating on it would be waived habitually. Measured over the 24
    most recent harness commits, a whole-tree digest changed on 24 of them.

    `harness_surface` digests only the modules Refractal depends on. Over the
    same range it changed on 3, and on exactly the two commits that altered
    behaviour Refractal cares about.
    """

    def _rows(self, versions, surfaces):
        rows = rows_for(success_rate=0.6, salt="h0")
        for i, row in enumerate(rows):
            row["harness_version"] = versions[i % len(versions)]
            row["harness_surface"] = surfaces[i % len(surfaces)]
        return rows

    def test_one_of_each_compares_normally(self):
        verdict = evaluate(
            build_units(self._rows(["v1"], ["s1"]), checkpoints=[A, B]), [A, B],
            resamples=200, seed=1,
        )
        self.assertEqual(verdict.blocking, [])
        self.assertNotEqual(verdict.tasks, [])

    def test_different_versions_with_one_surface_only_note(self):
        """The case that would have been waived habitually.

        A docs edit, a leaderboard refresh or a pin bump moves the version and
        not the surface. The comparison stands, and the reader is told.
        """
        verdict = evaluate(
            build_units(self._rows(["v1", "v2"], ["s1"]), checkpoints=[A, B]), [A, B],
            resamples=200, seed=1,
        )
        self.assertEqual(verdict.blocking, [])
        self.assertNotEqual(verdict.tasks, [])
        self.assertTrue(any("the comparison stands" in n for n in verdict.notes))

    def test_two_surfaces_block_by_default(self):
        verdict = evaluate(
            build_units(self._rows(["v1"], ["s1", "s2"]), checkpoints=[A, B]), [A, B],
            resamples=200, seed=1,
        )
        self.assertEqual(verdict.exit_code, 2)
        self.assertEqual(verdict.tasks, [])          # no number produced at all
        self.assertIn("something behavioural changed", verdict.blocking[0])

    def test_the_override_is_explicit_and_echoed(self):
        verdict = evaluate(
            build_units(self._rows(["v1"], ["s1", "s2"]), checkpoints=[A, B]), [A, B],
            resamples=200, seed=1, allow_harness_mismatch=True,
        )
        self.assertEqual(verdict.blocking, [])
        self.assertNotEqual(verdict.tasks, [])
        # Waiving a precondition must be visible in the output, not just in the
        # command someone typed.
        self.assertTrue(any("waived" in o for o in verdict.overrides))
        self.assertIn("OVERRIDDEN", render(verdict))

    def test_session_id_stays_provenance(self):
        """The contrast: differing sessions still produce a number, with a note."""
        rows = rows_for(success_rate=0.6, salt="h1")
        for row in rows[: len(rows) // 2]:
            row["session_id"] = "session-b"
        verdict = evaluate(
            build_units(rows, checkpoints=[A, B]), [A, B], resamples=200, seed=1
        )
        self.assertEqual(verdict.blocking, [])
        self.assertNotEqual(verdict.tasks, [])
        self.assertTrue(any("spans 2 sessions" in n for n in verdict.notes))


class TestVarianceEstimatorIsCalibrated(unittest.TestCase):
    """The estimator must recover known structure before it is trusted on a policy.

    Every number in the measurement write-up was measured against
    `interaction_spread`, a model authored to make the fixture discriminate --
    then the tests were scored against it. The first real policy is the first
    independent source of the effect, so a correction to that doc should be a
    *comparison* between measured and modelled rather than a replacement.

    Which only works if the estimator itself is known to be right. These assert it
    against the doc's false-positive rates, which are independent evidence: they
    were measured by counting rejections, not by reading a variance.
    """

    def _measure(self, scenario_spread, interaction_spread, salt="v0"):
        rows = rows_for(
            success_rate=0.5,
            scenario_spread=scenario_spread,
            interaction_spread=interaction_spread,
            salt=salt,
        )
        units = build_units(rows, checkpoints=[A, B]).units
        return measure_variance(units, A, B)

    def test_no_structure_gives_a_design_effect_near_one(self):
        report = self._measure(0.0, 0.0)
        self.assertLess(report.design_effect, 1.25)
        self.assertFalse(report.clustering_matters)

    def test_a_shared_effect_does_not_inflate_the_paired_design_effect(self):
        """The row that caught a bias in this estimator.

        Strong shared scenario difficulty, zero interaction. The doc measured the
        unclustered test as *conservative* here (2.7% against a nominal 5%), so a
        design effect meaningfully above 1 would be the estimator manufacturing
        the conclusion it exists to test. It did, at 1.38, until the binomial
        term used s-1 rather than s.
        """
        report = self._measure(0.45, 0.0)
        self.assertGreater(report.icc[A], 0.3, "the shared effect should show in ICC")
        self.assertLess(report.design_effect, 1.25, "but must not show in the design effect")
        self.assertFalse(report.clustering_matters)

    def test_an_interaction_does_inflate_it(self):
        report = self._measure(0.0, 0.35)
        self.assertGreater(report.design_effect, 1.5)
        self.assertTrue(report.clustering_matters)

    def test_icc_alone_cannot_distinguish_the_two(self):
        """Why both numbers are reported, not just one."""
        shared = self._measure(0.45, 0.0)
        interaction = self._measure(0.0, 0.35)
        # Higher ICC, lower design effect. Reading ICC alone inverts the answer.
        self.assertGreater(shared.icc[A], interaction.icc[A])
        self.assertLess(shared.design_effect, interaction.design_effect)

    def test_the_measured_excess_tracks_the_modelled_variance(self):
        """The comparison that makes a correction to the doc statable.

        Averaged over draws rather than asserted on one. A single draw's excess
        is itself a random variable -- on one reshuffle it came out at -0.0026
        against a modelled 0.0107, which is not a calibration failure, it is one
        sample of a noisy quantity. A calibration claim needs the mean, and
        asserting it on a single pinned draw was a claim about that draw.
        """
        for interaction in (0.20, 0.35):
            modelled = modelled_variance(
                scenario_spread=0.0, interaction_spread=interaction
            )["expected_difference_var"]
            excesses = []
            for draw in range(12):
                report = self._measure(0.0, interaction, salt=f"cal{draw}")
                excesses.append(report.observed_var - report.binomial_var)
            measured = sum(excesses) / len(excesses)
            # Same order of magnitude and same direction; a sanity bound, not a
            # precision claim at 119 scenarios.
            self.assertGreater(
                measured, 0.4 * modelled,
                f"mean excess {measured:.5f} over {len(excesses)} draws "
                f"(individual: {[round(e, 4) for e in excesses]})",
            )
            self.assertLess(measured, 2.0 * modelled)

    def test_only_the_interaction_term_is_expected_to_survive_pairing(self):
        m = modelled_variance(scenario_spread=0.45, interaction_spread=0.0)
        self.assertEqual(m["expected_difference_var"], 0.0)
        self.assertGreater(m["cancels_in_paired_difference"], 0.0)

    def test_it_degrades_rather_than_dividing_by_zero(self):
        rows = synthetic({A: {"x": [True] * 3}, B: {"x": [True] * 3}})
        report = measure_variance(build_units(rows, checkpoints=[A, B], min_seeds=2).units, A, B)
        self.assertEqual(report.units, 1)
        self.assertEqual(report.design_effect, 1.0)


class TestVarianceIsReportedUnconditionally(unittest.TestCase):
    """The baseline has to be recorded by default, not when it looks interesting.

    An agreement recorded is what makes a later disagreement legible. Behind a
    flag it would be skipped on exactly the runs where nothing seemed notable,
    which are the runs that establish what normal is.
    """

    def _verdict(self, **kw):
        rows = rows_for(success_rate=0.5, scenario_spread=0.30, salt="rv", **kw)
        return evaluate(build_units(rows, checkpoints=[A, B]), [A, B], resamples=300, seed=2)

    def test_every_task_carries_a_variance_report(self):
        verdict = self._verdict()
        for task in verdict.tasks:
            self.assertIsNotNone(task.variance)
            self.assertGreater(task.variance.units, 0)

    def test_the_report_reaches_the_rendered_output(self):
        text = render(self._verdict())
        self.assertIn("design effect on the paired difference", text)
        self.assertIn("within-cell ICC", text)

    def test_it_says_which_way_it_read(self):
        no_interaction = render(self._verdict())
        self.assertIn("not load-bearing", no_interaction)
        with_interaction = render(self._verdict(interaction_spread=0.35))
        self.assertIn("load-bearing here", with_interaction)


class TestDuplicateRowsBlock(unittest.TestCase):
    """A duplicated episode is evidence of a bug, not something to collapse.

    Third instance of the shape: a dict keyed by seed answers "what happened at
    this seed" and cannot answer "did anything arrive twice". The second row
    overwrote the first silently, even when they disagreed about the outcome.
    """

    def _rows(self, duplicate=False):
        def row(ckpt, seed, success):
            return {"scene_id": "s", "scene_hash": "sha256:a", "task_id": "t",
                    "task_hash": "sha256:t", "scenario_hash": "sha256:x",
                    "checkpoint_id": ckpt, "seed": seed, "session_id": "s",
                    "harness_version": "v", "harness_surface": "v",
                    "success": success, "is_infra_failure": False}
        rows = [row(A, 0, True), row(A, 1, True), row(B, 0, True), row(B, 1, True)]
        if duplicate:
            rows.insert(1, row(A, 0, False))  # same episode, opposite outcome
        return rows

    def test_a_clean_set_has_none(self):
        self.assertEqual(build_units(self._rows(), checkpoints=[A, B], min_seeds=2).duplicate_rows, [])

    def test_a_duplicate_is_recorded_rather_than_overwritten(self):
        eligibility = build_units(self._rows(duplicate=True), checkpoints=[A, B], min_seeds=2)
        self.assertEqual(len(eligibility.duplicate_rows), 1)
        self.assertEqual(eligibility.duplicate_rows[0], ("t", A, 0))

    def test_it_blocks_the_comparison(self):
        """Cannot be fixed in the analysis, so it must not produce a number."""
        eligibility = build_units(self._rows(duplicate=True), checkpoints=[A, B], min_seeds=2)
        verdict = evaluate(eligibility, [A, B], resamples=100, seed=1)
        self.assertEqual(verdict.exit_code, 2)
        self.assertEqual(verdict.tasks, [])
        self.assertIn("weights that scenario twice", verdict.blocking[0])
        self.assertIn("Fix the results, do not reinterpret them", verdict.blocking[0])

    def test_compare_checks_for_itself_rather_than_trusting_execute(self):
        """`compare` reads any results directory, including ones Refractal did not write."""
        from refractal.execute import OutputMissingError

        self.assertTrue(issubclass(OutputMissingError, Exception))
        # The two checks are independent on purpose: execute verifies what it
        # wrote, compare verifies what it reads.
        eligibility = build_units(self._rows(duplicate=True), checkpoints=[A, B], min_seeds=2)
        self.assertTrue(eligibility.duplicate_rows)


class TestTheBlockMessageNamesTheChangedFiles(unittest.TestCase):
    """A digest says something moved and leaves the reader to find out what."""

    def _rows(self, surfaces):
        rows = rows_for(success_rate=0.6, salt="sm")
        for i, row in enumerate(rows):
            row["harness_surface"] = surfaces[i % len(surfaces)]
        return rows

    def test_with_manifests_it_names_the_file(self):
        verdict = evaluate(
            build_units(self._rows(["s1", "s2"]), checkpoints=[A, B]), [A, B],
            resamples=100, seed=1,
            surface_manifests={
                "s1": {"orchestrator.py": "a", "runners/live_runner.py": "x"},
                "s2": {"orchestrator.py": "a", "runners/live_runner.py": "y"},
            },
        )
        self.assertEqual(verdict.exit_code, 2)
        self.assertIn("runners/live_runner.py", verdict.blocking[0])
        self.assertIn("before waiving", verdict.blocking[0])

    def test_it_names_what_refractal_assumes_about_the_changed_file(self):
        """A file name is not enough; the assumption at risk is the useful part.

        "The bridge swallows an assignment the base class makes deliberately" is
        not reconstructable from a hash, and it is the first thing a reader needs.
        """
        verdict = evaluate(
            build_units(self._rows(["s1", "s2"]), checkpoints=[A, B]), [A, B],
            resamples=100, seed=1,
            surface_manifests={
                "s1": {"orchestrator.py": "a"},
                "s2": {"orchestrator.py": "b"},
            },
        )
        message = verdict.blocking[0]
        self.assertIn("What Refractal assumes about them", message)
        self.assertIn("ASSIGNED INSIDE `run()`", message)
        self.assertIn("MOST INTRUSIVE ASSUMPTION", message)

    def test_a_file_with_no_recorded_assumptions_still_names_itself(self):
        verdict = evaluate(
            build_units(self._rows(["s1", "s2"]), checkpoints=[A, B]), [A, B],
            resamples=100, seed=1,
            surface_manifests={"s1": {"unknown.py": "a"}, "s2": {"unknown.py": "b"}},
        )
        self.assertIn("unknown.py", verdict.blocking[0])

    def test_without_manifests_it_says_so_rather_than_staying_silent(self):
        verdict = evaluate(
            build_units(self._rows(["s1", "s2"]), checkpoints=[A, B]), [A, B],
            resamples=100, seed=1,
        )
        self.assertIn("did not write one", verdict.blocking[0])


class TestDegenerateArmsAreBlocked(unittest.TestCase):
    """The second, independent guard against a checkpoint compared to itself.

    Measured: two arms drawn from one source give a design effect of 0.000 with
    sd 0.000, where a true null gives 0.973 with sd 0.117. That is roughly eight
    sigma down -- a signature, not a low reading.

    It also corrects the measurement write-up, which said a design effect
    below 1.0 is noise. Below 1 *near 1* is noise; near zero is two arms that are
    the same thing.
    """

    def _rows(self, identical):
        rows = []
        for i in range(40):
            for checkpoint in (A, B):
                for seed in range(3):
                    tag = f"{i}|{seed}" if identical else f"{checkpoint}|{i}|{seed}"
                    rows.append({
                        "scene_id": "s", "scene_hash": "h", "task_id": "t",
                        "task_hash": "th", "scenario_hash": f"s{i}",
                        "checkpoint_id": checkpoint, "seed": seed, "session_id": "x",
                        "harness_version": "v", "harness_surface": "v",
                        "is_infra_failure": False, "success": hash(tag) % 3 > 0,
                    })
        return rows

    def test_independent_arms_are_compared_normally(self):
        verdict = evaluate(
            build_units(self._rows(identical=False), checkpoints=[A, B]), [A, B],
            resamples=200, seed=1,
        )
        self.assertEqual(verdict.blocking, [])

    def test_identical_arms_are_blocked(self):
        verdict = evaluate(
            build_units(self._rows(identical=True), checkpoints=[A, B]), [A, B],
            resamples=200, seed=1,
        )
        self.assertEqual(verdict.exit_code, 2)
        self.assertEqual(verdict.tasks, [])
        self.assertIn("no scenario-level variation", verdict.blocking[0])
        self.assertIn("one model server", verdict.blocking[0])

    def test_it_does_not_fire_on_too_few_scenarios_to_tell(self):
        """Ten scenarios cannot distinguish a signature from a bad draw."""
        rows = [r for r in self._rows(identical=True) if int(r["scenario_hash"][1:]) < 5]
        verdict = evaluate(
            build_units(rows, checkpoints=[A, B]), [A, B], resamples=100, seed=1
        )
        self.assertEqual(verdict.blocking, [])
