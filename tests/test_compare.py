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

    SALT = "n8"  # pinned: a draw where the wrong tests declare a finding

    def setUp(self):
        rows = rows_for(
            success_rate=0.5,
            infra_failure_rate=0.03,
            scenario_spread=0.30,
            interaction_spread=0.35,
            salt=self.SALT,
            # No injected difference: any "finding" here is a false positive.
        )
        self.units = build_units(rows, checkpoints=[A, B]).units

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
        rows = rows_for(
            success_rate=0.5,
            infra_failure_rate=0.03,
            scenario_spread=0.30,
            salt="b0",
            per_task={(B, "vial-slot-7"): 0.57},
        )
        self.units = build_units(rows, checkpoints=[A, B]).units

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
        """The disagreement case, resolved in the bootstrap's favour."""
        verdict = self._verdict(-0.06, salt="r3")
        contrast = verdict.tasks[0].contrasts[0]
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
        """rates: {checkpoint: success_rate} for one task."""
        out = []
        for checkpoint, rate in rates.items():
            for row in rows_for(
                success_rate=0.5,
                scenario_spread=0.30,
                salt="m0",
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
                success_rate=0.5, scenario_spread=0.30, salt="u0",
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

    Every number in statistics-measurements.md was measured against
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
        """The comparison that makes a correction to the doc statable."""
        for interaction in (0.20, 0.35):
            report = self._measure(0.0, interaction)
            modelled = modelled_variance(
                scenario_spread=0.0, interaction_spread=interaction
            )["expected_difference_var"]
            measured = report.observed_var - report.binomial_var
            # Same order of magnitude and same direction; this is a sanity bound,
            # not a precision claim at 119 scenarios.
            self.assertGreater(measured, 0.4 * modelled)
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
