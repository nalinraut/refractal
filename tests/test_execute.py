import datetime as dt
import tempfile
import unittest
from pathlib import Path

from refractal.execute import (
    EPISODES_SCHEMA,
    FakeBenchmark,
    OutputMissingError,
    read_episodes,
    run_local,
)
from refractal.execute.results import comparison_prefix
from refractal.resolve import resolve

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "examples" / "catalog"
HARDWARE = "rtx5090"

SESSION = "0" * 32
LATER_SESSION = "1" * 32


def small_plan():
    """The example catalog at the smoke tier: enough episodes to be real, few enough to be fast."""
    import shutil

    import yaml

    tmp = tempfile.TemporaryDirectory(prefix="refractal-execute-")
    root = Path(tmp.name) / "catalog"
    shutil.copytree(CATALOG, root)
    run = root / "run.yaml"
    doc = yaml.safe_load(run.read_text(encoding="utf-8"))
    doc["run"]["tier"] = "smoke"
    run.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return tmp, root, resolve(root, hardware_profile=HARDWARE)


class ExecuteCase(unittest.TestCase):
    def setUp(self):
        self._catalog_tmp, self.catalog_root, self.plan = small_plan()
        self._results_tmp = tempfile.TemporaryDirectory(prefix="refractal-results-")
        self.results = self._results_tmp.name

    def tearDown(self):
        self._catalog_tmp.cleanup()
        self._results_tmp.cleanup()

    def run_once(self, **kwargs):
        kwargs.setdefault("session_id", SESSION)
        kwargs.setdefault("catalog_root", str(self.catalog_root))
        return run_local(self.plan, self.results, **kwargs)


class TestAReceiptMustSayWhatItMeans(unittest.TestCase):
    """A null `fired_step` with no reason is a mapping failure, and was
    indistinguishable from an episode that outran its trigger.

    Every guard checked that a receipt ARRIVED. None checked it arrived
    populated -- so a receipt that was present, structurally valid and entirely
    null passed every one of them, and a live sweep was needed to notice.
    """

    def _write(self, events):
        from refractal.execute.results import check_receipts

        check_receipts([{"perturbations_fired": events}])

    def test_null_with_a_reason_is_fine(self):
        """An episode that ended at 150 with a trigger at 200. Real, expected,
        and excluded from the curve by `compare` rather than by refusing it."""
        self._write([{"effect": "scale_actuator", "target": "g",
                      "specified_step": 200, "fired_step": None,
                      "reason": "the episode ended at step 150"}])

    def test_null_without_a_reason_is_refused(self):
        """The bug that shipped: field names that did not match the column's,
        so every declared field landed null."""
        from refractal.schema.errors import RefractalError

        with self.assertRaises(RefractalError) as ctx:
            self._write([{"effect": None, "target": None,
                          "specified_step": None, "fired_step": None,
                          "reason": None}])
        self.assertIn("mapping failure", str(ctx.exception))

    def test_a_fired_event_needs_no_reason(self):
        self._write([{"effect": "scale_actuator", "target": "g",
                      "specified_step": 0, "fired_step": 0, "reason": None}])

    def test_an_assigned_perturbation_with_no_entries_is_refused(self):
        """The emptiest receipt: assigned perturbations, and nothing recorded.

        Every per-event rule passes trivially over no events, so the narrowest
        check could not see it. Reachable, and reached: the adapter kept
        mutation events in its own list while wrapper windows lived on the
        timeline, so an episode perturbing only the observation reported
        nothing at all.
        """
        from refractal.schema.errors import RefractalError

        with self.assertRaises(RefractalError) as ctx:
            check = __import__("refractal.execute.results",
                               fromlist=["x"]).check_receipts
            check([{"episode_id": "e", "perturbation_count": 1,
                    "perturbations_fired": []}])
        self.assertIn("receipt is empty", str(ctx.exception))

    def test_an_unperturbed_episode_may_of_course_report_nothing(self):
        check = __import__("refractal.execute.results",
                           fromlist=["x"]).check_receipts
        check([{"episode_id": "e", "perturbation_count": 0,
                "perturbations_fired": []}])

    def test_a_wrapper_that_changed_nothing_is_refused(self):
        """The wrapper's own way of arriving empty.

        Every field populated, a full application count, and the observation
        identical on every one of them -- a transform hooked where its output
        is discarded looks exactly like this, and downstream it reads as a
        level that was measured.

        This also guards the guard's PLACEMENT. A wrapper always has a
        fired_step, so the first version of this check sat after the
        `continue` that skips fired events and could never run. Every test
        passed. This one fails if it moves back.
        """
        from refractal.schema.errors import RefractalError

        with self.assertRaises(RefractalError) as ctx:
            self._write([{"effect": "substitute_observation", "target": None,
                          "specified_step": 0, "fired_step": 0, "reason": None,
                          "before_digest": "aa", "after_digest": "aa",
                          "applications": 200, "applications_changed": 0}])
        self.assertIn("changed the observation on none", str(ctx.exception))

    def test_a_wrapper_identity_on_SOME_steps_is_fine(self):
        """Not every wrapper changes every input, and the first pair being
        equal is ordinary rather than suspicious.

        A rotation convention swap is identity whenever the quaternion already
        lies in the hemisphere it normalizes to. Measured on LIBERO, that is
        most steps -- so a check that demanded a change at step one, or on
        every step, would refuse a correctly applied effect.
        """
        self._write([{"effect": "substitute_observation", "target": None,
                      "specified_step": 0, "fired_step": 0, "reason": None,
                      "before_digest": "aa", "after_digest": "aa",
                      "applications": 200, "applications_changed": 45}])

    def test_an_unperturbed_row_is_untouched(self):
        self._write(None)
        self._write([])

    def test_the_WRITER_refuses_it_not_just_the_helper(self):
        """The link above this one, which the mutation found untested: every
        assertion here called `check_receipts` directly, so deleting the
        writer's call to it broke nothing. The writer is what a run reaches."""
        import tempfile

        from refractal.execute.results import ResultWriter
        from refractal.schema.errors import RefractalError

        row = {name: None for name in
               __import__("refractal.execute.results", fromlist=["x"]).EPISODES_SCHEMA.names}
        row.update({
            "episode_id": "e", "scenario_hash": "s", "base_scenario_hash": "b",
            "scene_id": "sc", "scene_hash": "sh", "task_id": "t",
            "task_hash": "th", "checkpoint_id": "pi0", "seed": 0,
            "session_id": "x", "worker_id": "w", "execution_mode": "serial",
            "success": True, "is_infra_failure": False, "steps": 1,
            "elapsed_sec": 1.0,
            "perturbations_fired": [{"effect": None, "target": None,
                                     "specified_step": None, "fired_step": None,
                                     "reason": None}],
        })
        with tempfile.TemporaryDirectory() as tmp:
            writer = ResultWriter(f"{tmp}/out", "sha256:deadbeef")
            with self.assertRaises(RefractalError):
                writer.write_episodes("pi0", "sc", [row], part="w-0")


class TestARealRunWritesTheCount(ExecuteCase):
    """A null count must only ever mean `the row predates the column`.

    If the runner left it unset, a fresh unperturbed episode and an old row
    would be indistinguishable -- and the old rows are the ones phase 1 made
    valid baselines. The distinction is only worth having if the runner is
    disciplined about it, so that is what this checks: through the real writer,
    read back from the Parquet it produced.
    """

    def test_an_unperturbed_run_records_zero_not_null(self):
        self.run_once()
        table = read_episodes(str(self.results), self.plan.plan_id)
        counts = set(table.column("perturbation_count").to_pylist())
        self.assertEqual(counts, {0}, "the runner must write 0, never leave it null")
        levels = set(table.column("perturbation_level").to_pylist())
        self.assertEqual(levels, {None}, "and no level, since nothing was perturbed")


class TestThePerturbationReceiptType(unittest.TestCase):
    """The receipt column, written and read back through Parquet.

    Asserted against the artifact rather than in memory, because the question
    is whether the declared type can hold what the runner will put in it --
    which an in-memory dict cannot answer.

    The column is empty today. Once a sweep writes to it the type is permanent,
    so the shape is settled first.
    """

    def _round_trip(self, events, level, count=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from refractal.execute.results import EPISODES_SCHEMA

        row = {name: None for name in EPISODES_SCHEMA.names}
        row.update({
            "episode_id": "sha256:e", "scenario_hash": "sha256:s",
            "base_scenario_hash": "sha256:b", "scene_id": "sc",
            "scene_hash": "sha256:sc", "task_id": "t", "task_hash": "sha256:t",
            "checkpoint_id": "pi0", "seed": 0, "session_id": "sess",
            "worker_id": "w", "execution_mode": "concurrent", "success": True,
            "is_infra_failure": False, "steps": 271, "elapsed_sec": 4.2,
            "perturbations_fired": events, "perturbation_level": level,
            "perturbation_count": count,
        })
        table = pa.Table.from_pylist([row], schema=EPISODES_SCHEMA)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.parquet"
            pq.write_table(table, path, compression="zstd")
            return pq.read_table(path).to_pylist()[0]

    @staticmethod
    def _event(**kw):
        base = {"effect": "scale_actuator", "target": "gripper0_finger1",
                "specified_step": 200, "fired_step": 200, "reason": None,
                "before": [20.0], "after": [10.0]}
        base.update(kw)
        return base

    def test_two_events_keep_their_own_values(self):
        """The case parallel columns cannot hold: which value belongs to which
        step. A struct per event keeps each one's facts together."""
        back = self._round_trip([
            self._event(specified_step=200, fired_step=200, before=[20.0], after=[10.0]),
            self._event(specified_step=250, fired_step=250, before=[10.0], after=[6.0]),
        ], 0.3)
        self.assertEqual(
            [(e["specified_step"], e["before"], e["after"])
             for e in back["perturbations_fired"]],
            [(200, [20.0], [10.0]), (250, [10.0], [6.0])],
        )

    def test_an_unfired_spec_rides_in_the_same_column(self):
        """So one column answers `did everything fire`, rather than a join."""
        back = self._round_trip([
            self._event(fired_step=None, reason="the episode ended at step 271",
                        after=None),
        ], None)
        (event,) = back["perturbations_fired"]
        self.assertIsNone(event["fired_step"])
        self.assertIn("ended", event["reason"])

    def test_before_and_after_hold_a_scalar_and_a_wrench(self):
        """scale_actuator reads one limit; apply_force reads six numbers out of
        xfrc_applied. One column, lengths 1 and 6, no column per effect."""
        back = self._round_trip([
            self._event(before=[20.0], after=[10.0]),
            self._event(effect="apply_force", target="bowl",
                        before=[0.0] * 6, after=[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        ], 0.5)
        lengths = [len(e["after"]) for e in back["perturbations_fired"]]
        self.assertEqual(lengths, [1, 6])

    def test_the_level_is_a_number_and_not_in_the_struct(self):
        """It is the grouping key: a curve groups on a plain value rather than
        reaching into an audit record. A string would sort `"20.0"` next to
        `"3.0"` and stop grouping without saying so."""
        back = self._round_trip([self._event()], 0.3)
        self.assertIsInstance(back["perturbation_level"], float)
        self.assertLess(back["perturbation_level"], 0.5)
        self.assertNotIn("level", back["perturbations_fired"][0])

    def test_an_unperturbed_episode_carries_nulls(self):
        """Every episode recorded so far, and every one in the unperturbed arm
        of a sweep."""
        back = self._round_trip(None, None)
        self.assertIsNone(back["perturbations_fired"])
        self.assertIsNone(back["perturbation_level"])

    def test_the_count_separates_two_meanings_of_a_null_level(self):
        """A null level means opposite things, and `compare` must tell them
        apart: an unperturbed episode is the BASELINE the curve is measured
        against, and a two-perturbation episode has no single level to place
        anywhere. Without the count both are the same null.
        """
        baseline = self._round_trip(None, None, count=0)
        single = self._round_trip([self._event()], 0.3, count=1)
        ambiguous = self._round_trip(
            [self._event(target="a"), self._event(target="b")], None, count=2)

        self.assertIsNone(baseline["perturbation_level"])
        self.assertIsNone(ambiguous["perturbation_level"])
        # ... and they are still distinguishable.
        self.assertEqual(baseline["perturbation_count"], 0)
        self.assertEqual(ambiguous["perturbation_count"], 2)
        self.assertEqual(single["perturbation_count"], 1)
        self.assertEqual(single["perturbation_level"], 0.3)

    def test_an_unperturbed_episode_cannot_just_carry_a_neutral_level(self):
        """Why the count exists rather than writing a neutral level.

        Neutral depends on the effect -- 1.0 for scale_actuator, 0.0 for
        apply_force -- so there is no one value an unperturbed row could carry
        that means `no perturbation` on every axis. Where the baseline sits
        comes from the effect's definition, which means `compare` has to know
        it is looking at a baseline.
        """
        from refractal.perturbations import effects

        self.assertIn("scale_actuator", effects())
        self.assertIn("apply_force", effects())
        # Two effects whose neutral values differ, in one registry. A single
        # neutral written into the row would be wrong for one of them.


class TestSchemaAndLayout(ExecuteCase):
    def test_every_episode_lands_exactly_once(self):
        summary = self.run_once()
        self.assertEqual(summary.written, self.plan.total_episodes)
        table = read_episodes(self.results, self.plan.plan_id)
        self.assertEqual(table.num_rows, self.plan.total_episodes)
        ids = table.column("episode_id").to_pylist()
        self.assertEqual(len(set(ids)), len(ids))

    def test_schema_matches_the_declared_one(self):
        # Declared, not inferred: a column that happens to be all-null in one
        # worker's output would otherwise land as null-typed and refuse to merge.
        self.run_once()
        self.assertEqual(read_episodes(self.results, self.plan.plan_id).schema, EPISODES_SCHEMA)

    def test_partitioned_by_comparison_then_checkpoint_then_scene(self):
        self.run_once()
        prefix = Path(comparison_prefix(self.results, self.plan.plan_id))
        self.assertTrue(prefix.is_dir())
        self.assertTrue((prefix / "plan.json").is_file())
        # Comparison is the partition unit: the common query joins across
        # checkpoints, and partitioning by run would force a cross-directory read.
        for checkpoint in self.plan.checkpoints:
            self.assertTrue((prefix / f"checkpoint={checkpoint.id}").is_dir())

    def test_catalog_travels_with_the_results(self):
        self.run_once()
        prefix = Path(comparison_prefix(self.results, self.plan.plan_id))
        for name in ("scenes.yaml", "tasks.yaml", "scenarios.yaml", "run.yaml"):
            self.assertTrue((prefix / "catalog" / name).is_file(), name)
        self.assertTrue((prefix / "catalog" / "assets" / "vial_rack.xml").is_file())

    def test_phase_outcomes_survive_two_tasks_with_different_phases(self):
        """The struct-vs-map fix, exercised.

        `vial-slot-4` declares four phases and `vial-slot-7` declares none. Both
        write into the same scene partition; a fixed struct has one schema per
        file and would reject the second.
        """
        self.run_once()
        table = read_episodes(self.results, self.plan.plan_id)
        by_task = {}
        for row in table.to_pylist():
            by_task.setdefault(row["task_id"], []).append(row["phase_outcomes"])
        self.assertIn("vial-slot-4", by_task)
        self.assertIn("vial-slot-7", by_task)


class TestFailureSemantics(ExecuteCase):
    def test_failure_reason_is_null_exactly_when_successful(self):
        # The API reference says null means a policy failure -- but null is also
        # what a success writes, so the two would be indistinguishable.
        self.run_once(benchmark=FakeBenchmark(success_rate=0.5))
        for row in read_episodes(self.results, self.plan.plan_id).to_pylist():
            if row["success"]:
                self.assertIsNone(row["failure_reason"])
            else:
                self.assertIsNotNone(row["failure_reason"])

    def test_infra_failure_is_its_own_column(self):
        self.run_once(benchmark=FakeBenchmark(success_rate=0.7, infra_failure_rate=0.2))
        rows = read_episodes(self.results, self.plan.plan_id).to_pylist()
        infra = [r for r in rows if r["is_infra_failure"]]
        self.assertTrue(infra, "expected some injected infra failures")
        for row in infra:
            # `compare` must be able to drop these without pattern-matching a
            # string, so the flag and the reason are separate facts.
            self.assertFalse(row["success"])
            self.assertEqual(row["failure_reason"], "worker_crashed")


class TestResume(ExecuteCase):
    def test_second_run_skips_completed_episodes(self):
        first = self.run_once()
        second = self.run_once(session_id=LATER_SESSION)
        self.assertEqual(second.written, 0)
        self.assertEqual(second.skipped, first.written)
        self.assertEqual(
            read_episodes(self.results, self.plan.plan_id).num_rows, self.plan.total_episodes
        )

    def test_resume_after_partial_completion(self):
        """Interrupt, resume, and require the result to be what an uninterrupted run gives.

        Outcomes are a deterministic function of `episode_id`, so this asserts
        correctness rather than merely that nothing crashed.
        """
        partial = self.plan.model_copy(
            update={"scenes": [self.plan.scenes[0].model_copy(
                update={"workers": self.plan.scenes[0].workers[:1]}
            )]}
        )
        run_local(
            partial, self.results, session_id=SESSION, catalog_root=str(self.catalog_root)
        )
        done_after_partial = read_episodes(self.results, self.plan.plan_id).num_rows
        self.assertLess(done_after_partial, self.plan.total_episodes)

        summary = self.run_once(session_id=LATER_SESSION)
        self.assertEqual(summary.skipped, done_after_partial)
        table = read_episodes(self.results, self.plan.plan_id)
        self.assertEqual(table.num_rows, self.plan.total_episodes)

        # Same episodes, same outcomes as a single clean run.
        with tempfile.TemporaryDirectory() as clean_dir:
            run_local(self.plan, clean_dir, session_id=SESSION)
            clean = read_episodes(clean_dir, self.plan.plan_id)
        resumed_outcomes = {r["episode_id"]: r["success"] for r in table.to_pylist()}
        clean_outcomes = {r["episode_id"]: r["success"] for r in clean.to_pylist()}
        self.assertEqual(resumed_outcomes, clean_outcomes)

    def test_session_id_is_recorded_per_episode(self):
        """A resumed comparison spans sessions; `compare` has to be able to see that."""
        partial = self.plan.model_copy(
            update={"scenes": [self.plan.scenes[0].model_copy(
                update={"workers": self.plan.scenes[0].workers[:1]}
            )]}
        )
        run_local(partial, self.results, session_id=SESSION)
        self.run_once(session_id=LATER_SESSION)
        sessions = set(
            read_episodes(self.results, self.plan.plan_id).column("session_id").to_pylist()
        )
        self.assertEqual(sessions, {SESSION, LATER_SESSION})


class TestAtomicity(ExecuteCase):
    def test_no_staging_files_survive(self):
        self.run_once()
        prefix = Path(comparison_prefix(self.results, self.plan.plan_id))
        leftovers = list(prefix.rglob(".tmp-*"))
        self.assertEqual(leftovers, [], "a partial write was left where a reader could see it")

    def test_parts_are_named_by_worker(self):
        self.run_once()
        prefix = Path(comparison_prefix(self.results, self.plan.plan_id))
        parts = list(prefix.rglob("part-*.parquet"))
        self.assertTrue(parts)
        for part in parts:
            self.assertNotIn("/", part.name.replace("part-", "", 1).split("-")[0])


class TestKnownGroundTruth(ExecuteCase):
    def test_a_constructed_difference_is_recoverable(self):
        """The fixture that makes step 3 testable at all.

        `compare` can only be trusted if it is checked against data whose answer
        is known in advance, and this is the only source where that is possible.
        """
        benchmark = FakeBenchmark(
            success_rate=0.5,
            per_task={
                ("ckpt-46", "vial-slot-4"): 0.30,
                ("ckpt-47", "vial-slot-4"): 0.90,
            },
        )
        self.run_once(benchmark=benchmark)
        rows = read_episodes(self.results, self.plan.plan_id).to_pylist()

        def rate(checkpoint, task):
            subset = [r for r in rows if r["checkpoint_id"] == checkpoint and r["task_id"] == task]
            return sum(r["success"] for r in subset) / len(subset)

        self.assertLess(rate("ckpt-46", "vial-slot-4"), 0.5)
        self.assertGreater(rate("ckpt-47", "vial-slot-4"), 0.7)
        # And unchanged where nothing was changed.
        self.assertAlmostEqual(
            rate("ckpt-46", "vial-slot-7"), rate("ckpt-47", "vial-slot-7"), delta=0.25
        )


if __name__ == "__main__":
    unittest.main()


class TestAWorkerThatWritesNothingIsCaught(ExecuteCase):
    """A run that completes, reports success and writes nothing.

    Indistinguishable from a correct run until someone tries to compare, by which
    point the compute is spent. The concrete route is the harness's
    `_build_recorder`, which returns NullEpisodeRecorder whenever `self._store is
    None` — override the recorder and not the store and every episode runs, every
    episode succeeds, and nothing is recorded.

    The guard is written against the symptom rather than that cause, so these
    tests break the writer in ways that have nothing to do with `_store`.
    """

    def test_a_writer_that_silently_discards_everything(self):
        """The null-recorder shape, without needing the harness to reproduce it."""
        import refractal.execute.local as local_mod

        class SilentWriter(local_mod.ResultWriter):
            def write_episodes(self, checkpoint_id, scene_id, rows, *, part):
                return None  # accepted, wrote nothing

        original = local_mod.ResultWriter
        local_mod.ResultWriter = SilentWriter
        try:
            with self.assertRaises(OutputMissingError) as ctx:
                self.run_once()
            message = str(ctx.exception)
            self.assertIn("wrote no", message)
            # The message must point at the gate, not just at the count.
            self.assertIn("recorder that was never activated", message)
        finally:
            local_mod.ResultWriter = original

    def test_a_writer_that_drops_some_rows(self):
        """Partial output is the worse case: it shrinks a denominator silently."""
        import refractal.execute.local as local_mod

        class LossyWriter(local_mod.ResultWriter):
            def write_episodes(self, checkpoint_id, scene_id, rows, *, part):
                return super().write_episodes(checkpoint_id, scene_id, rows[:-1], part=part)

        original = local_mod.ResultWriter
        local_mod.ResultWriter = LossyWriter
        try:
            with self.assertRaises(OutputMissingError) as ctx:
                self.run_once()
            self.assertIn("reached storage", str(ctx.exception))
            self.assertIn("shrinks a denominator", str(ctx.exception))
        finally:
            local_mod.ResultWriter = original

    def test_a_healthy_run_passes_the_check(self):
        summary = self.run_once()
        self.assertEqual(summary.written, self.plan.total_episodes)

    def test_a_fully_resumed_worker_does_not_trip_it(self):
        """Nothing attempted means nothing expected: skipping is not a shortfall."""
        self.run_once()
        second = self.run_once(session_id=LATER_SESSION)
        self.assertEqual(second.written, 0)
        self.assertGreater(second.skipped, 0)

    def test_an_id_that_was_never_planned(self):
        """episode_id comes from the plan; a writer must not invent one.

        Matters because the bridge is the first place ids are derived from a
        benchmark we did not write. An id not in the plan means the scene, task,
        scenario, seed or checkpoint did not survive the round trip.
        """
        import refractal.execute.local as local_mod

        class ForgetfulWriter(local_mod.ResultWriter):
            def write_episodes(self, checkpoint_id, scene_id, rows, *, part):
                rows = [dict(r) for r in rows]
                rows[0]["episode_id"] = "sha256:" + "ff" * 32
                return super().write_episodes(checkpoint_id, scene_id, rows, part=part)

        original = local_mod.ResultWriter
        local_mod.ResultWriter = ForgetfulWriter
        try:
            with self.assertRaises(OutputMissingError) as ctx:
                self.run_once()
            # Reported as missing, because the planned id is absent -- which is
            # the more actionable half of the same fact.
            self.assertIn("missing", str(ctx.exception))
        finally:
            local_mod.ResultWriter = original

    def test_a_duplicated_row_is_caught(self):
        """Invisible to a set comparison, and it double-weights a scenario."""
        import refractal.execute.local as local_mod

        class DoublingWriter(local_mod.ResultWriter):
            def write_episodes(self, checkpoint_id, scene_id, rows, *, part):
                return super().write_episodes(
                    checkpoint_id, scene_id, list(rows) + [dict(rows[0])], part=part
                )

        original = local_mod.ResultWriter
        local_mod.ResultWriter = DoublingWriter
        try:
            with self.assertRaises(OutputMissingError) as ctx:
                self.run_once()
            message = str(ctx.exception)
            self.assertIn("duplicate", message)
            self.assertIn("weights that scenario twice", message)
        finally:
            local_mod.ResultWriter = original

    def test_a_purely_extra_id_is_caught_as_unexpected(self):
        """Nothing planned goes missing, so only the unexpected branch can fire."""
        import refractal.execute.local as local_mod

        class ChattyWriter(local_mod.ResultWriter):
            def write_episodes(self, checkpoint_id, scene_id, rows, *, part):
                extra = dict(rows[0])
                extra["episode_id"] = "sha256:" + "ee" * 32
                return super().write_episodes(
                    checkpoint_id, scene_id, list(rows) + [extra], part=part
                )

        original = local_mod.ResultWriter
        local_mod.ResultWriter = ChattyWriter
        try:
            with self.assertRaises(OutputMissingError) as ctx:
                self.run_once()
            self.assertIn("never planned", str(ctx.exception))
        finally:
            local_mod.ResultWriter = original


class TestSurfaceManifests(ExecuteCase):
    """The digest gates; the manifest makes its verdict actionable.

    Moving the pin from 35f1200 to v0.6.0 moved the surface digest for a
    five-line change in `runners/live_runner.py`, which Refractal does not use.
    The right response was not to narrow the digest -- `runners/` is on the
    surface because async execution would make that file load-bearing at once --
    but to make the block name the file.
    """

    def test_a_manifest_round_trips(self):
        from refractal.execute import ResultWriter

        writer = ResultWriter(self.results, self.plan.plan_id)
        manifest = {"orchestrator.py": "aaaa", "runners/sync_runner.py": "bbbb"}
        writer.record_harness_manifest("surface-1", manifest)
        self.assertEqual(writer.read_harness_manifests(), {"surface-1": manifest})

    def test_recording_the_same_surface_twice_is_idempotent(self):
        """Written once per surface, not once per session."""
        from refractal.execute import ResultWriter

        writer = ResultWriter(self.results, self.plan.plan_id)
        writer.record_harness_manifest("surface-1", {"a.py": "1"})
        writer.record_harness_manifest("surface-1", {"a.py": "1"})
        self.assertEqual(len(writer.read_harness_manifests()), 1)

    def test_an_empty_manifest_writes_nothing(self):
        """The local backend has no harness, so there is nothing honest to record."""
        from refractal.execute import ResultWriter

        writer = ResultWriter(self.results, self.plan.plan_id)
        writer.record_harness_manifest("none/local-backend", {})
        self.assertEqual(writer.read_harness_manifests(), {})

    def test_the_diff_names_the_changed_file(self):
        from refractal.execute.harness import compare_manifests

        self.assertEqual(
            compare_manifests(
                {"orchestrator.py": "a", "runners/live_runner.py": "x"},
                {"orchestrator.py": "a", "runners/live_runner.py": "y"},
            ),
            {"changed": ["runners/live_runner.py"], "added": [], "removed": []},
        )


class TestWhatTheEpisodeActuallyReceived(unittest.TestCase):
    """`perturbation_level` is what the catalog asked for.
    `perturbation_exposure` is what the receipt says happened.

    They come apart for a transform whose effect is trajectory-dependent. Two
    rotation conventions agree exactly over half of all orientations, so an
    episode is perturbed only on the steps its trajectory spends in the half
    where they differ -- measured on LIBERO from a fixed action sequence, 2.5%
    to 15% of steps across start states alone.

    That is the `at_step: 0` problem arriving from geometry rather than timing,
    and unlike that one no declaration fixes it: nothing anyone writes in a
    catalog controls where a trajectory goes. Recording it is the whole
    remedy -- a result reads "N points at 12% exposure", and two checkpoints
    with different exposures are not read as having had the same treatment.
    """

    def exposure(self, receipt):
        from refractal.execute.vla_eval_runner import _measured_exposure

        return _measured_exposure(receipt)

    @staticmethod
    def _wrapper(applications=80, changed=12):
        return {"effect": "substitute_observation", "target": "states",
                "specified_step": 0, "fired_step": 0, "reason": None,
                "before_digest": "aa", "after_digest": "bb",
                "applications": applications, "applications_changed": changed}

    @staticmethod
    def _mutation():
        return {"effect": "scale_actuator", "target": "g", "specified_step": 0,
                "fired_step": 0, "reason": None, "before": [20.0],
                "after": [10.0]}

    def test_it_is_the_fraction_of_applications_that_changed(self):
        self.assertAlmostEqual(self.exposure([self._wrapper(80, 12)]), 0.15)

    def test_full_exposure_is_one(self):
        self.assertEqual(self.exposure([self._wrapper(40, 40)]), 1.0)

    def test_an_unperturbed_episode_has_none(self):
        self.assertIsNone(self.exposure(None))
        self.assertIsNone(self.exposure([]))

    def test_a_state_mutation_has_none_rather_than_a_constant(self):
        """It perturbs every step it applies to by construction, so a ratio
        would be 1.0 wearing the costume of a measurement."""
        self.assertIsNone(self.exposure([self._mutation()]))

    def test_two_perturbations_have_none(self):
        """The same rule the level follows: no single number to report."""
        self.assertIsNone(
            self.exposure([self._wrapper(), self._wrapper()]))
        self.assertIsNone(
            self.exposure([self._wrapper(), self._mutation()]))

    def test_zero_exposure_is_zero_and_not_absent(self):
        """An episode whose trajectory never entered the half where the
        conventions differ received nothing, and that is a measurement rather
        than a missing value. `check_receipts` refuses to write it, so this
        pins the arithmetic rather than the policy."""
        self.assertEqual(self.exposure([self._wrapper(80, 0)]), 0.0)


class TestCopyingACatalogThatContainsItsOwnResults(unittest.TestCase):
    """`-o ./results --catalog .` is the obvious way to run from a catalog
    directory, and it did not terminate.

    `os.walk` is lazy, so writing the provenance copy into the tree being
    walked makes it descend into what it has just created. The catalog nests
    inside itself once per level until the path exceeds the filesystem limit.

    What surfaced was `OSError: File name too long`, which reads as a
    path-length problem -- a wrong diagnosis that points at the digest in the
    directory name rather than at a copy that was never going to stop.
    """

    def _catalog(self, tmp):
        import os

        os.makedirs(f"{tmp}/cat/assets", exist_ok=True)
        with open(f"{tmp}/cat/scenes.yaml", "w") as fh:
            fh.write("scenes: []\n")
        with open(f"{tmp}/cat/assets/thing.xml", "w") as fh:
            fh.write("<mujoco/>\n")
        return f"{tmp}/cat"

    def _writer(self, results_uri):
        """With rows already written, which is what puts the results tree in
        the walk's listing.

        A first version of this created the writer and copied immediately. The
        results directory did not exist yet, so `os.walk` never saw it and the
        fixture could not reproduce the bug -- both mutations of the fix
        survived against it while the tests stayed green.

        Provenance is copied last in a real run, after the rows. The fixture
        has to be in that state or it is testing a different situation.
        """
        import os

        from refractal.execute.results import ResultWriter

        writer = ResultWriter(results_uri, "sha256:deadbeef")
        os.makedirs(f"{writer.prefix}/checkpoint=pi0/scene=s/episodes",
                    exist_ok=True)
        with open(f"{writer.prefix}/checkpoint=pi0/scene=s/episodes/part.parquet",
                  "w") as fh:
            fh.write("rows\n")
        return writer

    def test_it_terminates_and_copies_the_catalog(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            catalog = self._catalog(tmp)
            writer = self._writer(f"{catalog}/results")   # INSIDE the catalog
            writer.copy_catalog(catalog)
            copied = writer.fs.glob(f"{writer.prefix}/catalog/**")
            names = {p.rsplit("/", 1)[-1] for p in copied}
            self.assertIn("scenes.yaml", names)
            self.assertIn("thing.xml", names)

    def test_the_results_tree_is_not_copied_into_itself(self):
        """Not merely terminating -- the provenance copy must not contain a
        half-written copy of the results it is provenance for."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            catalog = self._catalog(tmp)
            writer = self._writer(f"{catalog}/results")
            writer.copy_catalog(catalog)
            marker = f"{writer.prefix}/catalog/"
            inside = [p[len(marker):] for p in writer.fs.glob(f"{marker}**")
                      if p.startswith(marker)]
            self.assertFalse(
                [p for p in inside if "comparison_id=" in p or p.startswith("results")],
                f"the results tree came along for the ride: {inside}",
            )
            self.assertEqual(sorted(inside), ["assets", "assets/thing.xml",
                                              "scenes.yaml"])

    def test_a_catalog_beside_the_results_still_copies_whole(self):
        """The ordinary case must not be pruned by the fix."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            catalog = self._catalog(tmp)
            writer = self._writer(f"{tmp}/elsewhere")
            writer.copy_catalog(catalog)
            names = {p.rsplit("/", 1)[-1]
                     for p in writer.fs.glob(f"{writer.prefix}/catalog/**")}
            self.assertIn("scenes.yaml", names)
            self.assertIn("thing.xml", names)
