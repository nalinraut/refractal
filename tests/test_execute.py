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
            self.assertIn("wrote none", message)
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
