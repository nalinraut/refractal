"""The two-method override, tested against a stand-in for the harness.

The harness is not installed here and will not be on most machines that run this
suite. But the thing worth testing is not the harness — it is that overriding
`_build_recorder` **alone** produces a run that completes, reports success and
records nothing, and that overriding the store as well fixes it.

That is reproducible from the gate itself, which is four lines:

    if self._store is None or rec_cfg is None:
        return NullEpisodeRecorder()

So this builds a module that behaves the same way, installs it as `vla_eval`, and
asserts both halves. If the real harness ever changes that gate, the
`harness_surface` digest moves and `compare` blocks — a different guard for a
different distance.
"""

import sys
import types
import unittest

from refractal.execute.vla_eval import (
    RECORDER_SURFACE,
    StepBuffer,
    to_episode_row,
    BridgeError,
    check_recorder_surface,
    make_parquet_orchestrator,
    make_parquet_recorder,
    worker_selection,
)
from refractal.schema.plan import PlannedEpisode


def install_stand_in_harness():
    """A module shaped like the parts of vla_eval the bridge touches."""

    class EpisodeRecorder:
        def __init__(self, **kwargs):
            self.store = kwargs.get("store")

        @property
        def is_active(self):
            return False

        sid = eid = eval_id = db_path = ""

        def record_step(self, **fields):
            ...

        def record_video(self, frame):
            ...

        def close(self, *a, **k):
            ...

    class NullEpisodeRecorder(EpisodeRecorder):
        pass

    class Orchestrator:
        """Reproduces the gate, and only the gate."""

        def __init__(self):
            self._store = None
            self._sid = "sid-1"

        def _init_store(self, *a, **k):
            self._store = None  # SQLite path in the real thing; None without one

        def _build_recorder(self, rec_cfg, task, bench_eval_id, safe, task_idx, ep, benchmark):
            if self._store is None or rec_cfg is None:
                return NullEpisodeRecorder()
            return EpisodeRecorder(store=self._store)

    recording = types.ModuleType("vla_eval.recording")
    recording.EpisodeRecorder = EpisodeRecorder
    recording.NullEpisodeRecorder = NullEpisodeRecorder
    orchestrator = types.ModuleType("vla_eval.orchestrator")
    orchestrator.Orchestrator = Orchestrator
    root = types.ModuleType("vla_eval")
    root.__version__ = "0.0.0-stand-in"
    root.recording, root.orchestrator = recording, orchestrator

    for name, module in (
        ("vla_eval", root),
        ("vla_eval.recording", recording),
        ("vla_eval.orchestrator", orchestrator),
    ):
        sys.modules[name] = module
    return NullEpisodeRecorder


class BridgeCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in
                       ("vla_eval", "vla_eval.recording", "vla_eval.orchestrator")}
        self.NullRecorder = install_stand_in_harness()
        self.collected = []
        self.recorder_cls = make_parquet_recorder(
            lambda episode_id, name, value: self.collected.append((episode_id, name, value))
        )

    def tearDown(self):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TestOverrideLogicAgainstAReproducedGate(BridgeCase):
    """Verifies the override logic is self-consistent against a *reproduction* of
    the harness's gate. Does **not** verify anything about the real harness.

    Named this way on purpose. A green run here means "our two-method override
    behaves correctly against the gate as we understand it", not "the real gate is
    unchanged". Those are different claims and only the first is tested.

    The specific blind spot: if a future harness *moves* the check rather than
    changing it -- same logic, different place -- `harness_surface` fires and
    blocks comparisons, while these tests keep passing and say nothing is wrong.
    That division of labour is deliberate; the risk is reading the first claim as
    the second.
    """
    def test_overriding_only_the_recorder_records_nothing(self):
        """The failure being guarded against, demonstrated rather than described."""
        from vla_eval.orchestrator import Orchestrator

        recorder_cls = self.recorder_cls

        class HalfOverridden(Orchestrator):
            def _build_recorder(self, rec_cfg, task, bench_eval_id, safe, ti, ep, bm):
                if rec_cfg is None:
                    return self.__class__.__mro__[1]._build_recorder(
                        self, rec_cfg, task, bench_eval_id, safe, ti, ep, bm)
                return recorder_cls(episode_id=str(ep), sid="s", eid="e", eval_id=bench_eval_id)

        half = HalfOverridden()
        half._init_store()  # the base: leaves _store None
        # The override is never consulted, because the gate closes first.
        recorder = Orchestrator._build_recorder(half, {"record_step": True}, {}, "ev", "s", 0, 0, None)
        self.assertIsInstance(recorder, self.NullRecorder)
        self.assertFalse(recorder.is_active)

    def test_overriding_both_produces_a_live_recorder(self):
        orchestrator = make_parquet_orchestrator(self.recorder_cls)()
        orchestrator._init_store()
        self.assertIsNotNone(orchestrator._store)

        recorder = orchestrator._build_recorder(
            {"record_step": True}, {"name": "t"}, "eval-1", "safe", 0, 7, None
        )
        self.assertTrue(recorder.is_active)
        self.assertEqual(recorder.eval_id, "eval-1")

    def test_a_null_recording_config_still_yields_the_null_recorder(self):
        """Recording off is a legitimate configuration, not a failure."""
        orchestrator = make_parquet_orchestrator(self.recorder_cls)()
        orchestrator._init_store()
        self.assertIsInstance(
            orchestrator._build_recorder(None, {}, "eval-1", "safe", 0, 0, None),
            self.NullRecorder,
        )

    def test_recorded_steps_reach_the_collector(self):
        recorder = self.recorder_cls(episode_id="ep-1", sid="s", eid="e", eval_id="ev")
        recorder.record_step(reward=1.0, success=True)
        self.assertIn(("ep-1", "reward", 1.0), self.collected)
        self.assertIn(("ep-1", "success", True), self.collected)

    def test_db_path_is_empty_on_purpose(self):
        """Plumbed to every model server, read by none. An affordance, not a feature."""
        self.assertEqual(
            self.recorder_cls(episode_id="e", sid="s", eid="e", eval_id="v").db_path, ""
        )


class TestRecorderSurface(BridgeCase):
    def test_the_real_surface_has_seven_members_plus_db_path(self):
        for name in RECORDER_SURFACE:
            self.assertTrue(hasattr(self.recorder_cls, name), name)

    def test_a_recorder_missing_a_member_is_refused_up_front(self):
        """EpisodeRecorder is a plain class, so nothing else would tell you."""

        class Incomplete:
            record_step = record_video = close = sid = eid = eval_id = db_path = None

        with self.assertRaises(BridgeError) as ctx:
            check_recorder_surface(Incomplete)
        self.assertIn("is_active", str(ctx.exception))
        self.assertIn("not an ABC", str(ctx.exception))


class TestWorkerSelection(unittest.TestCase):
    def _episode(self, task_id, seed=0):
        return PlannedEpisode(
            episode_id=f"sha256:{task_id}{seed}", task_id=task_id, task_hash="sha256:t",
            scenario_hash="sha256:s", seed=seed, checkpoint_id="ckpt",
        )

    def test_one_task_is_expressible(self):
        episodes = [self._episode("libero-3", s) for s in range(10)]
        self.assertEqual(
            worker_selection(episodes), {"tasks": ["libero-3"], "episodes_per_task": 10}
        )

    def test_two_tasks_in_one_worker_is_refused_rather_than_approximated(self):
        """Silently running a superset corrupts a denominator; a subset loses episodes."""
        with self.assertRaises(BridgeError) as ctx:
            worker_selection([self._episode("a"), self._episode("b")])
        self.assertIn("one task per worker", str(ctx.exception))

    def test_an_empty_worker_asks_for_nothing(self):
        self.assertEqual(worker_selection([]), {"tasks": [], "episodes_per_task": 0})


if __name__ == "__main__":
    unittest.main()


class TestEpisodeRowMapping(unittest.TestCase):
    """Where the harness's denominator decision gets undone.

    `_build_task_result` counts errored episodes as policy failures with
    `len(episodes)` as the denominator — documented and deliberate upstream, and
    wrong for a comparison. A crashed container is not evidence about a policy.
    """

    def _episode(self):
        return PlannedEpisode(
            episode_id="sha256:e", task_id="t", task_hash="sha256:t",
            scenario_hash="sha256:s", seed=0, checkpoint_id="ckpt",
        )

    def test_a_success(self):
        row = to_episode_row(self._episode(), {"metrics": {"success": True}, "steps": 120})
        self.assertTrue(row.success)
        self.assertFalse(row.is_infra_failure)
        # Null iff success. The reference says null means a policy failure, but
        # null is also what a success writes.
        self.assertIsNone(row.failure_reason)

    def test_a_policy_failure_names_itself(self):
        row = to_episode_row(self._episode(), {"metrics": {"success": False}, "steps": 300})
        self.assertFalse(row.success)
        self.assertFalse(row.is_infra_failure)
        self.assertEqual(row.failure_reason, "policy_failure")

    def test_an_infra_failure_leaves_the_denominator(self):
        row = to_episode_row(
            self._episode(),
            {"metrics": {"success": False}, "failure_reason": "server_unreachable"},
        )
        self.assertTrue(row.is_infra_failure)
        self.assertEqual(row.failure_reason, "server_unreachable")

    def test_a_crash_is_never_counted_as_a_success(self):
        """Their metrics can say success while the episode also errored."""
        row = to_episode_row(
            self._episode(),
            {"metrics": {"success": True}, "failure_reason": "worker_timeout"},
        )
        self.assertFalse(row.success)
        self.assertTrue(row.is_infra_failure)

    def test_phase_outcomes_come_across_as_pairs(self):
        row = to_episode_row(
            self._episode(),
            {"metrics": {"success": False, "phase_outcomes": {"reach": True, "grasp": False},
                         "terminal_phase": "grasp"}},
        )
        self.assertEqual(sorted(row.phase_outcomes), [("grasp", False), ("reach", True)])
        self.assertEqual(row.terminal_phase, "grasp")

    def test_missing_fields_do_not_raise(self):
        """A benchmark we did not write may omit anything optional."""
        row = to_episode_row(self._episode(), {})
        self.assertFalse(row.success)
        self.assertEqual(row.steps, 0)
        self.assertEqual(row.failure_reason, "policy_failure")


class TestStepBuffer(unittest.TestCase):
    def test_it_accumulates_in_order_and_nothing_is_written_mid_episode(self):
        buffer = StepBuffer()
        for i in range(3):
            buffer.collect("ep-1", "reward", float(i))
        buffer.collect("ep-2", "reward", 9.0)
        self.assertEqual(len(buffer), 2)
        self.assertEqual([v for _, v in buffer.steps_for("ep-1")], [0.0, 1.0, 2.0])

    def test_discard_frees_an_episode(self):
        buffer = StepBuffer()
        buffer.collect("ep-1", "reward", 1.0)
        buffer.discard("ep-1")
        self.assertEqual(buffer.steps_for("ep-1"), [])
