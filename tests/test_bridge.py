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
    NullRecordingStore,
    build_eval_config,
    check_index_contract,
    check_server_assignment,
    preflight_servers,
    group_by_seed,
    rows_from_benchmark_result,
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
        """Reproduces the real gate, including WHERE the store is assigned.

        This used to expose an `_init_store` method that the harness does not
        have -- invented here, then overridden in ParquetOrchestrator, and the
        tests passed. The harness assigns `_store` inside `run()`:

            async def run(self):
                if not self.no_save:
                    self._store = RecordingStore(db_path_for_eval(...))

        and `no_save` also decides whether `rec_cfg` is None, so the two gates are
        coupled and no flag opens both. The stand-in now mirrors that, so an
        override that only works against an imagined shape fails here.
        """

        def __init__(self, no_save=False):
            self._store = None
            self._sid = "sid-1"
            self.no_save = no_save

        def run(self):
            if not self.no_save:
                self._store = "sqlite-store"   # what the harness would build
            return self._store

        def effective_recording_config(self, raw):
            return None if self.no_save else {"record_step": True}

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
    def test_the_store_assignment_in_run_is_swallowed(self):
        """The gate is opened where the harness actually closes it.

        `run()` assigns a RecordingStore; the subclass must still see its own
        store afterwards, or the override only works in a test.
        """
        orchestrator = make_parquet_orchestrator(self.recorder_cls)()
        self.assertIsNotNone(orchestrator._store)
        orchestrator.run()                      # assigns "sqlite-store"
        self.assertIsInstance(orchestrator._store, NullRecordingStore)
        orchestrator._store = None              # the finally block
        self.assertIsInstance(orchestrator._store, NullRecordingStore)

    def test_the_null_store_answers_what_the_harness_calls_on_it(self):
        """A bare object() would have raised on the first benchmark."""
        store = NullRecordingStore()
        store.upsert_eval_metadata("eval", "safe", {})
        store.upsert_episode_result()
        store.upsert_step_rows()
        store.close()

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

        half = HalfOverridden(no_save=True)   # the only way to keep rec_cfg alive
        half.run()                            # leaves _store None
        # The override is never consulted, because the gate closes first.
        recorder = Orchestrator._build_recorder(half, {"record_step": True}, {}, "ev", "s", 0, 0, None)
        self.assertIsInstance(recorder, self.NullRecorder)
        self.assertFalse(recorder.is_active)

    def test_overriding_both_produces_a_live_recorder(self):
        orchestrator = make_parquet_orchestrator(self.recorder_cls)()
        orchestrator.run()
        self.assertIsNotNone(orchestrator._store)

        recorder = orchestrator._build_recorder(
            {"record_step": True}, {"name": "t"}, "eval-1", "safe", 0, 7, None
        )
        self.assertTrue(recorder.is_active)
        self.assertEqual(recorder.eval_id, "eval-1")

    def test_a_null_recording_config_still_yields_the_null_recorder(self):
        """Recording off is a legitimate configuration, not a failure."""
        orchestrator = make_parquet_orchestrator(self.recorder_cls)()
        orchestrator.run()
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
            max_steps=300,
            instruction="put the bowl on the plate",
            episode_id=f"sha256:{task_id}{seed}", task_id=task_id, task_hash="sha256:t",
            scenario_hash="sha256:s", seed=seed, checkpoint_id="ckpt",
        )

    def test_one_task_is_expressible(self):
        """`tasks` carries the instruction; `task_ids` carries ours.

        Both, because they answer different questions. The harness filters on the
        instruction, and a reader of the config needs to know which Refractal task
        that was -- a language string is not something you can look up in a
        catalog.
        """
        episodes = [self._episode("libero-3", s) for s in range(10)]
        self.assertEqual(
            worker_selection(episodes),
            {
                "tasks": ["put the bowl on the plate"],
                "episodes_per_task": 10,
                "task_ids": ["libero-3"],
            },
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
            max_steps=300,
            instruction="put the bowl on the plate",
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


class TestTheIndexContract(unittest.TestCase):
    """The sharpest thing in the bridge, and the easiest to miss.

    The harness runs `ep in range(episodes_per_task)` and hands `episode_idx = ep`
    to the benchmark, which does `initial_states[episode_idx]`. Refractal's
    scenario says which init state it means. Those are two different numbers that
    coincide only when a catalog uses `range: [0, N-1]`.
    """

    def _episodes(self, hashes):
        return [
            PlannedEpisode(
                max_steps=300,
                instruction="put the bowl on the plate",
                episode_id=f"sha256:{h}", task_id="t", task_hash="sha256:t",
                scenario_hash=h, seed=0, checkpoint_id="ckpt",
            )
            for h in hashes
        ]

    def _scenarios(self, indices):
        return {f"s{i}": {"init_state_index": i} for i in indices}

    def test_a_zero_based_contiguous_range_is_accepted(self):
        check_index_contract(
            self._episodes([f"s{i}" for i in range(5)]), self._scenarios(range(5))
        )

    def test_an_offset_range_is_refused(self):
        """[5, 14] would run init states 0..9 while every row claims 5..14."""
        with self.assertRaises(BridgeError) as ctx:
            check_index_contract(
                self._episodes([f"s{i}" for i in range(5, 10)]),
                self._scenarios(range(5, 10)),
            )
        message = str(ctx.exception)
        self.assertIn("never executed", message)
        self.assertIn("range: [0, N-1]", message)

    def test_a_gap_is_refused(self):
        with self.assertRaises(BridgeError):
            check_index_contract(
                self._episodes(["s0", "s1", "s3"]), self._scenarios([0, 1, 3])
            )

    def test_an_unknown_scenario_is_refused_rather_than_skipped(self):
        with self.assertRaises(BridgeError) as ctx:
            check_index_contract(self._episodes(["s0", "s9"]), self._scenarios([0]))
        self.assertIn("cannot be checked", str(ctx.exception))


class TestResultMapping(unittest.TestCase):
    def _episodes(self, n):
        return [
            PlannedEpisode(
                max_steps=300,
                instruction="put the bowl on the plate",
                episode_id=f"sha256:e{i}", task_id="t", task_hash="sha256:t",
                scenario_hash=f"s{i}", seed=0, checkpoint_id="ckpt",
            )
            for i in range(n)
        ]

    def _result(self, outcomes):
        return {
            "tasks": [
                {
                    "episodes": [
                        {"episode_id": i, "metrics": {"success": ok}, "steps": 100}
                        for i, ok in enumerate(outcomes)
                    ]
                }
            ]
        }

    def test_results_map_onto_planned_episodes_in_order(self):
        rows = rows_from_benchmark_result(
            self._result([True, False, True]), self._episodes(3)
        )
        self.assertEqual([r.success for r in rows], [True, False, True])
        self.assertEqual([r.episode.episode_id for r in rows],
                         ["sha256:e0", "sha256:e1", "sha256:e2"])

    def test_a_short_result_is_refused_not_padded(self):
        """Work asked for and not returned must not be written as completed."""
        with self.assertRaises(BridgeError) as ctx:
            rows_from_benchmark_result(self._result([True, False]), self._episodes(3))
        self.assertIn("must not be written as though it completed", str(ctx.exception))


class TestRefAndParamsAreDifferentJobs(unittest.TestCase):
    """`ref` says which scene; `params` says how to build it.

    One field did both at first, and `build_eval_config` passed the whole thing
    to the provider -- which raises, because LIBERO's `task_id` identifies a task
    and is not a constructor argument of the benchmark that owns it.
    """

    def _scene(self, **kwargs):
        from refractal.schema.models import ExternalScene

        base = {"provider": "pkg.mod:Bench", "ref": {}, "params": {}}
        base.update(kwargs)
        return ExternalScene.model_validate(base)

    def test_a_key_in_both_is_allowed_when_it_agrees(self):
        """LIBERO's `suite` genuinely belongs in both, so the rule cannot be
        "no key appears twice"."""
        scene = self._scene(
            ref={"suite": "libero_spatial", "task_id": 3},
            params={"suite": "libero_spatial", "send_state": True},
        )
        self.assertEqual(scene.ref["suite"], scene.params["suite"])

    def test_a_key_in_both_that_disagrees_is_refused(self):
        with self.assertRaises(Exception) as ctx:
            self._scene(
                ref={"suite": "libero_spatial"}, params={"suite": "libero_object"}
            )
        self.assertIn("disagree", str(ctx.exception))

    def test_params_participate_in_scene_identity(self):
        """`send_state: false` against a checkpoint trained with proprioception
        is the parameter that moved X-VLA on LIBERO from 97.8% to 42%. Two runs
        that disagree about it must not join."""
        from refractal.schema.identity import external_scene_ref_key
        from refractal.schema.models import Scene

        def scene(params):
            return Scene.model_validate({
                "id": "libero-0",
                "engine": "mujoco",
                "engine_version": "3.2.0",
                "external": {"provider": "pkg.mod:Bench",
                             "ref": {"suite": "s", "task_id": 0},
                             "params": params},
                "resource_shape": [{
                    "hardware_profile": "h", "envs_per_process": 1,
                    "vram_per_env_mb": 0, "cpu_cores": 1,
                    "sec_per_1k_steps": 10.0, "startup_sec": 1,
                }],
            })

        self.assertNotEqual(
            external_scene_ref_key(scene({"send_state": True})),
            external_scene_ref_key(scene({"send_state": False})),
        )


class TestEvalConfig(unittest.TestCase):
    class Scene:
        scene_id = "libero-spatial-0"
        external = types.SimpleNamespace(
            provider="vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark",
            ref={"suite": "libero_spatial", "task_id": 0},
            params={"suite": "libero_spatial", "send_state": True},
        )

    def _episodes(self, n, task="t"):
        return [
            PlannedEpisode(
                max_steps=300,
                instruction="put the bowl on the plate",
                episode_id=f"sha256:e{i}", task_id=task, task_hash="sha256:t",
                scenario_hash=f"s{i}", seed=0, checkpoint_id="ckpt",
            )
            for i in range(n)
        ]

    def test_it_speaks_the_harness_dialect(self):
        config = build_eval_config(
            scene=self.Scene(), episodes=self._episodes(10),
            server_url="ws://pi0:8000", output_dir="/tmp/out", max_steps=220,
        )
        benchmark = config["benchmarks"][0]
        self.assertEqual(benchmark["episodes_per_task"], 10)
        # The selector is the INSTRUCTION, not our task id: the harness filters
        # `get_tasks()` on `name`, which LIBERO sets to `task.language`.
        self.assertEqual(benchmark["tasks"], ["put the bowl on the plate"])
        self.assertEqual(benchmark["subname"], "libero_spatial")
        # `params` comes from `external.params`, never from `ref` -- `ref` holds
        # `task_id`, which is not a constructor argument and would raise.
        self.assertEqual(benchmark["params"], {"suite": "libero_spatial", "send_state": True})
        self.assertNotIn("task_id", benchmark["params"])
        self.assertEqual(config["server"]["url"], "ws://pi0:8000")

    def test_recording_is_enabled_or_the_other_gate_stays_shut(self):
        """`rec_cfg is None` closes the gate just as surely as `_store is None`."""
        config = build_eval_config(
            scene=self.Scene(), episodes=self._episodes(3),
            server_url="ws://x:8000", output_dir="/tmp/out", max_steps=220,
        )
        self.assertTrue(config["benchmarks"][0]["recording"]["record_step"])

    def test_a_non_external_scene_is_refused(self):
        with self.assertRaises(BridgeError):
            build_eval_config(
                scene=types.SimpleNamespace(scene_id="local", external=None),
                episodes=self._episodes(1), server_url="ws://x:8000",
                output_dir="/tmp/out", max_steps=220,
            )


class TestSeedsComeFromTheLoopNotTheCounter(unittest.TestCase):
    """The adjacent case the index contract implied but did not name.

    The counter that selects init states is the same counter that would have to
    encode a repeat. Ten scenarios x three seeds in one invocation runs 0..29 and
    reaches thirty *different* init states, while the plan claims ten scenarios
    run three times. LIBERO ships 50, so it does not even error.
    """

    def _episodes(self, scenarios, seeds):
        return [
            PlannedEpisode(
                max_steps=300,
                instruction="put the bowl on the plate",
                episode_id=f"sha256:e{i}s{seed}", task_id="t", task_hash="sha256:t",
                scenario_hash=f"s{i}", seed=seed, checkpoint_id="ckpt",
            )
            for seed in seeds
            for i in range(scenarios)
        ]

    def test_it_names_seeds_rather_than_blaming_the_indices(self):
        scenarios = {f"s{i}": {"init_state_index": i} for i in range(3)}
        with self.assertRaises(BridgeError) as ctx:
            check_index_contract(self._episodes(3, [0, 1]), scenarios)
        message = str(ctx.exception)
        self.assertIn("knows nothing about seeds", message)
        self.assertIn("group_by_seed", message)

    def test_each_seed_group_satisfies_the_contract(self):
        scenarios = {f"s{i}": {"init_state_index": i} for i in range(3)}
        for seed, group in group_by_seed(self._episodes(3, [0, 1, 2])).items():
            check_index_contract(group, scenarios)   # must not raise
            self.assertEqual(len(group), 3)

    def test_group_by_seed_partitions_without_loss(self):
        episodes = self._episodes(4, [0, 1, 2])
        groups = group_by_seed(episodes)
        self.assertEqual(sorted(groups), [0, 1, 2])
        self.assertEqual(sum(len(g) for g in groups.values()), len(episodes))

    def test_the_config_builder_refuses_a_mixed_group_too(self):
        """Two guards, because the caller might reach either first."""
        scene = types.SimpleNamespace(
            scene_id="s",
            external=types.SimpleNamespace(provider="m:C", ref={"suite": "x", "task_id": 0}),
        )
        with self.assertRaises(BridgeError) as ctx:
            build_eval_config(
                scene=scene, episodes=self._episodes(2, [0, 1]),
                server_url="ws://x:8000", output_dir="/tmp/out", max_steps=220,
            )
        self.assertIn("one seed's episodes", str(ctx.exception))

    def test_episodes_per_task_is_the_scenario_count_not_the_episode_count(self):
        scene = types.SimpleNamespace(
            scene_id="s",
            external=types.SimpleNamespace(provider="m:C", ref={"suite": "x", "task_id": 0}),
        )
        group = group_by_seed(self._episodes(5, [0, 1, 2]))[1]
        config = build_eval_config(
            scene=scene, episodes=group, server_url="ws://x:8000",
            output_dir="/tmp/out", max_steps=220,
        )
        self.assertEqual(config["benchmarks"][0]["episodes_per_task"], 5)


class TestServerAssignment(unittest.TestCase):
    """Both sides correct, the disagreement only in the relationship.

    Same shape as the index contract. Point two checkpoints at one port and every
    internal check passes -- well-formed rows, exact pairing, correct denominators
    -- and the verdict is a clean "no change" with a tight interval. The only
    thing wrong is which policy produced the rows, which the analysis cannot see.
    """

    def test_distinct_urls_are_accepted(self):
        check_server_assignment({"pi0": "ws://localhost:8000", "pi05": "ws://localhost:8001"})

    def test_a_shared_url_is_refused(self):
        with self.assertRaises(BridgeError) as ctx:
            check_server_assignment({"pi0": "ws://localhost:8000", "pi05": "ws://localhost:8000"})
        message = str(ctx.exception)
        self.assertIn("compared against themselves", message)
        self.assertIn("clean null", message)

    def test_it_names_which_checkpoints_collided(self):
        with self.assertRaises(BridgeError) as ctx:
            check_server_assignment(
                {"a": "ws://x:1", "b": "ws://x:1", "c": "ws://y:2"}
            )
        self.assertIn("'a', 'b'", str(ctx.exception))
        self.assertNotIn("'c'", str(ctx.exception))

    def test_no_servers_at_all_is_refused(self):
        with self.assertRaises(BridgeError):
            check_server_assignment({})


class TestServerPreflight(unittest.TestCase):
    """Fail before the first episode, not on first use."""

    def test_an_unreachable_server_is_refused_up_front(self):
        # Port 1 on localhost: nothing listens there.
        with self.assertRaises(BridgeError) as ctx:
            preflight_servers({"pi0": "ws://127.0.0.1:1"}, timeout=0.5)
        message = str(ctx.exception)
        self.assertIn("not accepting connections", message)
        # The reason servers are not supervised belongs in the error, because
        # this is where someone reads it.
        self.assertIn("warm-up", message)

    def test_a_reachable_server_passes(self):
        import socket
        import threading

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        threading.Thread(target=lambda: listener.accept(), daemon=True).start()
        try:
            result = preflight_servers({"pi0": f"ws://127.0.0.1:{port}"}, timeout=2.0)
            self.assertEqual(result, {"pi0": f"ws://127.0.0.1:{port}"})
        finally:
            listener.close()

    def test_it_checks_the_assignment_too(self):
        """One call, both preconditions: a collision never reaches a socket."""
        with self.assertRaises(BridgeError) as ctx:
            preflight_servers({"a": "ws://127.0.0.1:1", "b": "ws://127.0.0.1:1"}, timeout=0.5)
        self.assertIn("compared against themselves", str(ctx.exception))
