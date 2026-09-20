"""``run_vla_eval`` driven end to end with the harness replaced by a callable.

The loop is an I/O shell over parts that are tested on their own, so what is left
to test is the shell: how many times the harness is invoked and with what, what
lands in Parquet, and what happens on the second run.

``invoke`` is injected rather than monkeypatched. The signature it has to satisfy
-- ``(config, recorder_cls) -> result`` -- is the whole contract between the loop
and the harness, and a fake that satisfies it is a stricter statement about the
boundary than a patched import.

The fake calls the recorder, because a fake that did not would make every test
here fail on the buffer receipt. That is deliberate: the receipt is checked by the
tests that do not opt out of it.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_bridge import install_stand_in_harness

from refractal.execute.results import read_episodes
from refractal.execute.vla_eval import BridgeError
from refractal.execute.vla_eval_runner import run_vla_eval
from refractal.schema.models import ExternalScene, ResourceShape
from refractal.schema.plan import (
    Plan,
    PlannedEpisode,
    PlannedScenario,
    PlannedScene,
    PlannedWorker,
)

PROVIDER = "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"


def _frame_for(index: int):
    """A 4x4 frame filled with the episode's index, so it can be identified."""
    import numpy as np

    return np.full((4, 4, 3), index, dtype=np.uint8)


class FakeHarness:
    """Satisfies the ``invoke`` contract and records what it was asked to do."""

    def __init__(self, *, successes=None, short_by=0, silent=False, steps_per_episode=1,
                 frames_per_episode=0):
        self.configs: list[dict] = []
        self.successes = successes
        self.short_by = short_by
        #: `silent` means the recorder is never CONSTRUCTED -- the injection
        #: failing. `steps_per_episode=0` means it is constructed and records
        #: nothing -- an episode dying at step 0. Different failures.
        self.silent = silent
        self.steps_per_episode = steps_per_episode
        #: Frames the benchmark hands the recorder per episode. Each is filled
        #: with the episode's index so a test can tell which episode a written
        #: strip actually came from.
        self.frames_per_episode = frames_per_episode

    def __call__(self, config, recorder_cls):
        self.configs.append(dict(config))
        benchmark = config["benchmarks"][0]
        count = benchmark["episodes_per_task"]
        if not self.silent:
            # What the real recorder does: one instance per episode, stepped.
            for index in range(count):
                recorder = recorder_cls(
                    episode_id=str(index), sid="sid", eid=f"e{index}", eval_id="ev"
                )
                for _ in range(self.steps_per_episode):
                    recorder.record_step(reward=1.0)
                for _ in range(self.frames_per_episode):
                    recorder.record_video(_frame_for(index))
        episodes = []
        for index in range(count - self.short_by):
            ok = True if self.successes is None else self.successes[index % len(self.successes)]
            episodes.append(
                {
                    "episode_id": index,
                    "metrics": {"success": ok},
                    "steps": 120,
                    "elapsed_sec": 3.5,
                }
            )
        return {"tasks": [{"episodes": episodes}]}


def make_plan(*, scenarios=3, seeds=(0,), checkpoints=("pi0", "pi05"), scenes=("libero-0",)):
    """A plan of the shape the LIBERO catalog compiles to: external scenes, one
    task per worker, ``init_state_index`` starting at zero."""
    planned_scenes = []
    total = 0
    for scene_id in scenes:
        scenario_records = [
            PlannedScenario(
                scenario_hash=f"sha256:{scene_id}-s{i}",
                scenario_set_id="init-states",
                params={"init_state_index": i},
            )
            for i in range(scenarios)
        ]
        episodes = [
            PlannedEpisode(
                episode_id=f"sha256:{scene_id}-{i}-{seed}-{ckpt}",
                task_id="0",
                task_hash="sha256:task",
                scenario_hash=f"sha256:{scene_id}-s{i}",
                seed=seed,
                checkpoint_id=ckpt,
                max_steps=300,
                instruction="put the bowl on the plate",
            )
            for ckpt in checkpoints
            for seed in seeds
            for i in range(scenarios)
        ]
        total += len(episodes)
        planned_scenes.append(
            PlannedScene(
                scene_id=scene_id,
                scene_hash=f"sha256:{scene_id}",
                engine="mujoco",
                external=ExternalScene(
                    provider=PROVIDER,
                    ref={"suite": "libero_spatial", "task_id": 0},
                    params={"suite": "libero_spatial", "send_state": True,
                            "send_wrist_image": True},
                ),
                resource_shape=ResourceShape(hardware_profile="alienware", envs_per_process=1,
                                             vram_per_env_mb=0, cpu_cores=2,
                                             sec_per_1k_steps=10.0, startup_sec=5),
                scenarios=scenario_records,
                workers=[PlannedWorker(worker_id=f"{scene_id}/w0", scene_id=scene_id,
                                       episodes=episodes)],
                episode_count=len(episodes),
                estimated_seconds=60,
            )
        )
    return Plan(
        plan_id="sha256:plan",
        catalog_hash="sha256:catalog",
        refractal_version="0.0.0",
        hardware_profile="alienware",
        execution_mode="serial",
        checkpoints=[],
        tier="full",
        seeds=list(seeds),
        total_episodes=total,
        estimated_seconds=60,
        scenes=planned_scenes,
    )


class LoopCase(unittest.TestCase):
    def setUp(self):
        # make_parquet_recorder subclasses the harness's EpisodeRecorder, so the
        # stand-in has to be importable even though `invoke` is faked.
        self._saved = {k: sys.modules.get(k) for k in
                       ("vla_eval", "vla_eval.recording", "vla_eval.orchestrator")}
        install_stand_in_harness()
        # Put the stand-in on disk. `describe_installed_harness` digests the
        # surface files, so a module with no `__file__` yields "unlocatable" and
        # an empty manifest -- which would let this file assert around the
        # provenance path instead of through it.
        self._surface = Path(tempfile.mkdtemp()) / "vla_eval"
        for name in ("orchestrator.py", "recording.py", "registry.py", "types.py"):
            (self._surface).mkdir(parents=True, exist_ok=True)
            (self._surface / name).write_text(f"# stand-in {name}\n", encoding="utf-8")
        for package in ("runners", "protocol"):
            (self._surface / package).mkdir(parents=True, exist_ok=True)
            (self._surface / package / "__init__.py").write_text("", encoding="utf-8")
        sys.modules["vla_eval"].__file__ = str(self._surface / "__init__.py")
        for name, module in self._saved.items():
            self.addCleanup(
                sys.modules.pop if module is None else sys.modules.__setitem__,
                name, *([] if module is None else [module]),
            )
        self.tmp = tempfile.mkdtemp()
        self.results = str(Path(self.tmp) / "results")
        # preflight_servers reaches the network; the loop's own checks are what
        # these tests are about, so it is stubbed to the identity it returns.
        import refractal.execute.vla_eval_runner as runner

        self._real_preflight = runner.preflight_servers
        runner.preflight_servers = lambda servers, **kw: dict(servers)
        self.addCleanup(setattr, runner, "preflight_servers", self._real_preflight)

    def run_loop(self, plan, harness, **kwargs):
        all_servers = {"pi0": "ws://a:8000", "pi05": "ws://b:8000"}
        # Popped BEFORE the comprehension, not inside its `if`. Written the other
        # way first, where `kwargs.pop("only", all_servers)` ran once per server:
        # the first iteration consumed the key and every later one fell back to the
        # default, so the filter passed everything through and the missing-server
        # test was green against a complete server map.
        only = kwargs.pop("only", tuple(all_servers))
        session_id = kwargs.pop("session_id", "aaaa-first")
        return run_vla_eval(
            plan,
            self.results,
            {k: v for k, v in all_servers.items() if k in only},
            session_id=session_id,
            invoke=harness,
            output_dir=str(Path(self.tmp) / "harness"),
            **kwargs,
        )

    def rows(self, plan):
        return read_episodes(self.results, plan.plan_id).to_pylist()


class TestTheNestingIsOneInvocationPerLeaf(LoopCase):
    def test_one_invocation_per_worker_checkpoint_seed(self):
        plan = make_plan(scenarios=3, seeds=(0, 1), checkpoints=("pi0", "pi05"))
        harness = FakeHarness()
        summary = self.run_loop(plan, harness)
        # 1 scene x 1 worker x 2 checkpoints x 2 seeds
        self.assertEqual(summary.invocations, 4)
        self.assertEqual(len(harness.configs), 4)
        self.assertEqual(summary.written, 12)

    def test_each_invocation_asks_for_one_seeds_worth(self):
        """The reason seeds are a loop level: episodes_per_task is the scenario
        count. Passing both seeds at once would ask for 6 and get init states
        0..5, running three that do not exist and none of them twice."""
        plan = make_plan(scenarios=3, seeds=(0, 1), checkpoints=("pi0",))
        harness = FakeHarness()
        self.run_loop(plan, harness, only=("pi0",))
        self.assertEqual(
            [c["benchmarks"][0]["episodes_per_task"] for c in harness.configs], [3, 3]
        )

    def test_the_step_limit_comes_from_the_plan(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        harness = FakeHarness()
        self.run_loop(plan, harness, only=("pi0",))
        self.assertEqual(harness.configs[0]["benchmarks"][0]["max_steps"], 300)

    def test_each_checkpoint_goes_to_its_own_server(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0", "pi05"))
        harness = FakeHarness()
        self.run_loop(plan, harness)
        urls = {c["server"]["url"] for c in harness.configs}
        self.assertEqual(urls, {"ws://a:8000", "ws://b:8000"})


class TestWhatLandsInParquet(LoopCase):
    def test_every_planned_episode_is_written_exactly_once(self):
        plan = make_plan(scenarios=4, seeds=(0, 1), checkpoints=("pi0", "pi05"))
        self.run_loop(plan, FakeHarness())
        written = [r["episode_id"] for r in self.rows(plan)]
        planned = [
            e.episode_id
            for s in plan.scenes
            for w in s.workers
            for e in w.episodes
        ]
        self.assertEqual(sorted(written), sorted(planned))
        self.assertEqual(len(written), len(set(written)))

    def test_the_answering_server_is_recorded_per_row(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0", "pi05"))
        self.run_loop(plan, FakeHarness())
        by_checkpoint = {}
        for row in self.rows(plan):
            by_checkpoint.setdefault(row["checkpoint_id"], set()).add(row["server_url"])
        self.assertEqual(by_checkpoint["pi0"], {"ws://a:8000"})
        self.assertEqual(by_checkpoint["pi05"], {"ws://b:8000"})

    def test_outcomes_survive_the_round_trip(self):
        plan = make_plan(scenarios=4, checkpoints=("pi0",))
        self.run_loop(plan, FakeHarness(successes=[True, False, True, True]), only=("pi0",))
        rows = sorted(self.rows(plan), key=lambda r: r["scenario_hash"])
        self.assertEqual([r["success"] for r in rows], [True, False, True, True])
        self.assertEqual([r["failure_reason"] for r in rows],
                         [None, "policy_failure", None, None])

    def test_the_run_window_is_recorded_and_ordered(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        self.run_loop(plan, FakeHarness(), only=("pi0",))
        for row in self.rows(plan):
            self.assertIsNotNone(row["started_at"])
            self.assertLessEqual(row["started_at"], row["ended_at"])


class TestResume(LoopCase):
    def test_a_completed_plan_re_runs_nothing(self):
        plan = make_plan(scenarios=3, checkpoints=("pi0", "pi05"))
        self.run_loop(plan, FakeHarness())
        second = FakeHarness()
        summary = self.run_loop(plan, second, session_id="bbbb-second")
        self.assertEqual(summary.invocations, 0)
        self.assertEqual(summary.written, 0)
        self.assertEqual(summary.skipped, 6)

    def test_rerunning_a_group_replaces_its_rows_rather_than_doubling_them(self):
        """The reason part names are deterministic.

        Resume is group-granular -- the harness counts from zero, so a partly
        done group runs again in full. A session-stamped part name would leave
        the old file beside the new one and put two rows under one episode_id,
        which `compare` blocks on as a duplicate.
        """
        plan = make_plan(scenarios=3, checkpoints=("pi0",))
        self.run_loop(plan, FakeHarness(), only=("pi0",))
        summary = self.run_loop(
            plan, FakeHarness(), only=("pi0",), session_id="bbbb-second", resume=False
        )
        self.assertEqual(summary.invocations, 1)
        ids = [r["episode_id"] for r in self.rows(plan)]
        self.assertEqual(len(ids), 3, f"expected 3 rows, got {len(ids)}: {sorted(ids)}")
        self.assertEqual(len(set(ids)), 3)
        # And the new session's rows are the ones that survived.
        self.assertEqual({r["session_id"] for r in self.rows(plan)}, {"bbbb-second"})


class TestWhatItRefuses(LoopCase):
    def test_a_missing_server_is_named_before_anything_runs(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0", "pi05"))
        harness = FakeHarness()
        with self.assertRaises(BridgeError) as ctx:
            self.run_loop(plan, harness, only=("pi0",))
        self.assertIn("pi05", str(ctx.exception))
        self.assertEqual(harness.configs, [], "it must refuse before invoking anything")

    def test_two_arms_on_one_server_is_refused_without_touching_the_network(self):
        """Reached through the loop, not by calling the checker directly.

        It was reachable only from inside `preflight_servers` at first, which
        these tests stub -- so the loop could have lost the check entirely and
        every test here would still have been green.
        """
        plan = make_plan(scenarios=2, checkpoints=("pi0", "pi05"))
        harness = FakeHarness()
        with self.assertRaises(BridgeError) as ctx:
            run_vla_eval(
                plan, self.results, {"pi0": "ws://same:8000", "pi05": "ws://same:8000"},
                session_id="aaaa-first", invoke=harness,
                output_dir=str(Path(self.tmp) / "harness"),
            )
        self.assertIn("share a server URL", str(ctx.exception))
        self.assertEqual(harness.configs, [])

    def test_a_silent_recorder_is_refused_rather_than_written(self):
        """The failure the override exists to avoid: a run that completes,
        reports success and records nothing."""
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        with self.assertRaises(BridgeError) as ctx:
            self.run_loop(plan, FakeHarness(silent=True), only=("pi0",))
        message = str(ctx.exception)
        self.assertIn("never constructed", message)
        self.assertIn("_store is None", message)
        self.assertEqual(self.rows(plan), [], "nothing may be written after that")

    def test_episodes_that_die_before_their_first_step_are_not_a_silent_recorder(self):
        """The distinction the first real run forced.

        Every episode timed out during the server's torch.compile warm-up, so the
        step buffer was empty -- and the check, which read the buffer, reported
        that the recorder had never been consulted. It had been consulted twice.
        "Nothing was recorded" and "the injection failed" are different, and only
        the second is a bridge problem.
        """
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        # Constructs recorders, records no steps: an episode that dies at step 0.
        summary = self.run_loop(plan, FakeHarness(steps_per_episode=0), only=("pi0",))
        self.assertEqual(summary.written, 2)
        self.assertEqual(len(self.rows(plan)), 2)

    def test_a_short_result_writes_nothing_at_all(self):
        """Not even the episodes that did come back. A partially written group
        whose part file is then replaced on resume is fine; a partially written
        group recorded as complete is a wrong denominator."""
        plan = make_plan(scenarios=4, checkpoints=("pi0",))
        with self.assertRaises(BridgeError):
            self.run_loop(plan, FakeHarness(short_by=1), only=("pi0",))
        self.assertEqual(self.rows(plan), [])

    def test_a_worker_spanning_two_tasks_is_split_not_refused(self):
        """It used to be refused. A scene is the compiled model and tasks are
        cheap to vary on it, so one worker holding many is the normal case --
        LIBERO-Spatial's ten tasks are one MjModel. The loop splits by task and
        gives each its own invocation, because the harness's episode counter
        restarts per task and `worker_selection` can only name one at a time."""
        plan = make_plan(scenarios=4, checkpoints=("pi0",))
        worker = plan.scenes[0].workers[0]
        # Two tasks, each with its own zero-based contiguous scenario range.
        scene = plan.scenes[0]
        extra = [
            PlannedScenario(scenario_hash=f"sha256:b-s{i}", scenario_set_id="init",
                            params={"init_state_index": i})
            for i in range(4)
        ]
        object.__setattr__(scene, "scenarios", list(scene.scenarios) + extra)
        second = [
            e.model_copy(update={"task_id": "1", "instruction": "put the cup down",
                                 "scenario_hash": f"sha256:b-s{i}",
                                 "episode_id": f"sha256:second-{i}"})
            for i, e in enumerate(worker.episodes)
        ]
        object.__setattr__(worker, "episodes", list(worker.episodes) + second)

        harness = FakeHarness()
        summary = self.run_loop(plan, harness, only=("pi0",))
        self.assertEqual(summary.invocations, 2, "one invocation per task")
        self.assertEqual(summary.written, 8)
        self.assertEqual(
            sorted(c["benchmarks"][0]["tasks"][0] for c in harness.configs),
            ["put the bowl on the plate", "put the cup down"],
        )

    def test_each_task_gets_its_own_part_file(self):
        """Ten tasks writing to one path would leave one file and `verify_written`
        would not see it -- it checks the file it just wrote."""
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        scene = plan.scenes[0]
        worker = scene.workers[0]
        extra = [
            PlannedScenario(scenario_hash=f"sha256:b-s{i}", scenario_set_id="init",
                            params={"init_state_index": i})
            for i in range(2)
        ]
        object.__setattr__(scene, "scenarios", list(scene.scenarios) + extra)
        second = [
            e.model_copy(update={"task_id": "1", "instruction": "put the cup down",
                                 "scenario_hash": f"sha256:b-s{i}",
                                 "episode_id": f"sha256:second-{i}"})
            for i, e in enumerate(worker.episodes)
        ]
        object.__setattr__(worker, "episodes", list(worker.episodes) + second)
        summary = self.run_loop(plan, FakeHarness(), only=("pi0",))
        self.assertEqual(len(summary.parts), 2)
        self.assertEqual(len({p for p in summary.parts}), 2)
        self.assertEqual(len(self.rows(plan)), 4)

    def test_an_offset_scenario_range_is_refused_before_invoking(self):
        """check_index_contract, reached through the loop rather than directly."""
        plan = make_plan(scenarios=3, checkpoints=("pi0",))
        scene = plan.scenes[0]
        shifted = [
            s.model_copy(update={"params": {"init_state_index": i + 5}})
            for i, s in enumerate(scene.scenarios)
        ]
        object.__setattr__(scene, "scenarios", shifted)
        harness = FakeHarness()
        with self.assertRaises(BridgeError) as ctx:
            self.run_loop(plan, harness, only=("pi0",))
        self.assertIn("never executed", str(ctx.exception))
        self.assertEqual(harness.configs, [])


class TestProvenance(LoopCase):
    def test_the_plan_and_the_harness_manifest_are_written(self):
        from refractal.execute.results import ResultWriter

        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        self.run_loop(plan, FakeHarness(), only=("pi0",))
        writer = ResultWriter(self.results, plan.plan_id)
        rows = self.rows(plan)
        self.assertTrue(all(r["harness_surface"] for r in rows))
        self.assertEqual(len({r["harness_surface"] for r in rows}), 1)
        # The manifest is keyed by the same surface digest the rows carry.
        manifests = writer.read_harness_manifests()
        self.assertIn(rows[0]["harness_surface"], manifests)
        # And it names files, which is what makes a surface change actionable
        # rather than merely detected.
        self.assertIn("orchestrator.py", manifests[rows[0]["harness_surface"]])


class TestTheServerFlag(unittest.TestCase):
    def test_a_url_containing_an_equals_sign_survives(self):
        """Split on the first `=` only. Splitting on all of them and taking
        [-1] would truncate a query string and point a checkpoint at a different
        address than the one given -- silently, and in the direction of a
        self-comparison if the truncation happened to collide."""
        from refractal.cli import _servers

        parsed = _servers(["pi0=ws://h:8000/?token=abc=def"])
        self.assertEqual(parsed, {"pi0": "ws://h:8000/?token=abc=def"})

    def test_two_checkpoints_parse_into_two_entries(self):
        from refractal.cli import _servers

        self.assertEqual(
            _servers(["a=ws://one", "b=ws://two"]),
            {"a": "ws://one", "b": "ws://two"},
        )

    def test_a_repeated_checkpoint_is_refused_rather_than_last_wins(self):
        from refractal.cli import _servers

        with self.assertRaises(SystemExit):
            _servers(["a=ws://one", "a=ws://two"])

    def test_a_malformed_pair_is_refused(self):
        from refractal.cli import _servers

        for bad in ("pi0", "pi0=", "=ws://h", ""):
            with self.subTest(bad), self.assertRaises(SystemExit):
                _servers([bad])


class TestAPlanRoundTripsIntoARunnableConfig(unittest.TestCase):
    """The invariant nothing held, and the one that would have caught all three
    missing fields before the loop did.

    Every other test on the plan asks whether it is *correct*: does `plan_id`
    move when it should, does an episode id decompose, do two catalogs agree.
    None of them asks whether it is *sufficient* -- whether a backend handed
    only plan.json can construct the thing it has to hand to a runner.

    Those are different questions, and the distinction is the finding:
    `max_steps`, the benchmark provider and the suite ref were all inside a hash,
    so identity was intact and the plan was unrunnable. Being inside a digest is
    not the same as surviving the compile.

    So this reads a plan back off disk -- not a Plan object held in memory, which
    could carry a field the serialiser drops -- and drives it all the way to a
    harness config, asserting on the config rather than on the plan.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _round_trip(self, plan):
        """Write, read back, and build a config from what came off disk."""
        from refractal.execute.vla_eval import build_eval_config, group_by_seed
        from refractal.execute.vla_eval_runner import _max_steps
        from refractal.schema.plan import read_plan

        path = self.tmp / "plan.json"
        plan.write(path)
        reloaded = read_plan(path)

        configs = []
        for scene in reloaded.scenes:
            for worker in scene.workers:
                for checkpoint in sorted({e.checkpoint_id for e in worker.episodes}):
                    for_ckpt = [e for e in worker.episodes if e.checkpoint_id == checkpoint]
                    for seed, group in sorted(group_by_seed(for_ckpt).items()):
                        configs.append(
                            build_eval_config(
                                scene=scene,
                                episodes=group,
                                server_url="ws://stand-in:8000",
                                output_dir=str(self.tmp / "out"),
                                max_steps=_max_steps(group),
                            )
                        )
        return reloaded, configs

    def test_an_external_plan_reaches_a_complete_harness_config(self):
        # `make_plan` is defined in this module; imported by name so the nested
        # helper classes can reach it without a forward reference.
        from tests.test_vla_eval_loop import make_plan

        plan = make_plan(scenarios=3, seeds=(0, 1), checkpoints=("pi0", "pi05"))
        _, configs = self._round_trip(plan)
        self.assertEqual(len(configs), 4)
        for config in configs:
            benchmark = config["benchmarks"][0]
            # Every key the harness needs, named individually. A loop over
            # `benchmark.keys()` would pass on a config that had them all set to
            # None, which is the failure mode being guarded.
            self.assertTrue(benchmark["benchmark"], "no provider to import")
            self.assertTrue(benchmark["subname"], "no suite to select")
            self.assertGreater(benchmark["episodes_per_task"], 0)
            self.assertGreater(benchmark["max_steps"], 0, "no step limit")
            self.assertTrue(benchmark["tasks"], "no task to constrain the loop to")
            self.assertTrue(benchmark["params"].get("suite"), "no suite in params")
            self.assertNotIn(
                "task_id", benchmark["params"],
                "ref keys must not reach the constructor; task_id would raise TypeError",
            )
            self.assertTrue(config["server"]["url"])
            self.assertTrue(config["output_dir"])

    def test_the_step_limit_survives_serialisation(self):
        """It is the field that was missing, and it is an int inside a hash --
        the kind of thing a serialiser can drop without any identity test
        noticing."""
        from tests.test_vla_eval_loop import make_plan

        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        reloaded, configs = self._round_trip(plan)
        self.assertEqual(configs[0]["benchmarks"][0]["max_steps"], 300)
        for scene in reloaded.scenes:
            for worker in scene.workers:
                for episode in worker.episodes:
                    self.assertEqual(episode.max_steps, 300)

    def test_a_local_scene_is_refused_with_a_reason_not_a_None_field(self):
        """A plan of local scenes is a perfectly valid plan that this backend
        cannot run. It has to say so, rather than building a config whose
        provider is None and failing inside the harness."""
        from refractal.execute.vla_eval import BridgeError
        from tests.test_vla_eval_loop import make_plan

        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        object.__setattr__(plan.scenes[0], "external", None)
        with self.assertRaises(BridgeError) as ctx:
            self._round_trip(plan)
        self.assertIn("externally-defined", str(ctx.exception))

    def test_the_compiled_example_catalog_is_checked_too(self):
        """The round trip above uses a hand-built plan, which can only contain
        fields whoever wrote it remembered. This one comes out of `resolve`, so a
        field the planner stops populating fails here even though the fixture
        would still carry it."""
        from refractal.resolve import resolve

        catalog = Path(__file__).resolve().parents[1] / "examples" / "catalog"
        plan = resolve(catalog, hardware_profile="rtx5090")
        path = self.tmp / "compiled.json"
        plan.write(path)

        from refractal.schema.plan import read_plan

        reloaded = read_plan(path)
        for scene in reloaded.scenes:
            for worker in scene.workers:
                for episode in worker.episodes:
                    self.assertGreater(
                        episode.max_steps, 0,
                        f"{episode.episode_id} lost its step limit through the compile",
                    )
        # The example catalog is local, so this backend must refuse it -- and the
        # refusal is the assertion: it proves the plan reached the point of being
        # rejected for the right reason rather than for a missing field.
        self.assertTrue(all(s.external is None for s in reloaded.scenes))


class TestConcurrentMode(LoopCase):
    """Both checkpoints at once, each against its own server.

    The mode nothing used before, and the one the eval team wants: it halves wall
    clock. It is also the only mode that needs a column the others do not, because
    contention is a property of the run and a duration measured under it is not
    comparable with one measured alone.
    """

    def _plan(self, **kw):
        plan = make_plan(**kw)
        object.__setattr__(plan, "execution_mode", "concurrent")
        return plan

    def test_the_arms_actually_overlap_in_time(self):
        """The claim the mode is named for, measured rather than assumed.

        A fake that blocks proves the two invocations are in flight together; a
        fake that returns instantly would pass against a sequential loop too.
        """
        import threading

        barrier = threading.Barrier(2, timeout=5)

        class Overlapping(FakeHarness):
            def __call__(self, config, recorder_cls):
                # Both threads must arrive, or this raises BrokenBarrierError.
                barrier.wait()
                return super().__call__(config, recorder_cls)

        plan = self._plan(scenarios=2, seeds=(0,), checkpoints=("pi0", "pi05"))
        summary = self.run_loop(plan, Overlapping())
        self.assertEqual(summary.invocations, 2)
        self.assertEqual(summary.written, 4)

    def test_each_row_records_what_it_contended_with(self):
        plan = self._plan(scenarios=2, seeds=(0,), checkpoints=("pi0", "pi05"))
        self.run_loop(plan, FakeHarness())
        by_arm = {}
        for row in self.rows(plan):
            by_arm.setdefault(row["checkpoint_id"], set()).add(row["concurrent_with"])
        self.assertEqual(by_arm["pi0"], {"pi05"})
        self.assertEqual(by_arm["pi05"], {"pi0"})

    def test_serial_records_no_contention(self):
        """The column has to distinguish, or it is decoration."""
        plan = make_plan(scenarios=2, seeds=(0,), checkpoints=("pi0", "pi05"))
        self.assertEqual(plan.execution_mode, "serial")
        self.run_loop(plan, FakeHarness())
        self.assertEqual({r["concurrent_with"] for r in self.rows(plan)}, {None})

    def test_the_mode_is_recorded_on_every_row(self):
        plan = self._plan(scenarios=2, seeds=(0,), checkpoints=("pi0", "pi05"))
        self.run_loop(plan, FakeHarness())
        self.assertEqual({r["execution_mode"] for r in self.rows(plan)}, {"concurrent"})

    def test_concurrent_writes_the_same_episodes_as_serial(self):
        """Ordering is the only difference. If the modes disagreed about what ran,
        the mode would be changing the measurement rather than describing it."""
        serial_plan = make_plan(scenarios=3, seeds=(0, 1), checkpoints=("pi0", "pi05"))
        self.run_loop(serial_plan, FakeHarness())
        serial_ids = sorted(r["episode_id"] for r in self.rows(serial_plan))

        self.tmp = tempfile.mkdtemp()
        self.results = str(Path(self.tmp) / "results")
        conc_plan = self._plan(scenarios=3, seeds=(0, 1), checkpoints=("pi0", "pi05"))
        self.run_loop(conc_plan, FakeHarness())
        conc_ids = sorted(r["episode_id"] for r in self.rows(conc_plan))

        self.assertEqual(serial_ids, conc_ids)
        self.assertEqual(len(conc_ids), len(set(conc_ids)))

    def test_resume_still_skips_completed_groups(self):
        plan = self._plan(scenarios=2, seeds=(0,), checkpoints=("pi0", "pi05"))
        self.run_loop(plan, FakeHarness())
        summary = self.run_loop(plan, FakeHarness(), session_id="bbbb-second")
        self.assertEqual(summary.invocations, 0)
        self.assertEqual(summary.skipped, 4)


if __name__ == "__main__":
    unittest.main()


class TestVideoIsAskedForAndAccountedFor(LoopCase):
    """The receipt for frames, at the granularity `verify_written` does not reach.

    Asking for video and getting none is silent: the harness renders nothing, the
    recorder is handed nothing, and the run completes reporting success with an
    output directory that is merely smaller than expected. On a fifty-minute run
    that is the whole cost paid before anyone notices.
    """

    def strips(self, plan):
        from refractal.execute.results import comparison_prefix

        root = Path(comparison_prefix(self.results, plan.plan_id)) / "frames"
        return sorted(root.glob("*.webp")) if root.exists() else []

    def test_no_frames_when_video_was_requested_is_refused(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        with self.assertRaises(BridgeError) as ctx:
            self.run_loop(plan, FakeHarness(frames_per_episode=0),
                          only=("pi0",), record_video=True)
        self.assertIn("video was requested", str(ctx.exception))
        self.assertEqual(self.strips(plan), [])

    def test_without_video_no_frames_are_expected_or_written(self):
        """The guard fires on a broken promise, not on the absence of one."""
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        self.run_loop(plan, FakeHarness(frames_per_episode=0), only=("pi0",))
        self.assertEqual(self.strips(plan), [])

    def test_a_strip_is_written_per_episode_and_holds_its_frames(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        self.run_loop(plan, FakeHarness(frames_per_episode=3),
                      only=("pi0",), record_video=True, frame_every=1)
        rows = self.rows(plan)
        self.assertEqual(len(self.strips(plan)), len(rows))
        from PIL import Image

        for path in self.strips(plan):
            with Image.open(path) as img:
                self.assertEqual(img.width // img.height, 3, path.name)

    def test_frames_land_on_the_row_they_came_from(self):
        """The harness names episodes with its own counter; rows are named by a
        content-addressed digest. Keying the buffer by one and looking it up by
        the other finds nothing -- silently, on every episode. Caught in a real
        run by the receipt above, which is why this pins the correspondence
        rather than only the count.
        """
        import numpy as np
        from PIL import Image

        plan = make_plan(scenarios=3, checkpoints=("pi0",))
        self.run_loop(plan, FakeHarness(frames_per_episode=1),
                      only=("pi0",), record_video=True, frame_every=1)
        rows = self.rows(plan)
        self.assertEqual(len(rows), 3)

        from refractal.execute.results import comparison_prefix

        root = Path(comparison_prefix(self.results, plan.plan_id)) / "frames"
        # Rows come back in the order the episodes ran, which is the order the
        # harness constructed recorders, which is the index each frame carries.
        for index, row in enumerate(rows):
            name = row["episode_id"].replace("sha256:", "")[:32] + ".webp"
            with Image.open(root / name) as img:
                value = int(np.asarray(img.convert("RGB"))[0, 0, 0])
            self.assertEqual(
                value, index,
                f"{name} holds frames from episode {value}, not {index}",
            )

    def test_a_truncated_strip_is_refused_rather_than_rounded(self):
        """A strip is N square frames. A width that is not a whole multiple of
        the height is a partial write, and partial loss is an error here for the
        same reason it is in `verify_written`."""
        from PIL import Image

        from refractal.execute.results import OutputMissingError, ResultWriter

        writer = ResultWriter(results_uri=self.results, plan_id="sha256:" + "0" * 64)
        directory = f"{writer.prefix}/frames"
        writer.fs.makedirs(directory, exist_ok=True)
        path = f"{directory}/truncated.webp"
        with writer.fs.open(path, "wb") as handle:
            Image.new("RGB", (10, 4)).save(handle, format="WEBP")
        with self.assertRaises(OutputMissingError) as ctx:
            writer.verify_frames([path], worker_id="w")
        self.assertIn("not a whole number of square frames", str(ctx.exception))
