"""`refractal render --target compose`: a pure function of a plan.

No Docker anywhere in this file, which is the point of doing the renderer first:
the whole of it is assertions on a YAML document, and a rendered file somebody can
read is most of the value with none of the risk.
"""

from __future__ import annotations

import unittest

import yaml

from refractal.render import (
    CATALOG_PATH,
    PLAN_PATH,
    RESULTS_PATH,
    ComposeSettings,
    RenderError,
    compose_services,
    render_compose,
)
from refractal.schema.plan import PlanSchemaError, restrict_to_worker
from test_vla_eval_loop import make_plan

SERVERS = {"pi0": "ws://host.docker.internal:8000", "pi05": "ws://host.docker.internal:8001"}


def settings(**kw):
    base = dict(servers=dict(SERVERS), user="1000:1000")
    base.update(kw)
    return ComposeSettings(**base)


def with_workers(plan, count):
    """Split each scene's episodes across `count` workers, keeping every
    scenario whole so the index contract still holds."""
    from refractal.schema.plan import PlannedWorker

    for scene in plan.scenes:
        episodes = [e for w in scene.workers for e in w.episodes]
        # Split by task so each worker owns whole scenario ranges.
        tasks = sorted({e.task_id for e in episodes})
        workers = []
        for i in range(count):
            mine = [e for e in episodes if tasks.index(e.task_id) % count == i]
            workers.append(
                PlannedWorker(worker_id=f"{scene.scene_id}/{i}", scene_id=scene.scene_id,
                              cpuset=str(i), episodes=mine, estimated_seconds=60)
            )
        object.__setattr__(scene, "workers", [w for w in workers if w.episodes])
    return plan


class TestTheRenderedFileMatchesThePlan(unittest.TestCase):
    """The first of the two checks: what is rendered is what was planned."""

    def _doc(self, plan, **kw):
        return yaml.safe_load(render_compose(plan, settings(**kw)))

    def test_one_service_per_worker(self):
        plan = with_workers(make_plan(scenarios=4, checkpoints=("pi0", "pi05")), 1)
        doc = self._doc(plan)
        workers = [w for s in plan.scenes for w in s.workers]
        self.assertEqual(len(doc["services"]), len(workers))

    def test_each_service_names_a_real_worker(self):
        """A service whose --worker does not exist would exit 2 in the container,
        and the renderer knows every worker id."""
        plan = with_workers(make_plan(scenarios=2, checkpoints=("pi0",)), 1)
        doc = self._doc(plan)
        known = {w.worker_id for s in plan.scenes for w in s.workers}
        for service in doc["services"].values():
            command = service["command"]
            self.assertIn(command[command.index("--worker") + 1], known)

    def test_cpusets_do_not_overlap(self):
        """Compose has no scheduler. Two services on one core contend and the
        timings become noise."""
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        plan = with_workers(plan, 2)
        doc = self._doc(plan)
        cpusets = [s["cpuset"] for s in doc["services"].values() if "cpuset" in s]
        self.assertEqual(len(cpusets), len(set(cpusets)))

    def test_the_uid_is_a_literal_not_an_interpolation(self):
        """`${UID}:${GID}` is not exported by default in a POSIX shell, so it
        interpolates to ':' and Compose either errors or silently runs as root."""
        doc = self._doc(with_workers(make_plan(scenarios=2, checkpoints=("pi0",)), 1))
        for service in doc["services"].values():
            self.assertEqual(service["user"], "1000:1000")
            self.assertNotIn("$", service["user"])

    def test_no_service_is_a_model_server(self):
        """The servers stay outside: they are on the GPU, bare and warm, and a
        supervised one would put its model load inside the first episode of
        whichever arm started second."""
        plan = with_workers(make_plan(scenarios=2, checkpoints=("pi0", "pi05")), 1)
        doc = self._doc(plan)
        self.assertNotIn("pi0", doc["services"])
        self.assertNotIn("pi05", doc["services"])
        for service in doc["services"].values():
            for url in SERVERS.values():
                self.assertIn(f"--server", service["command"])
            self.assertTrue(
                any("host.docker.internal" in a for a in service["command"]),
                "the servers must be addressed on the host, not as a service name",
            )

    def test_the_mounts_are_what_the_command_refers_to(self):
        plan = with_workers(make_plan(scenarios=2, checkpoints=("pi0",)), 1)
        doc = self._doc(plan, catalog_dir="./cat", results_dir="./res",
                        plan_file="./p.json")
        service = next(iter(doc["services"].values()))
        targets = {v.split(":")[1] for v in service["volumes"]}
        self.assertEqual(targets, {PLAN_PATH, CATALOG_PATH, RESULTS_PATH})
        self.assertIn(f"./p.json:{PLAN_PATH}:ro", service["volumes"])
        # The results mount is the only writable one.
        self.assertTrue(any(v.endswith(f":{RESULTS_PATH}") for v in service["volumes"]))

    def test_no_obsolete_version_key(self):
        text = render_compose(
            with_workers(make_plan(scenarios=2, checkpoints=("pi0",)), 1), settings()
        )
        self.assertNotIn("version:", text)

    def test_restart_is_never(self):
        """Resume is group-granular, so a flapping container would re-run whole
        groups unattended."""
        doc = self._doc(with_workers(make_plan(scenarios=2, checkpoints=("pi0",)), 1))
        for service in doc["services"].values():
            self.assertEqual(service["restart"], "no")


class TestTheRendererRefusesRatherThanEmitting(unittest.TestCase):
    def test_a_plan_whose_workers_cannot_run_is_refused(self):
        """The check the renderer is the cheapest place for.

        Measured on the real LIBERO plan: twelve workers, 540 of 600 groups
        refused. Without this the renderer emits twelve services, eleven of which
        pull an image and construct a simulator before dying on preflight -- and
        the plan contained the answer all along.
        """
        from refractal.schema.plan import PlannedWorker

        plan = make_plan(scenarios=4, checkpoints=("pi0",))
        scene = plan.scenes[0]
        episodes = list(scene.workers[0].episodes)
        # Split the SCENARIO range between two workers, which is what the planner
        # does by default and what the harness's zero-based counter cannot express.
        object.__setattr__(scene, "workers", [
            PlannedWorker(worker_id=f"{scene.scene_id}/0", scene_id=scene.scene_id,
                          cpuset="0", episodes=episodes[:2]),
            PlannedWorker(worker_id=f"{scene.scene_id}/1", scene_id=scene.scene_id,
                          cpuset="1", episodes=episodes[2:]),
        ])
        with self.assertRaises(RenderError) as ctx:
            compose_services(plan, settings())
        message = str(ctx.exception)
        self.assertIn("cannot run on the vla-eval backend", message)
        self.assertIn("--workers-per-scene 1", message)

    def test_a_missing_server_is_refused_at_render_time(self):
        plan = with_workers(make_plan(scenarios=2, checkpoints=("pi0", "pi05")), 1)
        with self.assertRaises(RenderError) as ctx:
            compose_services(plan, settings(servers={"pi0": "ws://h:8000"}))
        self.assertIn("pi05", str(ctx.exception))

    def test_an_unknown_engine_is_refused(self):
        plan = with_workers(make_plan(scenarios=2, checkpoints=("pi0",)), 1)
        object.__setattr__(plan.scenes[0], "engine", "isaac")
        with self.assertRaises(RenderError) as ctx:
            compose_services(plan, settings())
        self.assertIn("isaac", str(ctx.exception))

    def test_a_plan_with_no_workers_is_refused(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        object.__setattr__(plan.scenes[0], "workers", [])
        with self.assertRaises(RenderError):
            compose_services(plan, settings())


class TestWorkerRestrictionIsAFilterNotAPlan(unittest.TestCase):
    """The second check: the backend changes where episodes run, not which exist.

    Same shape as the test asserting `serial` and `concurrent` write identical
    ids. If a per-worker run and a single-process run disagreed about which
    episodes exist, the deployment would be affecting the measurement.
    """

    def test_identity_is_untouched(self):
        plan = with_workers(make_plan(scenarios=4, seeds=(0, 1), checkpoints=("pi0", "pi05")), 2)
        one = restrict_to_worker(plan, plan.scenes[0].workers[0].worker_id)
        self.assertEqual(one.plan_id, plan.plan_id)
        self.assertEqual(one.catalog_hash, plan.catalog_hash)
        self.assertEqual(
            [s.scene_hash for s in one.scenes],
            [s.scene_hash for s in plan.scenes if any(
                w.worker_id == plan.scenes[0].workers[0].worker_id for w in s.workers)],
        )

    def test_the_union_of_workers_is_the_whole_plan(self):
        plan = with_workers(make_plan(scenarios=4, seeds=(0, 1), checkpoints=("pi0", "pi05")), 2)
        whole = sorted(e.episode_id for s in plan.scenes for w in s.workers
                       for e in w.episodes)
        pieces: list[str] = []
        for worker_id in [w.worker_id for s in plan.scenes for w in s.workers]:
            part = restrict_to_worker(plan, worker_id)
            pieces += [e.episode_id for s in part.scenes for w in s.workers
                       for e in w.episodes]
        self.assertEqual(sorted(pieces), whole)
        self.assertEqual(len(pieces), len(set(pieces)), "no episode in two workers")

    def test_accounting_is_recomputed_not_inherited(self):
        """Leaving total_episodes describing the whole plan would make a
        per-worker progress line lie."""
        # Two scenes, so restricting to one worker genuinely drops episodes.
        # `make_plan` gives one task per scene, and the helper splits by task, so
        # a single-scene plan puts everything in one worker and this would
        # compare 16 with 16 -- which it did, and the assertion caught it.
        plan = with_workers(
            make_plan(scenarios=4, seeds=(0, 1), checkpoints=("pi0", "pi05"),
                      scenes=("a", "b")), 1)
        one = restrict_to_worker(plan, plan.scenes[0].workers[0].worker_id)
        self.assertLess(one.total_episodes, plan.total_episodes)
        self.assertEqual(
            one.total_episodes,
            sum(len(w.episodes) for s in one.scenes for w in s.workers),
        )

    def test_scenes_with_no_workers_are_dropped(self):
        """A backend iterating scenes would otherwise construct one and find
        nothing to do, and a LIBERO scene costs twelve seconds to construct."""
        plan = with_workers(
            make_plan(scenarios=2, checkpoints=("pi0",), scenes=("a", "b")), 1
        )
        one = restrict_to_worker(plan, "a/0")
        self.assertEqual([s.scene_id for s in one.scenes], ["a"])

    def test_an_unknown_worker_names_the_real_ones(self):
        plan = with_workers(make_plan(scenarios=2, checkpoints=("pi0",)), 1)
        with self.assertRaises(PlanSchemaError) as ctx:
            restrict_to_worker(plan, "nope")
        self.assertIn("libero-0/0", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
