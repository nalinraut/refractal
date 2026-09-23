"""`refractal render --target compose`: a pure function of a plan.

No Docker anywhere in this file, which is the point of doing the renderer first:
the whole of it is assertions on a YAML document, and a rendered file somebody can
read is most of the value with none of the risk.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
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
    render_k8s,
)
from refractal.schema.plan import PlanSchemaError, restrict_to_worker
from tests.test_vla_eval_loop import make_plan

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
        whichever checkpoint started second."""
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
        self.assertIn("partition_unit: task", message)

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


class TestGpuReservationMatchesComposeSchema(unittest.TestCase):
    """The device-reservation shape is Docker's, and it validates the file.

    Written the wrong way first: `capabilities: [["gpu"]]`, nested, which
    renders and reads plausibly and which `docker compose up` rejects with
    "capabilities.0 must be a string". The renderer has no way to know, so the
    shape is pinned here rather than left to memory.
    """

    def _service(self, **kw):
        plan = make_plan(scenarios=1, checkpoints=("pi0",))
        doc = yaml.safe_load(render_compose(plan, ComposeSettings(
            servers={"pi0": "ws://h:8000"}, **kw)))
        return next(iter(doc["services"].values()))

    def test_no_deploy_key_without_gpus(self):
        self.assertNotIn("deploy", self._service())

    def test_capabilities_is_a_list_of_strings(self):
        device = self._service(gpus=1)["deploy"]["resources"]["reservations"]["devices"][0]
        self.assertEqual(device["capabilities"], ["gpu"])
        for item in device["capabilities"]:
            self.assertIsInstance(item, str)
        self.assertEqual(device["count"], 1)
        self.assertEqual(device["driver"], "nvidia")

    def test_video_flags_reach_the_container_command(self):
        command = self._service(record_video=True, frame_every=7)["command"]
        self.assertIn("--video", command)
        self.assertEqual(command[command.index("--frame-every") + 1], "7")

    def test_no_video_flags_when_not_asked_for(self):
        self.assertNotIn("--video", self._service()["command"])


class TestTheRenderCommandAcceptsWhatItEmits(unittest.TestCase):
    """`render` writes the file `run --backend compose` would run.

    They build ComposeSettings from the same fields, so a flag added to one
    parser and not the other leaves the other reading an attribute that does
    not exist. The render tests call `render_compose` directly, so nothing here
    exercised the command line until a plan was rendered from it.
    """

    def _render(self, tmp, *extra):
        from refractal.cli import main

        plan = Path(tmp) / "plan.json"
        plan.write_text(make_plan(scenarios=1, checkpoints=("pi0",)).to_json())
        out = Path(tmp) / "compose.yml"
        code = main(["render", str(plan), "-o", str(out),
                     "--server", "pi0=ws://h:8000", *extra])
        self.assertEqual(code, 0)
        return yaml.safe_load(out.read_text())

    def test_render_runs_without_the_run_only_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self._render(tmp)
            service = next(iter(doc["services"].values()))
            self.assertNotIn("--video", service["command"])
            self.assertNotIn("deploy", service)

    def test_render_emits_video_and_gpus_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self._render(tmp, "--video", "--frame-every", "5", "--gpus", "1")
            service = next(iter(doc["services"].values()))
            self.assertIn("--video", service["command"])
            self.assertEqual(
                service["command"][service["command"].index("--frame-every") + 1], "5")
            self.assertEqual(
                service["deploy"]["resources"]["reservations"]["devices"][0]["count"], 1)


def _write_minimal_plan(path):
    """The smallest plan the renderer accepts, written to disk."""
    import json

    plan = with_workers(make_plan(scenarios=2, checkpoints=("pi0",)), 1)
    json.dump(plan.model_dump(mode="json"), open(path, "w"))


class TestBothEntryPointsRenderTheSameOwnership(unittest.TestCase):
    """`run --backend compose` and `render` produce the same file, and one of
    them was producing containers that run as root.

    `run` defaulted `user` to the caller's uid:gid. `render` passed
    `args.user` straight through, so omitting the flag emitted no `user:` key
    at all -- and a container with no `user:` is root.

    **The failure is quiet in the worst direction.** The run succeeds, because
    root can write anywhere. What it leaves behind is a results tree the person
    who launched it does not own: they cannot delete it, cannot overwrite it,
    and a later run resuming into it fails on a permission error that says
    nothing about where the ownership came from.

    Found by a smoke test whose output directory could not be removed
    afterwards -- and the rendered file already carried a comment warning about
    root-owned bind mounts, which the renderer was itself creating.
    """

    def _render_via_cli(self, tmp, *extra):
        """Through the command handler, because that is where the defaulting
        lives -- ComposeSettings would happily take None either way."""
        import os

        from refractal.cli import main

        plan_path = f"{tmp}/plan.json"
        _write_minimal_plan(plan_path)
        out = f"{tmp}/compose.yml"
        code = main(["render", plan_path, "--target", "compose",
                     "--server", "pi0=ws://h:1", "--results", tmp,
                     "-o", out, *extra])
        self.assertEqual(code, 0)
        doc = yaml.safe_load(open(out).read())
        return next(iter(doc["services"].values()))

    def test_render_without_user_still_pins_the_caller(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            service = self._render_via_cli(tmp)
        self.assertEqual(service.get("user"), f"{os.getuid()}:{os.getgid()}",
                         "a container with no user: runs as root")

    def test_an_explicit_user_still_wins(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            service = self._render_via_cli(tmp, "--user", "4242:4242")
        self.assertEqual(service.get("user"), "4242:4242")


class TestTheKubernetesTarget(unittest.TestCase):
    """k8s was deferred, not rejected, and what deferred it was wrong.

    The execute module has always described k8s as "rendered from plan.json,
    never hand-written". What held it up was a claim in the design notes that
    the harness hardcodes its model server to localhost, making multi-host look
    blocked upstream -- a claim that came from a grep whose only hits were a
    docstring example and some help text, and which was false: the URL is an
    overridable default.
    """

    def _docs(self, **kw):
        # A cluster-reachable address by default. The shared `settings` helper
        # uses host.docker.internal, which this target refuses on purpose --
        # and which is why every one of these errored the first time they ran.
        kw.setdefault("servers", {"pi0": "ws://10.0.0.5:8000"})
        self.plan = with_workers(make_plan(scenarios=2, checkpoints=("pi0",)), 2)
        text = render_k8s(self.plan, settings(**kw))
        return [d for d in yaml.safe_load_all(text) if d]

    def test_one_job_per_worker(self):
        docs = self._docs()
        # Counted from the plan, not hardcoded: `with_workers` splits by TASK,
        # so asking for two workers on a one-task plan gives one worker with
        # episodes -- which the first version of this test asserted away.
        expected = sum(len(s.workers) for s in self.plan.scenes)
        self.assertEqual(len(docs), expected)
        self.assertEqual({d["kind"] for d in docs}, {"Job"})
        self.assertEqual(len({d["metadata"]["name"] for d in docs}), expected,
                         "a name collision would silently drop a worker")

    def test_the_worker_id_is_in_the_command(self):
        """Not derived from a completion index. An Indexed Job would need an
        index-to-worker lookup inside the container, putting part of the
        placement where nobody reading the manifest can see it."""
        docs = self._docs()
        command = docs[0]["spec"]["template"]["spec"]["containers"][0]["command"]
        self.assertIn("--worker", command)
        ids = {w.worker_id for s in self.plan.scenes for w in s.workers}
        self.assertIn(command[command.index("--worker") + 1], ids)

    def test_backoff_is_zero(self):
        """Resume is group-granular: a retried worker re-runs its whole group.
        Kubernetes' default backoffLimit of 6 would do that six times,
        unattended, which is the same reasoning behind Compose's restart: no."""
        for doc in self._docs():
            self.assertEqual(doc["spec"]["backoffLimit"], 0)
            self.assertEqual(
                doc["spec"]["template"]["spec"]["restartPolicy"], "Never")

    def test_the_user_reaches_the_pod_security_context(self):
        """The root-ownership hazard is the same one the Compose target had:
        the run succeeds and leaves results their owner cannot delete."""
        ctx = self._docs(user="4242:4243")[0]["spec"]["template"]["spec"]
        self.assertEqual(ctx["securityContext"]["runAsUser"], 4242)
        self.assertEqual(ctx["securityContext"]["runAsGroup"], 4243)
        self.assertEqual(ctx["securityContext"]["fsGroup"], 4243)

    def test_requests_equal_limits(self):
        """Which is what puts the pod in Guaranteed QoS -- the precondition for
        exclusive cores under cpuManagerPolicy=static. Without equality there
        is no pinning available at all, however the kubelet is configured."""
        res = self._docs()[0]["spec"]["template"]["spec"]["containers"][0]["resources"]
        self.assertEqual(res["requests"], res["limits"])

    def test_scratch_is_not_a_hostPath(self):
        """Per-pod and disposable. A hostPath would put two workers' harness
        scratch in one directory."""
        volumes = {v["name"]: v for v in
                   self._docs()[0]["spec"]["template"]["spec"]["volumes"]}
        self.assertIn("emptyDir", volumes["scratch"])
        self.assertNotIn("hostPath", volumes["scratch"])

    def test_a_docker_internal_server_is_refused(self):
        """`host.docker.internal` exists inside Docker and nowhere in a
        cluster. The pod would start, fail to connect, and look like a dead
        server rather than a name that was never going to resolve."""
        with self.assertRaises(RenderError) as ctx:
            self._docs(servers={"pi0": "ws://host.docker.internal:8000"})
        self.assertIn("host.docker.internal", str(ctx.exception))

    def test_the_plan_label_selects_one_experiment(self):
        """So every pod of one comparison is addressable:
        `kubectl get jobs -l refractal.dev/plan=<id>`."""
        docs = self._docs()
        labels = {d["metadata"]["labels"]["refractal.dev/plan"] for d in docs}
        self.assertEqual(len(labels), 1)
