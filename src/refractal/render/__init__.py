"""Turn a plan into something a runtime can start. Nothing about execution.

A renderer reads ``plan.json`` and emits a deployment description. It does not
decide what runs -- the planner did that, and every worker assignment, cpuset and
device field is already in the plan. This turns those into a Compose file whose
services invoke the ``--worker`` entrypoint.

The line this package sits on
-----------------------------

`plan` is the compiler and `execute` is the runtime. A renderer is neither: it is
a translation of a compiled artifact into one runtime's vocabulary, and it must
not be able to change the experiment. So it is a pure function of
``(plan, render-time settings)`` with no filesystem and no Docker, and the
`--backend compose` verb is a thin shell over it.

Which is also why **render-time settings are parameters and never plan fields**.
A uid, an image tag, a host address and a set of mount points are facts about
whoever is running, not about the experiment. The rule from the package docstring:
anything that differs between two people running the same experiment is
render-time. If a uid reached ``plan.json`` it would reach ``plan_id``, and two
people running one experiment would produce artifacts that could not join.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..schema.errors import RefractalError
from ..schema.plan import Plan, PlannedScene, PlannedWorker


class RenderError(RefractalError):
    """The plan cannot be expressed in this runtime's terms."""


#: Engine to container image. The engine decides the image because the engine is
#: what the image exists to provide -- a LIBERO worker needs robosuite, MuJoCo and
#: LIBERO's asset tree, and nothing about the checkpoint changes that.
DEFAULT_IMAGES = {
    "mujoco": "refractal-libero:local",
    "mjx": "refractal-libero:local",
}

#: Where a container finds the things it is given. Fixed, not configurable: they
#: are the container's own filesystem layout, so making them settings would invite
#: two people to disagree about them for no benefit.
PLAN_PATH = "/plan/plan.json"
CATALOG_PATH = "/catalog"
RESULTS_PATH = "/results"
SCRATCH_PATH = "/scratch"


@dataclass
class ComposeSettings:
    """Everything the renderer needs that is not in the plan.

    All of it render-time by the two-question test: would two people running this
    experiment reasonably differ on it (yes), and would a difference make their
    results incomparable (no).
    """

    #: ``checkpoint_id -> ws://host:port``. Points at the HOST, not at a service.
    #: The model servers stay outside Compose: they are on the GPU, bare and warm,
    #: and a supervised server would put its model load and first-inference JIT
    #: inside the first episode of whichever checkpoint started second.
    servers: dict[str, str]
    #: Emitted as a literal, e.g. ``"1000:1000"``. Never ``"${UID}:${GID}"`` --
    #: neither is exported by default in a POSIX shell, so the interpolation
    #: yields ``":"`` and Compose either errors or silently runs as root.
    user: str | None = None
    #: Host paths for the three mounts, relative or absolute.
    plan_file: str = "./plan.json"
    catalog_dir: str | None = "./catalog"
    results_dir: str = "./results"
    #: Engine -> image override.
    images: dict[str, str] = field(default_factory=dict)
    #: Extra read-only mounts, ``host:container``. For iteration: the image sets
    #: PYTHONPATH to /opt/refractal/src and /opt/refractal-libero/src rather than
    #: baking a wheel, so editing source does not mean rebuilding 8 GB. Bake it
    #: once it stops changing, and drop these.
    source_mounts: list[str] = field(default_factory=list)
    #: Added to every service when the servers are addressed as
    #: ``host.docker.internal``, which on Linux needs the mapping declared.
    host_gateway: bool = True
    #: Capture episode frames inside the container. Safe to set: the worker
    #: fails if frames were requested and none arrived, so a container that
    #: cannot render stops rather than finishing quietly with nothing.
    record_video: bool = False
    frame_every: int = 10
    #: GPUs reserved per worker. Rendering needs one -- the image runs MuJoCo
    #: under EGL -- and without a reservation the container starts, builds an
    #: environment and dies on its first frame rather than at preflight.
    gpus: int = 0

    def image_for(self, engine: str, provider: str | None = None) -> str:
        """The image for one scene: by provider if it has one, else by engine.

        Engine alone is too coarse, and the comment that used to sit here --
        "an engine decides the image because the image exists to provide the
        engine" -- described an image that does not exist.
        `refractal-libero:local` is 8.94 GB of MuJoCo AND LIBERO AND the
        harness: named for the engine, carrying the benchmark. That holds
        while one engine means one benchmark, and breaks the moment a catalog
        wants LIBERO and RoboCasa in one comparison, since both are `mujoco`.

        What belongs in the catalog is the REQUIREMENT, not the image, and it
        is already there: an external scene declares its provider, which is
        exactly "what this scene needs", is already part of identity, and
        already survives the compile into plan.json -- its own docstring says
        it is carried "so a backend can name the benchmark that owns this
        scene without the catalog".

        So the provider is the key when there is one, and the engine is the
        fallback, which is what a local MJCF scene has and all it needs.

        The provider is an import string, so a key may name any dotted or
        colon-separated token of it: ``--image libero=...`` matches
        ``vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark``. Two keys
        matching one provider is refused rather than resolved by declaration
        order -- picking one silently is how a scene ends up in the wrong
        image, and the wrong image is a different benchmark.
        """
        images = {**DEFAULT_IMAGES, **self.images}
        if provider:
            if provider in images:
                return images[provider]
            tokens = {t.lower() for t in re.split(r"[.:]", provider) if t}
            # Only user-supplied keys token-match. Engine names stay reserved
            # for the fallback, so a provider whose path happens to contain
            # "mujoco" cannot capture the engine default.
            hits = sorted(
                k for k in self.images
                if k not in DEFAULT_IMAGES and k.lower() in tokens
            )
            if len(hits) > 1:
                raise RenderError(
                    f"image keys {hits} all match provider {provider!r}. One "
                    "scene cannot take two images, and choosing by order would "
                    "put it in whichever happened to be declared first. Name a "
                    "key that identifies this provider uniquely."
                )
            if hits:
                return images[hits[0]]
        image = images.get(engine)
        if image is None:
            raise RenderError(
                f"no image for engine {engine!r}"
                + (f" or provider {provider!r}" if provider else "")
                + f". Known: {sorted(images)}. Key images by the provider when "
                "two benchmarks share an engine, or by the engine otherwise."
            )
        return image


def _service_name(worker_id: str) -> str:
    """A Compose service name from a worker id.

    Worker ids contain ``/``, which Compose does not allow in a service name. The
    substitution has to be injective or two workers could collapse into one
    service and the second would silently not run -- so it is checked across the
    whole plan rather than assumed from the shape of the ids.
    """
    return worker_id.replace("/", "-")


def compose_services(plan: Plan, settings: ComposeSettings) -> dict[str, Any]:
    """One service per worker. Pure: no filesystem, no Docker, no clock."""
    missing = sorted(
        {e.checkpoint_id for s in plan.scenes for w in s.workers for e in w.episodes}
        - set(settings.servers)
    )
    if missing:
        raise RenderError(
            f"no server URL for checkpoint(s) {missing}. Every checkpoint the plan runs "
            "needs one, and the renderer refuses rather than emitting a service that "
            "would fail preflight inside the container."
        )

    _check_runnable(plan)

    services: dict[str, Any] = {}
    for scene in plan.scenes:
        for worker in scene.workers:
            name = _service_name(worker.worker_id)
            if name in services:
                raise RenderError(
                    f"two workers render to the service name {name!r}. Worker ids are "
                    "unique but the '/'-to-'-' substitution is not injective over all "
                    "possible ids, and a collision would silently drop a worker's "
                    "episodes."
                )
            services[name] = _service(plan, scene, worker, settings)
    if not services:
        raise RenderError(
            "this plan has no workers, so there is nothing to render. A plan with no "
            "workers usually means every scene was filtered out."
        )
    return services


def _check_runnable(plan: Plan) -> None:
    """Refuse a plan whose workers cannot run, before emitting services for them.

    The renderer targets ``--backend vla-eval``, so it can check that backend's
    preconditions -- and the index contract is a pure function of the plan, which
    makes the renderer the cheapest place it can possibly be checked.

    Measured on the LIBERO catalog's default plan: twelve workers, and **540 of
    600 groups refused**. Without this the renderer emits twelve services, eleven
    of them start a container, pull an image, construct a LIBERO environment and
    then die on preflight. The information needed to know that was in the plan
    the whole time.

    The cause is always the same: the harness runs
    ``ep in range(episodes_per_task)`` and hands ``episode_idx = ep`` to the
    benchmark, so a worker holding a later slice of a scenario range runs the
    early indices while every row claims the late ones. One worker per scene
    fixes it, which is what ``partition_unit: task`` expresses.
    """
    from ..execute.vla_eval import BridgeError, check_index_contract, group_by_seed

    refused: list[str] = []
    total = 0
    for scene in plan.scenes:
        scenarios = {x.scenario_hash: x.params for x in scene.scenarios}
        for worker in scene.workers:
            for task_id in sorted({e.task_id for e in worker.episodes}):
                for_task = [e for e in worker.episodes if e.task_id == task_id]
                for checkpoint in sorted({e.checkpoint_id for e in for_task}):
                    for_ckpt = [e for e in for_task if e.checkpoint_id == checkpoint]
                    for seed, group in sorted(group_by_seed(for_ckpt).items()):
                        total += 1
                        try:
                            check_index_contract(group, scenarios)
                        except BridgeError:
                            refused.append(
                                f"{worker.worker_id}/{task_id}/{checkpoint}/seed{seed}"
                            )
    if refused:
        shown = ", ".join(refused[:3])
        raise RenderError(
            f"{len(refused)} of {total} episode group(s) in this plan cannot run on the "
            f"vla-eval backend, starting with {shown}. The harness counts episodes from "
            "zero within a task, so a worker holding a later slice of a scenario range "
            "runs the early init states while every row claims the late ones.\n\n"
            "Declare `partition_unit: task` on the scene's resource shape. Rendering this would start "
            f"{sum(len(s.workers) for s in plan.scenes)} containers, most of which would "
            "pull an image and construct a simulator before failing preflight -- and the "
            "plan said so all along."
        )


def _service(
    plan: Plan, scene: PlannedScene, worker: PlannedWorker, settings: ComposeSettings
) -> dict[str, Any]:
    command = [
        "refractal", "run", PLAN_PATH,
        "--worker", worker.worker_id,
        "--backend", "vla-eval",
        "-o", RESULTS_PATH,
        "--harness-output", SCRATCH_PATH,
    ]
    for checkpoint in sorted(settings.servers):
        command += ["--server", f"{checkpoint}={settings.servers[checkpoint]}"]
    if settings.catalog_dir is not None:
        command += ["--catalog", CATALOG_PATH]
    if settings.record_video:
        command += ["--video", "--frame-every", str(settings.frame_every)]

    volumes = [f"{settings.plan_file}:{PLAN_PATH}:ro"]
    if settings.catalog_dir is not None:
        volumes.append(f"{settings.catalog_dir}:{CATALOG_PATH}:ro")
    volumes.append(f"{settings.results_dir}:{RESULTS_PATH}")
    volumes += [f"{m}:ro" for m in settings.source_mounts]

    service: dict[str, Any] = {
        "image": settings.image_for(
            scene.engine, getattr(scene.external, "provider", None)),
        "command": command,
        "volumes": volumes,
        # Never restart. A crashed worker that restarts re-runs its whole group
        # -- resume is group-granular -- and would do it unattended, so a
        # flapping container could quietly burn a night of GPU on one worker.
        "restart": "no",
    }
    if settings.user:
        service["user"] = settings.user
    if worker.cpuset:
        # Compose has no scheduler. Without a cpuset the workers contend and the
        # timings become noise; the planner already computed the pinning.
        service["cpuset"] = worker.cpuset
    shape = scene.resource_shape
    service["mem_limit"] = f"{shape.memory_mb}m"
    if settings.host_gateway:
        service["extra_hosts"] = ["host.docker.internal:host-gateway"]
    if settings.gpus:
        # Compose's own device-reservation shape. `gpus:` as a top-level key is
        # the older form and is ignored by some versions without saying so.
        # `capabilities` is a flat list of strings. Nested, Compose rejects the
        # file at validation: "capabilities.0 must be a string". The schema is
        # Docker's, so the shape is pinned by a test rather than by memory.
        service["deploy"] = {"resources": {"reservations": {"devices": [
            {"driver": "nvidia", "count": settings.gpus,
             "capabilities": ["gpu"]}]}}}
    return service


def _k8s_name(worker_id: str) -> str:
    """A worker id as a Kubernetes object name.

    RFC 1123: lowercase alphanumerics and ``-``, starting and ending
    alphanumeric. Worker ids carry ``/``, and scene ids can carry anything a
    catalog author typed, so this is a substitution rather than a pass-through
    -- and like the Compose one it is checked for collisions across the plan
    rather than assumed injective.
    """
    safe = "".join(c if c.isalnum() else "-" for c in worker_id.lower())
    safe = safe.strip("-")
    return safe or "worker"


def render_k8s(plan: Plan, settings: ComposeSettings) -> str:
    """The plan as Kubernetes Jobs. One Job per worker, one YAML document each.

    **Deferred, not rejected.** The module docstring has always said k8s is
    "rendered from plan.json, never hand-written"; what held it up was a claim
    in the design notes that the harness hardcodes its model server to
    localhost, which made multi-host look blocked upstream. That claim was
    wrong -- the URL is a default, overridable -- and it came from a grep whose
    only hits were a docstring example and some help text.

    One Job per worker rather than a single Indexed Job. The shards are not
    interchangeable: each carries a specific worker id and its own pinning, so
    an Indexed Job would need a completion-index-to-worker lookup inside the
    container, which puts part of the placement somewhere nobody reading the
    manifest can see.

    Three places where Kubernetes cannot do what Compose does, all of them
    stated in the rendered file rather than discovered later:

    * **Pinning.** ``cpuset`` has no portable equivalent. Integer CPU limits on
      a Guaranteed pod get exclusive cores only when the kubelet runs
      ``cpuManagerPolicy: static``; otherwise the shares are advisory and the
      timings become noise the way they do with no pinning at all.
    * **Results.** ``hostPath`` is rendered because this is how a single node
      runs it, and it is wrong the moment two workers land on different nodes:
      each would write into a different filesystem and the comparison would be
      split in two without an error. A real cluster needs one ReadWriteMany
      volume.
    * **Restart.** ``backoffLimit: 0`` alongside ``restartPolicy: Never``,
      because resume is group-granular: a retried worker re-runs its whole
      group, unattended, and the default backoff of 6 would do it six times.
    """
    missing = sorted(
        {e.checkpoint_id for s in plan.scenes for w in s.workers for e in w.episodes}
        - set(settings.servers)
    )
    if missing:
        raise RenderError(
            f"no server URL for checkpoint(s) {missing}. Every checkpoint the "
            "plan runs needs one."
        )
    docker_only = sorted(
        c for c, url in settings.servers.items() if "host.docker.internal" in url
    )
    if docker_only:
        raise RenderError(
            f"checkpoint(s) {docker_only} address the model server as "
            "'host.docker.internal', which exists inside Docker and nowhere in "
            "a Kubernetes cluster. The pod would start, fail to connect, and "
            "the failure would look like a dead server rather than a name that "
            "was never going to resolve. Give a URL reachable from the cluster."
        )
    _check_runnable(plan)

    docs: list[str] = []
    seen: dict[str, str] = {}
    for scene in plan.scenes:
        for worker in scene.workers:
            name = _k8s_name(worker.worker_id)
            if name in seen:
                raise RenderError(
                    f"workers {seen[name]!r} and {worker.worker_id!r} both "
                    f"render to the object name {name!r}; one would silently "
                    "replace the other and its episodes would never run."
                )
            seen[name] = worker.worker_id
            docs.append(_job(plan, scene, worker, settings, name))
    if not docs:
        raise RenderError("this plan has no workers, so there is nothing to render.")

    header = (
        f"# Generated by `refractal render --target k8s` from plan "
        f"{plan.plan_id[:19]}...\n"
        f"# {sum(len(w.episodes) for s in plan.scenes for w in s.workers)} "
        f"episode(s) across {len(seen)} worker(s).\n"
        "#\n"
        "# The model servers are NOT here, exactly as in the Compose target:\n"
        "# they stay warm outside the batch, because a supervised server puts\n"
        "# its model load and first-inference JIT inside the first episode of\n"
        "# whichever checkpoint started second.\n"
        "#\n"
        "# THREE THINGS THIS FILE CANNOT PROMISE, and they are not footnotes:\n"
        "#\n"
        "#   hostPath results  -- correct on ONE node. Two workers on two nodes\n"
        "#     write into two filesystems and the comparison splits in half\n"
        "#     with no error anywhere. Use one ReadWriteMany volume.\n"
        "#\n"
        "#   cpu limits        -- NOT the cpuset the planner computed. Exclusive\n"
        "#     cores need kubelet cpuManagerPolicy=static and Guaranteed QoS;\n"
        "#     without it these are shares and the timings are noise.\n"
        "#\n"
        "#   backoffLimit: 0   -- deliberate. Resume is group-granular, so a\n"
        "#     retried worker re-runs its whole group. The default of 6 would\n"
        "#     do that six times, unattended.\n"
    )
    return header + "".join("---\n" + d for d in docs)


def _job(plan, scene, worker, settings: ComposeSettings, name: str) -> str:
    from pathlib import Path

    import yaml

    command = [
        "refractal", "run", PLAN_PATH,
        "--worker", worker.worker_id,
        "--backend", "vla-eval",
        "-o", RESULTS_PATH,
        "--harness-output", SCRATCH_PATH,
    ]
    for checkpoint in sorted(settings.servers):
        command += ["--server", f"{checkpoint}={settings.servers[checkpoint]}"]
    if settings.catalog_dir is not None:
        command += ["--catalog", CATALOG_PATH]
    if settings.record_video:
        command += ["--video", "--frame-every", str(settings.frame_every)]

    mounts = [{"name": "plan", "mountPath": PLAN_PATH,
               "subPath": "plan.json", "readOnly": True},
              {"name": "results", "mountPath": RESULTS_PATH},
              {"name": "scratch", "mountPath": SCRATCH_PATH}]
    volumes = [
        {"name": "plan", "hostPath": {
            "path": str(Path(settings.plan_file).resolve().parent),
            "type": "Directory"}},
        {"name": "results", "hostPath": {
            "path": str(Path(settings.results_dir).resolve()),
            "type": "DirectoryOrCreate"}},
        # Harness scratch is per-pod and disposable; it must not be a hostPath
        # or two workers would share one directory.
        {"name": "scratch", "emptyDir": {}},
    ]
    if settings.catalog_dir is not None:
        mounts.append({"name": "catalog", "mountPath": CATALOG_PATH,
                       "readOnly": True})
        volumes.append({"name": "catalog", "hostPath": {
            "path": str(Path(settings.catalog_dir).resolve()),
            "type": "Directory"}})
    for index, spec in enumerate(settings.source_mounts):
        host, _, container = spec.partition(":")
        mounts.append({"name": f"src{index}", "mountPath": container,
                       "readOnly": True})
        volumes.append({"name": f"src{index}",
                        "hostPath": {"path": host, "type": "Directory"}})

    shape = scene.resource_shape
    cpus = len(worker.cpuset.split(",")) if worker.cpuset else 1
    resources: dict[str, Any] = {
        # Requests equal limits, which is what puts the pod in Guaranteed QoS --
        # the precondition for exclusive cores under cpuManagerPolicy=static.
        # Without that policy this is a share, not a pin. Said in the header.
        "requests": {"cpu": str(cpus), "memory": f"{shape.memory_mb}Mi"},
        "limits": {"cpu": str(cpus), "memory": f"{shape.memory_mb}Mi"},
    }
    if settings.gpus:
        resources["limits"]["nvidia.com/gpu"] = str(settings.gpus)

    container: dict[str, Any] = {
        "name": "worker",
        "image": settings.image_for(
            scene.engine, getattr(scene.external, "provider", None)),
        "command": command,
        "volumeMounts": mounts,
        "resources": resources,
    }
    pod: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": volumes,
    }
    if settings.user:
        uid, _, gid = settings.user.partition(":")
        # The same hazard the Compose target carries: without this the
        # container runs as root and leaves a results tree its owner cannot
        # delete, while the run itself succeeds.
        pod["securityContext"] = {"runAsUser": int(uid),
                                  "runAsGroup": int(gid or uid),
                                  "fsGroup": int(gid or uid)}
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "labels": {
                # The comparison id, so every pod of one experiment is
                # selectable -- `kubectl get jobs -l refractal.dev/plan=<id>`.
                "refractal.dev/plan": plan.plan_id.split(":", 1)[-1][:32],
                "refractal.dev/scene": _k8s_name(scene.scene_id),
            },
        },
        "spec": {
            "completions": 1,
            "parallelism": 1,
            "backoffLimit": 0,
            "template": {"metadata": {"labels": {
                "refractal.dev/plan": plan.plan_id.split(":", 1)[-1][:32]}},
                "spec": pod},
        },
    }
    return yaml.safe_dump(job, sort_keys=False, default_flow_style=False)


def render_compose(plan: Plan, settings: ComposeSettings) -> str:
    """The Compose file, as YAML.

    No ``version:`` key -- it has been obsolete since Compose v2 and current
    versions warn about it.
    """
    import yaml

    document = {"services": compose_services(plan, settings)}
    header = (
        f"# Generated by `refractal render` from plan {plan.plan_id[:19]}...\n"
        f"# {plan.total_episodes} episode(s) across "
        f"{sum(len(s.workers) for s in plan.scenes)} worker(s).\n"
        "#\n"
        "# The model servers are NOT here. They run on the host, bare and warm,\n"
        "# because a supervised server would put its model load and first-inference\n"
        "# JIT inside the first episode of whichever checkpoint started second.\n"
        "#\n"
        "# Create the results directory before `docker compose up`. Docker creates a\n"
        "# missing bind-mount target as root whatever `user:` says, and a non-root\n"
        "# container then cannot write into it -- quietly, since the run completes and\n"
        "# `compare` fails later on somebody else's machine.\n"
    )
    return header + yaml.safe_dump(document, sort_keys=True, default_flow_style=False)


__all__ = [
    "render_k8s",
    "CATALOG_PATH",
    "ComposeSettings",
    "DEFAULT_IMAGES",
    "PLAN_PATH",
    "RESULTS_PATH",
    "RenderError",
    "SCRATCH_PATH",
    "compose_services",
    "render_compose",
]
