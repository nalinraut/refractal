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

    def image_for(self, engine: str) -> str:
        image = {**DEFAULT_IMAGES, **self.images}.get(engine)
        if image is None:
            raise RenderError(
                f"no image for engine {engine!r}. Known: "
                f"{sorted({**DEFAULT_IMAGES, **self.images})}. An engine decides the "
                "image because the image exists to provide the engine; pass "
                "images={...} to name one."
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
        "image": settings.image_for(scene.engine),
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
