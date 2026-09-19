"""Fitting work to machines: how many workers, on which device, at what cost.

This is the part the spec got wrong, so the correction is worth stating plainly.

The implementation prompt says::

    workers_for_scene = ceil(episodes / envs_per_process)

Under `mujoco`, ``envs_per_process`` is 1, so 204 episodes gives 204 workers --
against the API reference's own worked example, which prints 8. The formula
treats a *batch width* as a *capacity*.

``envs_per_process`` says how many environments one worker holds at once, and
therefore what that worker costs in CPU and VRAM. It does not say how many
workers exist. Worker count comes from what the host has:

    useful_ceiling  = ceil(episodes / envs_per_process)   # beyond this a worker
                                                          # cannot fill its batch
    affordable      = what CPU, RAM and VRAM allow after the model servers
    workers         = min(useful_ceiling, affordable share)

So the original formula is real, but it is an upper bound rather than a count --
past it you are paying for a worker that runs a part-empty tensor.

Allocation across scenes is greedy on makespan: hand the next worker to whichever
scene currently finishes last. Total time is the slowest scene, so minimising the
maximum is the objective, and one worker at a time is both optimal enough and
explainable in a sentence -- which matters for a file people are meant to read
and review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..schema.errors import RefractalError
from ..schema.models import Checkpoint, HardwareProfile, ResourceShape


class CapacityError(RefractalError):
    """The plan will not fit. Better than a run that dies at episode 300."""


@dataclass
class SceneDemand:
    """What one scene needs, before any placement decision."""

    scene_id: str
    episodes: int
    shape: ResourceShape
    #: Longest ``max_steps`` across the tasks on this scene. The cost estimate
    #: is an upper bound: it assumes every episode runs to the step limit, which
    #: only happens when nothing ever succeeds early.
    max_steps: int
    #: How many times a worker pays ``startup_sec``.
    #:
    #: Not once. A backend that can only address one checkpoint and one seed per
    #: invocation -- which the vla-eval bridge is, because the harness has no seed
    #: concept and one server per run -- pays scene construction once per
    #: (checkpoint x seed). Measured on the LIBERO catalog: 12 workers x 2
    #: checkpoints x 12s = 288s of env construction against a plan that estimated
    #: 95s for the whole run.
    #:
    #: A cost estimate whose only job is to be right before you spend, being 3x
    #: low on its dominant term, is the estimate failing at the one thing it is
    #: for.
    invocations_per_worker: int = 1

    def batch_seconds(self) -> float:
        """Wall-clock for one full batch of ``envs_per_process`` episodes."""
        return (self.max_steps / 1000.0) * self.shape.sec_per_1k_steps

    #: Tasks on this scene, needed when `partition_unit` is "task": a worker must
    #: then own a task's whole scenario range, so a scene cannot usefully have
    #: more workers than it has tasks.
    tasks: int = 1

    def useful_worker_ceiling(self) -> int:
        """How many workers this scene can usefully have.

        Two independent limits, and conflating them is what produced a plan whose
        twelve workers each held a slice of one task's scenario range -- 540 of
        600 episode groups refused, and 30 of 40 episodes would have been recorded
        against a scenario that never ran.

        * ``envs_per_process`` bounds how many workers are *useful*: past
          ``episodes / envs_per_process`` a worker has nothing to batch.
        * ``partition_unit`` bounds how many are *legal*: at ``task`` a worker
          must own a task's whole zero-based range, so the scene cannot split
          finer than one worker per task.

        Measured, and they point in opposite directions. LIBERO has
        ``envs_per_process: 1`` (useful ceiling = 600) and ``partition_unit:
        task`` (legal ceiling = 10). MJX has 4096 (useful = 1) and ``scenario``
        (legal = unbounded). The binding constraint is the *other* one on each
        engine, which is why neither field can be derived from the other.
        """
        useful = max(1, math.ceil(self.episodes / self.shape.envs_per_process))
        if self.shape.partition_unit == "task":
            return max(1, min(useful, self.tasks))
        return useful

    def makespan(self, workers: int) -> float:
        """Seconds for this scene at ``workers`` workers, including startup."""
        if workers <= 0:
            return math.inf
        per_worker_episodes = math.ceil(self.episodes / workers)
        batches = math.ceil(per_worker_episodes / self.shape.envs_per_process)
        startup = self.shape.startup_sec * self.invocations_per_worker
        return startup + batches * self.batch_seconds()

    def startup_cost(self, workers: int) -> float:
        """Total scene-construction time across all of this scene's workers."""
        return workers * self.shape.startup_sec * self.invocations_per_worker


@dataclass
class DeviceBudget:
    device_id: str
    capacity_mb: int
    used_mb: int = 0

    @property
    def free_mb(self) -> int:
        return self.capacity_mb - self.used_mb


@dataclass
class Allocation:
    workers: dict[str, int] = field(default_factory=dict)
    devices: dict[str, str] = field(default_factory=dict)
    cpusets: dict[str, str] = field(default_factory=dict)
    packed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def budget_model_servers(
    hardware: HardwareProfile, checkpoints: list[Checkpoint]
) -> tuple[list[DeviceBudget], dict[str, str]]:
    """Place one model server per checkpoint, first-fit, before anything else.

    Servers are reserved first because they are the inflexible consumer: a
    benchmark worker can be dropped or deferred, a policy cannot -- with no
    server there is nothing to evaluate. Reserving after would let workers claim
    memory the run cannot start without.
    """
    budgets = [DeviceBudget(d.id, d.vram_mb) for d in hardware.devices]
    placement: dict[str, str] = {}

    if not budgets:
        total = sum(c.vram_mb for c in checkpoints)
        if total:
            raise CapacityError(
                f"hardware profile {hardware.id!r} declares no GPU devices, but the run "
                f"needs {total} MB for {len(checkpoints)} model server(s). Add a device "
                "to hardware.yaml, or use a profile that has one."
            )
        return budgets, placement

    for checkpoint in sorted(checkpoints, key=lambda c: (-c.vram_mb, c.id)):
        target = max(budgets, key=lambda b: (b.free_mb, b.device_id))
        if checkpoint.vram_mb > target.free_mb:
            raise CapacityError(
                f"model server for checkpoint {checkpoint.id!r} needs {checkpoint.vram_mb} MB "
                f"but the roomiest device ({target.device_id}) has {target.free_mb} MB free of "
                f"{target.capacity_mb} MB. "
                f"{len(checkpoints)} concurrent checkpoints do not fit on this hardware -- "
                "reduce run.checkpoints, or set execution_mode: serial and compare across "
                "sessions (which costs you the same-session control)."
            )
        target.used_mb += checkpoint.vram_mb
        placement[checkpoint.id] = target.device_id

    return budgets, placement


def allocate_workers(
    demands: list[SceneDemand],
    hardware: HardwareProfile,
    budgets: list[DeviceBudget],
    *,
    pack_below_startup_sec: int = 30,
    workers_per_scene: int | None = None,
) -> Allocation:
    """Greedy makespan allocation, bounded by CPU, RAM, VRAM and useful ceiling.

    ``workers_per_scene`` caps the split. It exists because a backend can require
    that one worker own a scene's whole scenario range: the vla-eval harness
    counts episodes from zero within a task, so a worker holding init states 5..9
    would run 0..4 while every row claimed 5..9. ``check_index_contract`` refuses
    that, which means the *default* plan for a LIBERO catalog is unrunnable on
    that backend -- measured: 12 of 16 groups refused.

    A cap does not change ``plan_id``. Worker layout is placement, not identity:
    the same experiment split four ways and one way is the same experiment, which
    is why the same catalog can target a backend that shards and one that cannot.
    """
    allocation = Allocation()
    if not demands:
        return allocation

    cpu_free = hardware.cpu_cores
    mem_free = hardware.memory_mb
    max_workers = hardware.max_workers or 10**6
    per_scene_cap = workers_per_scene or 10**6

    for demand in demands:
        allocation.workers[demand.scene_id] = 0

    def affordable(demand: SceneDemand) -> str | None:
        """Which device can host one more worker of this scene, if any.

        Once a scene has a device, every later worker of that scene goes to the
        same one. Not an optimisation -- the VRAM accounting has to debit the
        device the worker actually lands on, and `PlannedScene` records one
        device per scene, so letting workers drift across devices would leave
        the budget describing a placement that is not the one in the plan.
        """
        shape = demand.shape
        if shape.cpu_cores > cpu_free or shape.memory_mb > mem_free:
            return None
        need = shape.vram_mb()
        if need == 0:
            return "cpu"
        pinned = allocation.devices.get(demand.scene_id)
        candidates = [
            b
            for b in budgets
            if b.free_mb >= need and (pinned is None or b.device_id == pinned)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda b: (b.free_mb, b.device_id)).device_id

    placed = 0
    while placed < max_workers:
        # Only scenes that can still use another worker are candidates. Without
        # the ceiling, a 3-episode scene would keep winning workers it cannot
        # fill while a 500-episode scene waits.
        candidates = [
            d
            for d in demands
            if allocation.workers[d.scene_id]
            < min(d.useful_worker_ceiling(), per_scene_cap)
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda d: (-d.makespan(allocation.workers[d.scene_id]), d.scene_id)
        )

        for demand in candidates:
            device = affordable(demand)
            if device is None:
                continue
            # Refuse a worker that costs more startup than the wall clock it
            # saves. Without this the allocator keeps adding workers while
            # makespan still falls by anything at all, and on a scene with
            # expensive construction it buys seconds with minutes.
            current = allocation.workers[demand.scene_id]
            gain = demand.makespan(current) - demand.makespan(current + 1)
            added_startup = demand.shape.startup_sec * demand.invocations_per_worker
            if current > 0 and gain < added_startup:
                allocation.warnings.append(
                    f"scene {demand.scene_id!r}: stopped at {current} worker(s) -- one more "
                    f"would save {gain:.0f}s of wall clock and cost {added_startup:.0f}s of "
                    "scene construction"
                )
                continue
            allocation.workers[demand.scene_id] += 1
            cpu_free -= demand.shape.cpu_cores
            mem_free -= demand.shape.memory_mb
            if device != "cpu":
                next(b for b in budgets if b.device_id == device).used_mb += (
                    demand.shape.vram_mb()
                )
            allocation.devices.setdefault(demand.scene_id, device)
            placed += 1
            break
        else:
            break  # nothing affordable for any scene; stop

    starved = [d for d in demands if allocation.workers[d.scene_id] == 0]
    if starved:
        _pack_or_fail(starved, allocation, pack_below_startup_sec)

    return allocation


def _pack_or_fail(
    starved: list[SceneDemand], allocation: Allocation, pack_below_startup_sec: int
) -> None:
    """Scenes that won no worker share one, sequentially -- if reloading is cheap.

    ``startup_sec`` decides, and it is decisive rather than a detail. At 90 s
    load, packing eleven small scenes costs sixteen minutes of pure loading; at
    3 s it is free. Same algorithm, opposite answer, which is why the threshold
    is a parameter rather than a constant.
    """
    expensive = [d for d in starved if d.shape.startup_sec > pack_below_startup_sec]
    if expensive:
        names = ", ".join(sorted(d.scene_id for d in expensive))
        raise CapacityError(
            f"no capacity left for scene(s) {names}, and their startup_sec exceeds the "
            f"packing threshold ({pack_below_startup_sec}s), so packing them into a shared "
            "worker would cost more in reloads than it saves. Raise max_workers, use "
            "hardware with more cores, or split the run."
        )

    for demand in starved:
        allocation.workers[demand.scene_id] = 0
        allocation.packed.append(demand.scene_id)
        allocation.devices.setdefault(demand.scene_id, "cpu")

    reload_cost = sum(d.shape.startup_sec for d in starved)
    allocation.warnings.append(
        f"{len(starved)} scene(s) packed sequentially into one shared worker "
        f"({', '.join(sorted(d.scene_id for d in starved))}); "
        f"{reload_cost}s of that worker's time is scene loading"
    )


def assign_cpusets(
    demands: list[SceneDemand], allocation: Allocation, hardware: HardwareProfile
) -> dict[str, str]:
    """Hand each worker distinct cores.

    Compose has no scheduler. Without explicit pinning the workers contend, and
    contention does not raise -- it just makes every timing measurement noise,
    differently on each run. For a tool whose output is a claim about a measured
    difference, that is the worst possible failure mode.
    """
    cpusets: dict[str, str] = {}
    cursor = 0
    for demand in sorted(demands, key=lambda d: d.scene_id):
        for shard in range(allocation.workers.get(demand.scene_id, 0)):
            cores = demand.shape.cpu_cores
            if cursor + cores > hardware.cpu_cores:
                cursor = 0  # wrap rather than fail; oversubscription is warned on
            lo, hi = cursor, cursor + cores - 1
            cpusets[f"{demand.scene_id}/{shard}"] = str(lo) if lo == hi else f"{lo}-{hi}"
            cursor += cores
    return cpusets


__all__ = [
    "Allocation",
    "CapacityError",
    "DeviceBudget",
    "SceneDemand",
    "allocate_workers",
    "assign_cpusets",
    "budget_model_servers",
]
