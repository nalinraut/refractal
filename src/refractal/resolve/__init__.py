"""``refractal.resolve`` -- catalog in, ``plan.json`` out.

The intellectual core, and pure functions on data. It imports no Docker, no GPU
library, no simulator and no ``vla_eval``; the only user code it touches is the
generator it must call to produce the scenario list.

It also never branches on ``engine``. If ``if engine == "isaac"`` appears in this
package the abstraction has broken, and the fix is to change what the resource
shape reports, not to add a branch. ``mujoco``, ``mjx`` and ``isaac`` differ here
only as different numbers in the same three columns.
"""

from __future__ import annotations

import math
from pathlib import Path

from .. import __version__
from ..schema.errors import CatalogError
from ..schema.loader import Catalog, load_catalog
from ..schema.plan import (
    PLAN_SCHEMA,
    Plan,
    PlannedScene,
    PlannedWorker,
    read_plan,
)
from .expand import (
    expand_episodes,
    find_type_only_collisions,
    generate_scenarios,
    subsample,
    task_hashes_for,
)
from .fit import (
    Allocation,
    CapacityError,
    SceneDemand,
    allocate_workers,
    assign_cpusets,
    budget_model_servers,
)
from .lock import BuildLock, load_lock


def resolve(
    catalog: Catalog | str | Path,
    *,
    hardware_profile: str,
    pack_below_startup_sec: int = 30,
    created_at: str | None = None,
) -> Plan:
    """Compile a catalog into a plan.

    ``created_at`` is a parameter rather than a call to the clock, so that the
    same catalog produces a byte-identical plan on demand. A planner that reads
    the wall clock cannot be tested for determinism, and determinism is the
    property the whole design leans on.
    """
    if not isinstance(catalog, Catalog):
        catalog = load_catalog(catalog)

    warnings = list(catalog.warnings)
    lock = load_lock(catalog.root)

    # Freshness first, then precedence. The lock wins over computed sources --
    # it has to, for an externally-defined scene -- so a stale entry would
    # otherwise become silently authoritative.
    recorded: dict[str, str] = {}
    if lock is not None:
        for scene in catalog.scenes:
            lock.check_scene_fresh(catalog.root, scene)
            entry = lock.scene_entry(scene.id)
            if entry is not None:
                recorded[scene.id] = entry.scene_hash
    scene_hashes = catalog.scene_hashes(recorded)
    task_hashes = task_hashes_for(catalog)
    hardware = catalog.hardware(hardware_profile)

    # --- scenarios, per scene -------------------------------------------
    scenarios_by_scene: dict[str, list] = {}
    for scenario_set in catalog.active_scenario_sets():
        if lock is not None:
            lock.check_filter_fresh(scenario_set, scene_hashes[scenario_set.scene])
            unverifiable = lock.check_filter_source(scenario_set)
            if unverifiable:
                warnings.append(unverifiable)
        produced = generate_scenarios(scenario_set, lock)
        if scenario_set.filter is not None and lock is not None:
            entry = next(f for f in lock.filters if f.scenario_set_id == scenario_set.id)
            warnings.append(
                f"scenario_set {scenario_set.id!r}: {entry.generated} generated, "
                f"{entry.dropped} dropped by filter, {len(produced)} kept"
            )
        # Dedupe across sets, not just within one. `generate_scenarios` dedupes a
        # single set's output, which is not enough: two sets on one scene can
        # produce the identical scenario, and then the same scenario_hash appears
        # twice, every episode_id derived from it appears twice, and the plan
        # silently runs and records the work twice. Found by checking whether the
        # type-collision case above had a same-type sibling -- it did, and it was
        # the worse of the two.
        bucket = scenarios_by_scene.setdefault(scenario_set.scene, [])
        already = {s.scenario_hash: s.scenario_set_id for s in bucket}
        for scenario in produced:
            prior = already.get(scenario.scenario_hash)
            if prior is not None:
                warnings.append(
                    f"scene {scenario_set.scene!r}: scenario set {scenario_set.id!r} produces a "
                    f"scenario already produced by {prior!r} ({scenario.params}); kept once. "
                    "Overlapping sets are allowed, but the duplicate does not become two "
                    "episodes."
                )
                continue
            already[scenario.scenario_hash] = scenario_set.id
            bucket.append(scenario)

    # --- tier, then expansion -------------------------------------------
    demands: list[SceneDemand] = []
    episodes_by_scene: dict[str, list] = {}
    kept_by_scene: dict[str, list] = {}

    for scene in catalog.scenes:
        raw = scenarios_by_scene.get(scene.id, [])
        if not raw:
            continue
        for first, second in find_type_only_collisions(raw):
            key = next(
                k
                for k in first.params
                if type(first.params[k]) is not type(second.params[k])
            )
            warnings.append(
                f"scene {scene.id!r}: scenario sets {first.scenario_set_id!r} and "
                f"{second.scenario_set_id!r} both produce {key}="
                f"{first.params[key]!r} / {second.params[key]!r}, which are numerically "
                "equal but written as different types, so they are two scenarios rather "
                "than one. If that was not intended, write both the same way."
            )
        kept = subsample(raw, catalog.run.tier)
        episodes = expand_episodes(catalog, scene_hashes[scene.id], kept, task_hashes)
        if not episodes:
            continue

        shape = scene.shape_for(hardware_profile)
        if shape is None:
            raise CatalogError(
                f"scene {scene.id!r} has no resource_shape for hardware profile "
                f"{hardware_profile!r}; declared profiles are "
                f"{[s.hardware_profile for s in scene.resource_shape] or 'none'}. "
                "Run 'refractal build --hardware " + hardware_profile + "' to probe it.",
                file="scenes.yaml",
            )

        tasks_on_scene = [t for t in catalog.tasks if t.scene == scene.id]
        kept_by_scene[scene.id] = kept
        episodes_by_scene[scene.id] = episodes
        demands.append(
            SceneDemand(
                scene_id=scene.id,
                episodes=len(episodes),
                shape=shape,
                max_steps=max(t.max_steps for t in tasks_on_scene),
            )
        )

    if not demands:
        raise CatalogError("this catalog expands to zero episodes; nothing to plan")

    # --- placement -------------------------------------------------------
    budgets, server_placement = budget_model_servers(hardware, catalog.run.checkpoints)
    allocation = allocate_workers(
        demands, hardware, budgets, pack_below_startup_sec=pack_below_startup_sec
    )
    warnings.extend(allocation.warnings)
    cpusets = assign_cpusets(demands, allocation, hardware)

    for device in budgets:
        if device.used_mb > device.capacity_mb:  # defence in depth; allocation refuses first
            raise CapacityError(
                f"device {device.device_id} oversubscribed: {device.used_mb} MB assigned of "
                f"{device.capacity_mb} MB"
            )

    # --- assemble --------------------------------------------------------
    planned_scenes: list[PlannedScene] = []
    total_seconds = 0

    for demand in demands:
        scene = catalog.scene(demand.scene_id)
        episodes = episodes_by_scene[scene.id]
        n_workers = allocation.workers[scene.id]
        device = allocation.devices.get(scene.id, "cpu")

        if n_workers:
            workers = _shard(scene.id, episodes, n_workers, device, cpusets, demand)
        else:
            workers = [
                PlannedWorker(
                    worker_id=f"packed/{scene.id}",
                    scene_id=scene.id,
                    device=device,
                    episodes=episodes,
                    packed_scenes=[scene.id],
                    estimated_seconds=int(demand.makespan(1)),
                )
            ]

        scene_seconds = max(w.estimated_seconds for w in workers)
        total_seconds = max(total_seconds, scene_seconds)
        planned_scenes.append(
            PlannedScene(
                scene_id=scene.id,
                scene_hash=scene_hashes[scene.id],
                engine=scene.engine,
                resource_shape=demand.shape,
                scenarios=kept_by_scene[scene.id],
                workers=workers,
                episode_count=len(episodes),
                estimated_seconds=scene_seconds,
            )
        )

    if server_placement:
        warnings.append(
            "model servers: "
            + ", ".join(f"{cid} -> {dev}" for cid, dev in sorted(server_placement.items()))
        )

    return Plan(
        plan_schema=PLAN_SCHEMA,
        plan_id=catalog.plan_id(recorded),
        catalog_hash=catalog.catalog_file_hash(),
        refractal_version=__version__,
        hardware_profile=hardware_profile,
        execution_mode=catalog.run.execution_mode,
        checkpoints=list(catalog.run.checkpoints),
        tier=catalog.run.tier,
        seeds=catalog.run.seed_values(),
        total_episodes=sum(s.episode_count for s in planned_scenes),
        estimated_seconds=total_seconds,
        scenes=planned_scenes,
        warnings=warnings,
        created_at=created_at,
    )


def _shard(scene_id, episodes, n_workers, device, cpusets, demand) -> list[PlannedWorker]:
    """Contiguous slices, not round-robin.

    Refractal assigns episodes explicitly rather than delegating to a shard
    index, which is what makes resume possible: on restart, `execute` subtracts
    the episode ids that already have results from the ones in this list. A
    round-robin index cannot express "all but these forty".

    Contiguous rather than interleaved keeps episodes for one task adjacent,
    which keeps scene reloads rare inside a worker.
    """
    per = math.ceil(len(episodes) / n_workers)
    workers = []
    for shard in range(n_workers):
        chunk = episodes[shard * per : (shard + 1) * per]
        if not chunk:
            continue
        worker_id = f"{scene_id}/{shard}"
        batches = math.ceil(len(chunk) / demand.shape.envs_per_process)
        workers.append(
            PlannedWorker(
                worker_id=worker_id,
                scene_id=scene_id,
                device=device,
                cpuset=cpusets.get(worker_id),
                episodes=chunk,
                estimated_seconds=int(
                    demand.shape.startup_sec + batches * demand.batch_seconds()
                ),
            )
        )
    return workers


__all__ = [
    "Allocation",
    "BuildLock",
    "CapacityError",
    "Plan",
    "read_plan",
    "resolve",
]
