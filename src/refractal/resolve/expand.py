"""Catalog -> the episode list.

Pure functions. The only impurity in the whole package is importing the user's
generator and filter, which happens here because a generator must be *called* to
produce the scenario list that `plan.json` records.

Order of operations matters and is not arbitrary:

1. generate  -- run the generator, quantized output only
2. filter    -- read survivors from the build lock, never evaluate here
3. subsample -- apply the tier, stratified
4. cross     -- tasks x scenarios x seeds x checkpoints

Filtering before subsampling, so a tier is a fraction of the *runnable*
scenarios rather than a fraction of a grid that is mostly unreachable.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from ..schema.errors import CatalogError, GeneratorError
from ..schema.identity import episode_id, scenario_hash, task_hash
from ..schema.importstr import resolve_import_string
from ..schema.loader import Catalog
from ..schema.models import ScenarioSet
from ..schema.plan import PlannedEpisode, PlannedScenario
from .lock import BuildLock

#: Fraction of scenarios each tier keeps. Thresholds on one value, so the smoke
#: set is a strict subset of the regression set, which is a strict subset of
#: full. Two independent samples would not have that property and the whole
#: point of a tier ladder is that promotion adds work rather than replacing it.
TIER_FRACTION = {"smoke": 0.10, "regression": 0.50, "full": 1.0}


def _hash_fraction(scenario_hash_value: str) -> float:
    """Map a hash onto [0, 1) deterministically, for stable subsampling."""
    digest = scenario_hash_value.split(":", 1)[-1]
    return int(digest[:12], 16) / float(1 << 48)


def subsample(scenarios: list[PlannedScenario], tier: str) -> list[PlannedScenario]:
    """Keep a stable fraction, stratified per scenario set, never fewer than one.

    Stratification is not cosmetic. A flat threshold across the whole catalog
    lets a set of three scenarios contribute zero to the smoke tier, which turns
    "smoke passed" into a statement about a scene nobody exercised.
    """
    fraction = TIER_FRACTION[tier]
    if fraction >= 1.0:
        return list(scenarios)

    by_set: dict[str, list[PlannedScenario]] = {}
    for scenario in scenarios:
        by_set.setdefault(scenario.scenario_set_id, []).append(scenario)

    kept: list[PlannedScenario] = []
    for set_id in by_set:
        members = by_set[set_id]
        selected = [s for s in members if _hash_fraction(s.scenario_hash) < fraction]
        if not selected:
            # Floor of one: the lowest-fraction member, so the choice is stable
            # and is still a subset of every larger tier.
            selected = [min(members, key=lambda s: _hash_fraction(s.scenario_hash))]
        kept.extend(selected)

    order = {s.scenario_hash: i for i, s in enumerate(scenarios)}
    return sorted(kept, key=lambda s: order[s.scenario_hash])


def generate_scenarios(scenario_set: ScenarioSet, lock: BuildLock | None) -> list[PlannedScenario]:
    """Run the generator, then apply the recorded filter survivors.

    The filter is **not** evaluated here. A reachability filter needs forward
    kinematics, a placement filter needs collision queries, and importing either
    would put a simulator inside `resolve` -- destroying the property that
    `refractal plan` runs on a bare laptop, which is the foundation the compiler
    architecture stands on. `refractal build` evaluates filters where the engine
    already exists and records the survivors; this reads that record and refuses
    if it is missing or stale.
    """
    generator = resolve_import_string(scenario_set.generator)
    rows = generator(scenario_set.params, scenario_set.generator_seed)
    if not isinstance(rows, list):
        raise GeneratorError(
            f"generator {scenario_set.generator!r} returned {type(rows).__name__}, "
            "expected a list of dicts"
        )

    scenarios = [
        PlannedScenario(
            scenario_hash=scenario_hash(row),
            scenario_set_id=scenario_set.id,
            params=row,
        )
        for row in rows
    ]

    seen: set[str] = set()
    deduped: list[PlannedScenario] = []
    for scenario in scenarios:
        # A generator that emits the same point twice would otherwise inflate
        # the episode count and put two identical rows in every comparison.
        if scenario.scenario_hash in seen:
            continue
        seen.add(scenario.scenario_hash)
        deduped.append(scenario)

    if scenario_set.filter is None:
        return deduped

    if lock is None:
        raise CatalogError(
            f"scenario_set {scenario_set.id!r} declares filter "
            f"{scenario_set.filter!r}, but there is no catalog/build.lock. "
            "Filters are evaluated by 'refractal build', which runs where the engine "
            "exists; 'refractal plan' only reads the result. Run 'refractal build' first.",
            file="scenarios.yaml",
        )

    survivors = lock.filter_survivors(scenario_set)
    kept = [s for s in deduped if s.scenario_hash in survivors]
    return kept


def expand_episodes(
    catalog: Catalog,
    scene_hash_value: str,
    scenarios: Iterable[PlannedScenario],
    task_hashes: Mapping[str, str],
) -> list[PlannedEpisode]:
    """Cross scenarios x tasks x seeds x checkpoints for one scene.

    Checkpoints are an expansion axis rather than a separate run, because the
    unit of work is a *comparison*: running two checkpoints in one session
    against the same scenario objects removes a whole class of confounds that
    running twice and joining afterwards cannot.

    ``scene_hash_value`` is passed in rather than recomputed. It costs file I/O
    over the model and every mesh, and doing that per episode would make
    expansion quadratic in the least interesting way.
    """
    run = catalog.run
    seeds = run.seed_values()
    tasks_by_set = {
        scenario_set.id: catalog.tasks_for(scenario_set)
        for scenario_set in catalog.active_scenario_sets()
    }
    episodes: list[PlannedEpisode] = []

    for scenario in scenarios:
        for task in tasks_by_set.get(scenario.scenario_set_id, ()):
            th = task_hashes[task.id]
            for seed in seeds:
                for checkpoint in run.checkpoints:
                    episodes.append(
                        PlannedEpisode(
                            episode_id=episode_id(
                                scene_hash=scene_hash_value,
                                task_hash=th,
                                scenario_hash=scenario.scenario_hash,
                                seed=seed,
                                checkpoint_id=checkpoint.id,
                            ),
                            task_id=task.id,
                            task_hash=th,
                            scenario_hash=scenario.scenario_hash,
                            seed=seed,
                            checkpoint_id=checkpoint.id,
                        )
                    )
    return episodes


def task_hashes_for(catalog: Catalog) -> dict[str, str]:
    return {task.id: task_hash(task) for task in catalog.tasks}


__all__ = [
    "TIER_FRACTION",
    "expand_episodes",
    "generate_scenarios",
    "subsample",
    "task_hashes_for",
]
