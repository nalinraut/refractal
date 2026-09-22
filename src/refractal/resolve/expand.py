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

from typing import Any, Iterable, Mapping, Sequence

from ..schema.errors import CatalogError, GeneratorError
from ..schema.canonical import hash_obj
from ..perturbations import declaration, has_inverse, invertible
from ..schema.identity import (
    episode_id,
    base_scenario_hash,
    scenario_hash,
    task_hash,
    task_identity,
)
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
            # Equal today, and deliberately computed by two calls rather than
            # one assignment: when a scenario set carries perturbations, the
            # first takes them and the second never does.
            scenario_hash=scenario_hash(row, scenario_set.perturbations),
            base_scenario_hash=base_scenario_hash(row),
            perturbations=[p.model_dump() for p in scenario_set.perturbations],
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


def _type_signature(params: Mapping[str, Any]) -> tuple:
    """A scenario's keys and the *type* of each numeric value."""
    return tuple(
        (k, type(v).__name__ if isinstance(v, (int, float)) and not isinstance(v, bool) else "_")
        for k, v in sorted(params.items())
    )


def _value_signature(params: Mapping[str, Any]) -> tuple:
    """A scenario's keys and values compared numerically, ignoring int/float."""
    return tuple(
        (k, float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
        for k, v in sorted(params.items())
    )


def find_type_only_collisions(
    scenarios: Sequence[PlannedScenario],
) -> list[tuple[PlannedScenario, PlannedScenario]]:
    """Scenarios that are numerically equal but differ in a written numeric type.

    Preserving the authored type means ``friction: 1`` and ``friction: 1.0`` are
    different scenarios. That is the right trade -- an index should print as an
    index -- but it leaves one way to be surprised: two scenario sets on one
    scene, differing only in how a number was typed, expand to two units where
    the author meant one, and the episode count quietly doubles.

    Worth being exact about where this can happen, because the obvious guess is
    wrong. A type *edit* to a catalog changes ``plan_id``, so the before and
    after land in different comparison directories and ``compare`` never sees
    both -- it cannot report them as non-overlapping. The confusion only exists
    *within a single plan*, which is why the check lives here rather than in the
    overlap report.
    """
    by_value: dict[tuple, list[PlannedScenario]] = {}
    for scenario in scenarios:
        by_value.setdefault(_value_signature(scenario.params), []).append(scenario)

    collisions = []
    for group in by_value.values():
        if len(group) < 2:
            continue
        for i, first in enumerate(group):
            for second in group[i + 1 :]:
                if _type_signature(first.params) != _type_signature(second.params):
                    collisions.append((first, second))
    return collisions


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
                            max_steps=task.max_steps,
                            instruction=task.instruction,
                            scenario_hash=scenario.scenario_hash,
                            base_scenario_hash=scenario.base_scenario_hash,
                            perturbations=list(scenario.perturbations),
                            seed=seed,
                            checkpoint_id=checkpoint.id,
                        )
                    )
    return episodes


def _refuse_undeclared_primitives(scenario_set, scene_id, spec, declared) -> None:
    """Compare what the effect says it needs against what the adapter supplies.

    **Capability-based, not identity-based.** The planner does not recognise
    effects -- it cannot inspect an unfamiliar one and conclude it needs a
    camera. It compares two declarations, so neither side has to know about the
    other, and an effect nobody here has heard of is refused for the right
    reason rather than for being unfamiliar.

    Three outcomes, because **silence and a negative answer are different
    claims** -- the same distinction the capability probe already draws between
    an actuator reported as unlimited and one nobody looked at:

    * the adapter lists the primitive          -> allowed
    * the adapter lists its primitives, and
      this is not among them                   -> refused: it said no
    * the adapter lists no primitives at all    -> refused: nobody asked it

    The second and third get different messages because they need different
    fixes. One is a catalog naming an effect this scene cannot run; the other is
    a build that never recorded what the scene can do.
    """
    known = declaration(spec.type)
    if known is None:
        return  # an unknown effect name is refused elsewhere, by name
    protocol, needs = known
    supplied = (declared or {}).get("primitives")

    if supplied is None:
        raise CatalogError(
            f"scenario_set {scenario_set.id!r} uses {spec.type!r}, which needs "
            f"{list(needs)} from the {protocol} protocol, but nothing has "
            f"recorded which primitives scene {scene_id!r} supplies. That is not "
            "the same as the scene not supplying them -- nobody has asked. Run "
            "'refractal build' where the adapter is importable.",
            file="scenarios.yaml",
        )

    missing = [primitive for primitive in needs if primitive not in supplied]
    if missing:
        raise CatalogError(
            f"scenario_set {scenario_set.id!r} uses {spec.type!r}, which needs "
            f"{missing} from the {protocol} protocol. Scene {scene_id!r} reports "
            f"supplying {sorted(supplied)}. The adapter has been asked and does "
            "not implement these, so the episode would fail partway rather than "
            "not start. Use an effect this scene can run, or implement the "
            "missing primitives.",
            file="scenarios.yaml",
        )


def refuse_unsupported_perturbations(catalog, lock=None) -> None:
    """Refuse a perturbation the scene cannot carry out.

    Capability is **per target, not per primitive**, and that distinction is the
    whole reason this exists. An adapter either supplies ``scale_actuator`` or it
    does not, which a probe could declare once -- but the primitive is
    meaningful on one actuator and meaningless on another in the same model.

    Measured on LIBERO: the gripper declares ``forcerange = [-20, 20]`` and
    every arm actuator declares ``[0, 0]``, which MuJoCo reads as *unlimited*.
    Scaling an unlimited limit by 0.3 leaves it unlimited. The call succeeds,
    the episode completes, and a fired-events log that recorded the call would
    report a perturbation that did nothing -- an entire sweep coming back
    looking like clean unperturbed runs.

    So the check is against what ``build`` recorded for this scene. A scene with
    no declaration is refused rather than allowed: "the probe did not say" and
    "the probe said yes" must not be the same answer, since the failure being
    prevented is silent and the cost of being wrong is a whole sweep.
    """
    entries = {e.scene_id: e for e in (lock.scenes if lock else [])}
    for scenario_set in catalog.active_scenario_sets():
        if not scenario_set.perturbations:
            continue
        scene_id = scenario_set.scene
        declared = (entries.get(scene_id).capabilities if scene_id in entries else {})
        actuators = (declared or {}).get("actuators")
        for spec in scenario_set.perturbations:
            _refuse_undeclared_primitives(scenario_set, scene_id, spec, declared)
            if spec.until_step is not None and not has_inverse(spec.type):
                raise CatalogError(
                    f"scenario_set {scenario_set.id!r} gives {spec.type!r} an "
                    f"until_step, but that effect cannot be ended: undoing it "
                    "would mean restoring a value rather than applying an "
                    "inverse, which clobbers anything else acting on the same "
                    f"target. Effects that can be sustained: {invertible()}. "
                    "Drop until_step to have it last to the end of the episode.",
                    file="scenarios.yaml",
                )
            if spec.type != "scale_actuator":
                continue
            if actuators is None:
                raise CatalogError(
                    f"scenario_set {scenario_set.id!r} scales the torque limit of "
                    f"{spec.target!r} on scene {scene_id!r}, but nothing has recorded "
                    "which of that scene's actuators have a torque limit. Run "
                    "'refractal build' where the benchmark is importable. Refused "
                    "rather than attempted: an actuator with no limit accepts the "
                    "call, changes nothing, and the run completes looking clean.",
                    file="scenarios.yaml",
                )
            known = actuators.get(spec.target)
            if known is None:
                near = sorted(n for n in actuators if spec.target.split("_")[0] in n)
                raise CatalogError(
                    f"scenario_set {scenario_set.id!r} perturbs actuator {spec.target!r}, "
                    f"which scene {scene_id!r} does not have. It has "
                    f"{sorted(actuators)[:6]}"
                    + (f"; did you mean {near[0]!r}?" if near else ""),
                    file="scenarios.yaml",
                )
            if not known.get("force_limited"):
                raise CatalogError(
                    f"scenario_set {scenario_set.id!r} scales the torque limit of "
                    f"{spec.target!r} on scene {scene_id!r}, but that actuator has no "
                    "limit to scale -- MuJoCo's forcerange [0, 0] means unlimited, and "
                    "scaling unlimited by any factor leaves it unlimited. The episode "
                    "would run, the log would say the perturbation fired, and nothing "
                    "would have changed. Perturb an actuator that has a limit, or give "
                    "this one an absolute limit.",
                    file="scenarios.yaml",
                )


def task_hashes_for(catalog: Catalog, lock=None) -> dict[str, str]:
    """Task hashes, taking provider-supplied content from the lock when present.

    Same shape as scene hashes for external scenes: the authored fields are
    readable on a laptop, the provider's facts are not, so `build` records them
    and `resolve` reads them back. A task with no lock entry hashes exactly as it
    did before the hook existed.

    **Staleness is checked, not assumed.** The entry records the authored fields
    it was built against; if those moved, the recorded facts may describe a
    different goal. Editing an instruction is the case that matters, because the
    instruction is also the harness's task selector -- so a stale entry is not a
    cosmetic mismatch, it can mean the recorded content belongs to a task the run
    will not execute.
    """
    hashes: dict[str, str] = {}
    for task in catalog.tasks:
        entry = lock.task_entry(task.id) if lock is not None else None
        if entry is None:
            hashes[task.id] = task_hash(task)
            continue
        if entry.authored_key is not None and entry.authored_key != hash_obj(
            task_identity(task)
        ):
            raise CatalogError(
                f"task {task.id!r} has been edited since 'refractal build' recorded its "
                f"provider facts. The lock holds {entry.facts!r}, which was recorded "
                "against different authored fields -- and the instruction is also the "
                "harness's task selector, so this is not necessarily cosmetic. Re-run "
                "'refractal build'."
            )
        hashes[task.id] = entry.task_hash
    return hashes


__all__ = [
    "TIER_FRACTION",
    "find_type_only_collisions",
    "expand_episodes",
    "generate_scenarios",
    "subsample",
    "task_hashes_for",
]
