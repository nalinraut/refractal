"""Content-addressed identity.

The project exists because ``vla-eval`` identifies an attempt as
``(task_name, 7)`` where 7 is a position in a list, so reordering the list
silently changes what episode 7 means. Every identity here is therefore a hash
of *what the thing is*, never of where it sits.

That principle has to be applied consistently or it leaks back in. The API
reference defines ``episode_id`` over ``task_id`` -- a string a human chose --
while defining it over ``scene_hash``, the content of the scene. Those are not
symmetric: editing ``predicate_args`` from ``{slot: 4}`` to ``{slot: 5}`` while
leaving the id as ``vial-slot-4`` produces an episode with an unchanged
identity and a changed meaning. That is the original bug, reintroduced one level
up. So this module hashes tasks by content and uses ``task_hash`` in
``episode_id``.

**Identity summary**

    scenario_hash = H({faults: [], params: <normalized>})
    task_hash     = H({instruction, predicate, predicate_args, max_steps, phases})
    scene_hash    = H({engine, engine_version, model+assets file digests})
    episode_id    = H({scene_hash, task_hash, scenario_hash, seed, checkpoint_id})
    plan_id       = H(<experiment identity: catalog minus placement and output>)

Every one is a hash of a *document*, not a concatenation. The reference writes
``sha256(scenario_hash + seed + checkpoint_id + scene_hash + task_id)``;
concatenating variable-length strings is ambiguous (``"a" + "b1"`` and
``"ab" + "1"`` collide), and while fixed-width hashes mostly hide that, the
checkpoint id is free-form and does not. A keyed document costs nothing and
cannot collide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, hash_file_set, hash_obj, normalize_params
from .errors import CatalogError
from .models import Checkpoint, Run, Scene, ScenarioSet, Task


def scenario_identity(
    params: Mapping[str, Any], faults: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """The canonical document a scenario hashes to.

    ``faults`` is folded in as an empty list from the first commit even though
    nothing in v1 can populate it. This is the actual mechanism behind the
    reserved-field decision: if faults were simply absent today and added to the
    document later, every scenario_hash in every recorded result would change
    the day Transect landed, and the reservation would have bought nothing.
    """
    return {
        "faults": [normalize_params(dict(f)) for f in faults],
        "params": normalize_params(dict(params)),
    }


def scenario_hash(
    params: Mapping[str, Any], faults: Sequence[Mapping[str, Any]] = ()
) -> str:
    return hash_obj(scenario_identity(params, faults))


def task_identity(task: Task) -> dict[str, Any]:
    """Task identity by content.

    ``id`` and ``description`` are excluded on purpose: renaming a task or
    editing its comment must not break the join against results already
    recorded, while changing its instruction, predicate, arguments, phases or
    step limit must.

    Two tasks whose identities collide are, by this definition, the same task
    run under two names -- which is true, and harmless.
    """
    return {
        "instruction": task.instruction,
        "predicate": task.predicate,
        "predicate_args": task.predicate_args,
        "max_steps": task.max_steps,
        "phases": [
            {"name": p.name, "predicate": p.predicate, "predicate_args": p.predicate_args}
            for p in task.phases
        ],
    }


def task_hash(task: Task) -> str:
    return hash_obj(task_identity(task))


def _scene_files(catalog_root: Path, scene: Scene) -> list[Path]:
    root = catalog_root.resolve()
    model_path = (root / scene.model).resolve()
    if not model_path.is_file():
        raise CatalogError(
            f"scene {scene.id!r} references model {scene.model!r}, which does not exist",
            file="scenes.yaml",
        )
    if not model_path.is_relative_to(root):
        raise CatalogError(
            f"scene {scene.id!r} references {scene.model!r} outside the catalog directory; "
            "the catalog must be self-contained because it is copied into every results "
            "directory as provenance",
            file="scenes.yaml",
        )
    files = [model_path]
    for pattern in scene.assets:
        matched = sorted(root.glob(pattern))
        if not matched:
            raise CatalogError(
                f"scene {scene.id!r} asset pattern {pattern!r} matched no files",
                file="scenes.yaml",
            )
        files.extend(p.resolve() for p in matched if p.is_file())
    return files


def scene_identity(catalog_root: Path, scene: Scene) -> dict[str, Any]:
    """Over *sources*, never the compiled model.

    Hashing the compiled model would require the engine at plan time, which
    breaks the property that ``refractal plan`` runs on a bare laptop with
    nothing installed -- the property the whole compiler architecture rests on.
    Compiling would only additionally catch two different source files that
    happen to compile identically, which is rare and not worth the dependency.

    The container verifies at startup that the hash it was handed matches what
    it just compiled, and fails loudly otherwise. That check runs where the
    engine already exists, so it is free.
    """
    return {
        "engine": scene.engine,
        "engine_version": scene.engine_version,
        "files": hash_file_set(catalog_root.resolve(), _scene_files(catalog_root, scene)),
    }


def scene_hash(catalog_root: Path, scene: Scene) -> str:
    return hash_obj(scene_identity(catalog_root, scene))


def episode_id(
    *,
    scene_hash: str,
    task_hash: str,
    scenario_hash: str,
    seed: int,
    checkpoint_id: str,
) -> str:
    return hash_obj(
        {
            "scene_hash": scene_hash,
            "task_hash": task_hash,
            "scenario_hash": scenario_hash,
            "seed": seed,
            "checkpoint_id": checkpoint_id,
        }
    )


def comparison_unit(*, scene_id: str, task_hash: str, scenario_hash: str) -> tuple[str, str, str]:
    """The key ``compare`` joins checkpoints on.

    DEVIATION, and an important one. The reference says the join key is
    ``scenario_hash``. It cannot be, for two independent reasons:

    * A scenario set is crossed with *every* task on its scene, so one
      ``scenario_hash`` covers ``vial-slot-4`` and ``vial-slot-7`` alike.
      Joining on it alone merges rows for two different goals into one 2x2 --
      and the reference's own 2x2 is specified as being reported per task, so
      the two statements contradict each other.
    * ``scenario_hash`` is over parameter values only, so two different scenes
      that both parameterize ``x`` and ``y`` collide.

    The unit is therefore ``(scene_id, task_hash, scenario_hash)``.

    ``scene_hash`` is deliberately *not* in the key: if it were, editing a mesh
    would produce an empty join rather than a loud one. Instead ``compare``
    should join on this key and then assert that ``scene_hash`` agrees across
    checkpoints, reporting a disagreement as an error -- "you are comparing
    results from two different geometries" is a sentence a user needs to read,
    not a silent zero-row result.
    """
    return (scene_id, task_hash, scenario_hash)


def experiment_identity(
    *,
    scenes: Iterable[Scene],
    tasks: Iterable[Task],
    scenario_sets: Iterable[ScenarioSet],
    run: Run,
    scene_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """What makes two runs "the same experiment". Hashes to ``plan_id``.

    DEVIATION from ``sha256(catalog_bytes + sorted(checkpoints) + plan_schema)``,
    on three counts:

    * **``plan_schema`` is excluded.** D9 argues that format compatibility and
      experiment identity are independent, and that conflating them would let a
      formatting change invalidate comparability with old results -- then the
      formula folds ``plan_schema`` into ``plan_id``, which does exactly that.
    * **``results_uri`` is excluded.** It lives in ``run.yaml``, so hashing raw
      catalog bytes makes the comparison id depend on the output directory --
      writing the same experiment to ``./results`` and to ``s3://...`` yields
      two ids that never join, and the output path ends up naming itself.
    * **Parsed models, not raw bytes.** Raw bytes churn on a reordered key, a
      changed comment, or trailing whitespace. That is the same objection D9
      raises against deriving ``plan_schema`` structurally, applied to the field
      that actually is derived.

    ``execution_mode``, ``max_concurrent_checkpoints`` and every resource shape
    are excluded for the same reason as hardware: they change how the work is
    placed and paced, not which episodes exist. Interleaved and serial runs of
    one experiment belong in one comparison, with the mode recorded per episode
    so ``compare`` can gate latency reporting on it.

    ``tier`` *is* included, because it changes which scenarios exist. That has a
    cost worth stating: a smoke run and a full run of the same catalog get
    different ids, so the smoke results cannot be reused as a head start on the
    full run even though the smoke set is a strict subset of it. Worth
    revisiting if tier-promotion turns out to matter in practice.
    """
    return {
        "scenes": sorted(
            (
                {
                    "id": s.id,
                    "engine": s.engine,
                    "scene_hash": scene_hashes[s.id],
                }
                for s in scenes
            ),
            key=lambda d: d["id"],
        ),
        "tasks": sorted(
            ({"id": t.id, "task_hash": task_hash(t), "scene": t.scene} for t in tasks),
            key=lambda d: d["id"],
        ),
        "scenario_sets": sorted(
            (
                {
                    "id": ss.id,
                    "scene": ss.scene,
                    "generator": ss.generator,
                    "generator_seed": ss.generator_seed,
                    "params": {
                        k: v.model_dump(exclude_unset=True) for k, v in ss.params.items()
                    },
                    "filter": ss.filter,
                    "tasks": ss.tasks,
                    "faults": [f.model_dump() for f in ss.faults],
                }
                for ss in scenario_sets
            ),
            key=lambda d: d["id"],
        ),
        "run": {
            "checkpoints": sorted(
                (
                    {
                        "id": c.id,
                        "path": c.path,
                        "server": c.server,
                        "server_args": c.server_args,
                    }
                    for c in run.checkpoints
                ),
                key=lambda d: d["id"],
            ),
            "seeds": run.seeds,
            "seed_base": run.seed_base,
            "tier": run.tier,
            "scenario_sets": sorted(run.scenario_sets) if run.scenario_sets else None,
        },
    }


def plan_id(identity: Mapping[str, Any]) -> str:
    return hash_obj(dict(identity))


__all__ = [
    "canonical_json",
    "comparison_unit",
    "episode_id",
    "experiment_identity",
    "plan_id",
    "scenario_hash",
    "scenario_identity",
    "scene_hash",
    "scene_identity",
    "task_hash",
    "task_identity",
]
