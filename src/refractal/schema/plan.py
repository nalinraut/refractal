"""``plan.json`` -- the compiler's output, and the contract between stages.

A plan is a file you can read, diff and commit. The whole schedule is decided up
front and recorded, rather than negotiated while running, which is what makes
placement part of the provenance instead of an accident of the day.

Two versions live here and they answer different questions. Conflating them is a
bug, so they are kept apart deliberately:

``plan_schema``
    **Manual.** A small integer bumped by hand when the format changes. Answers
    *"can this code read this file."* It cannot be derived, because only a human
    knows whether a change is breaking, and a structural hash would churn on
    cosmetic edits while telling a reader nothing about compatibility.

``plan_id``
    **Derived**, from the experiment identity in :mod:`refractal.schema.identity`.
    Answers *"are these the same experiment."* Doubles as the ``comparison_id``.
    Notably it does **not** include ``plan_schema``: format compatibility and
    experiment identity are independent, and folding one into the other would
    let a formatting change orphan every previously recorded result.

DEVIATION from the API reference: ``PlannedWorker.episodes`` holds episode
*records*, not bare episode ids. The reference gives each worker a list of
``episode_id`` strings and nothing that maps an id back to
``(task, scenario, seed, checkpoint)``. Since an id is a hash, that mapping is
not recoverable, and ``execute`` would have no way to run the plan it was
handed. The records are what make the plan self-contained -- and self-contained
is the property that lets a worker be handed its assignment and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .errors import CatalogError, RefractalError
from .models import Checkpoint, ExternalScene, ResourceShape, Strict

#: Bump by hand, and only when the format changes in a way older readers cannot
#: survive. See the module docstring for why this is not derived.
#:
#: 2 adds ``base_scenario_hash`` to every planned scenario and episode: the axis
#: a perturbation sweep joins on. Two plans previously declared 1 with different
#: field sets, which meant the number carried no information about shape, so
#: adding fields now moves it.
PLAN_SCHEMA = 2


class PlanSchemaError(RefractalError):
    """A plan declares a ``plan_schema`` this build does not know how to read."""


class PlannedScenario(Strict):
    scenario_hash: str
    #: What this scenario would hash to unperturbed. Equal to ``scenario_hash``
    #: when nothing is perturbed, which is every scenario today.
    #:
    #: Defaulted rather than required, so a plan written at plan_schema 1 still
    #: validates -- ``read_plan`` accepts older plans and they have no such key.
    base_scenario_hash: str = ""
    scenario_set_id: str
    params: dict[str, Any]


class PlannedEpisode(Strict):
    """One attempt, fully identified. The unit `execute` runs and `compare` joins."""

    episode_id: str
    task_id: str
    task_hash: str
    scenario_hash: str
    #: The unperturbed scenario this episode varies. ``compare`` holds it fixed
    #: and groups by perturbation level; ``episode_id`` derives from
    #: ``scenario_hash``, never from this, so two episodes differing only in
    #: perturbation are different episodes.
    base_scenario_hash: str = ""
    seed: int
    checkpoint_id: str
    #: The task's step limit, copied in so the plan is executable on its own.
    #: It is already inside ``task_hash``, so this is a restatement rather than a
    #: new degree of freedom -- but a backend has to hand a number to the harness,
    #: and reaching back into the catalog to find it would make `refractal run`
    #: need the catalog that `refractal plan` already compiled away.
    #:
    #: Required, not defaulted. ``Task.max_steps`` defaults to 400, so the planner
    #: always has a real value; a default here would let a plan that never recorded
    #: one hand 400 to a harness and produce a run that looks fine.
    max_steps: int = Field(gt=0)
    #: The task's instruction, carried for the same reason as ``max_steps``: it is
    #: inside ``task_hash`` and a backend needs it to act.
    #:
    #: Specifically, vla-eval selects tasks by natural-language name --
    #: ``get_tasks()`` returns ``name = task.language`` and the config filters on
    #: it -- so the instruction *is* the selector. Sending a Refractal task id
    #: instead matches nothing, which runs zero episodes.
    instruction: str = Field(min_length=1)


class PlannedWorker(Strict):
    #: ``"{scene_id}/{shard_index}"``. **Never a bare index** -- an index alone
    #: is meaningless once scenes have different worker counts, and using one
    #: bakes the single-scene assumption into every filename and log line.
    worker_id: str
    scene_id: str
    device: str = "cpu"
    cpuset: str | None = None
    episodes: list[PlannedEpisode] = Field(default_factory=list)
    #: Only for sequentially-packed small scenes: the worker loads each in turn,
    #: paying ``startup_sec`` per switch.
    packed_scenes: list[str] = Field(default_factory=list)
    estimated_seconds: int = 0


class PlannedScene(Strict):
    scene_id: str
    scene_hash: str
    engine: str
    #: Carried through from the catalog when the scene is externally defined, so a
    #: backend can name the benchmark that owns this scene without the catalog.
    #:
    #: Same reason as ``PlannedEpisode.max_steps``, and the same root cause: the
    #: plan was shaped by the local backend, whose executor needs nothing but
    #: identity. The vla-eval backend is the first consumer that needs the scene's
    #: *content*, and found three things absent. ``refractal plan`` compiles the
    #: catalog away, so anything an executor needs has to survive the compile.
    #:
    #: Not identity: ``scene_hash`` already covers the provider and ref, via
    #: ``external_scene_ref_key`` and the lock. This is the same information in an
    #: executable form.
    external: ExternalScene | None = None
    resource_shape: ResourceShape
    #: **The expanded list, not the generator spec.** A generator whose
    #: implementation drifts would otherwise make the provenance a lie.
    scenarios: list[PlannedScenario]
    workers: list[PlannedWorker]
    episode_count: int
    estimated_seconds: int


class Plan(Strict):
    #: Every shape this build can read, listed rather than bounded, so adding one
    #: is a deliberate edit here as well as to ``PLAN_SCHEMA``. ``read_plan``
    #: enforces the same range with a message; this is the backstop for a Plan
    #: built by any other route.
    plan_schema: Literal[1, 2] = PLAN_SCHEMA
    plan_id: str
    catalog_hash: str
    refractal_version: str
    hardware_profile: str
    execution_mode: str
    checkpoints: list[Checkpoint]
    tier: str
    seeds: list[int]
    total_episodes: int
    estimated_seconds: int
    scenes: list[PlannedScene]
    warnings: list[str] = Field(default_factory=list)
    #: The document ``plan_id`` was computed from.
    #:
    #: Recorded so two plans can say *why* their ids differ. A digest tells you
    #: something moved and nothing about what, which is fine until someone is
    #: holding two comparison directories and a question. Third time this has come
    #: up -- the harness surface records a per-file manifest beside its digest, an
    #: external scene records its facts beside its hash, and now this.
    #:
    #: Excluded from the hash it describes, obviously: it *is* the hash's input.
    identity: dict[str, Any] = Field(default_factory=dict)
    #: Informational, and deliberately excluded from every hash. It is also why
    #: the determinism gate compares ``plan_id`` and the scenario list rather
    #: than raw file bytes -- a timestamp cannot be byte-identical across runs.
    created_at: str | None = None

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent, sort_keys=False)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")


def explain_identity_difference(before: Plan, after: Plan) -> list[str]:
    """Name the fields that make two plans different experiments.

    Returns one line per difference, deepest key first, so the answer to "why did
    these not join" is a sentence rather than a bisect. Empty when the ids match.
    """
    if before.plan_id == after.plan_id:
        return []
    if not before.identity or not after.identity:
        return [
            "one of these plans predates identity recording, so the difference cannot "
            "be named — only that it exists."
        ]

    lines: list[str] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                walk(left.get(key), right.get(key), f"{path}.{key}" if path else key)
            return
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{path}[{index}]")
            return
        if left != right:
            lines.append(f"{path}: {left!r} -> {right!r}")

    walk(before.identity, after.identity, "")
    return lines or ["the identity documents differ in a way the walk did not reach"]


def locate_plan(target: str | Path) -> Path:
    """Resolve a plan file, a results directory, or a comparison directory.

    The plan is copied into every results directory as provenance, so someone
    holding results from three months ago has the plan whether or not they still
    have the file they planned from. Requiring them to know the layout to use it
    would waste the copy.

    That is also the case where this is most needed: two ids differ, one plan is
    to hand and the other is inside a results tree.
    """
    target = Path(target)
    if target.is_file():
        return target
    if (target / "plan.json").is_file():
        return target / "plan.json"
    comparisons = sorted(target.glob("comparison_id=*/plan.json"))
    if len(comparisons) == 1:
        return comparisons[0]
    if len(comparisons) > 1:
        ids = [p.parent.name for p in comparisons]
        raise CatalogError(
            f"{target} holds {len(ids)} comparisons; name one of them instead: {ids}"
        )
    raise CatalogError(
        f"no plan at {target}: expected a plan.json, a directory containing one, or a "
        "results directory with a single comparison_id=... in it"
    )


def restrict_to_worker(plan: "Plan", worker_id: str) -> "Plan":
    """The same plan with every worker but one removed.

    What ``--worker`` does, and it is a *filter* rather than a different plan:
    ``plan_id``, ``catalog_hash``, every scene hash and every episode id are
    untouched, so rows written by one worker join rows written by another as
    though one process had produced both. That is the property that makes a
    Compose run and a single-process run the same experiment.

    Scenes that keep no workers are dropped, because a backend that iterates
    scenes would otherwise construct one and find nothing to do -- and for the
    vla-eval backend, constructing a LIBERO scene costs twelve seconds.

    ``episode_count`` and ``estimated_seconds`` are recomputed for the scenes
    that remain. They are the plan's own accounting of itself, and leaving them
    describing episodes this process will not run would make a progress line
    lie.
    """
    known = [w.worker_id for s in plan.scenes for w in s.workers]
    if worker_id not in known:
        raise PlanSchemaError(
            f"this plan has no worker {worker_id!r}. It has {len(known)}: "
            f"{sorted(known)}. A worker id is assigned by `refractal plan`, so a "
            "mismatch usually means the plan was recompiled after the command "
            "referring to it was written."
        )

    scenes = []
    for scene in plan.scenes:
        mine = [w for w in scene.workers if w.worker_id == worker_id]
        if not mine:
            continue
        scenes.append(
            scene.model_copy(
                update={
                    "workers": mine,
                    "episode_count": sum(len(w.episodes) for w in mine),
                    "estimated_seconds": sum(w.estimated_seconds for w in mine),
                }
            )
        )
    return plan.model_copy(
        update={
            "scenes": scenes,
            "total_episodes": sum(s.episode_count for s in scenes),
            "estimated_seconds": sum(s.estimated_seconds for s in scenes),
        }
    )


def read_plan(path: str | Path) -> Plan:
    """Load a plan, refusing an unrecognised ``plan_schema`` before anything else.

    The check has to come first. Validating the body of a plan written by a
    future version produces a heap of confusing field errors; refusing on the
    version produces one sentence that says what to do.
    """
    path = locate_plan(path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CatalogError("plan.json must contain an object", file=str(path))
    declared = raw.get("plan_schema")
    # At or below, not equal. A reader knows the shapes that came before it --
    # every field added since is optional here, which is what makes that true --
    # and refusing an older plan would strand results whose plan is on disk
    # beside them. Above is still refused: those fields have no meaning yet.
    if not isinstance(declared, int) or declared > PLAN_SCHEMA or declared < 1:
        raise PlanSchemaError(
            f"plan declares plan_schema={declared!r}; this build of Refractal reads "
            f"plan_schema={PLAN_SCHEMA} and below. "
            + (
                "The plan was written by a newer Refractal -- upgrade."
                if isinstance(declared, int) and declared > PLAN_SCHEMA
                else "Re-run 'refractal plan' to regenerate it."
            )
        )
    return Plan.model_validate(raw)


__all__ = [
    "PLAN_SCHEMA",
    "Plan",
    "PlanSchemaError",
    "PlannedEpisode",
    "PlannedScenario",
    "PlannedScene",
    "PlannedWorker",
    "explain_identity_difference",
    "locate_plan",
    "read_plan",
    "restrict_to_worker",
]
