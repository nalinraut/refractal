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
from .models import Checkpoint, ResourceShape, Strict

#: Bump by hand, and only when the format changes in a way older readers cannot
#: survive. See the module docstring for why this is not derived.
PLAN_SCHEMA = 1


class PlanSchemaError(RefractalError):
    """A plan declares a ``plan_schema`` this build does not know how to read."""


class PlannedScenario(Strict):
    scenario_hash: str
    scenario_set_id: str
    params: dict[str, Any]


class PlannedEpisode(Strict):
    """One attempt, fully identified. The unit `execute` runs and `compare` joins."""

    episode_id: str
    task_id: str
    task_hash: str
    scenario_hash: str
    seed: int
    checkpoint_id: str


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
    resource_shape: ResourceShape
    #: **The expanded list, not the generator spec.** A generator whose
    #: implementation drifts would otherwise make the provenance a lie.
    scenarios: list[PlannedScenario]
    workers: list[PlannedWorker]
    episode_count: int
    estimated_seconds: int


class Plan(Strict):
    plan_schema: Literal[1] = PLAN_SCHEMA
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


def read_plan(path: str | Path) -> Plan:
    """Load a plan, refusing an unrecognised ``plan_schema`` before anything else.

    The check has to come first. Validating the body of a plan written by a
    future version produces a heap of confusing field errors; refusing on the
    version produces one sentence that says what to do.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CatalogError("plan.json must contain an object", file=str(path))
    declared = raw.get("plan_schema")
    if declared != PLAN_SCHEMA:
        raise PlanSchemaError(
            f"plan declares plan_schema={declared!r}; this build of Refractal reads "
            f"plan_schema={PLAN_SCHEMA}. "
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
    "read_plan",
]
