"""``catalog/build.lock`` -- facts that need the engine, recorded once.

Four things drift, and they all drift for the same reason: they are properties
of an environment that `refractal plan` deliberately does not have.

======================  =========================================================
`model_hash`            reads sources; verified against the compiled model
`engine_version`        only knowable where the engine is installed
`resource_shape`        probed by a calibration episode
filter survivors        may need FK, collision queries, penetration tests
======================  =========================================================

So `refractal build` produces all four, in one command, in one file. `resolve`
reads it and **refuses** when an entry is stale, exactly as it refuses to
oversubscribe VRAM: a plan built on a stale fact is worse than no plan.

Two decisions worth keeping:

**Never rewrite hand-authored YAML.** The API reference has `build` overwrite
`scenes.yaml` in place. Machine-written and human-written data in one file costs
comments and produces merge conflicts on every rebuild. Everything derived goes
here instead.

**Every entry carries the key it depends on**, so staleness is detected rather
than assumed. A filter's survivors depend on the scenario set that generated the
candidates, the geometry they were tested against, and the filter itself -- change
any one and the recorded answer is about a different question.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from ..schema.canonical import hash_obj
from ..schema.errors import CatalogError
from ..schema.models import ResourceShape, ScenarioSet, Strict

LOCK_FILENAME = "build.lock"
LOCK_SCHEMA = 1


def filter_key(scenario_set: ScenarioSet, scene_hash: str) -> str:
    """Identity of the question "which of these scenarios survive the filter?"."""
    return hash_obj(
        {
            "scenario_set": {
                "generator": scenario_set.generator,
                "generator_seed": scenario_set.generator_seed,
                "params": {
                    k: v.model_dump(exclude_unset=True) for k, v in scenario_set.params.items()
                },
            },
            "scene_hash": scene_hash,
            "filter": scenario_set.filter,
        }
    )


def filter_source_sha(callable_obj: Any) -> str | None:
    """Hash a callable's own source text, or None when it has none to read."""
    import inspect

    try:
        source = inspect.getsource(callable_obj)
    except (OSError, TypeError):
        return None
    return hash_obj({"source": source})


class FilterEntry(Strict):
    scenario_set_id: str
    key: str
    survivors: list[str]
    generated: int
    dropped: int
    #: sha256 of ``inspect.getsource`` of the resolved filter callable.
    #:
    #: Deliberately NOT part of ``key``. ``resolve`` has to be able to recompute
    #: the key to detect staleness, and recomputing a source hash means importing
    #: the filter -- which is the thing filters were moved to ``build`` to avoid,
    #: since a reachability filter drags in a kinematics stack. So the body is
    #: recorded here and verified separately, best-effort.
    #:
    #: **Known limit:** this covers the function's own source text. A filter that
    #: calls a helper, or imports a module whose behaviour changed, still hashes
    #: the same. It catches the common case -- someone edits the filter and
    #: re-plans -- and does not pretend to catch every case.
    source_sha: str | None = None


class ShapeEntry(Strict):
    scene_id: str
    key: str
    shape: ResourceShape


class SceneEntry(Strict):
    scene_id: str
    scene_hash: str
    model_hash: str
    engine_version: str


class BuildLock(Strict):
    lock_schema: int = LOCK_SCHEMA
    scenes: list[SceneEntry] = Field(default_factory=list)
    filters: list[FilterEntry] = Field(default_factory=list)
    shapes: list[ShapeEntry] = Field(default_factory=list)
    built_at: str | None = None

    #: Set by :func:`load_lock`; the scene hashes the filters were keyed against.
    def filter_survivors(self, scenario_set: ScenarioSet) -> set[str]:
        entry = next(
            (f for f in self.filters if f.scenario_set_id == scenario_set.id), None
        )
        if entry is None:
            raise CatalogError(
                f"scenario_set {scenario_set.id!r} declares a filter, but build.lock has no "
                "entry for it. Run 'refractal build'.",
                file=LOCK_FILENAME,
            )
        return set(entry.survivors)

    def check_filter_source(self, scenario_set: ScenarioSet) -> str | None:
        """Best-effort check that the filter's body is the one that ran.

        Importing the filter may fail -- that is the normal case on a laptop with
        no simulator, and it must stay normal, so a failed import returns a note
        rather than raising. On the machine where ``build`` ran the import
        succeeds and a body edit is caught, which is where the mistake actually
        gets made.
        """
        entry = next(
            (f for f in self.filters if f.scenario_set_id == scenario_set.id), None
        )
        if entry is None or entry.source_sha is None or scenario_set.filter is None:
            return None
        try:
            from ..schema.importstr import resolve_import_string

            current = filter_source_sha(resolve_import_string(scenario_set.filter))
        except Exception as exc:
            return (
                f"could not verify the body of filter {scenario_set.filter!r} "
                f"({type(exc).__name__}); the recorded survivors are trusted. Run "
                "'refractal build' where the filter is importable to re-verify."
            )
        if current is None:
            return f"filter {scenario_set.filter!r} has no readable source to verify"
        if current != entry.source_sha:
            raise CatalogError(
                f"the body of filter {scenario_set.filter!r} changed since build.lock was "
                f"written, though its name did not. The recorded survivors for "
                f"{scenario_set.id!r} were computed by different logic. Re-run "
                "'refractal build'.",
                file=LOCK_FILENAME,
            )
        return None

    def check_filter_fresh(self, scenario_set: ScenarioSet, scene_hash: str) -> None:
        entry = next(
            (f for f in self.filters if f.scenario_set_id == scenario_set.id), None
        )
        if entry is None:
            return
        expected = filter_key(scenario_set, scene_hash)
        if entry.key != expected:
            raise CatalogError(
                f"build.lock's filter result for scenario_set {scenario_set.id!r} is stale: "
                "the scenario set, the scene geometry, or the filter itself changed since it "
                "was recorded. Re-run 'refractal build'.",
                file=LOCK_FILENAME,
            )


def load_lock(catalog_root: Path) -> BuildLock | None:
    path = Path(catalog_root) / LOCK_FILENAME
    if not path.is_file():
        return None
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise CatalogError(f"is not valid UTF-8: {exc}", file=LOCK_FILENAME) from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"is not valid JSON: {exc}", file=LOCK_FILENAME) from exc
    if not isinstance(raw, dict):
        raise CatalogError("must contain an object", file=LOCK_FILENAME)
    declared = raw.get("lock_schema")
    if declared != LOCK_SCHEMA:
        raise CatalogError(
            f"declares lock_schema={declared!r}; this build reads {LOCK_SCHEMA}. "
            "Re-run 'refractal build'.",
            file=LOCK_FILENAME,
        )
    return BuildLock.model_validate(raw)


__all__ = [
    "LOCK_FILENAME",
    "LOCK_SCHEMA",
    "BuildLock",
    "FilterEntry",
    "SceneEntry",
    "ShapeEntry",
    "filter_key",
    "filter_source_sha",
    "load_lock",
]
