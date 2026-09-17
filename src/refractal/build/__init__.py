"""``refractal build`` -- the facts that need an environment `plan` does not have.

Four things drift, all for the same reason: they are properties of a machine
where the engine is installed, and `refractal plan` deliberately runs where it is
not.

=====================  ======================================================
``scene_hash``         over model sources, meshes and the engine version
``engine_version``     only knowable where the engine exists
``resource_shape``     measured by running episodes
filter survivors       may need FK, collision queries, penetration tests
=====================  ======================================================

So one command produces all four, into ``catalog/build.lock``, and `resolve`
reads that and **refuses when an entry is stale** -- exactly as it refuses to
oversubscribe VRAM. A plan built on a stale fact is worse than no plan.

Two rules this establishes
--------------------------

**`build` and `execute` are the infra-touching packages; `schema`, `resolve` and
`compare` are not.** The original rule named only `execute`, which was written
before `build` existed. Both run where the engine does, and nothing else may.

**Hand-authored YAML is never rewritten.** The API reference has `build`
overwrite ``scenes.yaml`` in place. Machine-written and human-written data in one
file costs comments, produces a merge conflict on every rebuild, and makes it
impossible to tell what a human asserted from what a machine measured. Everything
derived goes in the lock.

What is *not* here yet
----------------------

Probing ``resource_shape`` means running episodes, which needs the adapter that
does not exist. The seam is defined (:class:`ShapeProber`) and the default
carries forward whatever ``scenes.yaml`` declares, recording that it was declared
rather than measured. `resolve` still refuses when no shape exists for the
requested hardware, so the failure stays loud.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..resolve.lock import (
    LOCK_FILENAME,
    LOCK_SCHEMA,
    BuildLock,
    FilterEntry,
    SceneEntry,
    ShapeEntry,
    filter_key,
    filter_source_sha,
)
from ..schema.errors import CatalogError, RefractalError
from ..schema.canonical import hash_obj
from ..schema.identity import external_scene_ref_key, scenario_hash, scene_hash
from ..schema.importstr import check_arity, resolve_import_string
from ..schema.loader import Catalog, load_catalog
from ..schema.models import ResourceShape, Scene


class BuildError(RefractalError):
    """Something needed for the lock could not be established."""


class EngineProbe(Protocol):
    """Reports the engine version, and verifies a scene actually compiles.

    The compile check is the other half of hashing sources rather than the
    compiled model: `plan` hashes what it can read on a laptop, and whoever has
    the engine confirms that those sources still produce the model they claim to.
    """

    def version(self, engine: str) -> str: ...

    def verify(self, engine: str, model_path: Path) -> None: ...


class ExternalSceneProbe(Protocol):
    """Supplies the facts that define a scene living inside a wrapped benchmark.

    Returns a **document, not a digest**. Hashing is Refractal's job and happens
    in exactly one place: a probe that hashed for itself would be a second
    implementation of the canonicalisation, and two implementations of one rule
    is how identities fork silently.

    It also means the facts can be recorded in the lock alongside the hash, so a
    changed ``scene_hash`` can be explained rather than merely observed -- the
    same reason the harness surface records a manifest and not only a digest.

    And it leaves a probe with nothing to import from Refractal. A probe is a
    plugin *into* Refractal, unlike an adapter, which vla-eval loads and which
    genuinely never touches it -- but nothing here needs it to depend on us.
    """

    def external_scene_facts(self, scene: Scene) -> dict[str, Any]: ...


class ShapeProber(Protocol):
    """Measures how much machine one worker of a scene needs."""

    def measure(self, scene: Scene, hardware_profile: str) -> ResourceShape: ...


@dataclass
class DeclaredProbe:
    """No engine available: use what the catalog asserts, and record that.

    Not a stub. Declaring a version by hand is legitimate -- it is how a catalog
    stays reproducible on a machine that only plans -- and the lock records the
    provenance so nobody later mistakes an assertion for a measurement.
    """

    measured: bool = False

    def version(self, engine: str) -> str:
        raise BuildError(
            f"no engine_version for {engine!r}: declare it in scenes.yaml, or run "
            "'refractal build' where the engine is installed."
        )

    def verify(self, engine: str, model_path: Path) -> None:
        return None


@dataclass
class BuildReport:
    lock: BuildLock
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [f"  computed scene_hash for {len(self.lock.scenes)} scene(s)"]
        for entry in self.lock.filters:
            lines.append(
                f"  {entry.scenario_set_id}: {entry.generated} generated, "
                f"{entry.dropped} dropped by filter, {len(entry.survivors)} kept"
            )
        if self.lock.shapes:
            lines.append(f"  recorded resource_shape for {len(self.lock.shapes)} scene(s)")
        return lines


def build(
    catalog: Catalog | str | Path,
    *,
    hardware_profile: str | None = None,
    probe: EngineProbe | None = None,
    prober: ShapeProber | None = None,
    built_at: str | None = None,
    write: bool = True,
) -> BuildReport:
    """Establish the four facts and write ``catalog/build.lock``.

    ``built_at`` is a parameter rather than a clock read, for the same reason
    ``created_at`` is on the planner: a command whose output depends on the time
    cannot be tested for determinism.
    """
    if not isinstance(catalog, Catalog):
        catalog = load_catalog(catalog)
    probe = probe or DeclaredProbe()
    report = BuildReport(lock=BuildLock(lock_schema=LOCK_SCHEMA, built_at=built_at))

    # --- 1 & 2: engine version, then scene hash over sources ------------
    resolved_scene_hashes: dict[str, str] = {}
    for scene in catalog.scenes:
        version = scene.engine_version
        if version is None:
            version = probe.version(scene.engine)
            report.notes.append(f"probed engine_version for {scene.id!r}: {version}")
            scene = scene.model_copy(update={"engine_version": version})

        if scene.is_external:
            # The geometry lives in a wrapped benchmark. Only a probe that can
            # import the provider knows what it is; without one there is nothing
            # honest to record.
            if probe is None or not hasattr(probe, "external_scene_facts"):
                raise BuildError(
                    f"scene {scene.id!r} is defined by {scene.external.provider!r}, so its "
                    "facts must come from a probe that can import that provider. Run "
                    "'refractal build' where the benchmark is installed."
                )
            facts = probe.external_scene_facts(scene)
            if not isinstance(facts, dict) or not facts:
                raise BuildError(
                    f"the probe returned {type(facts).__name__} for scene {scene.id!r}; "
                    "external_scene_facts must return a non-empty dict of the facts that "
                    "define the scene. Refractal hashes it — a probe that returns a digest "
                    "would be a second implementation of the canonicalisation."
                )
            # The probe's facts AND the catalog's own assertions. The facts alone
            # were the first version, and they are not enough: they describe what
            # LIBERO contains, not how this catalog asks for it. `external.params`
            # carries the benchmark's constructor arguments -- which cameras are
            # sent, whether proprioception is sent, which of two quaternion
            # conventions the state uses -- and those decide what the policy
            # observes.
            #
            # Measured: `quat_no_antipodal` moves pi0 on one LIBERO task from 0/8
            # to 2/4. Under the facts-only digest, a run with it and a run without
            # it had the SAME scene_hash and the same plan_id, so they would have
            # joined into one comparison and been averaged. That is the exact
            # failure this project exists to make unrepresentable, sitting inside
            # the identity scheme itself.
            #
            # Caught by checking plan_id after adding the flag rather than by the
            # test, which asserted on `external_scene_ref_key` -- the helper --
            # while the build hashed something else.
            digest = hash_obj(
                {"facts": facts, "catalog_ref": external_scene_ref_key(scene)}
            )
            resolved_scene_hashes[scene.id] = digest
            report.lock.scenes.append(
                SceneEntry(
                    scene_id=scene.id,
                    scene_hash=digest,
                    model_hash=digest,
                    engine_version=version,
                    external=True,
                    ref_key=external_scene_ref_key(scene),
                    facts=facts,
                )
            )
            continue

        digest = scene_hash(catalog.root, scene)
        resolved_scene_hashes[scene.id] = digest

        if scene.model_hash and scene.model_hash != digest:
            # Loud: the sources moved since somebody wrote that value down, so
            # every result recorded under it describes a different world.
            report.warnings.append(
                f"scene {scene.id!r} declares model_hash {scene.model_hash[:19]}... but its "
                f"sources now hash to {digest[:19]}.... The geometry changed; results recorded "
                "under the old hash are not comparable with new ones."
            )

        probe.verify(scene.engine, catalog.root / scene.model)
        report.lock.scenes.append(
            SceneEntry(
                scene_id=scene.id,
                scene_hash=digest,
                model_hash=digest,
                engine_version=version,
            )
        )

    # --- 3: filters, evaluated here because they may need the engine ----
    for scenario_set in catalog.scenario_sets:
        if scenario_set.filter is None:
            continue
        generator = resolve_import_string(scenario_set.generator)
        candidates = generator(scenario_set.params, scenario_set.generator_seed)
        try:
            predicate = resolve_import_string(scenario_set.filter)
        except Exception as exc:
            raise BuildError(
                f"scenario_set {scenario_set.id!r} declares filter {scenario_set.filter!r}, "
                f"which cannot be imported here: {exc}. 'refractal build' must run where the "
                "filter's dependencies are installed."
            ) from exc

        # Arity before anything that runs the filter. Shape is a property of the
        # callable alone; rejecting everything is a property of the callable and
        # the grid together. Diagnose the simpler thing first, or a wrong-shaped
        # filter reports as a workspace problem. Do not reorder these.
        check_arity(predicate, "filter", scenario_set.filter)

        survivors = []
        for row in candidates:
            try:
                keep = bool(predicate(row))
            except Exception as exc:
                raise BuildError(
                    f"filter {scenario_set.filter!r} raised on scenario {row!r}: {exc}"
                ) from exc
            if keep:
                survivors.append(scenario_hash(row))

        if not survivors:
            # Naming the parameters is the difference between a useful error and
            # a shrug. The third failure mode -- right arity, wrong parameter
            # names -- is invisible otherwise: a filter reading `cube_x` from a
            # grid that carries `vial_x` gets its default from `.get()` and
            # rejects everything, which looks identical to an unreachable
            # workspace. This was hit for real while writing the tests.
            keys = sorted(candidates[0]) if candidates else []
            raise BuildError(
                f"filter {scenario_set.filter!r} rejected all {len(candidates)} scenarios in "
                f"{scenario_set.id!r}. Those scenarios carry the parameters {keys}. "
                "Either the filter reads parameters this grid does not define, the grid is "
                "entirely outside the workspace, or the filter is inverted; all three are "
                "worth knowing before a run, not after."
            )

        report.lock.filters.append(
            FilterEntry(
                scenario_set_id=scenario_set.id,
                key=filter_key(scenario_set, resolved_scene_hashes[scenario_set.scene]),
                survivors=survivors,
                generated=len(candidates),
                dropped=len(candidates) - len(survivors),
                source_sha=filter_source_sha(predicate),
            )
        )

    # --- 3b: predicates, checked where they are importable ---------------
    # Best-effort: the adapter lives on the machine `build` runs on, but a
    # catalog can legitimately be built before its adapter is installed. An
    # unimportable predicate is a note; an importable one with the wrong shape
    # is an error, because that is a real bug found for free.
    for task in catalog.tasks:
        for label, import_string in [("predicate", task.predicate)] + [
            ("predicate", phase.predicate) for phase in task.phases
        ]:
            try:
                fn = resolve_import_string(import_string)
            except Exception:
                report.notes.append(
                    f"could not import {label} {import_string!r} to check its shape"
                )
                continue
            check_arity(fn, label, import_string)

    # --- 4: resource shapes ---------------------------------------------
    if hardware_profile:
        for scene in catalog.scenes:
            if prober is not None:
                shape = prober.measure(scene, hardware_profile)
                source = "measured"
            else:
                shape = scene.shape_for(hardware_profile)
                source = "declared"
                if shape is None:
                    report.warnings.append(
                        f"scene {scene.id!r} has no resource_shape for {hardware_profile!r} and "
                        "no prober was supplied, so none was recorded. 'refractal plan' will "
                        "refuse to fit workers for it."
                    )
                    continue
            report.lock.shapes.append(
                ShapeEntry(
                    scene_id=scene.id,
                    key=_shape_key(resolved_scene_hashes[scene.id], scene, hardware_profile),
                    shape=shape.model_copy(update={"measured_at": built_at if source == "measured" else None}),
                )
            )
            report.notes.append(
                f"resource_shape for {scene.id!r} on {hardware_profile!r}: {source}"
            )

    if write:
        path = catalog.root / LOCK_FILENAME
        path.write_text(report.lock.model_dump_json(indent=2) + "\n", encoding="utf-8")

    return report


def _shape_key(scene_digest: str, scene: Scene, hardware_profile: str) -> str:
    from ..schema.canonical import hash_obj

    return hash_obj(
        {
            "scene_hash": scene_digest,
            "engine": scene.engine,
            "engine_version": scene.engine_version,
            "hardware_profile": hardware_profile,
        }
    )


__all__ = [
    "BuildError",
    "BuildReport",
    "DeclaredProbe",
    "EngineProbe",
    "ExternalSceneProbe",
    "ShapeProber",
    "build",
]
