"""Load and cross-validate a catalog directory.

Per-file validation is pydantic's job. This module does the part pydantic
cannot see: references between files, duplicate ids across files, files on disk
that the YAML points at, and the warnings that are better raised now than
discovered by ``resolve``.

Nothing here imports a generator, a filter, a predicate or a server. See
:mod:`refractal.schema.importstr` for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import CatalogError
from .identity import experiment_identity, plan_id, scene_hash
from .models import (
    HardwareFile,
    HardwareProfile,
    Run,
    RunFile,
    Scene,
    ScenariosFile,
    ScenarioSet,
    ScenesFile,
    Task,
    TasksFile,
)

CATALOG_FILES = ("scenes.yaml", "tasks.yaml", "scenarios.yaml", "run.yaml")
OPTIONAL_FILES = ("hardware.yaml",)


@dataclass(frozen=True)
class Catalog:
    """A validated catalog, plus the identities derived from it."""

    root: Path
    scenes: list[Scene]
    tasks: list[Task]
    scenario_sets: list[ScenarioSet]
    run: Run
    hardware_profiles: list[HardwareProfile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # -- lookups ----------------------------------------------------------

    def scene(self, scene_id: str) -> Scene:
        for s in self.scenes:
            if s.id == scene_id:
                return s
        raise KeyError(scene_id)

    def task(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(task_id)

    def hardware(self, profile_id: str) -> HardwareProfile:
        for h in self.hardware_profiles:
            if h.id == profile_id:
                return h
        raise CatalogError(
            f"no hardware profile {profile_id!r}; declared profiles are "
            f"{[h.id for h in self.hardware_profiles] or 'none'}",
            file="hardware.yaml",
        )

    def tasks_for(self, scenario_set: ScenarioSet) -> list[Task]:
        """Which tasks a scenario set is crossed with.

        Defaults to every task on the set's scene, which is the cheap axis:
        tasks are not in the physics model, so adding one costs episodes but
        never splits a batch.
        """
        on_scene = [t for t in self.tasks if t.scene == scenario_set.scene]
        if scenario_set.tasks is None:
            return on_scene
        wanted = set(scenario_set.tasks)
        return [t for t in on_scene if t.id in wanted]

    def active_scenario_sets(self) -> list[ScenarioSet]:
        if self.run.scenario_sets is None:
            return list(self.scenario_sets)
        wanted = set(self.run.scenario_sets)
        return [ss for ss in self.scenario_sets if ss.id in wanted]

    # -- identity ---------------------------------------------------------

    def scene_hashes(self, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
        """Scene hashes, with recorded ones taking precedence over computed ones.

        ``overrides`` comes from ``catalog/build.lock``. The lock wins because it
        is the only thing that *can* win for an externally-defined scene -- there
        are no catalog-local sources to hash -- and because a recorded hash that
        silently disagrees with the sources is a bug either way. ``resolve``
        checks the two agree before getting here; this function only applies the
        precedence.
        """
        overrides = overrides or {}
        out: dict[str, str] = {}
        for scene in self.scenes:
            if scene.id in overrides:
                out[scene.id] = overrides[scene.id]
            elif scene.is_external:
                raise CatalogError(
                    f"scene {scene.id!r} is defined by {scene.external.provider!r}, so its "
                    "hash comes from catalog/build.lock, which has no entry for it. "
                    "Run 'refractal build'.",
                    file="scenes.yaml",
                )
            else:
                out[scene.id] = scene_hash(self.root, scene)
        return out

    def experiment_identity(
        self, scene_hashes: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        return experiment_identity(
            scenes=self.scenes,
            tasks=self.tasks,
            scenario_sets=self.scenario_sets,
            run=self.run,
            scene_hashes=self.scene_hashes(scene_hashes),
        )

    def plan_id(self, scene_hashes: Mapping[str, str] | None = None) -> str:
        """Also the ``comparison_id``: results and the experiment share one name."""
        return plan_id(self.experiment_identity(scene_hashes))

    def catalog_file_hash(self) -> str:
        """Digest of the catalog files as they sit on disk.

        Informational only, and deliberately *not* ``plan_id``. This one churns
        on a reordered key or a changed comment, which is exactly why it cannot
        be the experiment identity -- but it is the right thing to record when
        the question is "which bytes did this run actually read".
        """
        from .canonical import hash_file_set

        present = [
            self.root / name
            for name in (*CATALOG_FILES, *OPTIONAL_FILES)
            if (self.root / name).is_file()
        ]
        return hash_file_set(self.root, present)


def _read_yaml(path: Path) -> Any:
    """Read one catalog file.

    ``encoding="utf-8"`` is not boilerplate and must not be removed. Without it
    Python decodes using the *host locale*, which makes catalog identity depend
    on an environment variable:

    * On a strict codec the read raises. Upstream #135 is exactly this --
      ``UnicodeDecodeError: 'gbk' codec`` on a CP936 Windows host, and
      ``LC_ALL=C`` reproduces it on Linux. Loud, and therefore harmless.
    * On a lenient codec -- latin-1, cp1252 -- **every** byte sequence decodes
      without error, into different text. An instruction containing any
      non-ASCII character then produces a different ``task_hash``, a different
      ``plan_id``, and results that silently refuse to join with everyone
      else's. No exception is raised anywhere.

    The second case is the one that matters: a host environment variable
    quietly forking content-addressed identity is the precise failure class
    this project exists to make impossible.
    """
    if not path.is_file():
        raise CatalogError("missing from the catalog directory", file=path.name)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise CatalogError(
            f"is not valid UTF-8: {exc}. Catalog files are always read as UTF-8, "
            "regardless of host locale.",
            file=path.name,
        ) from exc
    except yaml.YAMLError as exc:
        raise CatalogError(f"is not valid YAML: {exc}", file=path.name) from exc
    if not isinstance(data, dict):
        raise CatalogError("must contain a mapping at the top level", file=path.name)
    return data


def _validate(model, data, filename: str):
    try:
        return model.model_validate(data)
    except Exception as exc:  # pydantic ValidationError, or our own ValueErrors
        raise CatalogError(str(exc), file=filename) from exc


def load_catalog(root: str | Path) -> Catalog:
    root = Path(root)
    if not root.is_dir():
        raise CatalogError(f"{root} is not a directory")

    scenes = _validate(ScenesFile, _read_yaml(root / "scenes.yaml"), "scenes.yaml").scenes
    tasks = _validate(TasksFile, _read_yaml(root / "tasks.yaml"), "tasks.yaml").tasks
    sets = _validate(
        ScenariosFile, _read_yaml(root / "scenarios.yaml"), "scenarios.yaml"
    ).scenario_sets
    run = _validate(RunFile, _read_yaml(root / "run.yaml"), "run.yaml").run

    hardware: list[HardwareProfile] = []
    hardware_path = root / "hardware.yaml"
    if hardware_path.is_file():
        hardware = _validate(
            HardwareFile, _read_yaml(hardware_path), "hardware.yaml"
        ).hardware_profiles

    warnings: list[str] = []
    _check_unique(scenes, "scene", "scenes.yaml")
    _check_unique(tasks, "task", "tasks.yaml")
    _check_unique(sets, "scenario_set", "scenarios.yaml")
    _check_unique(hardware, "hardware_profile", "hardware.yaml")

    scene_ids = {s.id for s in scenes}
    task_ids = {t.id for t in tasks}
    set_ids = {ss.id for ss in sets}

    for t in tasks:
        if t.scene not in scene_ids:
            raise CatalogError(
                f"task {t.id!r} references scene {t.scene!r}, which is not declared "
                f"(declared: {sorted(scene_ids)})",
                file="tasks.yaml",
            )

    for ss in sets:
        if ss.scene not in scene_ids:
            raise CatalogError(
                f"scenario_set {ss.id!r} references scene {ss.scene!r}, which is not declared",
                file="scenarios.yaml",
            )
        if ss.tasks is not None:
            for tid in ss.tasks:
                if tid not in task_ids:
                    raise CatalogError(
                        f"scenario_set {ss.id!r} references task {tid!r}, which is not declared",
                        file="scenarios.yaml",
                    )
                task = next(t for t in tasks if t.id == tid)
                if task.scene != ss.scene:
                    # Not a nicety: a task belongs to exactly one scene because
                    # the scene is the affinity key. Crossing them would produce
                    # an episode with no batch it can legally run in.
                    raise CatalogError(
                        f"scenario_set {ss.id!r} is on scene {ss.scene!r} but names task "
                        f"{tid!r}, which is on scene {task.scene!r}. Work cannot cross scenes.",
                        file="scenarios.yaml",
                    )
        if not [t for t in tasks if t.scene == ss.scene] :
            warnings.append(
                f"scenario_set {ss.id!r} is on scene {ss.scene!r}, which has no tasks; "
                "it will expand to zero episodes"
            )

    if run.scenario_sets is not None:
        for sid in run.scenario_sets:
            if sid not in set_ids:
                raise CatalogError(
                    f"run selects scenario_set {sid!r}, which is not declared",
                    file="run.yaml",
                )

    for s in scenes:
        if s.is_external:
            continue
        if not (root / s.model).is_file():
            raise CatalogError(
                f"scene {s.id!r} references model {s.model!r}, which does not exist "
                f"under {root}",
                file="scenes.yaml",
            )
        if s.model_hash is None:
            warnings.append(
                f"scene {s.id!r} has no model_hash; run 'refractal build' to record one. "
                "Computing it from sources now, so the plan is still reproducible, but "
                "nothing pins the engine version this scene was measured against."
            )
        if s.engine_version is None:
            warnings.append(
                f"scene {s.id!r} has no engine_version, so its scene_hash cannot change "
                "when the engine does; results from two engine versions will join as if "
                "comparable. Run 'refractal build'."
            )
        if not s.resource_shape:
            warnings.append(
                f"scene {s.id!r} declares no resource_shape; 'refractal plan' will refuse "
                "to fit workers for it"
            )

    unused = scene_ids - {t.scene for t in tasks}
    if unused:
        warnings.append(f"scenes with no tasks, so no episodes: {sorted(unused)}")

    return Catalog(
        root=root,
        scenes=scenes,
        tasks=tasks,
        scenario_sets=sets,
        run=run,
        hardware_profiles=hardware,
        warnings=warnings,
    )


def _check_unique(items, kind: str, filename: str) -> None:
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise CatalogError(f"duplicate {kind} id {item.id!r}", file=filename)
        seen.add(item.id)


__all__ = ["CATALOG_FILES", "OPTIONAL_FILES", "Catalog", "load_catalog"]
