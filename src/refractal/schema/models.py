"""Catalog models.

Field-for-field from the API reference, with the deviations marked ``DEVIATION``
and argued in the docstring beside them. Nothing in this module imports a
simulator, a GPU library, Docker or the harness; nothing here does I/O beyond
what :mod:`refractal.schema.loader` hands it.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import NotImplementedInV1
from .importstr import validate_import_string

API_VERSION = "refractal.dev/v1alpha1"

_ID_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def _check_id(value: str) -> str:
    if not _ID_RE.match(value):
        raise ValueError(
            f"{value!r} is not a valid id: lowercase alphanumerics and hyphens, "
            "not starting or ending with a hyphen"
        )
    return value


Id = Annotated[str, Field(min_length=1, max_length=128)]
ImportString = Annotated[str, Field(min_length=3)]


class Strict(BaseModel):
    """Base config for every catalog model.

    ``extra="forbid"`` is the single highest-value line in this file. A typo in
    a YAML key is the most common failure mode in configuration-driven systems,
    and silently accepting it costs an afternoon.

    ``protected_namespaces=()`` is needed because the spec names two fields
    ``model`` and ``model_hash``, which collide with pydantic's reserved
    ``model_*`` namespace. That collision is a small argument for the rename
    proposed in the review notes: the value is called ``scene_hash`` everywhere
    downstream anyway.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
        frozen=True,
    )


class ApiObject(Strict):
    api_version: Literal[API_VERSION] = Field(alias="apiVersion")


# --------------------------------------------------------------------------
# scenes.yaml
# --------------------------------------------------------------------------


class ResourceShape(Strict):
    """Given this scene on this hardware: how much machine per worker, how wide a batch.

    A property of the scene, not of the run. Keyed by
    ``(scene, engine, hardware_profile)`` -- 512 envs on a 5090 is not 512 on an
    A100.
    """

    hardware_profile: str
    envs_per_process: int = Field(gt=0)
    vram_per_env_mb: int = Field(ge=0)
    vram_base_mb: int = Field(default=0, ge=0)
    cpu_cores: int = Field(gt=0)
    memory_mb: int = Field(default=2048, gt=0)
    #: Wall clock for 1000 steps of **one full batch, as the worker experiences
    #: it** -- not of the simulator.
    #:
    #: The distinction is load-bearing and was got wrong. Measuring the simulator
    #: alone gives 8.0s for LIBERO; the worker blocks on a model server every
    #: step, and the real figure from 600 episodes is 40.6s, of which ~33s is
    #: inference. Both numbers are correct measurements; only one is of this
    #: field.
    #:
    #: It INCLUDES amortised per-invocation setup, because that setup lands
    #: inside the episodes' own elapsed time -- LIBERO's first episode of each
    #: invocation runs 3.3 ms/step slower than the rest, which is the environment
    #: being constructed. Do not also count it in ``startup_sec``.
    sec_per_1k_steps: float = Field(gt=0)
    #: Cost paid once per worker *process*, before any episode runs: interpreter
    #: start, device context, model load, JIT compilation.
    #:
    #: NOT per-invocation setup, which for a worker driving vla-eval is 0.1s
    #: median and already inside ``sec_per_1k_steps``. Measured for MJX at 2.6s
    #: (0.2 import jax, 0.3 CUDA context, 0.8 model load, 1.2 first compile) and
    #: it does not vary with batch shape -- compile is 0.7s flat from 256 to
    #: 4096 envs.
    startup_sec: int = Field(ge=0)
    #: What a worker must own WHOLE. Not how many may share a process at once --
    #: that is ``envs_per_process``, and the two are independent.
    #:
    #: ``scenario``
    #:     Any subset of a scene's scenarios may be split off. MJX: the batch
    #:     genuinely holds arbitrary same-scene scenarios.
    #: ``task``
    #:     A worker must own a task's whole zero-based scenario range. Classic
    #:     MuJoCo under vla-eval: the harness rebuilds per ``task_id`` and counts
    #:     episodes from zero within one, so a worker holding a later slice runs
    #:     the early init states while every row claims the late ones.
    #:
    #: An enum rather than a number, and the reason is how it reads in three
    #: years rather than how it behaves. A number invites arithmetic --
    #: ``envs_per_process`` is a number and gets divided into things, which is the
    #: conflation that let a scenario range be split twelve ways. An enum says it
    #: is a boundary.
    #:
    #: **Declared, never derived.** Measured on the only two engines available:
    #:
    #:     scene              engine  envs/proc  partition_unit  workers chosen
    #:     libero-spatial     mujoco          1            task              12
    #:     panda-pick-cube    mjx          4096        scenario               1
    #:
    #: The planner splits the engine that cannot be split and refuses to split the
    #: one that can. So a formula linking the two fields is not underdetermined,
    #: it is wrong in the direction it points -- a derivation would have to
    #: invert, and a rule that inverts on a sample of two was not nearly right.
    #:
    #: It is a property of the engine-plus-BENCHMARK pair, not of the engine
    #: alone: a different driver over classic MuJoCo could answer ``scenario``.
    #: Which is why it lives on ``ResourceShape``, already keyed by
    #: ``(scene, engine, hardware_profile)``.
    partition_unit: Literal["scenario", "task"] = "task"
    #: A hard ceiling on the batch, when the hardware has one the planner cannot
    #: infer -- a driver limit, a licence, a known instability above N.
    #:
    #: NOT "a ceiling the planner may raise the batch to", which is what the
    #: validator used to say. The planner never raises a batch: it uses
    #: ``envs_per_process`` as measured. That clause described a feature nobody
    #: built, and the MJX measurement makes it uninteresting -- throughput is flat
    #: from 256 envs upward, so there is nothing to gain by raising toward a
    #: ceiling.
    #:
    #: Kept because declaring a real hardware limit is worth doing even when only
    #: the validator reads it; removed from the error message is the claim that
    #: something acts on it.
    max_envs: int | None = Field(default=None, gt=0)
    measured_at: str | None = None

    @model_validator(mode="after")
    def _check_max_envs(self) -> "ResourceShape":
        if self.max_envs is not None and self.max_envs < self.envs_per_process:
            raise ValueError(
                f"max_envs ({self.max_envs}) is below envs_per_process "
                f"({self.envs_per_process}). max_envs is a hard ceiling on the batch; "
                "a scene cannot be measured at a batch size its own ceiling forbids."
            )
        return self

    def vram_mb(self, envs: int | None = None) -> int:
        """Total VRAM for one worker at the given batch width."""
        return self.vram_base_mb + self.vram_per_env_mb * (envs or self.envs_per_process)


class ExternalScene(Strict):
    """A scene defined by someone else's benchmark, not by a file in this catalog.

    Wrapping a third-party benchmark means the geometry lives inside an installed
    package -- a LIBERO scene is a BDDL file inside `libero`, not an MJCF under
    ``catalog/assets``. There is nothing catalog-local to hash, and inventing a
    placeholder ``model:`` path would put a lie in the one artifact whose job is
    to not lie: the catalog copied into every results directory as provenance.

    So the scene says where it comes from instead, and ``refractal build``
    supplies its hash into the lock.
    """

    #: The benchmark class that defines this scene.
    provider: ImportString
    #: How that provider identifies it, e.g. ``{suite: libero_spatial, task_id: 3}``.
    #: Part of the scene's catalog-side identity: changing it means a different
    #: scene, and the lock goes stale.
    #:
    #: **Location only.** This says which scene; it is not passed to the provider.
    ref: dict[str, Any] = Field(default_factory=dict)
    #: Constructor arguments for the provider, e.g.
    #: ``{send_state: true, num_steps_wait: 10}``.
    #:
    #: Separate from ``ref`` because one field cannot do both jobs. ``ref`` held
    #: both at first, and a backend passed the whole thing to the provider --
    #: which raises, because ``task_id`` identifies a LIBERO task and is not a
    #: constructor argument of the benchmark that owns it.
    #:
    #: Both are hashed into the scene's identity, and for the same reason: these
    #: change what the policy observes. ``send_state: false`` against a checkpoint
    #: trained with proprioception is the parameter that moved X-VLA on LIBERO
    #: from 97.8% to 42% (arXiv 2603.13966v2 SS III-B). Two runs that disagree
    #: about it are not one experiment.
    #:
    #: The split is the same distinction as everywhere else here: what a thing
    #: *is* versus what is needed to *make* it. Conflating them is the shape of
    #: the original bug -- ``(task_name, 7)`` is a location standing in for an
    #: identity -- one level up.
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ref_and_params_must_agree(self) -> "ExternalScene":
        """A key in both must mean the same thing in both.

        Some keys legitimately appear twice: LIBERO's ``suite`` both identifies
        the scene and is a constructor argument of the benchmark that owns it. So
        the split cannot be "no key appears in both" -- it has to be "a key in
        both agrees".

        Without this, ``ref: {suite: libero_spatial}`` with
        ``params: {suite: libero_object}`` is a scene whose identity says one
        suite and whose execution uses another. That is not a mistake anyone would
        make deliberately, and it is exactly the kind that survives a review
        because both halves read correctly on their own.
        """
        conflicting = {
            key: (self.ref[key], self.params[key])
            for key in self.ref.keys() & self.params.keys()
            if self.ref[key] != self.params[key]
        }
        if conflicting:
            detail = "; ".join(
                f"{key}: ref={ref!r} params={param!r}"
                for key, (ref, param) in sorted(conflicting.items())
            )
            raise ValueError(
                f"external scene ref and params disagree -- {detail}. A key in both must "
                "mean the same thing in both: the identity would record one value and the "
                "run would use the other."
            )
        return self


    @model_validator(mode="after")
    def _check(self) -> "ExternalScene":
        validate_import_string(self.provider)
        return self


class Scene(Strict):
    """The physical world. **The affinity key.**

    Two scenarios belong to the same scene iff they compile to the same physics
    model and differ only in that model's data. Body, joint and geom counts and
    collision pairs live in the model, so they split scenes; position,
    orientation, mass, friction and damping live in the data, so they do not.
    """

    id: Id
    engine: Literal["mujoco", "mjx", "isaac"]
    #: Exactly one of ``model`` and ``external`` -- enforced, not merely optional.
    #: An optional field with no declared alternative is one people forget; a
    #: required choice is one they make.
    model: str | None = None
    external: ExternalScene | None = None
    #: DEVIATION: added. ``model_hash`` is specified as covering "the engine
    #: version string", but nothing in the schema carries an engine version and
    #: ``resolve`` runs where no engine is installed to be asked. So the version
    #: is a declared field, written by ``refractal build`` where the engine does
    #: exist, alongside the hash itself.
    engine_version: str | None = None
    model_hash: str | None = None
    assets: list[str] = Field(default_factory=list)
    resource_shape: list[ResourceShape] = Field(default_factory=list)
    description: str = ""

    @property
    def is_external(self) -> bool:
        return self.external is not None

    @model_validator(mode="after")
    def _check(self) -> "Scene":
        _check_id(self.id)
        if (self.model is None) == (self.external is None):
            raise ValueError(
                f"scene {self.id!r} must declare exactly one of 'model' (a file in this "
                "catalog) or 'external' (a scene defined by a wrapped benchmark); "
                + ("it declares both" if self.model else "it declares neither")
            )
        seen = set()
        for shape in self.resource_shape:
            if shape.hardware_profile in seen:
                raise ValueError(
                    f"scene {self.id!r} has two resource shapes for hardware profile "
                    f"{shape.hardware_profile!r}"
                )
            seen.add(shape.hardware_profile)
        return self

    def shape_for(self, hardware_profile: str) -> ResourceShape | None:
        for shape in self.resource_shape:
            if shape.hardware_profile == hardware_profile:
                return shape
        return None


class ScenesFile(ApiObject):
    scenes: list[Scene] = Field(min_length=1)


# --------------------------------------------------------------------------
# tasks.yaml
# --------------------------------------------------------------------------


class Phase(Strict):
    """A sub-goal, so an episode yields an outcome vector rather than one bit."""

    name: str = Field(min_length=1)
    predicate: ImportString
    predicate_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "Phase":
        validate_import_string(self.predicate)
        return self


class Task(Strict):
    """What the robot must achieve, and the predicate that decides success.

    Not in the physics model, which is why tasks are cheap to vary and scenes
    are expensive: 200 tasks over 10 scenes runs far faster than 10 tasks over
    200 scenes.
    """

    id: Id
    scene: Id
    instruction: str = Field(min_length=1)
    predicate: ImportString
    predicate_args: dict[str, Any] = Field(default_factory=dict)
    phases: list[Phase] = Field(default_factory=list)
    #: DEVIATION: ``max_steps`` is unmarked in the reference but participates in
    #: ``task_hash``. Raising it from 400 to 800 changes outcomes, so it changes
    #: comparability; leaving it out of identity would let a resumed run skip
    #: episodes recorded under the old limit.
    max_steps: int = Field(default=400, gt=0)
    #: How the scene's provider identifies this task, e.g. ``{task_id: 3}``.
    #:
    #: Only meaningful on an externally-defined scene, and needed once a scene
    #: carries more than one task -- which is the normal case, since a scene is
    #: the compiled model and tasks are cheap to vary on it. The scene's ``ref``
    #: says which model; this says which goal within it.
    #:
    #: Identity, via ``task_hash``: two tasks pointing at different provider
    #: tasks are different tasks even if their instructions somehow matched.
    provider_ref: dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Task":
        _check_id(self.id)
        validate_import_string(self.predicate)
        names = [p.name for p in self.phases]
        if len(names) != len(set(names)):
            raise ValueError(f"task {self.id!r} declares duplicate phase names: {names}")
        return self


class TasksFile(ApiObject):
    tasks: list[Task] = Field(min_length=1)


# --------------------------------------------------------------------------
# scenarios.yaml
# --------------------------------------------------------------------------


class ParamSpec(Strict):
    """One parameter axis. Exactly one form; mixing them is a validation error.

    | Form     | Fields                                       |
    |----------|----------------------------------------------|
    | range    | ``range: [min, max]``, ``steps``             |
    | choices  | ``choices: [...]``                           |
    | constant | ``value: x``                                 |
    | random   | ``range``, ``samples``, ``distribution``     |
    """

    #: ``int | float``, not ``float``: annotating this as float would make
    #: pydantic coerce ``range: [0, 49]`` to ``[0.0, 49.0]`` before a generator
    #: ever sees it, destroying the authored type that decides whether the axis
    #: yields indices or quantities.
    range: list[int | float] | None = None
    steps: int | None = Field(default=None, gt=0)
    choices: list[Any] | None = None
    value: Any = None
    samples: int | None = Field(default=None, gt=0)
    distribution: Literal["uniform", "normal"] | None = None
    mean: float | None = None
    std: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_form(self) -> "ParamSpec":
        form = self.form

        if form in ("range", "random"):
            if self.range is None or len(self.range) != 2:
                raise ValueError("range must be a two-element [min, max]")
            if self.range[0] > self.range[1]:
                raise ValueError(f"range min {self.range[0]} exceeds max {self.range[1]}")
        if form == "range" and self.steps is None:
            raise ValueError("range form requires 'steps'")
        if form == "random":
            if self.distribution is None:
                raise ValueError("random form requires 'distribution' (uniform or normal)")
            if self.distribution == "normal" and (self.mean is None or self.std is None):
                raise ValueError("normal distribution requires 'mean' and 'std'")
            if self.distribution == "uniform" and (self.mean is not None or self.std is not None):
                raise ValueError("'mean' and 'std' apply only to the normal distribution")
        if form == "choices" and not self.choices:
            raise ValueError("choices must be non-empty")
        return self

    @property
    def form(self) -> str:
        """Which of the four forms this spec uses.

        Derived from ``model_fields_set`` rather than from "is not None", so
        that an explicit ``value: null`` reads as a constant rather than as an
        absent field. Kept as a property rather than stored so the model stays
        frozen and round-trips through ``model_dump`` unchanged.
        """
        given = self.model_fields_set
        forms = []
        if "samples" in given:
            forms.append("random")
        elif "steps" in given or "range" in given:
            forms.append("range")
        if "choices" in given:
            forms.append("choices")
        if "value" in given:
            forms.append("constant")
        if len(forms) != 1:
            raise ValueError(
                "a parameter must use exactly one form -- range+steps, choices, "
                "value, or range+samples+distribution -- "
                f"but got keys {sorted(given)}"
            )
        return forms[0]

    def cardinality(self) -> int:
        """How many values this axis contributes to a full cross product."""
        if self.form == "range":
            return int(self.steps or 1)
        if self.form == "choices":
            return len(self.choices or ())
        if self.form == "random":
            return int(self.samples or 1)
        return 1


class FaultSpec(Strict):
    """**Reserved.** Validated, never executed in v1.

    The hook exists so that the scenario identity format does not change when
    Transect arrives. Note that this is not achieved by the field's presence
    alone -- see ``identity.scenario_identity``, which folds an (empty) faults
    list into every scenario's canonical form from the first commit. Without
    that, adding faults later would change the shape of the hashed document and
    invalidate every previously recorded scenario_hash, which is precisely what
    reserving the field was meant to prevent.
    """

    at_step: int = Field(ge=0)
    type: str
    target: str
    args: dict[str, Any] = Field(default_factory=dict)


class ScenarioSet(Strict):
    id: Id
    scene: Id
    generator: ImportString
    #: No default, deliberately: an unseeded generator cannot be reproduced, and
    #: a default would let you forget.
    generator_seed: int
    params: dict[str, ParamSpec] = Field(min_length=1)
    filter: ImportString | None = None
    tasks: list[Id] | None = None
    faults: list[FaultSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ScenarioSet":
        _check_id(self.id)
        validate_import_string(self.generator)
        if self.filter is not None:
            validate_import_string(self.filter)
        if self.faults:
            raise NotImplementedInV1(
                f"scenario_set {self.id!r} declares faults. Fault injection is Transect, "
                "not in this version. The key is reserved and validated so the scenario "
                "hash format will not change when it lands; leave it empty."
            )
        return self


class ScenariosFile(ApiObject):
    scenario_sets: list[ScenarioSet] = Field(min_length=1)


# --------------------------------------------------------------------------
# run.yaml
# --------------------------------------------------------------------------


class Checkpoint(Strict):
    id: Id
    path: str
    server: ImportString
    server_args: dict[str, Any] = Field(default_factory=dict)
    #: ``ge=0``, not ``gt=0``: a CPU-only policy needs no VRAM, and so does the
    #: echo server the `refractal init` example ships. Requiring a positive
    #: value would make the shipped example unplannable, which is how this was
    #: found.
    vram_mb: int = Field(default=8192, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "Checkpoint":
        _check_id(self.id)
        validate_import_string(self.server)
        return self


class Run(Strict):
    checkpoints: list[Checkpoint] = Field(min_length=1)
    results_uri: str = Field(min_length=1)
    seeds: int = Field(default=3, gt=0)
    seed_base: int = 0
    tier: Literal["smoke", "regression", "full"] = "full"
    scenario_sets: list[Id] | None = None
    #: Defined by episode ordering, not by how the servers are deployed.
    #:
    #: ``serial``
    #:     All tasks for checkpoint A, then all tasks for checkpoint B. The only
    #:     option when VRAM cannot hold both policies at once.
    #: ``concurrent``
    #:     Both checkpoints at the same time, each against its own server. Halves
    #:     wall clock. Checkpoints contend, so durations are not comparable and the rows
    #:     record which other checkpoints were running alongside.
    #:
    #: ``interleaved`` is deliberately absent. It existed to remove within-session
    #: drift, and drift was measured at zero: across ten tasks and three replicate
    #: positions, the pooled rank correlation between position and cell rate was
    #: +0.027 (p=0.839) for pi0 and +0.132 (p=0.230) for pi0.5, with the contrast
    #: at +0.046 (p=0.723), measured across ten tasks and three replicate
    #: positions. A third mode whose only
    #: justification does not hold is a default somebody picks for a reason that
    #: is not true.
    #:
    #: Never in ``plan_id``: two runs of one experiment in different modes belong
    #: in one comparison, so this is recorded per episode and `compare` gates
    #: latency reporting on it rather than refusing the join.
    execution_mode: Literal["serial", "concurrent"] = "serial"

    @field_validator("execution_mode", mode="before")
    @classmethod
    def _explain_interleaved(cls, value: Any) -> Any:
        """`interleaved` is refused with its history rather than aliased.

        Aliasing it to `serial` would be the right guess -- the old
        deployment-shaped definition was recorded on runs that executed serially
        -- and it would also relabel someone's catalog without telling them. The
        author asked for a mode that no longer exists; which of the two they
        meant is their call, not a default.
        """
        if value == "interleaved":
            raise ValueError(
                "execution_mode 'interleaved' no longer exists. It was defined by "
                "whether both servers stayed resident, which is a property of the "
                "deployment rather than of the run and distinguished nothing -- a "
                "serial run with two servers up had it too. Its successor was to be "
                "task-outer ordering, to remove within-session drift; drift was then "
                "measured at zero (ten tasks, three replicate positions, rank "
                "correlation +0.027 at p=0.839). Choose 'serial' (all tasks for one "
                "checkpoint, then the next) or 'concurrent' (both at once, each "
                "against its own server). See docs/execution-mode.md."
            )
        return value

    @model_validator(mode="after")
    def _check(self) -> "Run":
        ids = [c.id for c in self.checkpoints]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate checkpoint ids: {ids}")
        return self

    def seed_values(self) -> list[int]:
        """The seeds every checkpoint faces.

        Identical across checkpoints by construction -- that is what makes the
        comparison paired. If two checkpoints saw different seeds there would be
        nothing to run McNemar's test on.
        """
        return [self.seed_base + i for i in range(self.seeds)]


class RunFile(ApiObject):
    run: Run


# --------------------------------------------------------------------------
# hardware.yaml -- DEVIATION: proposed addition, see the review notes
# --------------------------------------------------------------------------


class Device(Strict):
    id: str  # "cuda:0", "cpu"
    vram_mb: int = Field(ge=0)


class HardwareProfile(Strict):
    """What a machine actually has.

    DEVIATION: not in the API reference, and the planner cannot work without it.
    ``--hardware rtx5090`` selects a *resource shape*, which describes what one
    worker needs; nothing anywhere describes what the host provides. Yet the
    reference's own worked example prints ``VRAM: cuda:0 16384 / 32768 MB`` and
    a total worker count, both of which require a capacity to divide into.

    Deliberately excluded from ``plan_id``: hardware determines placement, not
    experiment identity. The same experiment planned for a 5090 and for an A100
    is the same experiment, and its results must join.
    """

    id: str
    cpu_cores: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    devices: list[Device] = Field(default_factory=list)
    max_workers: int | None = Field(default=None, gt=0)

    def device(self, device_id: str) -> Device | None:
        for d in self.devices:
            if d.id == device_id:
                return d
        return None


class HardwareFile(ApiObject):
    hardware_profiles: list[HardwareProfile] = Field(min_length=1)


__all__ = [
    "API_VERSION",
    "ExternalScene",
    "ApiObject",
    "Checkpoint",
    "Device",
    "FaultSpec",
    "HardwareFile",
    "HardwareProfile",
    "ParamSpec",
    "Phase",
    "ResourceShape",
    "Run",
    "RunFile",
    "Scene",
    "ScenariosFile",
    "ScenarioSet",
    "ScenesFile",
    "Strict",
    "Task",
    "TasksFile",
]
