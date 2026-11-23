"""``refractal run --backend vla-eval``: the loop.

Thin on purpose. Everything it composes is tested without the harness --
``preflight_servers``, ``group_by_seed``, ``check_index_contract``,
``build_eval_config``, ``rows_from_benchmark_result``, ``StepBuffer``,
``verify_written`` -- because the pure parts are where the bugs were. What is left
here is the shape of the nesting and one I/O call.

The nesting, and why it is four deep
------------------------------------

``scene -> worker -> task -> checkpoint -> seed``, one harness invocation per leaf.

* **task**, because the harness's work-item loop can be constrained to one task
  at a time and its episode counter restarts per task. A scene is the compiled
  model and tasks are cheap to vary on it, so one worker holds many.
* **checkpoint**, because a harness run addresses one model server.
* **seed**, because the harness has no seed concept and the counter that would
  have to encode a repeat is the same counter that selects init states.

The task level is what lets a catalog declare one scene with ten tasks -- the
shape the design doc's own test requires, since LIBERO-Spatial's ten tasks
compile to one ``MjModel``. Without it, ``worker_selection`` refuses the worker
and the plan cannot run.

Resume is group-granular here
-----------------------------

The harness builds its own work list from ``episodes_per_task``, so a group cannot
be partially re-run -- ask for five episodes and it runs five, from zero. If any
episode in a group is missing, the whole group runs again.

That makes part-file naming load-bearing. A session-stamped name would leave the
old file beside the new one and put two rows under one ``episode_id``, which
``compare`` blocks on. So part names are deterministic per
``(worker, checkpoint, seed)`` and a re-run replaces the file -- ``fs.mv``
overwrites, verified rather than assumed.
"""

from __future__ import annotations

import datetime as dt
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..schema.plan import Plan, PlannedEpisode, PlannedScene
from ..perturbations import level_arg_of
from ..schema.outcome import chunk_lengths_consistent_with
from ..seeding import episode_seed
from .physics import ABSENT as PHYSICS_ABSENT
from .physics import actuator_facts, physics_digest, physics_manifest
from .harness import describe_installed_harness
from .resources import PeakWatcher
from .results import EPISODES_SCHEMA_VERSION, ResultWriter, read_episodes
from .vla_eval import (
    POLICY_SEEDS_KEY,
    ReceiptBuffer,
    BridgeError,
    EpisodeRow,
    check_server_assignment,
    StepBuffer,
    CallCounter,
    step_facts,
    build_eval_config,
    FrameBuffer,
    check_index_contract,
    check_reset_ordinals,
    check_engine_version,
    check_episode_boot_ids,
    episode_boot_id,
    check_server_revisions,
    server_identity,
    SERVER_IDENTITY_COLUMNS,
    boot_id_warnings,
    group_by_seed,
    make_parquet_orchestrator,
    make_parquet_recorder,
    preflight_servers,
    rows_from_benchmark_result,
)

#: ``(config, recorder_cls) -> benchmark result``. Injectable so the loop can be
#: driven without a harness; the default constructs and awaits the real one.
Invoke = Callable[[Mapping[str, Any], type], Mapping[str, Any]]


@dataclass
class VlaEvalSummary:
    session_id: str
    written: int = 0
    skipped: int = 0
    invocations: int = 0
    parts: list[str] = field(default_factory=list)
    #: (checkpoint, scenario) groups whose replicates span server processes.
    boot_id_warnings: list[str] = field(default_factory=list)


def default_invoke(
    config: Mapping[str, Any], recorder_cls: type, physics: Any = None
) -> Mapping[str, Any]:
    """Construct ParquetOrchestrator, run it, return the one benchmark result.

    ``anyio.run`` rather than ``asyncio.run`` because that is what the harness's
    own CLI uses, and its runners are written against anyio's cancellation
    semantics.
    """
    import anyio  # type: ignore

    config = dict(config)
    seeds = config.pop(POLICY_SEEDS_KEY, None)
    orchestrator_cls = make_parquet_orchestrator(recorder_cls, physics, seeds)
    orchestrator = orchestrator_cls(config)
    results = anyio.run(orchestrator.run)
    if not results:
        raise BridgeError(
            "the harness returned no benchmark results at all. Its config named one "
            f"benchmark ({config.get('benchmarks', [{}])[0].get('benchmark')!r}); check the "
            "server connected and the benchmark imported."
        )
    return results[0]


def _refractal_version() -> str:
    """Imported lazily: this module is the runtime, and a package-level import
    here would be a cycle."""
    from .. import __version__

    return __version__


def _measured_exposure(receipt: list | None) -> float | None:
    """How much of its perturbation the episode actually received.

    The counterpart to `_declared_level`, and deliberately not derived from the
    same place: the level is what the catalog asked for, this is what the
    simulator's own record says happened.

    Only meaningful for a transform, and only for exactly one of them. A state
    mutation perturbs every step it applies to by construction, so the ratio
    would be a constant wearing the costume of a measurement.
    """
    if not receipt:
        return None
    measured = [
        event for event in receipt
        if event.get("applications")
    ]
    if len(measured) != 1 or len(receipt) != 1:
        return None
    event = measured[0]
    return float(event.get("applications_changed") or 0) / float(event["applications"])


def _declared_level(specs: list | None) -> float | None:
    """The single level to group a curve on, or None when there is not one.

    Read from the effect's DECLARATION of which argument is its level, not
    guessed from argument names. Guessing meant an effect calling its argument
    anything but `factor`, `level` or `scale` silently lost the column a curve
    groups by -- a continuous axis becoming categorical by accident.

    Four cases, told apart by `perturbation_count` and the effect's own
    declaration:

        count 0                     -> the baseline. On the curve, at its top.
        count 1, sweepable effect   -> its level.
        count 1, categorical effect -> no level, CORRECTLY. A state source is
                                       correct-or-wrong; an axis with two points
                                       is a comparison, not a curve, and compare
                                       already does matched comparisons.
        count 2+                    -> no single level; off any single axis.

    The third case is why this reads a declaration. Without one, a categorical
    perturbation and a continuous one whose argument went unrecognised are the
    same row, and only the first is correct.
    """
    if not specs or len(specs) != 1:
        return None
    # A declared dose IS the level. It is the axis the curve groups on, and it
    # belongs to the spec rather than to the effect -- any wrapper can carry
    # one, so no effect declares it as its level argument.
    probability = specs[0].get("probability")
    if probability is not None:
        return float(probability)
    key = level_arg_of(specs[0].get("type", ""))
    if key is None:
        return None          # categorical, by the effect's own declaration
    try:
        return float(dict(specs[0].get("args") or {})[key])
    except (KeyError, TypeError, ValueError):
        return None


def _specs_for(outcome: Any) -> list:
    """The specs planned for the episode this outcome describes.

    Straight off the outcome, which carries its `PlannedEpisode`. An earlier
    version looked the episode up by id in the worker's list -- a positional
    correspondence where a direct reference already existed, which is the
    failure mode this file warns about elsewhere and silently returned nothing.
    """
    return list(outcome.episode.perturbations)


class PhysicsSurface:
    """What the engine turned out to be, filled in the first time it is seen.

    Mutable and shared across a worker's invocations, because the benchmark is
    constructed inside the harness and the first episode is the earliest moment
    anything can read its model. Every row written after that carries the real
    digest; rows written before it -- there are none in practice, since the
    capture happens while the first recorder is built -- would carry ABSENT.

    Read once and cached. The model does not change within a worker, and
    reconstructing the reading per episode would cost a model walk on every one.
    """

    __slots__ = ("version", "surface", "manifest")

    def __init__(self) -> None:
        self.version = PHYSICS_ABSENT
        self.surface = PHYSICS_ABSENT
        self.manifest: dict = {}

    @property
    def seen(self) -> bool:
        return self.surface != PHYSICS_ABSENT

    def observe(self, benchmark: Any) -> None:
        """Read the engine's actuators off a live benchmark, once.

        Never raises. A benchmark that hides its model, an engine that is not
        MuJoCo, a version string that cannot be found -- all leave the surface
        ABSENT, which `compare` reads as "nothing looked" rather than as a
        fact. Failing a run because the provenance could not be collected would
        trade a real result for a missing annotation.
        """
        if self.seen:
            return
        try:
            model = benchmark._env.sim.model      # noqa: SLF001 - the bridge's job
            facts = actuator_facts(model)
            if not facts:
                return
            self.surface = physics_digest(facts)
            self.manifest = physics_manifest(facts)
            try:
                import robosuite  # type: ignore

                self.version = str(getattr(robosuite, "__version__", "unknown"))
            except Exception:  # noqa: BLE001
                self.version = "unknown"
        except Exception:  # noqa: BLE001
            return


def _row(
    row: EpisodeRow,
    scene: PlannedScene,
    worker_id: str,
    *,
    session_id: str,
    execution_mode: str,
    harness_version: str,
    harness_surface: str,
    physics_version: str,
    physics_surface: str,
    provider_class: str | None = None,
    receipt: Any = None,
    episode_perturbations: list | None = None,
    server_url: str,
    started_at: dt.datetime,
    ended_at: dt.datetime,
    concurrent_with: str | None,
    peaks: Any = None,
    step_facts: Mapping[str, Any] | None = None,
    step_budget: int | None = None,
    decisions: int | None = None,
    decisions_absent: str | None = None,
    server: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One Parquet row.

    ``started_at`` / ``ended_at`` are the *invocation* window, identical across
    the group, because that is the only timing the harness's aggregate supports.
    Per-episode duration is real and lives in ``elapsed_sec``; inventing
    per-episode wall-clock by accumulating it would read as measured when it was
    derived. The group window is a true statement about a coarser thing.
    """
    episode = row.episode
    return {
        "episode_id": episode.episode_id,
        "scenario_hash": episode.scenario_hash,
        # Falls back to scenario_hash, which is what it equals when nothing
        # is perturbed -- and what a plan written at plan_schema 1 implies,
        # since it has no such field and could carry no perturbation.
        "base_scenario_hash": episode.base_scenario_hash or episode.scenario_hash,
        "scene_id": scene.scene_id,
        "scene_hash": scene.scene_hash,
        "task_id": episode.task_id,
        "task_hash": episode.task_hash,
        "checkpoint_id": episode.checkpoint_id,
        "seed": episode.seed,
        "session_id": session_id,
        "worker_id": worker_id,
        "execution_mode": execution_mode,
        "server_url": server_url,
        "concurrent_with": concurrent_with,
        # The resource receipt. Absent rather than zero when nothing sampled,
        # for the same reason the VRAM figure is null rather than zero when the
        # platform will not attribute it: "nobody looked" and "used none" are
        # different claims, and reading the first as the second is what removed
        # the planner's GPU constraint.
        "peak_rss_mb": getattr(peaks, "rss_mb", None) or None,
        "peak_vram_mb": getattr(peaks, "vram_mb", None),
        "harness_version": harness_version,
        "harness_surface": harness_surface,
        # Written, never left unset: a null here must only ever mean the row
        # predates the column, not that nobody counted.
        # Declared: what was ASKED, from the spec, which the runner knows even
        # when the episode ended before its trigger. Whether it actually fired
        # is the receipt's job, and `compare` decides between them -- an episode
        # that outran its trigger is not a point on its level's curve.
        "provider_class": provider_class,
        "perturbation_count": len(episode_perturbations or ()),
        "perturbation_level": _declared_level(episode_perturbations),
        "perturbation_exposure": _measured_exposure(receipt),
        "perturbations_fired": receipt or None,
        "physics_version": physics_version,
        "physics_surface": physics_surface,
        "success": row.success,
        "phase_outcomes": row.phase_outcomes,
        # RENAMED, and it was being thrown away. This wrote `terminal_phase`,
        # which schema v2 renamed to `termination_reason`; `from_pylist` takes
        # a schema and silently ignores keys outside it, so every vla-eval row
        # recorded a null termination reason while the writer believed it had
        # written one. The same shape as a recorder whose store stays null.
        "termination_reason": row.terminal_phase,
        "failure_reason": row.failure_reason,
        "is_infra_failure": row.is_infra_failure,
        "reset_ordinal": row.reset_ordinal,
        "engine_version": row.engine_version,
        **{column: (server or {}).get(column) for column in SERVER_IDENTITY_COLUMNS},
        "server_boot_id": episode_boot_id(row),
        "steps": row.steps,
        "elapsed_sec": row.elapsed_sec,
        "started_at": started_at,
        "ended_at": ended_at,
        "artifact_uri": None,
        **_distributional(row, step_facts, step_budget, decisions,
                          decisions_absent),
    }


#: Arms whose action-hash log has already been reported on, so the canary
#: prints once per arm rather than once per group.
_CANARY_SEEN: set = set()


def _canary(
    counter: "CallCounter", log: str | None, checkpoint_id: str, outcomes: list,
    declared_chunk: int | None = None, per_step: list | None = None,
) -> None:
    """Say, after the first group, whether the declared log is the one in use.

    The catalog declares where to READ and `vla-eval serve --arg` decides
    where to WRITE, so the two can name different files and nothing in a run
    would notice: `decisions` comes out null with an honest reason, at the end.

    This is purely about timing. It refuses nothing -- null with a stated
    reason is already the honest degradation -- but discovering that the
    server writes elsewhere in the first minute is worth a great deal more
    than discovering it after a thousand episodes. For a probe whose primary
    check IS `decisions`, an unmeasured one leaves it unavailable, and that is
    better known before the GPU time than after.
    """
    if checkpoint_id in _CANARY_SEEN:
        return
    _CANARY_SEEN.add(checkpoint_id)
    if not log:
        print(f"  {checkpoint_id}: no action_hash_log declared, so `decisions` "
              "will be null and the declared chunk length cannot be checked")
        return
    counted = [o for o in outcomes
               if counter.calls(o.episode.episode_id) is not None]
    if counted:
        first = counted[0]
        calls = counter.calls(first.episode.episode_id)
        print(f"  {checkpoint_id}: action-hash log is live -- "
              f"{calls} inference call(s) "
              f"recorded for the first episode, {len(counted)}/{len(outcomes)} "
              "of this group attributed")
        _chunk_canary(checkpoint_id, first, calls, declared_chunk)
        _step_canary(checkpoint_id, outcomes, per_step)
        return
    ambiguous = [o for o in outcomes
                 if o.episode.episode_id in counter.ambiguous()]
    if ambiguous:
        print(f"  {checkpoint_id}: WARNING -- {log} already contains "
              f"{len(ambiguous)} of this group's episodes from an earlier run. "
              "Counts are ambiguous and `decisions` will be null; rotate the "
              "log and re-run to measure the chunk length.")
        return
    print(f"  {checkpoint_id}: WARNING -- {log} records no call for any of this "
          "group's episodes. The catalog says where to read and the serve "
          "command decides where to write; check they name the same file, or "
          "`decisions` stays null for the whole run and the declared chunk "
          "length cannot be checked.")


def _chunk_canary(
    checkpoint_id: str, first: Any, calls: int, declared: int | None
) -> None:
    """Does the server appear to have loaded the config the arm declares?

    The canary above proves the log is live. It says nothing about WHICH
    configuration the server came up on, and that is a separate failure:
    `server_args` is a declaration Refractal never sends -- the chunk size
    comes from the server's own config file at launch -- so a chunk-5 arm can
    come up on a chunk-10 config and nothing notices until the
    declared-versus-realized check runs after the fact.

    One episode is enough, and it is the same arithmetic as that check:
    `decisions == ceil(steps / k)` admits an interval of k per episode, and
    104 steps over 11 calls gives 10..10 while 104 over 21 gives 5..5. It will
    not always pin one value -- 81 steps over 9 calls gives 9..10 -- but it
    always separates 5 from 10, which is the only distinction a replan-interval
    probe needs. Declared 5 against an interval of 9..10 is caught in the
    first minute rather than the third hour.

    A warning rather than a refusal, because the post-hoc check is the one
    with the whole run behind it. This is about when you learn, not whether.
    """
    if declared is None:
        return
    steps = getattr(first, "steps", None)
    if not steps or not calls:
        return
    window = chunk_lengths_consistent_with(steps, calls)
    if window is None:
        return
    lo, hi = window
    span = f"{lo}" if hi == lo else f"{lo}..{hi if hi is not None else 'unbounded'}"
    if lo <= declared and (hi is None or declared <= hi):
        print(f"  {checkpoint_id}: first episode ran {steps} step(s) in {calls} "
              f"inference call(s), consistent with chunk length {span}; "
              f"declares {declared}")
        return
    print(f"  {checkpoint_id}: WARNING -- declares chunk_length={declared}, but "
          f"its first episode ran {steps} step(s) in {calls} inference call(s), "
          f"which is only consistent with {span}. The server was most likely "
          "launched on another config: Refractal never sends `server_args`, so "
          "the chunk size comes from the server's own config file. Stop now "
          "rather than finding this in the post-hoc check.")


def _step_canary(checkpoint_id: str, outcomes: list, per_step: list | None) -> None:
    """Check the newly measured `step_to_success` against `steps`.

    Two routes to one number, one new and one long established. The per-step
    measurement is the least exercised code in this path; `steps` comes from
    the harness's aggregate and has been written for far longer. On a harness
    that ends the episode when the predicate holds they must agree on a
    success row, so agreement validates the new measurement against the old
    field, and disagreement is a finding either way -- it would mean either
    that the per-step attribution is off, or that this harness keeps stepping
    past success, which changes what `steps` means on every row ever written.
    """
    if not per_step:
        return
    agree = disagree = 0
    example = None
    for index, outcome in enumerate(outcomes):
        if index >= len(per_step) or not outcome.success:
            continue
        measured = (per_step[index] or {}).get("step_to_success")
        if measured is None:
            continue
        if measured == outcome.steps:
            agree += 1
        else:
            disagree += 1
            example = example or (measured, outcome.steps)
    if not (agree or disagree):
        return
    if not disagree:
        print(f"  {checkpoint_id}: step_to_success agrees with steps on all "
              f"{agree} success row(s) of the first group -- the per-step "
              "measurement checks out against the harness aggregate")
        return
    print(f"  {checkpoint_id}: NOTE -- step_to_success differs from steps on "
          f"{disagree} of {agree + disagree} success row(s), e.g. {example[0]} "
          f"vs {example[1]}. Either the per-step attribution is off, or this "
          "harness keeps stepping past success -- which would change what "
          "`steps` means on every row written to date.")


def _why_no_count(
    counter: "CallCounter | None", log: str | None, episode_id: str
) -> str | None:
    """Which of the four reasons `decisions` is absent, where it is.

    They are not interchangeable and a reader acts on them differently: a
    missing declaration is a catalog fix, an ambiguous count is a log that
    needs rotating, and an episode the log never mentions means the server was
    launched pointing somewhere else -- which is invisible from here because
    the catalog says where to READ and the serve command says where to WRITE.
    """
    if counter is None:
        return "unmeasured:this run counted no inference calls"
    if not log:
        return ("unmeasured:this arm declares no action_hash_log, and the "
                "harness records per environment step rather than per "
                "inference call")
    if episode_id in counter.ambiguous():
        return ("unmeasured:this episode appears more than once in the action "
                "hash log, so its call count is ambiguous. The server appends, "
                "so a re-run against an un-rotated log doubles every count -- "
                "rotate it and re-read rather than trusting the doubled value")
    if counter.calls(episode_id) is None:
        return (f"unmeasured:{log} records no call for this episode. The "
                "catalog declares where to read and the serve command decides "
                "where to write; check they name the same file")
    return None


def _distributional(
    row: EpisodeRow, facts: Mapping[str, Any] | None, step_budget: int | None,
    decisions: int | None = None, decisions_absent: str | None = None,
) -> dict[str, Any]:
    """The v2 columns, each written only where this backend can answer it.

    Every one of these was null on every vla-eval row, which put the whole
    distributional layer out of reach of the only experiment that needed it:
    no ECDF, no median, no quantile, no restricted mean, and no budget sweep
    without a re-run. The rows were thin while the real data sat in a
    side-channel log.

    `decisions` comes from the model server's own log, because the harness
    records per ENVIRONMENT STEP -- reward, done, success, action, state --
    and never per inference call, so nothing inside a run can count them.
    Measured rather than derived: the declared chunk size would make
    `steps / decisions` a restatement of the declaration, and a check that
    compares a declaration against itself always passes. Null with the reason
    recorded when the server was launched without a log.
    """
    facts = facts or {}
    provenance: dict[str, str] = {}
    first = facts.get("step_to_success")
    samples = facts.get("success_samples") or 0

    if row.success and first is not None:
        step_to_success = first
        provenance["step_to_success"] = "measured:per-step success field"
    elif row.success and samples == 0:
        # Observed success, no per-step record to say when. `steps` is the
        # success step on a harness that stops at success, and that is a
        # property of the backend -- so it is recorded as the derivation it is.
        step_to_success = row.steps
        provenance["step_to_success"] = (
            "derived:steps; this harness ends the episode at success")
    else:
        step_to_success = None
        if row.success:
            provenance["step_to_success"] = (
                "unmeasured:success recorded but no step carried it")

    if facts.get("ended_early") and not row.success:
        # Never seen on LIBERO: all 315 failures in the recorded run sat at
        # exactly the budget. Detected rather than assumed away, and left null
        # because naming it TERMINAL needs a basis, which needs a termination
        # map this backend does not declare.
        outcome = None
        provenance["outcome"] = (
            "unmeasured:ended before budget without success, and this backend "
            "declares no termination map to name the basis")
    elif row.success:
        outcome = "SUCCESS"
        provenance["outcome"] = "derived:success flag"
    elif step_budget is not None and row.steps >= step_budget:
        outcome = "BUDGET_EXHAUSTED"
        provenance["outcome"] = "derived:ran to the declared budget"
    else:
        outcome = None
        provenance["outcome"] = "unmeasured:no budget declared for this group"

    # WHY it is absent, not a single reason covering every way it can be.
    # "this arm declares no action_hash_log" was printed for an episode whose
    # arm declared one and whose count was refused as ambiguous -- a reason
    # that reads as an explanation and names the wrong cause, which is the
    # same defect as a refusal that convincingly blames the wrong thing.
    provenance["decisions"] = (
        "measured:model server action-hash log, one line per inference call"
        if decisions is not None else
        decisions_absent or
        "unmeasured:this harness records per environment step and never per "
        "inference call, and no action_hash_log was read")
    if step_budget is not None:
        provenance["step_budget"] = "declared:the group's step limit"

    return {
        "outcome": outcome,
        "step_to_success": step_to_success,
        "step_budget": step_budget,
        "decisions": decisions,
        "schema_version": EPISODES_SCHEMA_VERSION,
        "field_provenance": sorted(provenance.items()),
    }


def _max_steps(episodes: list[PlannedEpisode]) -> int:
    """The group's step limit, which is one task's, which is one number.

    ``worker_selection`` has already refused a worker spanning two tasks, so this
    cannot disagree -- but it is checked rather than assumed, because the whole
    point of carrying ``max_steps`` in the plan is that the number handed to the
    harness is the one that is inside ``task_hash``.
    """
    limits = {e.max_steps for e in episodes}
    if len(limits) != 1:
        raise BridgeError(
            f"this group spans {len(limits)} step limits ({sorted(limits)}); one harness "
            "invocation takes one max_steps, and a limit that differs from the one in "
            "task_hash would make the recorded identity a lie."
        )
    return limits.pop()


def _run_group(
    group: list[PlannedEpisode],
    *,
    scene: PlannedScene,
    worker_id: str,
    task_id: str,
    checkpoint_id: str,
    seed: int,
    scenarios: Mapping[str, Any],
    servers: Mapping[str, str],
    output_dir: str,
    invoke: "Invoke",
    writer: ResultWriter,
    record_video: bool = False,
    frame_every: int = 10,
    session_id: str,
    execution_mode: str,
    harness_version: str,
    harness_surface: str,
    physics: Any,
    benchmark_override: str | None = None,
    concurrent_with: str | None,
    policy_seeding: str | None = None,
    #: Per-arm, because each arm's server writes its own log. Shared across
    #: groups so a long run reads each log forward rather than from the start
    #: every time, and thread-safe by being per (arm, path) rather than global.
    call_counts: "CallCounter | None" = None,
    action_hash_log: str | None = None,
    declared_chunk: int | None = None,
    #: (lo, hi, writer): rows whose init_state_index is in [lo, hi] go to this writer.
    route: "tuple[int, int, ResultWriter] | None" = None,
) -> tuple[int, str | None]:
    """One harness invocation, start to written rows. Returns (rows, part path).

    Extracted so `serial` and `concurrent` share it exactly. The modes differ in
    *ordering* and nothing else -- if they differed in what an invocation does,
    the mode would be changing the measurement rather than describing it.

    Thread-safe by not sharing anything: each call builds its own recorder class,
    its own buffer and its own config, and `ResultWriter.write_episodes` writes to
    a path that includes the task, checkpoint and seed, so two concurrent calls
    cannot target one file.
    """
    check_index_contract(group, scenarios)
    # A scratch directory per invocation, not per run.
    #
    # The harness writes a progress file at `<output_dir>/<benchmark_name>.tmp`
    # and then `os.replace`s it into place. The name depends only on the
    # benchmark, so two orchestrators sharing an output_dir race on one temp
    # path: the first replace succeeds, the second finds nothing and the run dies
    # with FileNotFoundError on `LIBEROBenchmark_libero_spatial.tmp`.
    #
    # Found by the first real `concurrent` run -- serial shares the directory too
    # and never collides, because it is never in two places at once.
    #
    # THE WORKER ID IS IN THE PATH, and was not at first. Without it the key is
    # (checkpoint, task, seed), which two workers on one scene share whenever the
    # planner splits a scene's scenarios between them -- the default plan for the
    # LIBERO catalog had four workers per scene. A worker cap hid it,
    # and `--backend compose` runs one container per worker, which would have
    # collided at exactly that boundary.
    #
    # Fixed here rather than upstream because output_dir is ours to choose, and
    # a harness scratch directory being single-orchestrator is a reasonable thing
    # for it to assume. Refractal's own results do not go here; they go to the
    # results URI as Parquet.
    scratch = (
        f"{output_dir.rstrip('/')}/{worker_id.replace('/', '-')}"
        f"/{checkpoint_id}/{task_id}/seed{seed}"
    )
    config = build_eval_config(
        scene=scene,
        episodes=group,
        server_url=servers[checkpoint_id],
        output_dir=scratch,
        max_steps=_max_steps(group),
        record_video=record_video,
        benchmark_override=benchmark_override,
    )
    if policy_seeding:
        # Position -> the episode's seed, for `_build_recorder` to put on the
        # task dict the harness sends with EPISODE_START. Position, because the
        # harness only knows its counter; `check_index_contract` above is what
        # makes position i this group's i-th planned episode. A private key,
        # removed by `default_invoke` before the harness sees the config.
        config[POLICY_SEEDS_KEY] = {
            index: {
                "policy_seed": episode_seed(
                    episode.episode_id, policy_seeding,
                    replicate=episode.seed if policy_seeding.endswith("/v2") else None,
                ),
                "policy_seed_scheme": policy_seeding,
                "policy_replicate": episode.seed,
                "refractal_episode_id": episode.episode_id,
            }
            for index, episode in enumerate(group)
        }
    buffer = StepBuffer()
    frames = FrameBuffer(keep_every=frame_every) if record_video else None
    receipts = ReceiptBuffer()
    recorder_cls = make_parquet_recorder(buffer.collect, frames, receipts)

    # Per invocation, not once per run: a server restarted between groups on
    # another environment shows up on the rows it produced.
    server = server_identity(servers[checkpoint_id])
    started_at = dt.datetime.now(dt.timezone.utc)
    result = invoke(config, recorder_cls)
    ended_at = dt.datetime.now(dt.timezone.utc)

    outcomes = rows_from_benchmark_result(result, group)
    check_reset_ordinals(group, outcomes, scenarios,
                         provider=getattr(getattr(scene, "external", None), "provider", None))
    check_engine_version(scene, outcomes)
    check_episode_boot_ids(outcomes)
    if not recorder_cls.constructed:
        raise BridgeError(
            f"the harness ran {len(group)} episode(s) and never constructed the injected "
            "recorder, so `_build_recorder` was not consulted. The run would have "
            "completed, reported success and recorded nothing. Check whether the recorder "
            "gate in orchestrator.py still reads `self._store is None` -- "
            "scripts/verify_harness_claims.py checks exactly this."
        )
    # Read BEFORE the buffer is cleared, and this is the whole reason the
    # order matters: the per-step record is the only place the step a predicate
    # first held is observable, and `clear()` is two lines away from throwing
    # it out. Keyed positionally, like the frames and the receipts.
    per_step = [step_facts(buffer.steps_for(key)) for key in buffer.in_order()]
    buffer.clear()

    # After the invocation, so every call this group made has been written.
    if call_counts is not None:
        call_counts.update(action_hash_log)
        _canary(call_counts, action_hash_log, checkpoint_id, outcomes,
                declared_chunk=declared_chunk, per_step=per_step)

    # Positional, like the frames: the buffer is keyed by the harness's episode
    # counter and the rows by Refractal's content-addressed id. Zipped rather
    # than looked up, which is the same correspondence rows_from_benchmark_result
    # already relies on.
    # Sampled once per harness invocation, at the point the episodes for this
    # group have just finished and the process is still the one that ran them.
    # Not a background thread: a sampler would contend for the single core the
    # worker is pinned to, and the cost of that pinning is what is being
    # measured.
    peaks = PeakWatcher()
    peaks.sample()

    collected = receipts.in_order()
    if collected and len(collected) != len(outcomes):
        raise BridgeError(
            f"{len(collected)} perturbation receipt(s) for {len(outcomes)} "
            "episode(s). The positional match between receipts and rows is not "
            "safe, so nothing is written rather than attaching each receipt to "
            "whichever row happens to line up."
        )
    rows = [
        _row(
            outcome,
            scene,
            worker_id,
            receipt=(collected[index] if collected else None),
            provider_class=config["benchmarks"][0]["benchmark"],
            episode_perturbations=_specs_for(outcome),
            session_id=session_id,
            execution_mode=execution_mode,
            harness_version=harness_version,
            harness_surface=harness_surface,
            physics_version=physics.version,
            physics_surface=physics.surface,
            server_url=servers[checkpoint_id],
            started_at=started_at,
            ended_at=ended_at,
            concurrent_with=concurrent_with,
            peaks=peaks,
            step_facts=(per_step[index] if index < len(per_step) else None),
            step_budget=_max_steps(group),
            decisions=(call_counts.calls(outcome.episode.episode_id)
                       if call_counts is not None else None),
            decisions_absent=_why_no_count(
                call_counts, action_hash_log, outcome.episode.episode_id),
            server=server,
        )
        for index, outcome in enumerate(outcomes)
    ]
    status = _step_log_status()
    for r in rows:
        r["step_log_complete"] = None if status is None else bool(status(r["episode_id"]))
    part = (
        f"{worker_id.replace('/', '-')}"
        f"-{task_id.replace('/', '-')}"
        f"-{checkpoint_id}-seed{seed}"
    )
    label = f"{worker_id}/{task_id}/{checkpoint_id}/seed{seed}"
    targets = [(writer, rows)]
    if route is not None:
        # Split by init state AFTER the group ran whole, in one fresh env: routing
        # changes where a row is written, never which episodes run or in what order.
        lo, hi, routed_writer = route
        index = [(scenarios.get(o.episode.scenario_hash) or {}).get("init_state_index")
                 for o in outcomes]
        inside = [i is not None and lo <= int(i) <= hi for i in index]
        targets = [(writer, [r for r, x in zip(rows, inside) if not x]),
                   (routed_writer, [r for r, x in zip(rows, inside) if x])]
    path = None
    for target, chunk in targets:
        if not chunk:
            continue
        written_path = target.write_episodes(checkpoint_id, scene.scene_id, chunk, part=part)
        target.verify_written(
            {r["episode_id"] for r in chunk}, [written_path] if written_path else [], worker_id=label
        )
        path = path or written_path

    # The receipt for video, at the granularity `verify_written` does not reach.
    #
    # Asking for frames and getting none is silent: the harness renders nothing,
    # the recorder is handed nothing, and the run completes reporting success
    # with an output directory that is merely smaller than expected. On a
    # fifty-minute run that is the whole cost paid before anyone notices.
    #
    # So compare the request against what arrived, and read the strips back
    # rather than trusting a count kept by the code that wrote them.
    if record_video:
        strip_paths = writer.write_strips(frames, [r["episode_id"] for r in rows])
        if not strip_paths:
            raise BridgeError(
                f"{label}: video was requested and the harness produced no frames "
                f"across {len(rows)} episode(s). Either `record_video` is not "
                "reaching the harness spec, or the recorder's `record_video` is "
                "not being called. Nothing downstream would have noticed: the "
                "episodes wrote rows and the run would have reported success."
            )
        kept = writer.verify_frames(strip_paths, worker_id=label)
        if kept == 0:
            raise BridgeError(
                f"{label}: {len(strip_paths)} strip(s) were written and decode to "
                "zero frames."
            )
    if frames is not None:
        frames.clear()
    return len(rows), path


def _step_log_status():
    """The step-log status hook named by $REFRACTAL_STEP_LOG_STATUS (``module:callable``, called with
    an episode id, returning whether its per-step log is complete), or None when unset. Resolved per
    group: the hook lives in the worker process that ran the group's episodes."""
    name = os.environ.get("REFRACTAL_STEP_LOG_STATUS")
    if not name:
        return None
    from ..schema.importstr import resolve_import_string

    return resolve_import_string(name)


def run_vla_eval(
    plan: Plan,
    results_uri: str,
    servers: Mapping[str, str],
    *,
    session_id: str,
    catalog_root: str | None = None,
    output_dir: str = "./vla-eval-output",
    invoke: Invoke | None = None,
    resume: bool = True,
    record_video: bool = False,
    frame_every: int = 10,
    #: The class to construct for a PERTURBED episode. Placement, so it comes
    #: from the run rather than the catalog -- declaring it in a scene's
    #: `external` block would move scene_hash and strand every existing result
    #: as a sweep's baseline, for a class measured inert when unperturbed.
    benchmark_override: str | None = None,
    provenance_plan: Plan | None = None,
    #: (lo, hi, uri): rows of init states lo..hi are written to a second results root
    #: (``refractal run --route-states``), e.g. a sealed confirmation split.
    route_states: "tuple[int, int, str] | None" = None,
) -> VlaEvalSummary:
    """Drive the harness once per (worker, checkpoint, seed) and write Parquet."""
    # Derived from the episodes, not from ``plan.checkpoints``. The episodes are
    # what will be run, and they are the set that needs an address; reading the
    # declared list would pass vacuously on a plan whose list is empty -- which is
    # how this was first written, and how the test for it passed while checking
    # nothing.
    wanted = sorted(
        {e.checkpoint_id for s in plan.scenes for w in s.workers for e in w.episodes}
    )
    missing = [c for c in wanted if c not in servers]
    if missing:
        raise BridgeError(
            f"no server URL for checkpoint(s) {missing}. Every checkpoint the plan runs "
            f"needs one: --server <checkpoint>=<url>. Given: {sorted(servers)}."
        )
    # Hoisted out of `preflight_servers` deliberately. Two checkpoints pointed at one
    # server is a self-comparison -- a difference near zero with a tight interval
    # and every internal check passing -- and noticing it needs no network, so it
    # must not wait behind a connect that might itself fail for another reason.
    check_server_assignment(servers)
    servers = preflight_servers(servers)
    check_server_revisions(plan.checkpoints, servers)

    # Per arm: each server writes its own log, and a shared counter across
    # groups is what lets a long run read each one forward rather than from
    # the start every time. Declared on the checkpoint because the server is
    # launched outside Refractal, so the path cannot be discovered.
    #
    # Read from the PLAN's checkpoints rather than the catalog's, so a run
    # filtered to one worker still finds its own arm's log.
    hash_logs = {
        c.id: getattr(c, "action_hash_log", None)
        for c in (getattr(plan, "checkpoints", None) or [])
    }
    call_counts: dict[str, CallCounter] = {}
    declared_chunks = {
        c.id: getattr(c, "chunk_length", None)
        for c in (getattr(plan, "checkpoints", None) or [])
    }

    if route_states is not None and record_video:
        raise BridgeError(
            "--route-states with --video: frame strips are matched to rows by position "
            "within a group, and a routed group's rows go to two roots. Run without video."
        )
    writer = ResultWriter(results_uri, plan.plan_id)
    routed = None
    if route_states is not None:
        lo, hi, routed_uri = route_states
        routed = (int(lo), int(hi), ResultWriter(routed_uri, plan.plan_id))
    writers = [writer] + ([routed[2]] if routed else [])
    # The WHOLE plan, not this worker's slice of it.
    #
    # Under `--backend compose` every container runs `--worker`, and `--worker`
    # is a filter: same plan_id, same episode ids, fewer workers. Each container
    # then writes its own restriction to the same path and the last one wins, so
    # the results directory ends up carrying a plan that says 60 episodes across
    # 1 worker while its plan_id digests 600 across 10.
    #
    # An identity document that contradicts its own digest is worse than none.
    for w in writers:
        w.write_plan((provenance_plan or plan).to_json())
    # What was supplied at run time, beside what was planned. Server addresses
    # and the backend shaped this run and appeared in no artifact, so two runs
    # against different servers produced rows nobody could tell apart.
    #
    # Recorded, not hashed: an address is a property of a machine, and folding
    # it into plan_id would make one experiment on two machines two
    # experiments.
    render = {
        "backend": "vla-eval",
        "servers": dict(sorted(servers.items())),
        "catalog_root": catalog_root,
        "results_uri": results_uri,
        "benchmark_override": benchmark_override,
        "record_video": bool(record_video),
        "frame_every": frame_every,
        "refractal_version": _refractal_version(),
    }
    if routed:
        # Both roots say where the other half of every group went.
        render["route_states"] = {"states": [routed[0], routed[1]],
                                  "routed_uri": route_states[2], "unrouted_uri": results_uri}
    for w in writers:
        w.write_render_manifest(render)
    if catalog_root is not None:
        for w in writers:
            w.copy_catalog(catalog_root)

    harness_version, harness_surface, manifest = describe_installed_harness()
    for w in writers:
        w.record_harness_manifest(harness_surface, manifest)
    # Filled in by the bridge the first time it sees a live benchmark, because
    # what matters is the model that actually ran rather than what the catalog
    # claimed. Until then it reads absent, which is honest: nothing has looked.
    physics = PhysicsSurface()

    already = set().union(*(w.completed_episode_ids() for w in writers)) if resume else set()
    # Bound here rather than passed through `Invoke`, which is the seam a test
    # substitutes: a fake invoke should not have to know about provenance
    # collection to be a valid stand-in for the real one.
    invoke = invoke or functools.partial(default_invoke, physics=physics)
    summary = VlaEvalSummary(session_id=session_id)

    for scene in plan.scenes:
        scenarios = {s.scenario_hash: s.params for s in scene.scenarios}
        for worker in scene.workers:
            for task_id in sorted({e.task_id for e in worker.episodes}):
                for_task = [e for e in worker.episodes if e.task_id == task_id]
                checkpoints = sorted({e.checkpoint_id for e in for_task})

                def job(checkpoint_id: str, seed: int, group: list[PlannedEpisode],
                        alongside: list[str]):
                    return _run_group(
                        group,
                        scene=scene,
                        worker_id=worker.worker_id,
                        task_id=task_id,
                        checkpoint_id=checkpoint_id,
                        seed=seed,
                        scenarios=scenarios,
                        servers=servers,
                        output_dir=output_dir,
                        invoke=invoke,
                        writer=writer,
                        record_video=record_video,
                        frame_every=frame_every,
                        session_id=session_id,
                        execution_mode=plan.execution_mode,
                        harness_version=harness_version,
                        harness_surface=harness_surface,
                        physics=physics,
                        benchmark_override=benchmark_override,
                        concurrent_with=",".join(sorted(alongside)) or None,
                        policy_seeding=getattr(plan, "policy_seeding", None),
                        call_counts=call_counts.setdefault(
                            checkpoint_id, CallCounter()),
                        action_hash_log=hash_logs.get(checkpoint_id),
                        declared_chunk=declared_chunks.get(checkpoint_id),
                        route=routed,
                    )

                if plan.execution_mode == "concurrent":
                    # Both checkpoints at once, each against its own server. One
                    # thread per checkpoint: `invoke` is synchronous and the real
                    # one calls anyio.run, which needs its own thread rather than
                    # a shared event loop.
                    #
                    # Ordered seed-outer so the checkpoints contend with each other
                    # rather than with a different seed of themselves.
                    for seed in sorted({e.seed for e in for_task}):
                        batch = []
                        for checkpoint_id in checkpoints:
                            group = [
                                e for e in for_task
                                if e.checkpoint_id == checkpoint_id and e.seed == seed
                            ]
                            if not group or all(e.episode_id in already for e in group):
                                summary.skipped += len(group)
                                continue
                            batch.append((checkpoint_id, group))
                        if not batch:
                            continue
                        alongside = [c for c, _ in batch]
                        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                            futures = [
                                pool.submit(job, c, seed, g,
                                            [o for o in alongside if o != c])
                                for c, g in batch
                            ]
                            for future in futures:
                                written, path = future.result()
                                summary.written += written
                                if path:
                                    summary.parts.append(path)
                                summary.invocations += 1
                else:
                    # serial: all seeds for one checkpoint, then the next. The
                    # ordering is the mode, so this stays checkpoint-outer --
                    # reordering it to alternate would be interleaved execution
                    # recorded as serial, which is the mislabelling this whole
                    # definition exists to remove.
                    for checkpoint_id in checkpoints:
                        for_checkpoint = [
                            e for e in for_task if e.checkpoint_id == checkpoint_id
                        ]
                        for seed, group in sorted(group_by_seed(for_checkpoint).items()):
                            if all(e.episode_id in already for e in group):
                                summary.skipped += len(group)
                                continue
                            written, path = job(checkpoint_id, seed, group, [])
                            summary.written += written
                            if path:
                                summary.parts.append(path)
                            summary.invocations += 1

    # After the run, because the engine is only readable once a benchmark has
    # been constructed -- which happens inside the harness, during the first
    # episode. Nothing to record if no engine was ever seen.
    for w in writers:
        w.record_physics_manifest(physics.surface, physics.manifest)
    boot_rows = []
    for uri in [results_uri] + ([route_states[2]] if routed else []):
        try:
            boot_rows += read_episodes(uri, plan.plan_id).select(
                ["checkpoint_id", "scenario_hash", "server_boot_id"]).to_pylist()
        except Exception:  # noqa: BLE001 -- a root this worker wrote nothing to
            if uri == results_uri and not routed:
                raise
    summary.boot_id_warnings = boot_id_warnings(boot_rows)
    if summary.boot_id_warnings:
        print("\n  " + "!" * 72)
        print("  WARNING: replicates of one state did not all run on one server process.")
        print("  A seed pins the noise, not the numerics, so these replicates differ by")
        print("  more than their seed:")
        for line in summary.boot_id_warnings:
            print(f"    {line}")
        print("  " + "!" * 72 + "\n")
    return summary


__all__ = ["Invoke", "VlaEvalSummary", "default_invoke", "run_vla_eval"]
