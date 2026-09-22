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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..schema.plan import Plan, PlannedEpisode, PlannedScene
from ..perturbations import level_arg_of
from .physics import ABSENT as PHYSICS_ABSENT
from .physics import actuator_facts, physics_digest, physics_manifest
from .harness import describe_installed_harness
from .results import ResultWriter
from .vla_eval import (
    ReceiptBuffer,
    BridgeError,
    EpisodeRow,
    check_server_assignment,
    StepBuffer,
    build_eval_config,
    FrameBuffer,
    check_index_contract,
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


def default_invoke(
    config: Mapping[str, Any], recorder_cls: type, physics: Any = None
) -> Mapping[str, Any]:
    """Construct ParquetOrchestrator, run it, return the one benchmark result.

    ``anyio.run`` rather than ``asyncio.run`` because that is what the harness's
    own CLI uses, and its runners are written against anyio's cancellation
    semantics.
    """
    import anyio  # type: ignore

    orchestrator_cls = make_parquet_orchestrator(recorder_cls, physics)
    orchestrator = orchestrator_cls(dict(config))
    results = anyio.run(orchestrator.run)
    if not results:
        raise BridgeError(
            "the harness returned no benchmark results at all. Its config named one "
            f"benchmark ({config.get('benchmarks', [{}])[0].get('benchmark')!r}); check the "
            "server connected and the benchmark imported."
        )
    return results[0]


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
    benchmark_class: str | None = None,
    receipt: Any = None,
    episode_perturbations: list | None = None,
    server_url: str,
    started_at: dt.datetime,
    ended_at: dt.datetime,
    concurrent_with: str | None,
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
        "harness_version": harness_version,
        "harness_surface": harness_surface,
        # Written, never left unset: a null here must only ever mean the row
        # predates the column, not that nobody counted.
        # Declared: what was ASKED, from the spec, which the runner knows even
        # when the episode ended before its trigger. Whether it actually fired
        # is the receipt's job, and `compare` decides between them -- an episode
        # that outran its trigger is not a point on its level's curve.
        "benchmark_class": benchmark_class,
        "perturbation_count": len(episode_perturbations or ()),
        "perturbation_level": _declared_level(episode_perturbations),
        "perturbations_fired": receipt or None,
        "physics_version": physics_version,
        "physics_surface": physics_surface,
        "success": row.success,
        "phase_outcomes": row.phase_outcomes,
        "terminal_phase": row.terminal_phase,
        "failure_reason": row.failure_reason,
        "is_infra_failure": row.is_infra_failure,
        "steps": row.steps,
        "elapsed_sec": row.elapsed_sec,
        "started_at": started_at,
        "ended_at": ended_at,
        "artifact_uri": None,
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
    buffer = StepBuffer()
    frames = FrameBuffer(keep_every=frame_every) if record_video else None
    receipts = ReceiptBuffer()
    recorder_cls = make_parquet_recorder(buffer.collect, frames, receipts)

    started_at = dt.datetime.now(dt.timezone.utc)
    result = invoke(config, recorder_cls)
    ended_at = dt.datetime.now(dt.timezone.utc)

    outcomes = rows_from_benchmark_result(result, group)
    if not recorder_cls.constructed:
        raise BridgeError(
            f"the harness ran {len(group)} episode(s) and never constructed the injected "
            "recorder, so `_build_recorder` was not consulted. The run would have "
            "completed, reported success and recorded nothing. Check whether the recorder "
            "gate in orchestrator.py still reads `self._store is None` -- "
            "scripts/verify_harness_claims.py checks exactly this."
        )
    buffer.clear()

    # Positional, like the frames: the buffer is keyed by the harness's episode
    # counter and the rows by Refractal's content-addressed id. Zipped rather
    # than looked up, which is the same correspondence rows_from_benchmark_result
    # already relies on.
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
            benchmark_class=config["benchmarks"][0]["benchmark"],
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
        )
        for index, outcome in enumerate(outcomes)
    ]
    part = (
        f"{worker_id.replace('/', '-')}"
        f"-{task_id.replace('/', '-')}"
        f"-{checkpoint_id}-seed{seed}"
    )
    path = writer.write_episodes(checkpoint_id, scene.scene_id, rows, part=part)
    label = f"{worker_id}/{task_id}/{checkpoint_id}/seed{seed}"
    writer.verify_written(
        {r["episode_id"] for r in rows}, [path] if path else [], worker_id=label
    )

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

    writer = ResultWriter(results_uri, plan.plan_id)
    # The WHOLE plan, not this worker's slice of it.
    #
    # Under `--backend compose` every container runs `--worker`, and `--worker`
    # is a filter: same plan_id, same episode ids, fewer workers. Each container
    # then writes its own restriction to the same path and the last one wins, so
    # the results directory ends up carrying a plan that says 60 episodes across
    # 1 worker while its plan_id digests 600 across 10.
    #
    # An identity document that contradicts its own digest is worse than none.
    writer.write_plan((provenance_plan or plan).to_json())
    if catalog_root is not None:
        writer.copy_catalog(catalog_root)

    harness_version, harness_surface, manifest = describe_installed_harness()
    writer.record_harness_manifest(harness_surface, manifest)
    # Filled in by the bridge the first time it sees a live benchmark, because
    # what matters is the model that actually ran rather than what the catalog
    # claimed. Until then it reads absent, which is honest: nothing has looked.
    physics = PhysicsSurface()

    already = writer.completed_episode_ids() if resume else set()
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
    writer.record_physics_manifest(physics.surface, physics.manifest)
    return summary


__all__ = ["Invoke", "VlaEvalSummary", "default_invoke", "run_vla_eval"]
