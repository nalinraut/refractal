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
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..schema.plan import Plan, PlannedEpisode, PlannedScene
from .harness import describe_installed_harness
from .results import ResultWriter
from .vla_eval import (
    BridgeError,
    EpisodeRow,
    check_server_assignment,
    StepBuffer,
    build_eval_config,
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


def default_invoke(config: Mapping[str, Any], recorder_cls: type) -> Mapping[str, Any]:
    """Construct ParquetOrchestrator, run it, return the one benchmark result.

    ``anyio.run`` rather than ``asyncio.run`` because that is what the harness's
    own CLI uses, and its runners are written against anyio's cancellation
    semantics.
    """
    import anyio  # type: ignore

    orchestrator_cls = make_parquet_orchestrator(recorder_cls)
    orchestrator = orchestrator_cls(dict(config))
    results = anyio.run(orchestrator.run)
    if not results:
        raise BridgeError(
            "the harness returned no benchmark results at all. Its config named one "
            f"benchmark ({config.get('benchmarks', [{}])[0].get('benchmark')!r}); check the "
            "server connected and the benchmark imported."
        )
    return results[0]


def _row(
    row: EpisodeRow,
    scene: PlannedScene,
    worker_id: str,
    *,
    session_id: str,
    execution_mode: str,
    harness_version: str,
    harness_surface: str,
    server_url: str,
    started_at: dt.datetime,
    ended_at: dt.datetime,
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
        "harness_version": harness_version,
        "harness_surface": harness_surface,
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
    # Hoisted out of `preflight_servers` deliberately. Two arms pointed at one
    # server is a self-comparison -- a difference near zero with a tight interval
    # and every internal check passing -- and noticing it needs no network, so it
    # must not wait behind a connect that might itself fail for another reason.
    check_server_assignment(servers)
    servers = preflight_servers(servers)

    writer = ResultWriter(results_uri, plan.plan_id)
    writer.write_plan(plan.to_json())
    if catalog_root is not None:
        writer.copy_catalog(catalog_root)

    harness_version, harness_surface, manifest = describe_installed_harness()
    writer.record_harness_manifest(harness_surface, manifest)

    already = writer.completed_episode_ids() if resume else set()
    invoke = invoke or default_invoke
    summary = VlaEvalSummary(session_id=session_id)

    for scene in plan.scenes:
        scenarios = {s.scenario_hash: s.params for s in scene.scenarios}
        for worker in scene.workers:
            for task_id in sorted({e.task_id for e in worker.episodes}):
                for_task = [e for e in worker.episodes if e.task_id == task_id]
                for checkpoint_id in sorted({e.checkpoint_id for e in for_task}):
                    for_checkpoint = [
                        e for e in for_task if e.checkpoint_id == checkpoint_id
                    ]
                    for seed, group in sorted(group_by_seed(for_checkpoint).items()):
                        if all(e.episode_id in already for e in group):
                            summary.skipped += len(group)
                            continue

                        # Whole group or nothing: the harness counts from zero, so a
                        # partially-done group has to be re-run in full and its part
                        # file replaced rather than added to.
                        check_index_contract(group, scenarios)
                        config = build_eval_config(
                            scene=scene,
                            episodes=group,
                            server_url=servers[checkpoint_id],
                            output_dir=output_dir,
                            max_steps=_max_steps(group),
                        )
                        # steps.parquet is deferred until after the first real run,
                        # so nothing drains this buffer. The receipt for the injection
                        # is `recorder_cls.constructed`, not the buffer -- see below.
                        buffer = StepBuffer()
                        recorder_cls = make_parquet_recorder(buffer.collect)

                        started_at = dt.datetime.now(dt.timezone.utc)
                        result = invoke(config, recorder_cls)
                        ended_at = dt.datetime.now(dt.timezone.utc)

                        # Raises if the harness returned fewer results than the plan
                        # asked for, before anything is written. A short run must not
                        # be recorded as though it completed.
                        outcomes = rows_from_benchmark_result(result, group)
                        if not recorder_cls.constructed:
                            raise BridgeError(
                                f"the harness ran {len(group)} episode(s) and never constructed "
                                "the injected recorder, so `_build_recorder` was not consulted. "
                                "The run would have completed, reported success and recorded "
                                "nothing. Check whether the recorder gate in orchestrator.py "
                                "still reads `self._store is None` -- "
                                "scripts/verify_harness_claims.py checks exactly this."
                            )
                        buffer.clear()
                        rows = [
                            _row(
                                outcome,
                                scene,
                                worker.worker_id,
                                session_id=session_id,
                                execution_mode=plan.execution_mode,
                                harness_version=harness_version,
                                harness_surface=harness_surface,
                                server_url=servers[checkpoint_id],
                                started_at=started_at,
                                ended_at=ended_at,
                            )
                            for outcome in outcomes
                        ]
                        # The task is in the name because a worker now holds
                        # many. Without it, ten tasks would write to one path and
                        # each would replace the last -- 60 groups producing 6
                        # files, and `verify_written` would not see it, because it
                        # checks the file it just wrote.
                        part = (
                            f"{worker.worker_id.replace('/', '-')}"
                            f"-{task_id.replace('/', '-')}"
                            f"-{checkpoint_id}-seed{seed}"
                        )
                        path = writer.write_episodes(
                            checkpoint_id, scene.scene_id, rows, part=part
                        )
                        written_paths = [path] if path else []
                        writer.verify_written(
                            {r["episode_id"] for r in rows},
                            written_paths,
                            worker_id=f"{worker.worker_id}/{checkpoint_id}/seed{seed}",
                        )
                        summary.parts.extend(written_paths)
                        summary.written += len(rows)
                        summary.invocations += 1

    return summary


__all__ = ["Invoke", "VlaEvalSummary", "default_invoke", "run_vla_eval"]
