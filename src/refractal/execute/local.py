"""The ``local`` backend: runs a plan in-process against a fake benchmark.

No Docker, no simulator, no model server. It exercises the real writer, the real
schema, the real partition layout and the real resume path -- everything except
the physics.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..schema.plan import Plan, PlannedEpisode, PlannedScene, PlannedWorker
from .fake import FakeBenchmark
from .harness import LOCAL, describe_installed_harness
from .results import ResultWriter

#: Written to every row. Under the `local` backend nothing talks to the harness,
#: and saying so is more honest than recording a version that never ran.
LOCAL_HARNESS_VERSION = LOCAL


@dataclass
class RunSummary:
    session_id: str
    written: int
    skipped: int
    parts: list[str]


def _row(
    episode: PlannedEpisode,
    scene: PlannedScene,
    worker: PlannedWorker,
    outcome: dict,
    *,
    session_id: str,
    execution_mode: str,
    started_at: dt.datetime,
    harness_version: str,
    harness_surface: str,
) -> dict:
    ended_at = started_at + dt.timedelta(seconds=outcome["elapsed_sec"])
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
        "worker_id": worker.worker_id,
        "execution_mode": execution_mode,
        "server_url": None,  # nothing was served; the local backend simulates
        "concurrent_with": None,  # one process, nothing to contend with
        "harness_version": harness_version,
        "harness_surface": harness_surface,
        "success": outcome["success"],
        "phase_outcomes": outcome["phase_outcomes"],
        "terminal_phase": outcome["terminal_phase"],
        "failure_reason": outcome["failure_reason"],
        "is_infra_failure": outcome["is_infra_failure"],
        "steps": outcome["steps"],
        "elapsed_sec": outcome["elapsed_sec"],
        "started_at": started_at,
        "ended_at": ended_at,
        "artifact_uri": None,
    }


def run_local(
    plan: Plan,
    results_uri: str,
    *,
    catalog_root: str | None = None,
    benchmark: FakeBenchmark | None = None,
    session_id: str,
    started_at: dt.datetime | None = None,
    resume: bool = True,
) -> RunSummary:
    """Execute a plan in-process.

    ``session_id`` and ``started_at`` are injected rather than generated. A
    backend that reaches for ``uuid4()`` and the wall clock cannot be tested for
    reproducibility, and the resume path is precisely the thing that has to be
    proven rather than assumed.
    """
    benchmark = benchmark or FakeBenchmark()
    started_at = started_at or dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    writer = ResultWriter(results_uri, plan.plan_id)

    # Always: a results directory that cannot say what produced it is not
    # provenance, it is a pile of Parquet.
    writer.write_plan(plan.to_json())
    if catalog_root is not None:
        writer.copy_catalog(catalog_root)

    already = writer.completed_episode_ids() if resume else set()
    written = skipped = 0
    parts: list[str] = []

    for scene in plan.scenes:
        # Partitioned by checkpoint, so one worker's episodes fan out into one
        # part file per checkpoint it touched.
        for worker in scene.workers:
            by_checkpoint: dict[str, list[dict]] = {}
            cursor = started_at
            for episode in worker.episodes:
                if episode.episode_id in already:
                    skipped += 1
                    continue
                outcome = benchmark.run(episode)
                by_checkpoint.setdefault(episode.checkpoint_id, []).append(
                    _row(
                        episode,
                        scene,
                        worker,
                        outcome,
                        session_id=session_id,
                        execution_mode=plan.execution_mode,
                        started_at=cursor,
                        harness_version=LOCAL_HARNESS_VERSION,
                        harness_surface=LOCAL,
                    )
                )
                cursor += dt.timedelta(seconds=outcome["elapsed_sec"])
                written += 1

            attempted: set[str] = set()
            written_paths: list[str] = []
            for checkpoint_id, rows in sorted(by_checkpoint.items()):
                part = f"{worker.worker_id.replace('/', '-')}-{session_id[:8]}"
                path = writer.write_episodes(checkpoint_id, scene.scene_id, rows, part=part)
                attempted.update(r["episode_id"] for r in rows)
                if path:
                    written_paths.append(path)
                    parts.append(path)

            # Post-condition against the artifact, scoped to this worker's own
            # files so the comparison can be exact in both directions.
            writer.verify_written(attempted, written_paths, worker_id=worker.worker_id)

    return RunSummary(session_id=session_id, written=written, skipped=skipped, parts=parts)


__all__ = ["LOCAL_HARNESS_VERSION", "RunSummary", "run_local"]
