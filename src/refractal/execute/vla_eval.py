"""Driving `allenai/vla-evaluation-harness` without forking it.

Two subclasses and no patches. The harness is a pinned dependency; everything
Refractal needs is reachable by overriding a method.

The two-method problem
----------------------

``Orchestrator._build_recorder`` is the obvious hook, and overriding it alone
produces a run that completes, reports success and writes nothing:

.. code-block:: python

    if self._store is None or rec_cfg is None:
        return NullEpisodeRecorder()

``self._store`` is a SQLite ``RecordingStore``, built from a filesystem path. A
Parquet backend has no SQLite path to give it, so the gate stays closed and the
override never runs. Both have to move, which is why this class overrides
``_build_recorder`` *and* the store setup.

``ResultWriter.verify_written`` is the belt to this braces: it asserts that a
worker which ran episodes actually put rows on disk, checked by reading the part
files back. Getting this override wrong now fails on the first worker rather
than at compare time.

What is NOT overridden, and why
-------------------------------

The setup before the harness's episode loop does three things, all of which are
the silent-wrongness kind, and none of which are reimplemented here:

* ``apply_render_mode`` runs **before** the benchmark class is constructed,
  because the renderer binds at the first simulator import. An ordering
  constraint, not a movable call.
* ``_merge_observation_params`` negotiates observation parameters between server
  and benchmark. This is the class of bug their own paper measures at 55
  percentage points.
* 33 lines of action/observation spec cross-validation, which is what catches an
  absolute-versus-delta action space -- the mismatch that produces 0%.

So Refractal does not hand-roll a driver. It subclasses, and expresses its
per-worker episode assignment in the harness's own terms (``tasks`` plus
``episodes_per_task``) rather than fighting the work-item loop.

**Documented cost:** resume is task-granular here, not episode-granular. The
harness builds its own work list inline, so a restart re-runs a whole task.
Nothing corrupts -- the writer dedupes on ``episode_id`` and
``verify_written`` would catch it if it did not -- but compute is wasted. That is
a smaller price than reimplementing the three things above against a benchmark we
did not write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..schema.errors import RefractalError
from ..schema.plan import PlannedEpisode


class BridgeError(RefractalError):
    """The harness is not present, or is not shaped the way the bridge expects."""


#: Surface members `EpisodeRecorder` exposes that a replacement must provide.
#: Seven, not six: `is_active` is easy to miss from the prose description and the
#: runner calls it.
RECORDER_SURFACE = (
    "record_step",
    "record_video",
    "close",
    "is_active",
    "sid",
    "eid",
    "eval_id",
    "db_path",
)


@dataclass
class EpisodeRow:
    """One episode's worth of recorded facts, in Refractal's terms.

    Deliberately not the harness's ``EpisodeResult``: this is what a
    ``ResultWriter`` consumes, and keeping the shapes separate means their type
    changing is a compile-time problem here rather than a schema problem in
    Parquet.
    """

    episode: PlannedEpisode
    success: bool
    steps: int
    elapsed_sec: float
    failure_reason: str | None
    is_infra_failure: bool
    phase_outcomes: list[tuple[str, bool]] = field(default_factory=list)
    terminal_phase: str | None = None


def require_harness() -> Any:
    """Import the harness, or explain what is missing rather than raising ImportError."""
    try:
        import vla_eval  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BridgeError(
            "the vla-eval bridge needs the harness installed: "
            "pip install 'refractal[vla-eval]', or run inside the benchmark image. "
            f"({exc})"
        ) from exc
    return vla_eval


def check_recorder_surface(recorder_cls: type) -> None:
    """Refuse a recorder that does not implement everything the runner calls.

    Checked explicitly because the harness's ``EpisodeRecorder`` is a plain class
    rather than an ABC -- there is no abstract-method error to rely on, and a
    missing member surfaces as an ``AttributeError`` several hundred episodes into
    a run instead of before the first one.
    """
    missing = [name for name in RECORDER_SURFACE if not hasattr(recorder_cls, name)]
    if missing:
        raise BridgeError(
            f"{recorder_cls.__name__} is missing {missing} from the recorder surface. "
            "EpisodeRecorder is a plain class, not an ABC, so nothing else would have "
            "told you until a runner called one of them mid-episode."
        )


def worker_selection(episodes: list[PlannedEpisode]) -> dict[str, Any]:
    """Express a worker's episode assignment in the harness's own terms.

    The harness builds its work list inline from ``tasks`` x
    ``episodes_per_task`` and shards it round-robin; that construction is not a
    method and cannot be overridden without copying the loop around it. So rather
    than fighting it, a Refractal worker is constrained to one task's episodes and
    described the way the harness already understands.

    Returns the config fragment, and raises if the assignment cannot be expressed
    -- which is the honest failure. Silently running a superset would corrupt the
    denominator; silently running a subset would lose episodes the plan promised.
    """
    if not episodes:
        return {"tasks": [], "episodes_per_task": 0}

    task_ids = {e.task_id for e in episodes}
    if len(task_ids) != 1:
        raise BridgeError(
            f"this worker is assigned {len(task_ids)} tasks ({sorted(task_ids)}), and the "
            "harness's work-item loop can only be constrained to one at a time. Plan with "
            "one task per worker for this backend, or the assignment cannot be expressed "
            "without re-running episodes that belong to another worker."
        )
    return {"tasks": sorted(task_ids), "episodes_per_task": len(episodes)}


class StepBuffer:
    """Accumulates recorded step fields per episode, in memory, in order.

    Nothing is written from inside an episode. Buffering here keeps the
    temp-then-rename guarantee in one place -- a recorder that wrote as it went
    would be producing partial files that a concurrent resume could read as
    complete.
    """

    def __init__(self) -> None:
        self._fields: dict[str, list[tuple[str, Any]]] = {}

    def collect(self, episode_id: str, name: str, value: Any) -> None:
        self._fields.setdefault(episode_id, []).append((name, value))

    def steps_for(self, episode_id: str) -> list[tuple[str, Any]]:
        return self._fields.get(episode_id, [])

    def discard(self, episode_id: str) -> None:
        self._fields.pop(episode_id, None)

    def __len__(self) -> int:
        return len(self._fields)


def to_episode_row(episode: PlannedEpisode, result: Mapping[str, Any]) -> EpisodeRow:
    """Map one harness ``EpisodeResult`` onto Refractal's episode row.

    **This function is where their denominator decision gets undone**, so it is
    worth being explicit about rather than burying in a dict comprehension.

    ``_build_task_result`` counts errored episodes as policy failures, with
    ``len(episodes)`` as the denominator -- documented and deliberate upstream,
    and wrong for a comparison. A crashed container is not evidence about a
    policy.

    So the harness's ``failure_reason`` becomes ``is_infra_failure``: it is set
    exactly when the episode did not complete for a reason outside the policy --
    ``server_unreachable``, a wedged container, a timeout. The distinction is kept
    as its own column so ``compare`` can drop those from a denominator without
    pattern-matching a string.

    And a genuine policy failure names itself. The reference says a null
    ``failure_reason`` means a real failure, but null is also what a success
    writes, so the two would be indistinguishable; here null means success and
    nothing else.
    """
    metrics = result.get("metrics") or {}
    infra_reason = result.get("failure_reason") or None
    success = bool(metrics.get("success", False)) and infra_reason is None

    if success:
        failure_reason = None
    elif infra_reason is not None:
        failure_reason = str(infra_reason)
    else:
        failure_reason = "policy_failure"

    phases = metrics.get("phase_outcomes") or {}
    return EpisodeRow(
        episode=episode,
        success=success,
        steps=int(result.get("steps") or 0),
        elapsed_sec=float(result.get("elapsed_sec") or 0.0),
        failure_reason=failure_reason,
        is_infra_failure=infra_reason is not None,
        phase_outcomes=[(str(k), bool(v)) for k, v in phases.items()],
        terminal_phase=metrics.get("terminal_phase"),
    )


def check_index_contract(episodes: list[PlannedEpisode], scenarios: Mapping[str, Any]) -> None:
    """The harness indexes init states by its OWN episode counter. Refuse a
    plan where that counter would not select the scenario the plan names.

    This is the sharpest thing in the bridge and it is easy to miss. The harness
    runs ``ep in range(cfg.episodes_per_task)`` and hands ``episode_idx = ep`` to
    the benchmark, which does ``initial_states[episode_idx]``. Refractal's
    scenario says which init state it means, in ``init_state_index``.

    Those are two different numbers that happen to coincide when a catalog uses
    ``range: [0, N-1]``. If it uses ``[5, 14]``, the harness still counts 0..9 and
    runs init states 0..9 while every recorded row claims 5..14 — a comparison
    built on scenarios that were never executed, with nothing anywhere
    disagreeing.

    So it is checked rather than assumed, and it fails loudly rather than
    quietly running the wrong thing. The alternative -- teaching the harness to
    take an explicit index -- means the work-item loop, which is inline and not a
    method.
    """
    wanted = sorted(
        scenarios[e.scenario_hash]["init_state_index"]
        for e in episodes
        if e.scenario_hash in scenarios
    )
    if len(wanted) != len(episodes):
        raise BridgeError(
            "some episodes reference scenarios that are not in the plan's scenario list, "
            "so their init-state index cannot be checked against the harness's counter."
        )
    expected = list(range(len(wanted)))
    if wanted != expected:
        raise BridgeError(
            f"this worker's init_state_index values are {wanted[:6]}"
            f"{'...' if len(wanted) > 6 else ''}, but the harness selects init states by its "
            f"own episode counter, which will run {expected[:6]}"
            f"{'...' if len(expected) > 6 else ''}. Every row would claim a scenario that was "
            "never executed. Use `range: [0, N-1]` for init_state_index, or run a subset by "
            "narrowing the tier rather than by offsetting the index."
        )


def build_eval_config(
    *,
    scene: Any,
    episodes: list[PlannedEpisode],
    server_url: str,
    output_dir: str,
    max_steps: int,
) -> dict[str, Any]:
    """The vla-eval config for one worker against one checkpoint.

    Expressed in the harness's own terms rather than by overriding its work-item
    loop, which is inline and not a method. ``worker_selection`` refuses a worker
    spanning two tasks, because the loop can only be constrained to one.
    """
    selection = worker_selection(episodes)
    external = getattr(scene, "external", None)
    if external is None:
        raise BridgeError(
            f"scene {scene.scene_id if hasattr(scene, 'scene_id') else scene!r} is not an "
            "externally-defined scene, so there is no provider to name as the benchmark."
        )
    return {
        "server": {"url": server_url},
        "output_dir": output_dir,
        "benchmarks": [
            {
                "benchmark": external.provider,
                "subname": dict(external.ref).get("suite"),
                "episodes_per_task": selection["episodes_per_task"],
                "tasks": selection["tasks"],
                "max_steps": max_steps,
                "params": dict(external.ref),
                # Recording on: the gate is `rec_cfg is None or self._store is
                # None`, and ParquetOrchestrator moves the second. Leaving this
                # unset would close the first and record nothing.
                "recording": {"record_step": True, "record_video": False},
            }
        ],
    }


def rows_from_benchmark_result(
    result: Mapping[str, Any], episodes: list[PlannedEpisode]
) -> list[EpisodeRow]:
    """Map the harness's aggregate back onto the plan's episodes, positionally.

    Positional because that is the only correspondence available: the harness
    stamps ``episode_id`` with its own integer counter, which is the index this
    bridge just checked against ``init_state_index``. The check is what makes the
    position meaningful; without it this would be guessing.
    """
    by_index: dict[int, Mapping[str, Any]] = {}
    for task in result.get("tasks") or []:
        for episode in task.get("episodes") or []:
            index = episode.get("episode_id")
            if isinstance(index, int):
                by_index[index] = episode

    missing = [i for i in range(len(episodes)) if i not in by_index]
    if missing:
        raise BridgeError(
            f"the harness returned {len(by_index)} episode result(s) for "
            f"{len(episodes)} planned episode(s); {len(missing)} have no result "
            f"(first index {missing[0]}). A plan that asked for work it did not get back "
            "must not be written as though it completed."
        )
    return [to_episode_row(episode, by_index[i]) for i, episode in enumerate(episodes)]


def make_parquet_recorder(collect) -> type:
    """Build the recorder subclass, lazily.

    Lazily because the base class lives in the harness, and this module has to
    import on a machine that does not have it -- `refractal.execute` is importable
    wherever `refractal` is. Subclassing at call time keeps the import boundary
    where the dependency rule puts it.

    ``collect`` receives ``(episode_id, field_name, value)`` for every recorded
    step field, so nothing is written to disk from inside an episode. Buffering
    and writing are the caller's job, which keeps the atomic temp-then-rename
    guarantee in one place.
    """
    vla_eval = require_harness()
    from vla_eval.recording import EpisodeRecorder  # type: ignore

    class ParquetEpisodeRecorder(EpisodeRecorder):  # type: ignore[misc]
        """Matches the recorder surface; writes nothing itself."""

        def __init__(self, episode_id: str, sid: str, eid: str, eval_id: str) -> None:
            # Deliberately does not call super().__init__: the base sets up a
            # SQLite store this recorder has no use for.
            self._episode_id = episode_id
            self._sid, self._eid, self._eval_id = sid, eid, eval_id

        @property
        def is_active(self) -> bool:
            return True

        @property
        def sid(self) -> str:
            return self._sid

        @property
        def eid(self) -> str:
            return self._eid

        @property
        def eval_id(self) -> str:
            return self._eval_id

        @property
        def db_path(self) -> str:
            # Empty on purpose. The harness forwards this to model servers in
            # EPISODE_START so they *could* open their own StepRecorder against
            # the same SQLite, and across every shipped model server there is no
            # caller. An affordance, not a feature -- and "a path to one SQLite
            # file" does not generalise to a URI plus a partition key anyway.
            return ""

        def record_step(self, **fields: Any) -> None:
            for name, value in fields.items():
                collect(self._episode_id, name, value)

        def record_video(self, frame: Any) -> None:
            return None

        def close(self, *args: Any, **kwargs: Any) -> None:
            return None

    check_recorder_surface(ParquetEpisodeRecorder)
    return ParquetEpisodeRecorder


def make_parquet_orchestrator(recorder_cls) -> type:
    """Build the Orchestrator subclass, overriding **both** required methods.

    ``_build_recorder`` is the obvious one. The other is whatever sets
    ``self._store``: the base returns ``NullEpisodeRecorder`` whenever the store
    is ``None``, so overriding the recorder alone yields a run that completes,
    reports success and records nothing.

    A sentinel is used rather than a real ``RecordingStore`` because the gate
    only tests for ``None``. Constructing a SQLite store to satisfy a check and
    then never writing to it would leave a stray file that looks like results.
    """
    vla_eval = require_harness()
    from vla_eval.orchestrator import Orchestrator  # type: ignore

    class ParquetOrchestrator(Orchestrator):  # type: ignore[misc]
        #: Truthy, unused, and not a SQLite handle. Exists only to open the gate.
        _PARQUET_STORE = object()

        def _init_store(self, *args: Any, **kwargs: Any) -> None:
            self._store = self._PARQUET_STORE

        def _build_recorder(
            self, rec_cfg, task, bench_eval_id, benchmark_safe_name, task_idx, episode_id, benchmark
        ):
            if rec_cfg is None:
                from vla_eval.recording import NullEpisodeRecorder  # type: ignore

                return NullEpisodeRecorder()
            return recorder_cls(
                episode_id=str(episode_id),
                sid=str(getattr(self, "_sid", "")),
                eid=f"{bench_eval_id}-{task_idx}-{episode_id}",
                eval_id=bench_eval_id,
            )

    return ParquetOrchestrator


__all__ = [
    "RECORDER_SURFACE",
    "BridgeError",
    "EpisodeRow",
    "StepBuffer",
    "to_episode_row",
    "build_eval_config",
    "check_index_contract",
    "check_recorder_surface",
    "rows_from_benchmark_result",
    "make_parquet_orchestrator",
    "make_parquet_recorder",
    "require_harness",
    "worker_selection",
]
