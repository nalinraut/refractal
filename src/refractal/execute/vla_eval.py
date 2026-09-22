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

    **"The harness's own terms" means its task names, not ours.** The work-item
    loop filters like this::

        tasks = benchmark.get_tasks()
        if cfg.tasks:
            tasks = [t for t in tasks if t.get("suite") in cfg.tasks
                                      or t.get("name") in cfg.tasks]

    and LIBERO's ``get_tasks`` sets ``name = task.language``. So the selector is
    the instruction string. Sending ``task_id`` -- which this returned first --
    matches nothing, leaves ``tasks`` empty, and runs zero episodes. That fails
    loudly, because ``rows_from_benchmark_result`` refuses a short result, but it
    fails after the servers are warm rather than before anything is spent.

    Filtering by name also gets the right *index*: after the filter there is one
    task, so the harness's ``episode_idx`` counts ``0..episodes_per_task-1``
    against that task's own init states -- which is the contract
    ``check_index_contract`` checks.

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
    instructions = {e.instruction for e in episodes}
    if len(instructions) != 1:
        # One task id with two instructions cannot happen through `resolve`, but
        # it can through a hand-written plan -- and it would select two tasks
        # from one worker, which is the thing the check above exists to prevent.
        raise BridgeError(
            f"task {sorted(task_ids)[0]!r} carries {len(instructions)} different "
            f"instructions ({sorted(instructions)}). The harness selects tasks by their "
            "instruction, so this would select more than one."
        )
    return {
        "tasks": sorted(instructions),
        "episodes_per_task": len(episodes),
        "task_ids": sorted(task_ids),
    }


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

    def clear(self) -> None:
        """Drop everything buffered.

        Until ``steps.parquet`` lands, the runner calls this after each harness
        invocation: the buffer's only job today is to prove the recorder override
        took effect, and holding every step of every episode for the whole run to
        prove that would be a memory leak wearing a receipt.
        """
        self._fields.clear()

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


def check_server_assignment(servers: Mapping[str, str]) -> None:
    """Distinct checkpoints must answer on distinct URLs.

    Point two checkpoints at one port and you get a comparison of a checkpoint
    against itself: plausible rates, a difference near zero, a tight interval and
    a verdict of "no change". Nothing downstream would object, because every
    internal check passes — the rows are well-formed, the pairing is exact, the
    denominators are right. The only thing wrong is which policy produced them,
    which is not recorded anywhere the analysis can see.

    Same shape as the index contract: both sides correct, the disagreement living
    only in the relationship, and the result fully formed and wrong.
    """
    if not servers:
        raise BridgeError("no server URLs were given; every checkpoint needs one.")
    by_url: dict[str, list[str]] = {}
    for checkpoint, url in servers.items():
        by_url.setdefault(url, []).append(checkpoint)
    collisions = {url: names for url, names in by_url.items() if len(names) > 1}
    if collisions:
        detail = "; ".join(f"{url} -> {sorted(names)}" for url, names in collisions.items())
        raise BridgeError(
            f"these checkpoints share a server URL: {detail}. They would be compared "
            "against themselves, and the result would look like a clean null: a "
            "difference near zero with a tight interval and every internal check passing."
        )


def preflight_servers(servers: Mapping[str, str], *, timeout: float = 5.0) -> dict[str, str]:
    """Check every declared server before the first episode, not on first use.

    A run that gets forty episodes in and dies because the second server was never
    started has wasted the compute and left a partial comparison behind. One
    connect attempt per server turns a confusing mid-run failure into a stated
    precondition.

    **What this checks and what it does not.** A TCP connect proves something is
    listening on that port. It does not prove it is a model server, or the right
    checkpoint — that is what the harness's HELLO handshake and spec
    cross-validation are for, and they happen per episode. This is the cheap half,
    and its job is to fail before anything is spent.
    """
    import socket
    from urllib.parse import urlparse

    check_server_assignment(servers)
    unreachable: dict[str, str] = {}
    for checkpoint, url in sorted(servers.items()):
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except OSError as exc:
            unreachable[checkpoint] = f"{url} ({exc.__class__.__name__}: {exc})"
    if unreachable:
        lines = "\n  ".join(f"{c}: {d}" for c, d in sorted(unreachable.items()))
        raise BridgeError(
            "these model servers are not accepting connections:\n  "
            + lines
            + "\n\nStart them before the run rather than letting the run start them. "
            "A supervised server loads during the first episode, so two interleaved "
            "checkpoints would carry different warm-up costs -- model load, CUDA "
            "context, first-inference JIT -- inside the thing being measured. Servers "
            "up and warm first is the measurement protocol, not a concession to "
            "architecture."
        )
    return dict(servers)


def group_by_seed(episodes: list[PlannedEpisode]) -> dict[int, list[PlannedEpisode]]:
    """One harness invocation per seed, because the harness has no seed concept.

    The counter that selects init states is the same counter that would have to
    encode a repeat. With 10 scenarios and 3 seeds in one invocation it runs
    0..29 and reaches init states 0..29 — thirty *different* scenarios, while the
    plan claims ten scenarios run three times each. LIBERO ships 50 init states,
    so it does not even error.

    So seeds come from the loop, not from the counter: one invocation per
    ``(worker, checkpoint, seed)``, each with ``episodes_per_task`` equal to the
    scenario count, and the counter maps 1:1 onto ``init_state_index``.

    **What a seed means on this backend, stated because it is not what the design
    doc means.** Refractal defines a seed as the policy's noise draw, pinned so a
    failure is reproducible. The harness exposes no per-episode policy seeding —
    that lives inside each model server and is server-specific. So here a seed is
    a **repeat index**: it labels repetitions and does not make them reproducible.

    Repetition still does the statistical work, separating "is this scenario
    hard" from "did the policy get lucky", which is what the clustered bootstrap
    consumes. It does not give deterministic replay. Recording it as though it
    did would be the mislabelling this project exists to catch.
    """
    groups: dict[int, list[PlannedEpisode]] = {}
    for episode in episodes:
        groups.setdefault(episode.seed, []).append(episode)
    return groups


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
    seeds = {e.seed for e in episodes}
    if len(seeds) > 1:
        # Name the cause. The index list would otherwise look like the problem,
        # when it is a symptom of running several seeds in one invocation.
        raise BridgeError(
            f"this group spans {len(seeds)} seeds ({sorted(seeds)}). The harness selects "
            "init states by a counter that knows nothing about seeds, so a single "
            "invocation covering N scenarios x S seeds would run N*S *different* init "
            "states rather than repeating N of them. Split by seed first — "
            "group_by_seed() — and run one invocation per seed."
        )

    # A scenario that names no index is not selecting from a pinned array, and
    # this contract does not apply to it. An MJX scene varies `cube_x` and
    # `cube_y`; there is no counter to disagree with, and demanding the key
    # raised KeyError on every such plan -- including at render time, which is
    # meant to be a pure function of the plan and answerable anywhere.
    #
    # Checked across the whole group rather than per episode: a group where some
    # scenarios carry an index and some do not is a catalog error worth naming,
    # not something to skip past.
    # An episode naming a scenario the plan does not define is a different and
    # worse failure than one naming a scenario without an index. Checked first,
    # because `.get(...) or {}` would quietly turn the first into the second --
    # which it did, and a test that existed for the unknown-scenario case caught
    # it.
    unknown = [e for e in episodes if e.scenario_hash not in scenarios]
    if unknown:
        raise BridgeError(
            f"{len(unknown)} episode(s) name a scenario this scene does not define, "
            f"for example {unknown[0].scenario_hash}. The index contract cannot be "
            "checked against a scenario that is not in the plan."
        )

    indexed = [e for e in episodes if "init_state_index" in scenarios[e.scenario_hash]]
    if not indexed:
        return
    if len(indexed) != len(episodes):
        raise BridgeError(
            f"{len(indexed)} of {len(episodes)} episode(s) in this group name an "
            "`init_state_index` and the rest do not. Either every scenario on a scene "
            "selects from the provider's pinned array or none does; a mixture means "
            "the ones without an index would run whichever state the counter reached."
        )

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


def perturbation_map(episodes: list[PlannedEpisode]) -> dict[str, Any]:
    """Episode position -> what to do to it, and the id to seed its RNG from.

    Position rather than ``episode_id`` because the benchmark only knows the
    harness's counter. The real id rides along as a value so the perturbation
    RNG is seeded from the episode's identity -- the one seed in this system
    that is a genuine reproducibility guarantee, and useless if it is seeded
    from a position that means something different in the next run.
    """
    return {
        str(index): {
            "episode_id": episode.episode_id,
            "specs": [dict(spec) for spec in episode.perturbations],
        }
        for index, episode in enumerate(episodes)
        if episode.perturbations
    }


def build_eval_config(
    *,
    scene: Any,
    episodes: list[PlannedEpisode],
    server_url: str,
    output_dir: str,
    max_steps: int,
    record_video: bool = False,
    benchmark_override: str | None = None,
) -> dict[str, Any]:
    """The vla-eval config for one worker against one checkpoint.

    Expressed in the harness's own terms rather than by overriding its work-item
    loop, which is inline and not a method. ``worker_selection`` refuses a worker
    spanning two tasks, because the loop can only be constrained to one.
    """
    if len({e.seed for e in episodes}) > 1:
        raise BridgeError(
            "build_eval_config takes one seed's episodes; split with group_by_seed() "
            "and run one invocation per seed. See check_index_contract for why."
        )
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
                # The class the harness constructs. Overridden, never derived:
                # running a perturbed episode needs a benchmark that can run a
                # timeline, and WHICH class does that is placement -- measured,
                # not asserted, by a claim that the perturbed LIBERO subclass
                # produces bit-identical trajectories when nothing is specified.
                #
                # So it cannot live in the scene's `external` block, which is
                # hashed: declaring it there would move scene_hash and strand
                # every existing result as a sweep's baseline. It comes from the
                # run instead, and the row records what actually ran.
                "benchmark": (
                    benchmark_override
                    if benchmark_override and any(e.perturbations for e in episodes)
                    else external.provider
                ),
                "subname": dict(external.ref).get("suite"),
                "episodes_per_task": selection["episodes_per_task"],
                "tasks": selection["tasks"],
                "max_steps": max_steps,
                # `params`, not `ref`. The ref identifies the scene -- LIBERO's
                # `task_id` lives there -- and the harness does
                # `benchmark_cls(**params)`, where an identifying key that is not
                # a constructor argument raises TypeError.
                "params": {
                    **dict(getattr(external, "params", None) or {}),
                    # The spec channel. The harness does `benchmark_cls(**params)`,
                    # so this is how a perturbation reaches a benchmark the
                    # harness constructs -- and it is keyed by POSITION, because
                    # the harness identifies an episode by its index within a
                    # task and Refractal identifies it by content. The two never
                    # match, so the correspondence is the same positional one
                    # every other channel here relies on.
                    #
                    # Omitted entirely when nothing is perturbed, so an
                    # unperturbed run hands the benchmark exactly what it always
                    # did and a benchmark with no such parameter still
                    # constructs.
                    **(
                        {"perturbations": perturbation_map(episodes)}
                        if any(e.perturbations for e in episodes)
                        else {}
                    ),
                },
                # Recording on: the gate is `rec_cfg is None or self._store is
                # None`, and ParquetOrchestrator moves the second. Leaving this
                # unset would close the first and record nothing.
                # `record_video` is what makes the harness render and forward
                # frames at all. With it False the recorder's `record_video` is
                # never called, so turning the recorder into a frame sink
                # without also turning this on changes nothing.
                "recording": {"record_step": True, "record_video": record_video},
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


class ReceiptBuffer:
    """One perturbation receipt per episode, in the order episodes were built.

    Positional, for exactly the reason ``FrameBuffer`` is: the recorder is
    constructed with the harness's own episode counter, while a row carries the
    content-addressed id. Looking one up by the other finds nothing, silently --
    which is how the frame version of this bug was found, by a receipt reporting
    zero frames on a run that had captured them.

    Registered at construction rather than at first use, so an episode that dies
    before it can hand anything over still occupies its place in the order. Its
    receipt is ``None``: nothing was handed over, which is not the same as a
    benchmark that perturbs nothing and hands over an empty list.
    """

    def __init__(self) -> None:
        self._receipts: dict[str, Any] = {}
        self._order: list[str] = []

    def begin(self, harness_episode_id: str) -> None:
        key = str(harness_episode_id)
        if key not in self._receipts:
            self._order.append(key)
            self._receipts[key] = None

    def collect(self, harness_episode_id: str, receipt: Any) -> None:
        self._receipts[str(harness_episode_id)] = receipt

    def in_order(self) -> list[Any]:
        return [self._receipts[key] for key in self._order]

    def clear(self) -> None:
        self._receipts.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._order)


class FrameBuffer:
    """Every `keep_every`-th frame of each episode, in capture order.

    Subsampled at capture rather than afterwards: a 220-step episode at 224px
    is ~33 MB of raw RGB held per worker, and only every tenth frame is ever
    written. Keeping all of them costs memory for frames that are discarded.

    KEYED BY THE HARNESS'S EPISODE ID, which is not Refractal's. The recorder is
    constructed with whatever the harness calls the episode -- its own counter --
    while a row carries the content-addressed `episode_id` that identifies the
    episode across runs. The two never match, and looking one up by the other
    silently finds nothing, which is how this was found: the receipt reported
    zero frames on a run that had captured them.

    So the correspondence here is positional, and it is the same positional
    correspondence `rows_from_benchmark_result` already relies on: episodes are
    constructed, recorded and reported in one order. `in_order` exposes that
    order so a caller can zip it against rows rather than guess at a key.
    """

    def __init__(self, keep_every: int = 10) -> None:
        self.keep_every = keep_every
        self._frames: dict[str, list[Any]] = {}
        self._seen: dict[str, int] = {}

    def begin(self, harness_episode_id: str) -> None:
        """Register an episode when its recorder is built, not at its first frame.

        An episode that dies before producing one still happened and still has a
        row. Registering on the first frame instead would leave it out of the
        order entirely, and the positional match against rows would then be off
        by one for every episode after it.
        """
        self._seen.setdefault(harness_episode_id, 0)

    def collect(self, harness_episode_id: str, frame: Any) -> None:
        n = self._seen.get(harness_episode_id, 0)
        self._seen[harness_episode_id] = n + 1
        # A benchmark that cannot produce a frame passes None. Dropping it here
        # keeps the buffer encodable, and an episode that yields only None ends
        # up with no frames, which the receipt then reports.
        if n % self.keep_every == 0 and frame is not None:
            self._frames.setdefault(harness_episode_id, []).append(frame)

    def in_order(self) -> list[list[Any]]:
        """Frame lists, one per episode, in the order the episodes ran."""
        return [self._frames.get(k, []) for k in self._seen]

    @property
    def episodes_seen(self) -> int:
        return len(self._seen)

    @property
    def episodes_with_frames(self) -> int:
        return sum(1 for v in self._frames.values() if v)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self._frames.values())

    def clear(self) -> None:
        self._frames.clear()
        self._seen.clear()


def make_parquet_recorder(collect, frames=None, receipts=None) -> type:
    """Build the recorder subclass, lazily.

    Lazily because the base class lives in the harness, and this module has to
    import on a machine that does not have it -- `refractal.execute` is importable
    wherever `refractal` is. Subclassing at call time keeps the import boundary
    where the dependency rule puts it.

    ``collect`` receives ``(episode_id, field_name, value)`` for every recorded
    step field, so nothing is written to disk from inside an episode. Buffering
    and writing are the caller's job, which keeps the atomic temp-then-rename
    guarantee in one place.

    The returned class counts its own instantiations in ``constructed``. That is
    the receipt for the injection working -- and it has to be separate from
    whether anything was *collected*, because an episode that dies before its
    first step records nothing while the injection was fine. The first real run
    against two pi0 servers did exactly that: every episode timed out during
    torch.compile warm-up, the step buffer was empty, and a check that read the
    buffer reported the recorder had never been consulted. It had been consulted
    twice.
    """
    vla_eval = require_harness()
    from vla_eval.recording import EpisodeRecorder  # type: ignore

    class ParquetEpisodeRecorder(EpisodeRecorder):  # type: ignore[misc]
        """Matches the recorder surface; writes nothing itself."""

        #: How many times the harness asked for one. Zero means `_build_recorder`
        #: was never consulted, which is the failure the override exists to avoid.
        constructed = 0

        def __init__(self, episode_id: str, sid: str, eid: str, eval_id: str) -> None:
            type(self).constructed += 1
            # Deliberately does not call super().__init__: the base sets up a
            # SQLite store this recorder has no use for.
            self._episode_id = episode_id
            self._sid, self._eid, self._eval_id = sid, eid, eval_id
            if frames is not None:
                frames.begin(episode_id)
            if receipts is not None:
                # Registered at construction, like the frame buffer, so the
                # positional correspondence to rows holds even for an episode
                # that dies before recording anything. Registering on first use
                # would silently shift every later episode's receipt onto the
                # wrong row.
                receipts.begin(episode_id)

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

        def record_perturbations(self, receipt: Any) -> None:
            """Take the episode's perturbation receipt from the benchmark.

            NOT part of `RECORDER_SURFACE`, which is what the harness expects a
            recorder to have. This is Refractal's own addition, called only by a
            benchmark that perturbs -- so it travels the same benchmark ->
            recorder -> bridge channel the frames do, rather than needing a
            second route back out of the harness.

            A benchmark that never perturbs never calls it, and that is not the
            same as calling it with nothing: an absent receipt and an empty one
            are different claims, and the buffer keeps them apart.
            """
            if receipts is not None:
                receipts.collect(self._episode_id, receipt)

        def record_step(self, **fields: Any) -> None:
            for name, value in fields.items():
                collect(self._episode_id, name, value)

        def record_video(self, frame: Any) -> None:
            # Was a no-op. A recorder that silently discards what it is handed
            # is the same shape as a store that silently stays null: the run
            # completes, reports success, and the frames are gone.
            if frames is not None:
                frames.collect(self._episode_id, frame)

        def close(self, *args: Any, **kwargs: Any) -> None:
            return None

    check_recorder_surface(ParquetEpisodeRecorder)
    return ParquetEpisodeRecorder


class NullRecordingStore:
    """Truthy, no-op, and not a SQLite handle.

    The gate is ``self._store is None``, so something has to be there. A real
    ``RecordingStore`` would create a SQLite file that gets written to and then
    ignored -- a file that looks like results and is not.

    It is not a bare sentinel either, which is what this was first. The harness
    calls ``upsert_eval_metadata`` on the store before the episode loop and
    ``close`` in the ``finally``, so ``object()`` would have raised on the first
    benchmark. Found by reading orchestrator.py rather than by running anything:
    the stand-in tests passed with the bare sentinel, because the stand-in did
    not call those methods.
    """

    def upsert_eval_metadata(self, *args: Any, **kwargs: Any) -> None: ...
    def upsert_episode_result(self, *args: Any, **kwargs: Any) -> None: ...
    def upsert_step_rows(self, *args: Any, **kwargs: Any) -> None: ...
    def close(self) -> None: ...


def make_parquet_orchestrator(recorder_cls, physics=None) -> type:
    """Build the Orchestrator subclass. The store override is not a method.

    ``_build_recorder`` is the obvious hook. The other gate is ``self._store``,
    and finding where it is set turned out to matter more than expected:

    .. code-block:: python

        async def run(self):
            if not self.no_save:
                self._store = RecordingStore(db_path_for_eval(...))

    It is assigned **inside ``run()``**, not in a method there is anything to
    override. An earlier version of this class overrode ``_init_store``, which
    does not exist in the harness at all -- it existed only in the test
    stand-in, where it had been invented. The stand-in passed throughout.

    Worse, the two gates are coupled. ``no_save=True`` makes
    ``_effective_recording_config`` return ``None``, which shuts the *other*
    gate; ``no_save=False`` makes ``run()`` build a real SQLite store. There is
    no flag combination that opens both.

    So ``_store`` becomes a property: reads return the null store, and writes are
    swallowed, including ``run()``'s assignment and the ``finally`` block's
    ``self._store = None``. Intrusive, and the alternative is reimplementing
    ``run()`` -- which is the method that also does render-mode setup,
    observation-param negotiation and spec cross-validation, i.e. the three
    things this bridge exists in order not to reimplement.
    """
    vla_eval = require_harness()
    from vla_eval.orchestrator import Orchestrator  # type: ignore

    class ParquetOrchestrator(Orchestrator):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._parquet_store = NullRecordingStore()
            super().__init__(*args, **kwargs)

        @property
        def _store(self) -> Any:
            return self._parquet_store

        @_store.setter
        def _store(self, value: Any) -> None:
            # Swallowed on purpose: `run()` assigns a RecordingStore here and the
            # `finally` assigns None. Neither is wanted, and neither is
            # preventable without reimplementing `run()`.
            return None

        def _build_recorder(
            self, rec_cfg, task, bench_eval_id, benchmark_safe_name, task_idx, episode_id, benchmark
        ):
            # The earliest moment anything here can see a constructed benchmark,
            # and therefore its engine's actuators. Read once per worker; the
            # model does not change underneath a running worker.
            if physics is not None:
                physics.observe(benchmark)
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
    "NullRecordingStore",
    "ReceiptBuffer",
    "StepBuffer",
    "to_episode_row",
    "build_eval_config",
    "check_index_contract",
    "check_server_assignment",
    "preflight_servers",
    "group_by_seed",
    "check_recorder_surface",
    "rows_from_benchmark_result",
    "make_parquet_orchestrator",
    "make_parquet_recorder",
    "require_harness",
    "worker_selection",
]
