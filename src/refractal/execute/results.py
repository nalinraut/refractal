"""Writing results: Parquet, at a URI, atomically.

Three rules, each one load-bearing.

**Everything is a URI, through fsspec, from the first commit.** ``./results`` and
``s3://bucket/results`` are the same code path. The moment any code calls
``open("results.db")`` a single-host system has been built, and unbuilding one
later means touching every write site.

**Every write is atomic: temp name, then rename.** Rename is atomic on POSIX and
on object stores, so a half-written file never looks complete. This is the whole
mechanism behind resume -- if a partial file could be mistaken for a finished
one, "list what exists and subtract" would silently drop episodes.

**The schema is declared, not inferred.** Parquet files written from Python
dicts get their types guessed per file, so a column that happens to be all-null
in one worker's output lands as null-typed and refuses to merge with another
worker's. Declaring it once means every part file is readable as one dataset.

DEVIATION from the API reference, forced by atomicity: ``episodes.parquet`` is a
*directory* of part files, not a single file. You cannot have both incremental
atomic writes and one file -- rewriting a single file on each flush is neither
atomic across concurrent workers nor better than quadratic. Part files are how
every columnar store handles this, and Hive-style partitioning means DuckDB and
Polars read the directory as one table regardless.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..schema.errors import RefractalError

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

#: One row per episode. Order is the read order; keep identity columns first so
#: a human running `head` sees what the row *is* before what happened to it.
#: Every ``read_table`` here passes ``use_threads=False``, and it is not a
#: performance choice.
#:
#: pyarrow's threaded reader over a Python file object -- which is what an fsspec
#: handle is -- leaves a thread pool unjoined at interpreter shutdown, and CPython
#: exits with SIGABRT and ``terminate called without an active exception``. The
#: rows are written correctly and the process still dies: `refractal run` exited
#: 134 on the happy path, after printing its success line.
#:
#: Reproduced outside Refractal in twelve lines -- write twelve Parquet files
#: through ``fs.open``, read them back the same way -- so it is pyarrow and fsspec
#: rather than anything here. Threading buys nothing at part-file sizes, which are
#: tens to hundreds of rows.
READ_THREADS = False

EPISODES_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        # The comparison join key is (scene_id, task_hash, scenario_hash).
        # `task_hash` is here rather than only `task_id` because a task id is a
        # label someone chose: renaming it must not break a join, and changing
        # its meaning must.
        pa.field("scenario_hash", pa.string(), nullable=False),
        # The same scenario with nothing perturbed, and the axis a perturbation
        # sweep joins on: hold this fixed, vary the level, get a curve. Equal to
        # `scenario_hash` for an unperturbed episode, which makes every row
        # written before this column existed a valid member of that join.
        #
        # Identity, not provenance -- but identity for a different question than
        # `scenario_hash` answers. `episode_id` derives from `scenario_hash`, so
        # two episodes differing only in torque scale are different episodes;
        # this is what lets `compare` still see them as the same base.
        pa.field("base_scenario_hash", pa.string(), nullable=False),
        # The receipt: one struct per perturbation, fired or not.
        #
        # A list of structs rather than parallel columns, because two
        # perturbations in one episode would otherwise need parallel arrays to
        # say which value belongs to which step -- and parallel arrays drift.
        # Each event's facts stay together.
        #
        # Not a map of strings either. Every query would parse them, and "20.0"
        # sorts and compares differently from 20.0; stringly-typed columns are
        # how a groupable value silently stops grouping.
        #
        # `before` and `after` are lists because they are not always scalars:
        # scale_actuator reads one torque limit, apply_force reads a six-element
        # wrench out of xfrc_applied. One column holds both at length 1 and 6,
        # rather than a column per effect.
        #
        # Unfired specs live here too, with a null `fired_step` and a `reason`.
        # One column then answers "did everything fire", instead of a join
        # between what was asked and what happened. A trigger at step 200 in an
        # episode that ended at 150 legitimately did not fire -- absence alone
        # is not the error -- but a sweep where nothing fired anywhere is, the
        # same way a filter that rejects every scenario is.
        pa.field(
            "perturbations_fired",
            pa.list_(
                pa.struct(
                    [
                        pa.field("effect", pa.string(), nullable=False),
                        pa.field("target", pa.string(), nullable=False),
                        pa.field("specified_step", pa.int32(), nullable=False),
                        #: Null when it never came due.
                        pa.field("fired_step", pa.int32()),
                        #: Why not, when `fired_step` is null.
                        pa.field("reason", pa.string()),
                        #: A MUTATION's evidence: the values either side of it.
                        pa.field("before", pa.list_(pa.float64())),
                        pa.field("after", pa.list_(pa.float64())),
                        #: A WRAPPER's evidence, which is a different KIND
                        #: rather than a different shape. There is no numeric
                        #: "before" for an observation -- it is an image or a
                        #: state vector -- so the pair is digests, and they get
                        #: their own fields instead of being coerced into the
                        #: float list. Coercion is not an option rather than a
                        #: style choice: a hex digest through `float()` raises,
                        #: which is how this was found.
                        pa.field("before_digest", pa.string()),
                        pa.field("after_digest", pa.string()),
                        #: How many steps the wrapper applied on, and on how
                        #: many of those the value actually CHANGED.
                        #:
                        #: Two counts rather than one, because a wrapper can be
                        #: legitimately identity on some steps. See
                        #: `check_receipts` for why the difference is the whole
                        #: signal.
                        pa.field("applications", pa.int32()),
                        pa.field("applications_changed", pa.int32()),
                    ]
                )
            ),
        ),
        # The grouping key, kept OUT of the receipt on purpose.
        #
        # The receipt records what happened; `compare` groups by what was asked.
        # A torque-margin curve should group on a plain number rather than
        # reaching into an audit struct -- the receipt is for checking and the
        # level is for analysis, and conflating them makes both worse.
        #
        # Null when an episode has no perturbation, and also when it has more
        # than one: there is then no single level to group on, and silently
        # picking one would be worse than saying so. A sweep over two
        # simultaneous levels needs a key this column cannot express, which is
        # an open question rather than something to fudge here.
        pa.field("perturbation_level", pa.float64()),
        # What the episode actually RECEIVED, as against what it was assigned.
        #
        # The fraction of applications on which the perturbation changed the
        # observation. `perturbation_level` is read from the catalog before the
        # run; this is read from the receipt after it, and they answer different
        # questions.
        #
        # It exists because a transform's effect can be trajectory-dependent in
        # a way a state mutation's is not. Two rotation conventions agree
        # exactly over half of all orientations, so an episode is perturbed only
        # on the steps its trajectory spends in the half where they differ.
        # Measured on LIBERO from a fixed action sequence, that ranged from 2.5%
        # to 15% of steps across start states alone.
        #
        # This is the `at_step: 0` problem again, arriving from geometry rather
        # than timing -- and unlike that one it cannot be declared away, because
        # no declaration controls where a trajectory goes. What it can be is
        # RECORDED, so a result reads "N points at 12% exposure" rather than
        # "N points", and two checkpoints with different exposures are not
        # compared as though they had received the same treatment.
        #
        # Null when the episode carries no perturbation, more than one, or one
        # whose effect is a state mutation -- for which every applicable step is
        # perturbed by construction and the number would be a constant 1.0
        # dressed as a measurement.
        pa.field("perturbation_exposure", pa.float64()),
        # How many perturbations the episode carried, because a null level
        # means two opposite things and `compare` has to tell them apart.
        #
        #   count 0  -> unperturbed. Null level, and it BELONGS on the curve:
        #               it is the baseline every other point is measured
        #               against, and phase 1 made every result recorded before
        #               perturbations existed a valid base for exactly this.
        #   count 1  -> one level, in `perturbation_level`.
        #   count 2+ -> null level, and it does NOT belong on a single axis:
        #               there is no one level to place it at.
        #
        # Without this, "on the curve at the origin" and "off the curve" are the
        # same null. The same distinction as `absent` versus an empty digest in
        # the physics surface: nothing looked and there was nothing there are
        # different claims.
        #
        # It also avoids a trap in the other direction. An unperturbed episode
        # cannot simply carry a neutral level, because neutral depends on the
        # effect -- 1.0 for scale_actuator, 0.0 for apply_force. Where the
        # baseline sits on an axis comes from the effect's own definition, so
        # `compare` has to know it is looking at a baseline in order to place
        # it, and that is what this column tells it.
        #
        # Null only ever means "written before this column existed". The runner
        # writes 0 for an unperturbed episode rather than leaving it unset, so a
        # null is always about the schema and never about the episode.
        pa.field("perturbation_count", pa.int32()),
        # Which benchmark class actually ran this episode.
        #
        # Provenance, never identity, and that is a measured claim rather than a
        # preference: the same unperturbed episode through LIBEROBenchmark and
        # through PerturbedLIBEROBenchmark produces bit-identical qpos over 20
        # steps. The subclass adds a timeline to `step` and an empty timeline
        # fires nothing, so which class ran is HOW the scene was run -- the same
        # category as the backend, the execution mode and the cpuset.
        #
        # Putting it in scene_hash would have cost exactly what phase 1 bought:
        # base_scenario_hash equals scenario_hash for an unperturbed scenario so
        # that every result already recorded is the baseline end of a torque
        # curve. Moving scene_hash to name the executing class would strand
        # those and make every sweep re-run its own baseline.
        #
        # Recorded anyway, because "it does not change identity" is not the same
        # as "nobody needs to know", and the claim it rests on is checked by
        # refractal-libero's verify_libero_claims.py rather than assumed.
        pa.field("benchmark_class", pa.string()),
        pa.field("scene_id", pa.string(), nullable=False),
        pa.field("scene_hash", pa.string(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("task_hash", pa.string(), nullable=False),
        pa.field("checkpoint_id", pa.string(), nullable=False),
        # A real column. Buried in a JSON blob you cannot group per scenario,
        # and a multi-seed sweep collapses into unrelated run-level averages.
        pa.field("seed", pa.int32(), nullable=False),
        # Provenance of the *observation*, not of the experiment: a resumed
        # comparison spans sessions with different warmup and thermal state.
        # Fine for success rates, wrong for latency. `compare` warns.
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("worker_id", pa.string(), nullable=False),
        pa.field("execution_mode", pa.string(), nullable=False),
        # Which model server answered. Provenance by the two-question test: the
        # same checkpoint served from two addresses is the same checkpoint (the
        # args that would change that are already in `plan_id`), so this must
        # never gate a join. It is here because a self-comparison -- two checkpoints
        # pointed at one server -- is preventable before a run and otherwise
        # unreconstructable after it, and `artifact_uri` is for artifacts.
        pa.field("server_url", pa.string()),
        # Which OTHER checkpoints were running while this episode ran, sorted and
        # comma-joined; empty when nothing else was.
        #
        # Contention is a property of the run, not of the deployment. A success
        # rate measured under contention is fine; a duration measured under it is
        # not comparable with one measured alone, and `elapsed_sec` is in the same
        # row. Without this column the only way to know is to trust
        # `execution_mode`, which is exactly the field that was wrong.
        #
        # Provenance, not identity: contention does not change what an episode
        # means, so it must never gate a join.
        pa.field("concurrent_with", pa.string()),
        # Two columns, one of each kind. `harness_version` is provenance: human
        # readable, never gates. `harness_surface` is a precondition: a digest
        # over only the harness modules Refractal depends on, so it stays
        # constant through docs edits and data refreshes and moves when
        # something behavioural does. Gating on the version instead would fire
        # on every dependency bump -- measured at 100% of commits -- and a check
        # that always fires gets waived habitually.
        pa.field("harness_version", pa.string()),
        pa.field("harness_surface", pa.string()),
        # The same pair, one layer down. robosuite sits between MuJoCo and the
        # benchmark and owns the robot model -- actuator gains, force ranges,
        # friction -- none of which are in a BDDL file, a suite name or an
        # engine version. A release that edits a gripper's forcerange changes
        # the physics of every episode and moves no identity here.
        #
        # `physics_version` is provenance and never gates. `physics_surface` is
        # a digest of the VALUES, so a version bump that changed nothing does
        # not block a comparison and a forcerange edited in a patch release
        # does. It is a precondition, not identity, for the same reason as the
        # harness: putting it in plan_id would orphan every result on every
        # dependency bump.
        #
        # This matters now rather than in principle. A torque-margin sweep is a
        # curve over exactly this number, so a silent change does not perturb
        # the results -- it rewrites the axis.
        pa.field("physics_version", pa.string()),
        pa.field("physics_surface", pa.string()),
        pa.field("success", pa.bool_(), nullable=False),
        # map<string,bool>, not a fixed struct. Two tasks on one scene write to
        # the same file and may declare different phases; a struct has one
        # schema per file and would refuse the second task.
        pa.field("phase_outcomes", pa.map_(pa.string(), pa.bool_())),
        pa.field("terminal_phase", pa.string()),
        # Null if and only if `success`. The API reference says "null means a
        # genuine policy failure", but null is also what a success writes, so
        # the two are indistinguishable. A failure always names itself:
        # "policy_failure", "timeout", or an infra reason.
        pa.field("failure_reason", pa.string()),
        # Its own column, never inferred from a string. `compare` excludes these
        # from success-rate denominators and must not have to pattern-match.
        pa.field("is_infra_failure", pa.bool_(), nullable=False),
        pa.field("steps", pa.int32(), nullable=False),
        pa.field("elapsed_sec", pa.float64(), nullable=False),
        pa.field("started_at", pa.timestamp("us", tz="UTC")),
        pa.field("ended_at", pa.timestamp("us", tz="UTC")),
        pa.field("artifact_uri", pa.string()),
    ]
)

STEPS_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("step", pa.int32(), nullable=False),
        pa.field("qpos", pa.list_(pa.float32())),
        pa.field("qvel", pa.list_(pa.float32())),
        pa.field("action", pa.list_(pa.float32())),
        pa.field("ee_pose", pa.list_(pa.float32())),
        # Null unless the model server reports it. This is what the chunk-timing
        # experiment needs, and it is obtainable without touching the benchmark.
        pa.field("chunk_index", pa.int32()),
    ]
)


class OutputMissingError(RefractalError):
    """A worker ran episodes and its results are not on disk.

    The failure this exists for: a run that completes, reports success, and
    writes nothing is indistinguishable from a correct run until someone tries
    to compare — by which point the compute is spent and the session is gone.

    The concrete way to get there is the harness's ``_build_recorder``, which
    returns ``NullEpisodeRecorder`` whenever ``self._store is None``. Override the
    recorder and not the store, and every episode runs, every episode succeeds,
    and nothing is recorded. But the guard is written against the *symptom*
    rather than that cause, so it also catches the variants nobody has thought of:
    a writer pointed at the wrong prefix, a filesystem that accepted a write and
    dropped it, a backend that forgot to flush its last batch.
    """


def comparison_prefix(results_uri: str, plan_id: str) -> str:
    """``comparison_id=<hex>`` -- the plan id, minus the algorithm prefix.

    The colon in ``sha256:...`` is legal in a POSIX path and trouble almost
    everywhere else: it is a drive separator on Windows and needs escaping in
    several object-store tools. The digest alone is unambiguous.
    """
    digest = plan_id.split(":", 1)[-1]
    return f"{results_uri.rstrip('/')}/comparison_id={digest}"


def check_receipts(rows: Iterable[dict[str, Any]]) -> None:
    """A null ``fired_step`` must carry a reason. Refuse the write otherwise.

    The two nulls mean opposite things and were indistinguishable in the
    column:

      null + reason -> the episode outran its trigger. Real, expected, and the
                       row is excluded from its level's curve.
      null + none   -> nothing mapped. The receipt arrived, is structurally
                       valid, and says nothing.

    The second is what shipped. ``fired_events`` emitted the Timeline's field
    names -- ``type``, ``specified_at``, ``fired_at`` -- against a struct
    declaring ``effect``, ``specified_step``, ``fired_step``, so every declared
    field landed null. Every guard checked that a receipt ARRIVED; none checked
    that it arrived populated, and a live sweep was needed to notice.

    A wrapper has a third way to arrive empty, and it needs its own check.
    Its receipt is a digest pair and two counts, and a transform that ran on
    every step while never changing anything has a full `applications` count
    and a `applications_changed` of zero. Every field is populated; nothing
    happened.

    Note what is NOT checked: the first digest pair being equal. A wrapper may
    be identity on some inputs and not others -- a rotation convention swap is
    exactly that -- so equal digests at step one is ordinary rather than
    suspicious. Only zero changes across the whole window is the failure.

    Checked at the write, which is the earliest point that can see it and the
    loudest place to say so: the alternative is discovering it at analysis, or
    not at all.
    """
    for row in rows:
        # An episode that was ASSIGNED perturbations and reports none is the
        # emptiest receipt there is, and the narrowest check missed it: every
        # per-event rule passes trivially over no events.
        #
        # It is reachable. The adapter accumulated mutation events in its own
        # list while wrapper windows lived on the timeline, so an episode
        # perturbing only the observation reported nothing at all.
        if row.get("perturbation_count") and not row.get("perturbations_fired"):
            raise RefractalError(
                f"episode {row.get('episode_id')!r} was assigned "
                f"{row['perturbation_count']} perturbation(s) and its receipt "
                "is empty -- not a perturbation that did not fire, which "
                "records a reason, but no entry of any kind. Nothing collected "
                "what happened."
            )
        for event in row.get("perturbations_fired") or ():
            # Before the fired_step guard below, not after it. A wrapper always
            # HAS a fired step, so a check placed after that `continue` can
            # never run -- which is what the first version of this did, and
            # every test still passed because nothing reached it.
            applications = event.get("applications")
            if applications and not event.get("applications_changed"):
                raise RefractalError(
                    f"perturbation {event.get('effect')!r} applied on "
                    f"{applications} steps and changed the observation on none "
                    "of them. The receipt is fully populated and records a "
                    "transform that passed its input through unchanged, which "
                    "reads downstream as a level that was measured. Either the "
                    "effect is a no-op on this scene, or it is hooked somewhere "
                    "its output is discarded."
                )
            if event.get("fired_step") is not None:
                continue
            if not event.get("reason"):
                raise RefractalError(
                    "a perturbation receipt has fired_step=None and no reason, "
                    f"for {event.get('effect')!r} on {event.get('target')!r}. "
                    "An episode that outran its trigger records a reason; a "
                    "receipt with neither is a mapping failure -- most likely "
                    "field names that do not match the column's. Refusing to "
                    "write a receipt that is present, valid and empty."
                )


@dataclass
class ResultWriter:
    """Appends episode rows under a comparison prefix, one part file per flush."""

    results_uri: str
    plan_id: str

    def __post_init__(self) -> None:
        self.fs, _ = fsspec.core.url_to_fs(self.results_uri)
        self.prefix = comparison_prefix(self.results_uri, self.plan_id)

    def scene_dir(self, checkpoint_id: str, scene_id: str) -> str:
        return f"{self.prefix}/checkpoint={checkpoint_id}/scene={scene_id}/episodes"

    def write_episodes(
        self, checkpoint_id: str, scene_id: str, rows: list[dict[str, Any]], *, part: str
    ) -> str | None:
        """Write one part file atomically. Returns its path, or None if no rows."""
        if not rows:
            return None
        check_receipts(rows)
        directory = self.scene_dir(checkpoint_id, scene_id)
        self.fs.makedirs(directory, exist_ok=True)
        final = f"{directory}/part-{part}.parquet"
        # A temp name in the same directory, so the rename stays within one
        # filesystem and therefore stays atomic.
        staging = f"{directory}/.tmp-{uuid.uuid4().hex}.parquet"

        table = pa.Table.from_pylist(rows, schema=EPISODES_SCHEMA)
        with self.fs.open(staging, "wb") as handle:
            pq.write_table(table, handle, compression="zstd")
        self.fs.mv(staging, final)
        return final

    def write_strips(self, frames, episode_ids, *, quality: int = 80) -> list[str]:
        """One sprite strip per episode: frames left to right in a single WebP.

        One file per episode rather than one per frame, because a 600-episode
        run at every tenth step is ~7,600 frames and a static site serving that
        many files pays for each one. A strip is also a single decode, so the
        page can drive it from one clock and cannot desynchronise.

        Written temp-then-rename, for the same reason the Parquet parts are.
        """
        try:
            from PIL import Image
        except ImportError as exc:      # pragma: no cover - environment-dependent
            raise RuntimeError(
                "video was requested but Pillow is not installed, so frames "
                "cannot be encoded. Install it, or run without video."
            ) from exc

        directory = f"{self.prefix}/frames"
        self.fs.makedirs(directory, exist_ok=True)
        written: list[str] = []
        # Positional, because the buffer is keyed by the harness's episode id and
        # the rows by Refractal's. Same correspondence the outcome mapping uses.
        by_episode = frames.in_order()
        # Nothing at all is the receipt's case, not this one: the caller turns
        # an empty result into "video was requested and none arrived", which
        # names the actual problem. A mismatch means SOME episodes recorded and
        # the positional match is therefore unsafe, which is this one.
        if not by_episode:
            return []
        if len(by_episode) != len(episode_ids):
            raise OutputMissingError(
                f"the recorder saw {len(by_episode)} episode(s) and the plan named "
                f"{len(episode_ids)}. Frames cannot be matched to rows positionally "
                "when the counts disagree."
            )
        for episode_id, got in zip(episode_ids, by_episode):
            if not got:
                continue
            tiles = [Image.fromarray(f) for f in got]
            w, h = tiles[0].size
            strip = Image.new("RGB", (w * len(tiles), h))
            for i, tile in enumerate(tiles):
                strip.paste(tile, (i * w, 0))
            name = episode_id.replace("sha256:", "")[:32]
            final = f"{directory}/{name}.webp"
            staging = f"{directory}/.tmp-{uuid.uuid4().hex}.webp"
            with self.fs.open(staging, "wb") as handle:
                strip.save(handle, format="WEBP", quality=quality, method=5)
            self.fs.mv(staging, final)
            written.append(final)
        return written

    def verify_frames(self, paths: Iterable[str], *, worker_id: str) -> int:
        """Read the strips back and confirm they decode.

        Reads the artifact, not a counter. A tally kept by the code that did the
        writing is produced by the same path that would have failed, which is
        the whole reason `verify_written` reads the part files rather than
        trusting its own count.
        """
        from PIL import Image

        total = 0
        for path in paths:
            with self.fs.open(path, "rb") as handle:
                img = Image.open(handle)
                img.load()
                if img.height == 0 or img.width % img.height:
                    raise OutputMissingError(
                        f"{worker_id}: {path} is {img.width}x{img.height}, which is "
                        "not a whole number of square frames. The strip is truncated "
                        "or was written from frames of differing size."
                    )
                total += img.width // img.height
        return total

    def episode_ids_in(self, paths: Iterable[str]) -> list[str]:
        """Episode ids from specific part files, **with duplicates preserved**.

        Deliberately a list rather than a set. A set cannot see a row written
        twice, and a doubled episode is the same failure as the cross-set
        scenario duplication: one scenario counted twice, silently weighted
        double in a comparison.
        """
        ids: list[str] = []
        for path in paths:
            with self.fs.open(path, "rb") as handle:
                ids.extend(pq.read_table(handle, columns=["episode_id"],
                                     use_threads=READ_THREADS)
                           .column("episode_id").to_pylist())
        return ids

    def verify_written(
        self, expected_ids: set[str], paths: Iterable[str], *, worker_id: str
    ) -> None:
        """Require the ids on disk to be exactly the ids that were planned.

        Three distinct failures, and the naive version catches only the first:

        * **missing** — planned and not written. The null-recorder case.
        * **unexpected** — written and not planned. An id the writer invented,
          which matters because ``episode_id`` is derived from things the harness
          supplies, and the bridge is the first place those come from a benchmark
          we did not write.
        * **duplicated** — one id, two rows. Invisible to a set comparison, and
          it double-weights a scenario in every downstream statistic.

        The evidence comes from the part files, not from a counter incremented
        alongside the writes: a count is produced by the same code path that did
        the writing, so it cannot disagree with it. Verification is only worth
        anything when the two sides come from different paths.
        """
        if not expected_ids:
            return

        paths = list(paths)
        if not paths:
            raise OutputMissingError(
                f"worker {worker_id!r} ran {len(expected_ids)} episode(s) and wrote no files "
                "at all. The run reported success and produced no results, which is the shape "
                "of a recorder that was never activated — check that whatever gates the "
                "recorder is set, not just that the recorder class was overridden."
            )

        ids = self.episode_ids_in(paths)
        found = set(ids)
        missing = expected_ids - found
        unexpected = found - expected_ids
        duplicated = len(ids) - len(found)

        if missing and not found:
            raise OutputMissingError(
                f"worker {worker_id!r} ran {len(expected_ids)} episode(s) and wrote none. "
                "The run reported success and produced no results, which is the shape of a "
                "recorder that was never activated — check that whatever gates the recorder "
                "is set, not just that the recorder class was overridden."
            )
        if missing:
            raise OutputMissingError(
                f"worker {worker_id!r} ran {len(expected_ids)} episode(s) but only "
                f"{len(expected_ids) - len(missing)} reached storage; {len(missing)} missing "
                f"(first: {sorted(missing)[0]}). A partial result silently shrinks a denominator."
            )
        if unexpected:
            raise OutputMissingError(
                f"worker {worker_id!r} wrote {len(unexpected)} episode id(s) that were never "
                f"planned (first: {sorted(unexpected)[0]}). episode_id is derived from the "
                "scene, task, scenario, seed and checkpoint; an id that is not in the plan "
                "means one of those did not survive the round trip intact."
            )
        if duplicated:
            raise OutputMissingError(
                f"worker {worker_id!r} wrote {len(ids)} rows for {len(found)} episode(s): "
                f"{duplicated} duplicate(s). A repeated episode_id weights that scenario twice "
                "in every downstream statistic, which no later stage can detect."
            )

    def completed_episode_ids(self) -> set[str]:
        """Which episodes already have a result, for resume.

        Reads exactly one column out of every part file. This is the property
        Parquet was chosen for -- the same question against a sequential log
        means reading every file start to finish.
        """
        pattern = f"{self.prefix}/checkpoint=*/scene=*/episodes/part-*.parquet"
        try:
            paths = self.fs.glob(pattern)
        except FileNotFoundError:
            return set()
        done: set[str] = set()
        for path in paths:
            with self.fs.open(path, "rb") as handle:
                table = pq.read_table(handle, columns=["episode_id"],
                                      use_threads=READ_THREADS)
            done.update(table.column("episode_id").to_pylist())
        return done

    def record_physics_manifest(self, surface: str, manifest: dict[str, str]) -> None:
        """The per-actuator lines behind a physics surface digest.

        Same shape and same reason as the harness manifest: the digest says
        something moved, and whoever reads the comparison months later no longer
        has both engines installed to find out what.
        """
        if not manifest or surface == "absent":
            return
        directory = f"{self.prefix}/physics"
        self.fs.makedirs(directory, exist_ok=True)
        target = f"{directory}/{surface}.json"
        if self.fs.exists(target):
            return
        staging = f"{directory}/.tmp-{uuid.uuid4().hex}.json"
        with self.fs.open(staging, "wb") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        self.fs.mv(staging, target)

    def record_harness_manifest(self, surface: str, manifest: dict[str, str]) -> None:
        """Store the per-file surface digests for this run, once.

        Keyed by the surface digest, so two sessions on the same harness write
        one file and a comparison spanning two surfaces has both to diff. Written
        once per session rather than per row because it is provenance, not data.
        """
        if not manifest:
            return
        directory = f"{self.prefix}/harness"
        self.fs.makedirs(directory, exist_ok=True)
        target = f"{directory}/{surface}.json"
        if self.fs.exists(target):
            return
        staging = f"{directory}/.tmp-{uuid.uuid4().hex}.json"
        with self.fs.open(staging, "wb") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        self.fs.mv(staging, target)

    def read_harness_manifests(self) -> dict[str, dict[str, str]]:
        """Every recorded surface manifest, by digest. Empty when none were written."""
        try:
            paths = self.fs.glob(f"{self.prefix}/harness/*.json")
        except FileNotFoundError:
            return {}
        out: dict[str, dict[str, str]] = {}
        for path in paths:
            with self.fs.open(path, "rb") as handle:
                out[Path(path).stem] = json.loads(handle.read().decode("utf-8"))
        return out

    def write_plan(self, plan_json: str) -> None:
        """Always, independent of whether a catalog was supplied.

        Separate from ``copy_catalog`` because they answer different questions and
        one is much cheaper. The plan is what makes a results directory
        interpretable at all -- it carries the identity document, so the directory
        can say why it is a different experiment from another one. The catalog is
        the fuller provenance.

        They used to be one call, which meant a run without a catalog produced
        results that could not explain themselves. Found by trying to read a plan
        back out of a results tree.
        """
        self.fs.makedirs(self.prefix, exist_ok=True)
        with self.fs.open(f"{self.prefix}/plan.json", "wb") as handle:
            handle.write(plan_json.encode("utf-8"))

    def copy_catalog(self, catalog_root: str) -> None:
        """Results and the definitions that produced them travel together."""
        import os

        for dirpath, _, filenames in os.walk(catalog_root):
            rel_dir = os.path.relpath(dirpath, catalog_root)
            for name in filenames:
                if name.startswith("."):
                    continue
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                target = f"{self.prefix}/catalog/{rel}"
                self.fs.makedirs(target.rsplit("/", 1)[0], exist_ok=True)
                with open(os.path.join(dirpath, name), "rb") as src:
                    payload = src.read()
                with self.fs.open(target, "wb") as dst:
                    dst.write(payload)


def read_episodes(results_uri: str, plan_id: str) -> "pa.Table":
    """Read every episode row for one comparison as a single table."""
    fs, _ = fsspec.core.url_to_fs(results_uri)
    prefix = comparison_prefix(results_uri, plan_id)
    paths = sorted(fs.glob(f"{prefix}/checkpoint=*/scene=*/episodes/part-*.parquet"))
    if not paths:
        return EPISODES_SCHEMA.empty_table()
    tables = []
    for path in paths:
        with fs.open(path, "rb") as handle:
            tables.append(pq.read_table(handle, use_threads=READ_THREADS))
    return pa.concat_tables(tables)


__all__ = [
    "EPISODES_SCHEMA",
    "STEPS_SCHEMA",
    "ResultWriter",
    "comparison_prefix",
    "read_episodes",
]
