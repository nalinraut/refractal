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
EPISODES_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        # The comparison join key is (scene_id, task_hash, scenario_hash).
        # `task_hash` is here rather than only `task_id` because a task id is a
        # label someone chose: renaming it must not break a join, and changing
        # its meaning must.
        pa.field("scenario_hash", pa.string(), nullable=False),
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
        # never gate a join. It is here because a self-comparison -- two arms
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
                ids.extend(pq.read_table(handle, columns=["episode_id"])
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
                table = pq.read_table(handle, columns=["episode_id"])
            done.update(table.column("episode_id").to_pylist())
        return done

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
            tables.append(pq.read_table(handle))
    return pa.concat_tables(tables)


__all__ = [
    "EPISODES_SCHEMA",
    "STEPS_SCHEMA",
    "ResultWriter",
    "comparison_prefix",
    "read_episodes",
]
