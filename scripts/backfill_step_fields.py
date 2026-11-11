#!/usr/bin/env python3
"""Recover the v2 distributional columns on rows written without them.

Every vla-eval row ever written has ``outcome``, ``step_to_success``,
``step_budget``, ``decisions`` and ``schema_version`` null: the row path wrote
none of them. That put the distributional layer out of reach of the only
experiment that needed it -- no ECDF, no median, no quantile, no restricted
mean, and no budget sweep without re-running.

Three of those are recoverable exactly, and one is not:

``step_to_success``
    ``steps`` on a success row. Sound only because this harness ends the
    episode when the predicate holds, which is a property of the backend and
    is CHECKED here rather than assumed -- see the guard below.
``step_budget``
    DECLARED, from the plan's ``max_steps``, and verified against the value
    every failure sat at. Inferring it from the failures alone works only on a
    run that had some, and reads an observation as a declaration; reading the
    plan and checking it is the same declared-versus-realized pattern as
    everywhere else. A disagreement refuses.
``outcome``
    SUCCESS where ``success``, BUDGET_EXHAUSTED otherwise, which holds only
    under the same partition the guard enforces.
``decisions``
    NOT derived. It is the count of inference calls, which the harness never
    records -- it records per environment step. Taken from the model server's
    action-hash log where one is given, and left null with a reason otherwise.
    Deriving it from the declared chunk size would make ``steps / decisions``
    a restatement of the declaration.

THE GUARD, and it is the reason this is a script with a refusal rather than a
query. The partition is "steps below the budget means it succeeded, steps at
the budget means it ran out". That is true of the recorded data -- successes
span 63..210 and all 315 failures sit at exactly 220 -- and nothing stops a
future run from producing a success that lands exactly on the budget. Such a
row would be silently reclassified as BUDGET_EXHAUSTED and its success time
discarded. So a success at the budget refuses the whole file rather than
deriving.

Nothing is written in place. Each input produces a sibling file, so the
observed rows stay exactly as recorded and a backfill can be thrown away.

Usage::

    python scripts/backfill_step_fields.py RESULTS_DIR [--hash-log LOG ...] [--write]

Without ``--write`` it reports what it would do and refuses nothing silently.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from refractal.execute.results import EPISODES_SCHEMA  # noqa: E402

SUFFIX = ".backfilled.parquet"


class BackfillRefused(Exception):
    """The partition this backfill rests on does not hold for these rows."""


def calls_per_episode(paths: list[pathlib.Path]) -> dict[str, int]:
    """Inference calls per episode, from the model server's hash log.

    One line per call, each carrying the episode id Refractal sent with
    EPISODE_START. A measurement from a different artefact than the row, which
    is why it is recorded as such rather than folded in with the derivations.
    """
    counts: collections.Counter = collections.Counter()
    for path in paths:
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("event") or not record.get("episode_id"):
                    continue
                counts[record["episode_id"]] += 1
    return dict(counts)


def declared_budget(results: pathlib.Path) -> tuple[int | None, str]:
    """``max_steps`` from the plan beside these rows, with where it came from.

    The plan is the only thing that can answer this on a run with no failures,
    and it is a declaration rather than an inference either way. Returns None
    and a reason when no plan is in reach -- a results tree someone else
    wrote, or one comparison's rows copied out of their directory.
    """
    # THREE outcomes, kept apart on purpose: no plan file, a plan file that
    # will not parse, and a plan file that parses but does not carry the field
    # where this looked. The first version of this collapsed the third into
    # the first and reported "no plan.json in reach" against a tree that had
    # one -- it was reading `scenes[].episodes[]`, and the planner groups
    # episodes under the WORKER that will run them.
    #
    # A lookup that says "absent" when it means "I looked somewhere that does
    # not exist" is indistinguishable from a real absence, and the next change
    # to the planner's shape reproduces it. So the path searched is named in
    # the message, which makes a wrong path visible the first time it is read.
    PATH = "scenes[].workers[].episodes[].max_steps"
    plans = sorted(results.rglob("plan.json"))
    if not plans:
        return None, f"unmeasured:no plan.json anywhere under {results}"

    unreadable: list[str] = []
    for candidate in plans:
        try:
            document = json.loads(candidate.read_text())
        except (ValueError, OSError) as exc:
            unreadable.append(f"{candidate.name}: {type(exc).__name__}")
            continue
        limits = {
            episode.get("max_steps")
            for scene in (document.get("scenes") or [])
            for worker in (scene.get("workers") or [])
            for episode in (worker.get("episodes") or [])
            if episode.get("max_steps") is not None
        }
        if len(limits) == 1:
            return limits.pop(), f"declared:{candidate.name} {PATH}"
        if limits:
            return None, (f"unmeasured:{candidate.name} declares {len(limits)} "
                          "different max_steps, so there is no single budget")
        return None, (f"unmeasured:{candidate.name} parsed but carries no "
                      f"{PATH}. The file is there and the field is not where "
                      "this looked -- check the planner's shape before reading "
                      "this as an absent declaration.")
    return None, (f"unmeasured:{len(plans)} plan.json file(s) found and none "
                  f"could be read ({'; '.join(unreadable[:3])})")


def budget_of(steps: list[int], successes: list[bool]) -> int | None:
    """The one step limit these rows ran under, or None if they cannot say.

    Read off the failures: a failure ran until it was stopped, so its step
    count IS the limit. Rows with no failure cannot say what the limit was,
    and get None rather than max(steps) -- the longest success is a lower
    bound on the budget, not the budget.

    ACROSS THE WHOLE INPUT, not per file. A results tree holds one file per
    (task, seed) group of ten episodes, and a group where all ten succeeded
    contains no failure at all -- 4 such episodes in the recorded run. Scoped
    per file they would get a null budget, and the distributional layer then
    refuses the whole comparison on them, which is how this was found: the
    budget is a property of the RUN, and the other files are what reveal it.
    """
    failures = {s for s, ok in zip(steps, successes) if not ok}
    if len(failures) == 1:
        return failures.pop()
    return None


REQUIRED = tuple(f.name for f in EPISODES_SCHEMA if not f.nullable)


def derive(table: "pa.Table", calls: dict[str, int],
           budget: int | None = None) -> tuple["pa.Table", dict[str, int]]:
    # Refused here rather than left to pyarrow, which reports
    # "Column 'scenario_hash' is declared non-nullable but contains nulls"
    # from inside the writer -- after every other file has been processed, and
    # without saying that the INPUT was short rather than the derivation wrong.
    absent = [name for name in REQUIRED if name not in table.column_names]
    if absent:
        raise BackfillRefused(
            f"these rows are missing {absent}, which the episode schema "
            "requires. A backfill can add the columns a writer omitted; it "
            "cannot invent the identity of an episode."
        )
    steps = table.column("steps").to_pylist()
    successes = [bool(v) for v in table.column("success").to_pylist()]
    episode_ids = table.column("episode_id").to_pylist()
    if budget is None:
        budget = budget_of(steps, successes)

    if budget is not None:
        landed = [
            (episode_ids[i], steps[i])
            for i, ok in enumerate(successes)
            if ok and steps[i] >= budget
        ]
        if landed:
            raise BackfillRefused(
                f"{len(landed)} success row(s) have steps == the budget "
                f"({budget}); first {landed[0][0]} at {landed[0][1]} steps. "
                "The derivation reads 'steps below the budget' as success and "
                "'steps at the budget' as exhaustion, so such a row would be "
                "reclassified and its success time discarded. Refusing rather "
                "than deriving."
            )

    tally: collections.Counter = collections.Counter()
    outcome, step_to_success, step_budget, decisions, provenance = [], [], [], [], []
    for i, ok in enumerate(successes):
        route: dict[str, str] = {}
        if ok:
            outcome.append("SUCCESS")
            step_to_success.append(steps[i])
            route["outcome"] = "derived:success flag"
            route["step_to_success"] = (
                "derived:steps; this harness ends the episode at success")
            tally["success"] += 1
        elif budget is not None:
            outcome.append("BUDGET_EXHAUSTED")
            step_to_success.append(None)
            route["outcome"] = "derived:steps equals the budget every failure sat at"
            tally["budget_exhausted"] += 1
        else:
            outcome.append(None)
            step_to_success.append(None)
            route["outcome"] = (
                "unmeasured:no failure anywhere in this run, so the budget is "
                "unknown")
            tally["outcome unknown"] += 1

        step_budget.append(budget)
        if budget is not None:
            route["step_budget"] = "derived:the value every failure ran to"

        count = calls.get(episode_ids[i])
        decisions.append(count)
        route["decisions"] = (
            "measured:model server action-hash log, one line per inference call"
            if count is not None else
            "unmeasured:the harness records per environment step, never per "
            "inference call, and no hash log was given")
        if count is not None:
            tally["decisions measured"] += 1
        provenance.append(sorted(route.items()))

    columns = {name: table.column(name) for name in table.column_names}
    columns["outcome"] = pa.array(outcome, pa.string())
    columns["step_to_success"] = pa.array(step_to_success, pa.int32())
    columns["step_budget"] = pa.array(step_budget, pa.int32())
    columns["decisions"] = pa.array(decisions, pa.int32())
    # NOT 2. A native v2 row was written by a writer that observed these
    # fields; these were reconstructed afterwards. A reader asking "was this
    # measured or inferred?" must not get the same answer from both.
    columns["schema_version"] = pa.array([None] * table.num_rows, pa.int32())
    columns["field_provenance"] = pa.array(
        provenance, pa.map_(pa.string(), pa.string()))

    for field in EPISODES_SCHEMA:
        columns.setdefault(field.name, pa.nulls(table.num_rows, field.type))
    rebuilt = pa.table({f.name: columns[f.name] for f in EPISODES_SCHEMA},
                       schema=EPISODES_SCHEMA)
    return rebuilt, dict(tally)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--hash-log", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--write", action="store_true",
                        help="write sibling .backfilled.parquet files")
    args = parser.parse_args(argv[1:])

    calls = calls_per_episode(args.hash_log) if args.hash_log else {}
    print(f"inference-call counts from {len(args.hash_log)} hash log(s): "
          f"{len(calls)} episode(s)\n")

    files = [p for p in sorted(args.results.rglob("*.parquet"))
             if not p.name.endswith(SUFFIX)]

    # Pass one: the run's budget, from every failure in the tree.
    all_steps: list[int] = []
    all_success: list[bool] = []
    for path in files:
        table = pq.read_table(path)
        if "steps" not in table.column_names or "success" not in table.column_names:
            continue
        all_steps += table.column("steps").to_pylist()
        all_success += [bool(v) for v in table.column("success").to_pylist()]
    observed_budget = budget_of(all_steps, all_success)
    observed = sorted({s for s, ok in zip(all_steps, all_success) if not ok})
    if observed_budget is None and observed:
        print(f"REFUSED: failures in this tree ran to {len(observed)} different "
              f"step counts ({observed[:6]}), so there is no single budget. Two "
              "budgets in one tree are two experiments; back-fill them "
              "separately.", file=sys.stderr)
        return 1

    declared, route = declared_budget(args.results)
    if declared is not None and observed_budget is not None and declared != observed_budget:
        print(f"REFUSED: the plan declares max_steps={declared} and every "
              f"failure ran to {observed_budget}. One of the two does not "
              "describe this run, and deriving an outcome from either would "
              "record a budget nothing was evaluated against.", file=sys.stderr)
        return 1

    budget = declared if declared is not None else observed_budget
    if declared is not None:
        detail = route + (f", verified against {len(observed)} failure value(s)"
                          if observed_budget is not None else
                          ", with no failure to verify it against")
    else:
        detail = (f"derived:the value every failure ran to ({route})"
                  if observed_budget is not None else route)
    print(f"step budget: {budget}  [{detail}]\n")

    totals: collections.Counter = collections.Counter()
    refused: list[tuple[pathlib.Path, str]] = []
    written = 0
    for path in files:
        table = pq.read_table(path)
        try:
            rebuilt, tally = derive(table, calls, budget)
        except BackfillRefused as exc:
            refused.append((path, str(exc)))
            continue
        totals.update(tally)
        totals["rows"] += table.num_rows
        if args.write:
            # with_suffix() replaces only the final suffix and rejects a
            # dotted one, so the sibling name is built explicitly.
            pq.write_table(rebuilt, path.with_name(path.stem + SUFFIX))
            written += 1

    print(f"{len(files)} file(s), {totals['rows']} row(s)")
    for key in ("success", "budget_exhausted", "outcome unknown", "decisions measured"):
        if totals.get(key):
            print(f"  {key:20s} {totals[key]}")
    if written:
        print(f"\nwrote {written} sibling file(s) ending {SUFFIX}; "
              "nothing was modified in place")
    elif not refused:
        print("\nnothing written (no --write). Re-run with --write to produce "
              "sibling files.")

    if refused:
        print(f"\n{len(refused)} file(s) REFUSED:\n")
        for path, reason in refused[:5]:
            print(f"  {path}\n    {reason}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
