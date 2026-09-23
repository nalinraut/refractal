#!/usr/bin/env python3
"""Mutation checks that cannot report a result from a run that did not happen.

Breaking a check to watch it fail is how this project decides whether a test is
real. That makes the mutation runner a checker, and this project has now found
three checkers with no check on themselves -- so this one has one.

The incident
------------

    ./scripts/test.sh tests.test_physics tests.test_compare

Two positional arguments override ``unittest discover``'s start directory, so
the run found **zero tests** and reported **zero failures**. Three mutations
were recorded as "did not bite" when nothing had been executed at all. One
argument happens to run the whole suite; two run nothing. Neither is visible in
a failure count.

So ``0 failures from 0 tests`` is an error here, not a pass -- the same rule as
``verify_written``, which refuses to believe a worker wrote its rows because a
counter says so. The test *count* is checked against a baseline taken from the
unmutated tree, so a run that executes a subset is refused as loudly as one that
executes nothing.

A surviving mutation is not yet a finding
-----------------------------------------

A mutation that nothing notices means one of two very different things:

    the check it breaks is not tested        -- a real gap
    the mutation does not change behaviour   -- nothing at all

These look identical from a failure count, and the second is easy to write by
accident: a rewrite that moves a guard past a filter it was already past, an
operator swapped where the operands are equal. One was nearly recorded as a
coverage gap here.

So survival now requires a **witness**: a snippet shown to produce different
output under the original and the mutant. Without one the result is
UNDETERMINED rather than a gap -- the same rule as ``0 failures from 0 tests``,
one level up. A mutation is an instrument, and an instrument that did not move
says nothing about what it was pointed at.

Restoring
---------

From an in-memory copy, never from git. ``git checkout`` restores HEAD, and a
mutation applied to uncommitted work is undone by reverting the work itself --
which happened, silently deleting a gate that had just been written. Commit
before mutating as well; this makes it survivable either way.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rewrite import RewriteError, subs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: Written while a file is mutated, removed when it is restored. Its presence at
#: startup means a previous sweep died mid-mutation.
SENTINEL = ROOT / ".mutation-in-progress"
RAN = re.compile(r"^Ran (\d+) tests?", re.M)
OUTCOME = re.compile(r"^(FAIL|ERROR): ", re.M)


class MutationError(RuntimeError):
    """The run that was supposed to judge a mutation did not happen properly."""


def refuse_if_interrupted() -> None:
    """A previous sweep died with a file still mutated. Say which."""
    if SENTINEL.exists():
        raise MutationError(
            f"a previous mutation run did not restore:\n    {SENTINEL.read_text().strip()}\n"
            "Restore that file (git checkout, if the real work is committed) and "
            f"delete {SENTINEL.name}. Refusing to mutate on top of a mutation."
        )


def run_suite(python: str | None = None) -> tuple[int, int, str]:
    """The whole suite, always. Returns (tests run, failing tests, output)."""
    env_prefix = [f"PYTHON={python}"] if python else []
    proc = subprocess.run(
        ["env", *env_prefix, "./scripts/test.sh"],
        cwd=ROOT, capture_output=True, text=True,
    )
    output = proc.stdout + proc.stderr
    match = RAN.search(output)
    if match is None:
        raise MutationError(
            "the test run printed no 'Ran N tests' line, so nothing can be "
            f"concluded from it.\n{output[-800:]}"
        )
    return int(match.group(1)), len(OUTCOME.findall(output)), output


class Baseline(int):
    """A test count from a tree that was green when it was taken.

    A plain int cannot carry that, and every caller reaching for `check`
    directly was left to remember it. One did not -- it destructured the
    failure count away and swept a red tree, so two runs of verdicts meant
    nothing. The precondition is a type now rather than a convention.
    """

    __slots__ = ()


def green_baseline(python: str | None = None) -> Baseline:
    """Run the suite and return its size, refusing to return one if it is red."""
    tests, failures, _ = run_suite(python)
    if failures:
        raise MutationError(
            f"the tree is already failing {failures} test(s). Nothing can be "
            "concluded about any mutation from a run that was red to begin "
            "with. Fix the tree, then take a baseline."
        )
    return Baseline(tests)


def run_witness(source: str, python: str | None = None) -> str:
    """Run a snippet against the tree as it currently stands.

    Its whole job is to show that a mutation changed something. Failure is a
    result rather than an error -- a mutant that makes the witness raise has
    demonstrably changed behaviour -- so the return code and stderr are part of
    the answer, not an exception.
    """
    proc = subprocess.run(
        [python or sys.executable, "-c", source],
        cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    return f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr.strip()[-300:]}"


def check(
    name: str,
    path: str,
    pairs: list[tuple[str, str]],
    *,
    baseline: Baseline,
    witness: str | None = None,
    python: str | None = None,
) -> int:
    """Apply a mutation, run the suite, restore, and report how many tests died.

    Raises rather than returning zero when the run itself was not trustworthy:
    a suite that executed a different number of tests than the baseline is not
    evidence about the mutation, whichever direction it moved.

    ``witness`` is a snippet that must behave differently under the mutant. It
    is what separates a coverage gap from a mutation that changed nothing, and
    it is only consulted when the suite stays green -- a mutation the tests
    already killed needs no further proof that it did something.
    """
    target = ROOT / path
    original = target.read_text(encoding="utf-8")
    if not isinstance(baseline, Baseline):
        raise MutationError(
            f"{name}: `baseline` must come from `green_baseline()`, which "
            "refuses a tree that is already failing. Passing a bare count lets "
            "a red baseline through, and then a kill may be the pre-existing "
            "failure while a survival is masked by it -- which happened, from "
            "a caller that took the test count and discarded the failures."
        )
    before = run_witness(witness, python) if witness else None
    # A sentinel, because the restore lives in a `finally` and SIGKILL does not
    # run one. An interrupted sweep leaves the tree mutated and the next run's
    # baseline fails for a reason that looks like a real regression -- which is
    # exactly what happened, and cost a confusing minute.
    SENTINEL.write_text(f"{path}\n{name}\n", encoding="utf-8")
    try:
        subs(target, pairs)
    except RewriteError as exc:
        raise MutationError(f"{name}: the mutation did not apply -- {exc}") from exc
    try:
        tests, failures, output = run_suite(python)
        after = run_witness(witness, python) if witness else None
    finally:
        target.write_text(original, encoding="utf-8")
        SENTINEL.unlink(missing_ok=True)

    if tests != baseline:
        raise MutationError(
            f"{name}: the mutated run executed {tests} tests, the baseline "
            f"executed {int(baseline)}. A different set of tests is not "
            "evidence about this mutation. Refusing to report a number."
        )
    if failures == 0:
        if witness is None:
            raise MutationError(
                f"{name}: UNDETERMINED. The mutation applied and nothing "
                "failed, which means either the check it breaks is untested or "
                "the mutation changes no behaviour at all. Those are opposite "
                "findings and a failure count cannot tell them apart. Supply a "
                "witness -- a snippet that behaves differently under the "
                "mutant -- and this can say which."
            )
        if after == before:
            raise MutationError(
                f"{name}: EQUIVALENT, so this says nothing about coverage. The "
                "witness behaved identically with and without the mutation, so "
                "the mutation did not change what the code does. Write a "
                "stronger mutation, or accept that this line has no observable "
                f"effect to break.\n    witness gave: {before!r}"
            )
        raise MutationError(
            f"{name}: SURVIVED, and the witness proves it is a real gap. The "
            "mutation demonstrably changes behaviour and the whole suite stayed "
            f"green.\n    original: {before!r}\n    mutant:   {after!r}"
        )
    return failures


def main(argv: list[str]) -> int:
    python = argv[1] if len(argv) > 1 else None
    try:
        refuse_if_interrupted()
    except MutationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        base = green_baseline(python)
    except MutationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"baseline: {int(base)} tests, green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
