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


def check(
    name: str,
    path: str,
    pairs: list[tuple[str, str]],
    *,
    baseline_tests: int,
    python: str | None = None,
) -> int:
    """Apply a mutation, run the suite, restore, and report how many tests died.

    Raises rather than returning zero when the run itself was not trustworthy:
    a suite that executed a different number of tests than the baseline is not
    evidence about the mutation, whichever direction it moved.
    """
    target = ROOT / path
    original = target.read_text(encoding="utf-8")
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
    finally:
        target.write_text(original, encoding="utf-8")
        SENTINEL.unlink(missing_ok=True)

    if tests != baseline_tests:
        raise MutationError(
            f"{name}: the mutated run executed {tests} tests, the baseline "
            f"executed {baseline_tests}. A different set of tests is not "
            "evidence about this mutation. Refusing to report a number."
        )
    if failures == 0:
        raise MutationError(
            f"{name}: the mutation applied and NOTHING FAILED. Either the "
            "check it breaks is not tested, or the mutation does not change "
            "behaviour. Both are findings; neither is a pass."
        )
    return failures


def main(argv: list[str]) -> int:
    python = argv[1] if len(argv) > 1 else None
    try:
        refuse_if_interrupted()
    except MutationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    tests, failures, _ = run_suite(python)
    if failures:
        print(f"the tree is not green ({failures} failing); fix before mutating",
              file=sys.stderr)
        return 1
    print(f"baseline: {tests} tests, green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
