#!/usr/bin/env python3
"""Substitutions that cannot silently do nothing.

Every mechanical edit to this codebase goes through here. Not as a convenience
-- as the removal of a decision that kept being got wrong.

The habit it replaces
---------------------

A script would do ``s = s.replace(old, new)`` and write the file. When ``old``
had drifted by a space, the replace matched nothing, returned the string
unchanged, and the write succeeded. Nothing raised. The edit was reported as
done and was not done.

That happened at least six times here. The last was the worst: ``plan_id``
gained a parameter and kept passing the old argument list, so the fix looked
complete, the tests passed, and ``plan_id`` did not move when it should have.
It was caught by a measurement disagreeing with the code, not by the edit.

The fix that did not work
-------------------------

"Assert after every replacement." That is discipline, and discipline applied by
hand is applied unevenly: the very script that introduced the ``plan_id`` bug
asserted on two of its four substitutions. Remembering to guard is the thing
that fails.

So the guard moves into the only form available. ``sub`` always counts. There is
no unchecked path through this module, and reaching for bare ``str.replace``
becomes one visible choice rather than N invisible ones.

All or nothing
--------------

``subs`` applies a whole batch or writes nothing. A script that edits three
places and matches two used to leave the file half-changed and the failure
downstream, where it reads as a different bug.
"""

from __future__ import annotations

import sys
from pathlib import Path


class RewriteError(RuntimeError):
    """A substitution matched a different number of times than it claimed."""


def _check(path: Path, text: str, old: str, expected: int | None) -> int:
    found = text.count(old)
    if found == 0:
        raise RewriteError(
            f"{path}: no match for\n    {old[:200]!r}\n"
            "Nothing was written. The text has drifted, or it was never there."
        )
    if expected is not None and found != expected:
        raise RewriteError(
            f"{path}: expected {expected} occurrence(s) of\n    {old[:160]!r}\n"
            f"but found {found}. Refusing rather than guessing which you meant."
        )
    return found


def sub(path: str | Path, old: str, new: str, *, count: int | None = 1) -> int:
    """Replace ``old`` with ``new`` in ``path``. Raise if it matches nothing.

    ``count`` is what you expect to find, and it is checked. Pass ``None`` to
    accept any non-zero number -- which is still not zero, because zero is the
    failure this exists to catch.
    """
    return subs(path, [(old, new)], count=count)


def subs(path: str | Path, pairs, *, count: int | None = 1) -> int:
    """Apply every substitution, or none of them.

    Each pair is checked against the text as it stands after the previous one,
    so overlapping edits behave the way reading them top to bottom suggests.
    """
    target = Path(path)
    original = target.read_text(encoding="utf-8")
    text = original
    total = 0
    for old, new in pairs:
        total += _check(target, text, old, count)
        text = text.replace(old, new)
    if text == original:
        raise RewriteError(
            f"{target}: every substitution matched but the file is unchanged. "
            "The replacements are identical to what is already there."
        )
    target.write_text(text, encoding="utf-8")
    return total


def insert_before(path: str | Path, anchor: str, addition: str) -> int:
    """Put ``addition`` immediately before ``anchor``. Raise if absent.

    ``addition`` must NOT repeat the anchor -- this function adds it. Writing
    the anchor at the end of the addition duplicates it, which is a syntax
    error in the next command rather than a silent wrong answer, but it is a
    mistake made twice here in one afternoon and it is cheap to refuse.
    """
    if addition.rstrip().endswith(anchor.rstrip()):
        raise RewriteError(
            "the addition already ends with the anchor, so inserting would "
            "duplicate it. Drop the trailing copy: insert_before adds it.\n"
            f"    anchor: {anchor[:120]!r}"
        )
    return sub(path, anchor, addition + anchor)


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        print("usage: rewrite.py FILE OLD NEW", file=sys.stderr)
        return 2
    try:
        found = sub(argv[1], argv[2], argv[3])
    except RewriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{argv[1]}: {found} replacement(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
