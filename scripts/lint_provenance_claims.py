#!/usr/bin/env python3
"""Refuse prose claims about provenance in a schema that records provenance.

`ResourceShape` carries `measured_at`, set by `refractal build` only when a
prober actually measured the shape:

    shape.model_copy(update={"measured_at": built_at if source == "measured" else None})

So the file already answers "was this measured". A comment that *also* answers it
creates a conflict the file cannot resolve — and in the one case that was caught,
the prose won:

    startup_sec: 25           # JAX compilation, measured cold

`measured_at` was absent, meaning declared. The comment said measured. The real
value was 2.6 seconds, the measurement had never happened, and the planner spent
its life reasoning about that scene with a number ten times too large. It
survived because "measured" is the word that stops anyone checking.

WHAT THIS DOES AND DOES NOT FLAG
--------------------------------

It does **not** object to recording what a measurement found. A comment holding a
table of throughput numbers is the most valuable thing in a catalog, and deleting
it would be the wrong lesson.

It objects to a comment **asserting that a value was obtained by measurement**
while the field that records that is empty. Fix it in whichever direction is
true: set `measured_at` if it really was measured, or drop the claim if it was
not.

Usage::

    python scripts/lint_provenance_claims.py CATALOG [CATALOG ...]

Exit 0 if every claim is backed, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Words that assert HOW a value was obtained, as opposed to what it is.
CLAIMS = re.compile(
    r"\b(measured|benchmarked|profiled|timed|observed on|measured at)\b", re.I
)

#: A comment is exempt when it is explicitly disclaiming rather than claiming.
DISCLAIMERS = re.compile(
    r"\b(not measured|never measured|unmeasured|declared|estimate[ds]?|guess\w*|"
    r"asserted and wrong|was not measured)\b",
    re.I,
)


def _shape_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Line ranges of `resource_shape:` blocks, by indentation.

    Scoped to those blocks deliberately. The first version of this check matched
    `measured` anywhere in any YAML and flagged ten things, of which one was
    real -- comments about a BDDL diff, a GPU's true VRAM, and a quaternion
    convention, none of which are claims about a resource shape field. That is
    the fire-on-a-class mistake this project already has an entry about: a check
    that observes a symptom shared by the cause and by everything else.
    """
    blocks = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("resource_shape:"):
            continue
        indent = len(line) - len(line.lstrip())
        end = len(lines)
        for later in range(index + 1, len(lines)):
            text = lines[later]
            if text.strip() and not text.strip().startswith("#"):
                if len(text) - len(text.lstrip()) <= indent:
                    end = later
                    break
        blocks.append((index, end))
    return blocks


def lint(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    problems = []
    for start, end in _shape_blocks(lines):
        body = "\n".join(lines[start:end])
        if re.search(r"^\s*measured_at\s*:", body, re.M):
            continue          # the shape says how it was obtained; prose may agree
        for offset, line in enumerate(lines[start:end], start + 1):
            comment = line.split("#", 1)[1] if "#" in line else ""
            if not comment:
                continue
            if CLAIMS.search(comment) and not DISCLAIMERS.search(comment):
                problems.append(f"{path}:{offset}: {comment.strip()[:78]}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    problems: list[str] = []
    checked = 0
    for root in argv[1:]:
        for path in sorted(Path(root).rglob("*.yaml")):
            checked += 1
            problems += lint(path)

    if not problems:
        print(f"  {checked} file(s): no unbacked provenance claims.")
        return 0

    print(f"  {len(problems)} prose claim(s) of measurement in file(s) with no "
          f"`measured_at`:\n", file=sys.stderr)
    for problem in problems:
        print(f"    {problem}", file=sys.stderr)
    print(
        "\n  The schema records provenance in `measured_at`, which `refractal build`\n"
        "  sets only when a prober actually measured the shape. A comment claiming\n"
        "  measurement while that field is empty is a conflict the file cannot\n"
        "  resolve, and prose wins in the reader's head.\n\n"
        "  Fix in whichever direction is true: set `measured_at`, or drop the claim.\n"
        "  Recording WHAT a measurement found is fine and wanted -- it is asserting\n"
        "  THAT one happened which has to be backed.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
