#!/usr/bin/env python3
"""Does the wheel contain everything the source tree has?

Derived, not listed. The previous version of this check was a hardcoded list of
five subpackages; `render` became a sixth and the assertion was not updated, so
it would have **failed on a correct wheel**. A stale allow-list fails safe, which
is the right direction, but it still has to be maintained by hand and it was not.

Comparing against the source tree needs no maintenance and catches the failure it
was written for. That failure was real: an unanchored `build/` line in
`.gitignore` matched `src/refractal/build/`, so an entire subpackage was never
committed and the wheel shipped five of six while `git status` stayed clean.

What this does NOT check is whether the source tree is what you think it is --
step 1 of the checklist is `git status --short`, and it is first for this reason.
A wheel that faithfully contains an incomplete checkout passes here.

Usage::

    python scripts/verify_wheel.py dist/refractal-0.1.0a1-py3-none-any.whl
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "refractal"


def source_subpackages() -> set[str]:
    src = ROOT / "src" / PACKAGE
    return {
        d.name
        for d in src.iterdir()
        if d.is_dir() and (d / "__init__.py").is_file() and not d.name.startswith("_")
    }


def wheel_subpackages(path: Path) -> set[str]:
    names = zipfile.ZipFile(path).namelist()
    prefix = f"{PACKAGE}/"
    return {
        name[len(prefix):].split("/", 1)[0]
        for name in names
        if name.startswith(prefix) and "/" in name[len(prefix):]
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"no such wheel: {path}", file=sys.stderr)
        return 2

    expected = source_subpackages()
    found = wheel_subpackages(path)
    missing = sorted(expected - found)
    extra = sorted(found - expected)

    print(f"  {path.name}")
    print(f"  source has {len(expected)}: {sorted(expected)}")
    print(f"  wheel has  {len(found)}: {sorted(found)}")

    if missing:
        print(
            f"\n  MISSING FROM THE WHEEL: {missing}\n"
            "  The source tree has these and the artifact does not. Check .gitignore "
            "for an unanchored directory pattern -- `build/` matches "
            "src/refractal/build/ as well as ./build/, and the result is a wheel "
            "short an entire subpackage while `git status` stays clean.",
            file=sys.stderr,
        )
    if extra:
        print(
            f"\n  IN THE WHEEL AND NOT THE SOURCE: {extra}\n"
            "  Almost certainly a stale dist/. Step 2 of the checklist says delete "
            "and rebuild rather than reuse, for exactly this.",
            file=sys.stderr,
        )
    if missing or extra:
        return 1

    # Not just the packages: the CLI has to be reachable from the artifact.
    names = set(zipfile.ZipFile(path).namelist())
    for required in (f"{PACKAGE}/cli.py", f"{PACKAGE}/__init__.py"):
        if required not in names:
            print(f"\n  MISSING: {required}", file=sys.stderr)
            return 1

    print(f"\n  Wheel matches the source tree ({len(expected)} subpackages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
