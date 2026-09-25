#!/usr/bin/env python3
"""Every guard is exercised, recorded as untested, or marked unreachable.

A guard nobody has seen fire is a guard nobody knows works. This project has
shipped that exact defect: an error message asserting a guarantee about
``max_envs`` that the code did not provide, found by accident rather than by
review, because nothing had ever run the line.

Three states, and the point is that a guard must be in exactly one of them:

``exercised``
    A test reaches the ``raise``. Nothing to record.

``reachable-but-untested``
    No test reaches it, but some configuration would. Most guards are here and
    that is the honest label: invalid YAML, a missing lock file, a plan with no
    workers. Listed in the baseline below, which may shrink and may not grow.

``structurally unreachable``
    No configuration can reach it, and it is kept anyway as protection against
    a future change. Marked in the source with ``# unreachable: <reason>``.

Why a script and not a document
-------------------------------

A classification written down once rots the same way ``startup_sec: 25 # JAX
compilation, measured cold`` rotted: the comment outlived the code and nothing
was positioned to notice. Both directions are failures here and both are
detected:

* a guard marked unreachable that a test *does* reach means the marker is
  false, and a false "this cannot happen" is worse than no marker at all
* a new unreached guard that is neither marked nor in the baseline means a
  guard shipped without anyone deciding which of the three states it is in

The baseline is an exact set rather than a ceiling. Writing a test for a
guard therefore fails this check until the entry is removed, which is the
ratchet working: progress has to be recorded to be kept.

Keyed on ``file::function::exception`` and never on line numbers. A baseline
keyed on lines churns on every edit above it, and a baseline that churns is a
baseline somebody regenerates without reading.

Usage::

    coverage run --source=src/refractal -m pytest tests/
    coverage json -o coverage.json
    python scripts/lint_guards.py coverage.json
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

#: Guards no test reaches today, which some configuration could. Not a
#: shortcoming to be embarrassed about: it is the set of error paths whose
#: message has never been read by anyone, which is worth knowing precisely.
#: Remove an entry when you write the test. Adding one requires deciding,
#: deliberately, that a new error path ships unexercised.
BASELINE_FILE = pathlib.Path(__file__).parent / "guards-baseline.txt"

#: Not a guard: the module-level entry point, which pytest never executes
#: because it imports rather than runs the module.
SKIP_FUNCTIONS = {"<module>"}


def enclosing(tree: ast.AST) -> dict[int, str]:
    """Line number to the name of the function containing it."""
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                out[line] = node.name
    return out


def guards(root: pathlib.Path) -> dict[tuple[str, int], tuple[str, str, str]]:
    """Every ``raise`` in the tree, keyed by (file, line).

    Value is (key, exception name, marker reason or "").
    """
    found = {}
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        where = enclosing(tree)
        rel = str(path.relative_to(root.parent))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc
            name = ""
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                name = exc.func.id
            elif isinstance(exc, ast.Name):
                name = exc.id
            elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Attribute):
                name = exc.func.attr
            fn = where.get(node.lineno, "<module>")
            # `# unreachable: reason` on the raise line or the three above it.
            reason = ""
            for probe in range(max(0, node.lineno - 4), node.lineno):
                if probe < len(lines) and "# unreachable:" in lines[probe]:
                    reason = lines[probe].split("# unreachable:", 1)[1].strip()
            if "# unreachable:" in lines[node.lineno - 1]:
                reason = lines[node.lineno - 1].split("# unreachable:", 1)[1].strip()
            found[(rel, node.lineno)] = (f"{rel}::{fn}::{name}", name, reason)
    return found


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: lint_guards.py COVERAGE_JSON", file=sys.stderr)
        return 2
    cov_path = pathlib.Path(sys.argv[1])
    cov = json.loads(cov_path.read_text())
    root = pathlib.Path(__file__).parent.parent / "src" / "refractal"

    # Coverage records line NUMBERS, so editing a source file after the run
    # silently shifts every guard below the edit and the comparison becomes
    # nonsense. Found by mutation: inserting one line made this script report
    # two untouched guards in a different function as newly reached, with no
    # indication anything was wrong.
    #
    # A wrong answer delivered confidently is the failure mode this whole lint
    # exists to prevent, so refuse rather than compare.
    cov_mtime = cov_path.stat().st_mtime
    stale = [p for p in sorted(root.rglob("*.py")) if p.stat().st_mtime > cov_mtime]
    if stale:
        print(f"error: {len(stale)} source file(s) changed after {cov_path.name} "
              "was written, so its line numbers no longer match.", file=sys.stderr)
        for p in stale[:5]:
            print(f"  {p}", file=sys.stderr)
        print("Re-run coverage, then this script.", file=sys.stderr)
        return 2

    all_guards = guards(root)

    missed: dict[str, set[int]] = {}
    for fname, data in cov["files"].items():
        rel = str(pathlib.Path(fname))
        for suffix in ("src/", ""):
            if rel.startswith(suffix):
                rel = rel[len(suffix):]
                break
        missed[rel] = set(data["missing_lines"])

    exercised, untested, unreachable, lying = [], [], [], []
    for (rel, line), (key, _exc, reason) in sorted(all_guards.items()):
        fn = key.split("::")[1]
        if fn in SKIP_FUNCTIONS:
            continue
        was_missed = line in missed.get(rel, set())
        if reason and not was_missed:
            lying.append((key, reason))
        elif reason:
            unreachable.append((key, reason))
        elif was_missed:
            untested.append(key)
        else:
            exercised.append(key)

    baseline = set()
    if BASELINE_FILE.exists():
        baseline = {
            ln.split("#")[0].strip()
            for ln in BASELINE_FILE.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }

    now = set(untested)
    added, removed = now - baseline, baseline - now
    total = len(exercised) + len(untested) + len(unreachable)
    print(f"  {total} guard(s) in src/refractal")
    print(f"    exercised by tests      : {len(exercised)}")
    print(f"    reachable but untested  : {len(untested)}")
    print(f"    structurally unreachable: {len(unreachable)}")

    bad = False
    if lying:
        bad = True
        print(f"\n  {len(lying)} guard(s) marked unreachable that a test DOES reach.")
        print("  The marker is false, which is worse than no marker.")
        for key, reason in lying:
            print(f"    {key}\n      marked: {reason}")
    if added:
        bad = True
        print(f"\n  {len(added)} new unreached guard(s), neither marked nor in the baseline.")
        print("  Decide which state each is in: write a test, or mark it")
        print("  `# unreachable: <reason>`, or add it to scripts/guards-baseline.txt.")
        for key in sorted(added):
            print(f"    {key}")
    if removed:
        bad = True
        print(f"\n  {len(removed)} baseline entry(s) no longer unreached.")
        print("  If you wrote the test, delete these lines: the baseline records")
        print("  what is untested, and a stale entry understates the coverage.")
        for key in sorted(removed):
            print(f"    {key}")
    if not bad:
        print("\n  Every guard is exercised, recorded, or marked.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
