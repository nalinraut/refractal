#!/usr/bin/env python3
"""Null the 360 rows whose termination_reason is a phase name.

FakeProvider wrote `self.phases[reached]` -- how far the episode got -- into a
column documented as "the reason, where success is the verdict". That is a
provider writing the wrong semantic into a documented column, which is a
defect rather than a second legitimate use, and it is fixed at the source.

The rows it already wrote are a separate question. They are not rewritten into
the compat layer: 360 synthetic rows are not worth a permanent special case in
every future read. They are migrated once, here, and the value removed is pure
duplication -- `phase_outcomes` carries the same information, verified per-row
rather than in aggregate before anything is written.

REFUSES rather than inferring. A row whose phase-name value is not a key in
its own `phase_outcomes` is not duplication, and this script stops instead of
guessing what it was.

Rewrites part files in place, preserving every other column. Idempotent: a
second run finds nothing to do.
"""

import json
import pathlib
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else "/home/nalin/Develop/refractal-runs")
#: The only vocabulary this affects. Deliberately a literal list rather than
#: "anything that is also a phase_outcomes key": a real termination flag that
#: happened to share a name with a phase would otherwise be silently erased.
PHASE_NAMES = {"reach", "grasp", "transport", "place"}
COLUMN = "terminal_phase"          # pre-rename name, which is what is on disk


def main() -> int:
    files = sorted(ROOT.rglob("episodes/part-*.parquet"))
    touched = rows_nulled = 0
    refusals = []

    for path in files:
        table = pq.read_table(path)
        if COLUMN not in table.column_names:
            continue
        values = table.column(COLUMN).to_pylist()
        if not any(v in PHASE_NAMES for v in values):
            continue

        outcomes = table.column("phase_outcomes").to_pylist()
        new = []
        for value, po in zip(values, outcomes):
            if value not in PHASE_NAMES:
                new.append(value)
                continue
            keys = {k for k, _ in (po or [])} if isinstance(po, list) else set(po or {})
            if value not in keys:
                refusals.append(f"{path}: {value!r} is not a phase_outcomes key")
                new.append(value)
                continue
            new.append(None)

        if refusals:
            break
        idx = table.column_names.index(COLUMN)
        table = table.set_column(idx, COLUMN, pa.array(new, type=pa.string()))
        pq.write_table(table, path)
        touched += 1
        rows_nulled += sum(1 for a, b in zip(values, new) if a != b)

    if refusals:
        print(f"  REFUSED, {len(refusals)} row(s) are not duplication:", file=sys.stderr)
        for r in refusals[:5]:
            print(f"    {r}", file=sys.stderr)
        return 1

    print(f"  {len(files)} part files scanned, {touched} rewritten, "
          f"{rows_nulled} rows nulled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
