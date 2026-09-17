#!/usr/bin/env python3
"""Does within-session drift exist at the timescale a run takes?

`interleaved` exists to remove it. If a run of this length has no measurable
drift, interleaved pays invocation overhead for nothing and should not be
anyone's default -- so the mode should not be built before this is answered.

Two tests, because they probe different timescales and only one is clean.

**1. Replicate trend (clean).** Within one (task, checkpoint) group, the three
replicates ran back to back, minutes apart, on identical init states with
identical everything. Any trend across replicate index is drift, because nothing
else varies. This is the test with no confound.

**2. Contrast versus session position (confounded, reported anyway).** For each
task, the paired difference against when in the session it ran. Task difficulty
varies and is not controlled, so a trend here is suggestive rather than
conclusive -- but a *flat* result here alongside a flat result above is a
stronger statement than either alone.

Both are reported with their sample sizes, because ten tasks and three replicates
is not much and a null from a small sample is weak evidence for absence. Saying
"no drift detected at n=10" is different from "no drift", and only the first is
available here.

Usage::

    python scripts/check_session_drift.py RESULTS_URI PLAN_ID
"""

from __future__ import annotations

import sys
from collections import defaultdict


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, no scipy. Ties averaged."""
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    from refractal.execute.results import read_episodes

    rows = read_episodes(argv[1], argv[2]).to_pylist()
    if not rows:
        print(f"no episodes under {argv[1]} for {argv[2]}", file=sys.stderr)
        return 2

    # --- test 1: does a checkpoint's rate trend across its own replicates? ---
    cells: dict[tuple, list[bool]] = defaultdict(list)
    for r in rows:
        cells[(r["task_hash"], r["checkpoint_id"], r["seed"])].append(r["success"])

    by_seed: dict[int, list[float]] = defaultdict(list)
    per_arm: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (task, checkpoint, seed), outcomes in cells.items():
        rate = sum(outcomes) / len(outcomes)
        by_seed[seed].append(rate)
        per_arm[checkpoint][seed].append(rate)

    print(f"  {len(rows)} episodes, {len(cells)} (task, checkpoint, replicate) cells\n")
    print("  1. Replicate trend -- the clean test")
    print("     Same task, same checkpoint, same init states, minutes apart.")
    print("     Nothing varies but time, so any trend is drift.\n")
    seeds = sorted(by_seed)
    print(f"     {'arm':8} " + "  ".join(f"rep{s}" for s in seeds) + "   spread")
    for arm in sorted(per_arm):
        means = [sum(per_arm[arm][s]) / len(per_arm[arm][s]) for s in seeds]
        print(f"     {arm:8} " + "  ".join(f"{m:5.1%}" for m in means)
              + f"   {max(means) - min(means):+.1%}")
    pooled = [sum(by_seed[s]) / len(by_seed[s]) for s in seeds]
    print(f"     {'pooled':8} " + "  ".join(f"{m:5.1%}" for m in pooled)
          + f"   {max(pooled) - min(pooled):+.1%}")

    xs = [float(s) for (_, _, s) in cells for _ in (0,)]
    xs, ys = [], []
    for (task, checkpoint, seed), outcomes in cells.items():
        xs.append(float(seed))
        ys.append(sum(outcomes) / len(outcomes))
    rho = spearman(xs, ys)
    print(f"\n     Spearman(replicate index, cell rate) = {rho:+.3f}  (n={len(xs)} cells)")

    # --- test 2: does the contrast trend with session position? --------------
    starts: dict[tuple, object] = {}
    for r in rows:
        key = (r["task_hash"], r["checkpoint_id"])
        if r["started_at"] is not None:
            starts[key] = min(starts.get(key, r["started_at"]), r["started_at"])

    task_rates: dict[tuple, list[bool]] = defaultdict(list)
    for r in rows:
        task_rates[(r["task_hash"], r["checkpoint_id"])].append(r["success"])

    arms = sorted({r["checkpoint_id"] for r in rows})
    print("\n  2. Contrast versus session position -- confounded by task difficulty")
    if len(arms) != 2:
        print(f"     skipped: needs exactly two arms, found {arms}")
    else:
        a, b = arms
        points = []
        for task in {t for (t, _) in task_rates}:
            if (task, a) not in starts or (task, b) not in starts:
                continue
            ra = sum(task_rates[(task, a)]) / len(task_rates[(task, a)])
            rb = sum(task_rates[(task, b)]) / len(task_rates[(task, b)])
            points.append((min(starts[(task, a)], starts[(task, b)]), rb - ra, task))
        points.sort()
        print(f"     {'order':6} {'contrast':>10}   ({b} - {a})")
        for i, (_, diff, _) in enumerate(points):
            print(f"     {i:<6} {diff:+10.1%}")
        if len(points) >= 3:
            rho2 = spearman([float(i) for i in range(len(points))],
                            [d for _, d, _ in points])
            print(f"\n     Spearman(session order, contrast) = {rho2:+.3f}  (n={len(points)} tasks)")
            print("     Task difficulty is not controlled here, so read this as "
                  "corroboration\n     of test 1 rather than as evidence on its own.")

    print("\n  A null at these sample sizes is weak evidence for absence, not proof of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
