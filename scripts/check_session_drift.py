#!/usr/bin/env python3
"""Does within-session drift exist at the timescale a run takes?

`interleaved` exists to remove it. If a run of this length has no measurable
drift, interleaved pays invocation overhead for nothing and should not be
anyone's default -- so the mode should not be built before this is answered.

Three questions, and they are not the same question.

**1. Does replicate POSITION predict outcome, pooled across tasks?** Not "is each
task's sequence monotone" -- non-increasing across three points happens by
accident often enough that two tasks agreeing is two observations, not six. Ten
tasks at three replicate positions is still three positions, so the test is a
rank correlation between position and cell rate, pooled, with a permutation
p-value over task-internal relabelings.

**2. Is it DIFFERENTIAL?** "Drift exists in pi0" and "drift affects the arms
differently" are different claims and only the second justifies `interleaved`. So
the same test on the per-replicate contrast.

**3. Can question 2 even be answered?** If one arm sits at the ceiling on most
tasks, its cells have no room to move, the contrast is `1 - other_arm` by
arithmetic, and the differential test is not independent evidence -- it is
question 1 wearing different units. The ceiling count is printed first because it
decides whether the rest means anything.

A further caution the ordering supplies for free. The current loop is
checkpoint-outer with checkpoints sorted, so pi0 runs entirely before pi05.
Monotone session-wide degradation therefore predicts the LATER arm is worse. If
the earlier arm declines within its own block and the later arm does not, that is
evidence against session-wide drift and for something arm-local -- an
accumulating server, not a warming machine.

Sample sizes are printed with everything, because a null here is weak evidence
for absence rather than evidence of it.

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


def _permutation_p(xs, ys, groups, observed, rounds: int = 2000) -> float:
    """Two-sided p by shuffling replicate labels *within each task*.

    Within-task because that is what isolates position: task difficulty is held
    fixed by construction, so the only thing the shuffle destroys is the
    position-outcome association. A global shuffle would also destroy the
    task structure and answer a question nobody asked.

    Deterministic PRNG, seeded, because a script whose verdict wobbles between
    runs is not a check.
    """
    import random
    from collections import defaultdict as dd

    index_by_group = dd(list)
    for i, g in enumerate(groups):
        index_by_group[g].append(i)

    rng = random.Random(0)
    extreme = 0
    for _ in range(rounds):
        shuffled = list(xs)
        for indices in index_by_group.values():
            values = [xs[i] for i in indices]
            rng.shuffle(values)
            for i, v in zip(indices, values):
                shuffled[i] = v
        if abs(spearman(shuffled, ys)) >= abs(observed) - 1e-12:
            extreme += 1
    return extreme / rounds


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    from refractal.execute.results import read_episodes

    rows = read_episodes(argv[1], argv[2]).to_pylist()
    if not rows:
        print(f"no episodes under {argv[1]} for {argv[2]}", file=sys.stderr)
        return 2

    # --- test 0: is either arm at a ceiling, and on how many tasks? --------
    per_task: dict[tuple, list[bool]] = defaultdict(list)
    for r in rows:
        per_task[(r["task_hash"], r["checkpoint_id"])].append(r["success"])
    arms_all = sorted({r["checkpoint_id"] for r in rows})
    tasks_all = sorted({t for (t, _) in per_task})
    print(f"  {len(rows)} episodes, {len(tasks_all)} tasks, arms {arms_all}\n")
    print("  0. Ceiling census -- decides whether the differential test can speak")
    pinned: dict[str, int] = {}
    for arm in arms_all:
        n = sum(
            1 for t in tasks_all
            if (t, arm) in per_task
            and (all(per_task[(t, arm)]) or not any(per_task[(t, arm)]))
        )
        pinned[arm] = n
        print(f"     {arm:8} pinned at 0% or 100% on {n}/{len(tasks_all)} tasks")
    print()

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

    print("\n     Pooled rank correlation, per arm. Permutation p-value shuffles")
    print("     replicate labels WITHIN each task, so task difficulty cannot")
    print("     contribute -- only position can.\n")
    for arm in arms_all:
        xs, ys, groups = [], [], []
        for (task, checkpoint, seed), outcomes in cells.items():
            if checkpoint != arm:
                continue
            xs.append(float(seed))
            ys.append(sum(outcomes) / len(outcomes))
            groups.append(task)
        if len(set(ys)) <= 1:
            print(f"     {arm:8} every cell identical ({ys[0]:.0%}) -- no variance, "
                  "position cannot predict anything")
            continue
        rho = spearman(xs, ys)
        p = _permutation_p(xs, ys, groups, rho)
        print(f"     {arm:8} Spearman = {rho:+.3f}  permutation p = {p:.3f}  "
              f"(n={len(xs)} cells, {len(set(groups))} tasks)")

    # --- test 2: is the CONTRAST trending? the differential claim -----------
    print("\n  2. Differential drift -- the claim that justifies `interleaved`")
    if len(arms_all) != 2:
        print(f"     skipped: needs exactly two arms, found {arms_all}")
    else:
        a, b = arms_all
        xs, ys, groups = [], [], []
        for task in tasks_all:
            for seed in seeds:
                ka, kb = (task, a, seed), (task, b, seed)
                if ka not in cells or kb not in cells:
                    continue
                ra = sum(cells[ka]) / len(cells[ka])
                rb = sum(cells[kb]) / len(cells[kb])
                xs.append(float(seed))
                ys.append(rb - ra)
                groups.append(task)
        if len(set(ys)) <= 1:
            print("     the contrast is constant -- nothing to trend")
        else:
            rho = spearman(xs, ys)
            p = _permutation_p(xs, ys, groups, rho)
            print(f"     Spearman(replicate index, {b} - {a}) = {rho:+.3f}  "
                  f"permutation p = {p:.3f}  (n={len(xs)})")
        # The honest version: only tasks where BOTH arms have room to move.
        # On a task where one arm is pinned, the contrast is `1 - other_arm` by
        # arithmetic and carries no information test 1 does not already have.
        # Restricting is underpowered rather than uninformative, and the two are
        # worth distinguishing.
        interior = [
            t for t in tasks_all
            if all(
                (t, arm) in per_task
                and any(per_task[(t, arm)])
                and not all(per_task[(t, arm)])
                for arm in arms_all
            )
        ]
        print(f"\n     Tasks where BOTH arms are interior: {len(interior)}/{len(tasks_all)}")
        if not interior:
            print("     None. The differential claim is not measurable from this run at "
                  "all:\n     on every task one arm is pinned, so the contrast restates "
                  "test 1.")
        else:
            xs, ys, groups = [], [], []
            for task in interior:
                for seed in seeds:
                    ka, kb = (task, a, seed), (task, b, seed)
                    if ka not in cells or kb not in cells:
                        continue
                    xs.append(float(seed))
                    ys.append(sum(cells[kb]) / len(cells[kb])
                              - sum(cells[ka]) / len(cells[ka]))
                    groups.append(task)
            if len(set(ys)) <= 1:
                print("     contrast constant on those tasks -- nothing to trend")
            else:
                rho = spearman(xs, ys)
                pv = _permutation_p(xs, ys, groups, rho)
                print(f"     Spearman on those only = {rho:+.3f}  permutation p = {pv:.3f}  "
                      f"(n={len(xs)} cells, {len(interior)} tasks)")
            print("     This subset IS independent evidence. It is also small, so a null")
            print("     here is weak. Both statements are true at once.")

        if max(pinned.values()) > len(tasks_all) / 2:
            worst = max(pinned, key=lambda k: pinned[k])
            print(f"\n     CEILING WARNING: {worst} is pinned on "
                  f"{pinned[worst]}/{len(tasks_all)} tasks. The all-task number above is")
            print("     mostly test 1 in different units. Read the both-interior subset,")
            print("     and read the ceiling itself as the finding: this suite is too easy")
            print(f"     for {worst}, so the comparison wants a harder one rather than "
                  "more scenarios.")

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
    print("\n  3. Contrast versus session position -- confounded by task difficulty")
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
