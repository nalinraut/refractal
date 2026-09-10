"""Turning episode rows into paired units, and deciding what is comparable.

The comparison unit is ``(scene_id, task_hash, scenario_hash)``. Every
checkpoint faces the same scenarios by construction -- that is a property of the
plan, not a hope -- so the data is *paired*, and paired data needs a paired test.

The rule this module exists for
-------------------------------

Infra failures are excluded from success-rate denominators. That is right, and
it breaks the pairing, which the spec never addressed: if ckpt-46's episode at
seed 1 crashed and ckpt-47's did not, the scenario no longer has a matched pair
at that seed. McNemar needs both members.

Three ways out, and they are not equivalent:

1. **Drop the seed across every checkpoint.** Pairing stays exact. Costs one
   seed on that scenario and leaves scenarios with unequal seed counts.
2. **Drop the whole scenario if any checkpoint errored on any seed.** Clean and
   uniform, and far more expensive: a scenario survives only if all ``s * k``
   of its episodes are clean, so the loss is ``1 - (1-p)^(s*k)`` -- at p=3%,
   5 seeds and 2 checkpoints that is 26% of scenarios, against 5.9% of seed
   slots under (1).
3. **Keep it with unequal seeds.** Cheapest, and it makes the per-scenario pass
   rate a proportion over differing denominators -- exactly what bites in the
   clustered bootstrap.

**Decision: (1), with the surviving seed count recorded per unit and a floor
below which the unit is refused.** The pairing stays exact, and the loss is
visible in the report rather than silent -- the same principle as the overlap
counts. The floor matters because the dichotomisation rule is "majority", and a
majority of two is not the same test as a majority of three.

Cost, so nobody has to guess how many checkpoints they can afford: a seed slot
survives only if all ``k`` checkpoints were clean, so expected loss is
``1 - (1-p)^k``, which is about ``k*p`` for small ``p``. Linear in checkpoint
count -- 5.9% at k=2 and 11.5% at k=4 for a 3% infra rate. It only hurts when
the infra rate is itself bad (19% loss at k=2 when p=10%), which is the right
incentive.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

Dichotomy = Literal["majority", "all"]


@dataclass(frozen=True)
class UnitKey:
    scene_id: str
    task_id: str
    task_hash: str
    scenario_hash: str

    def as_tuple(self) -> tuple[str, str, str]:
        # `task_id` is a label and deliberately not part of the key: renaming a
        # task must not break a join, changing its meaning must.
        return (self.scene_id, self.task_hash, self.scenario_hash)


@dataclass
class Unit:
    """One scenario faced by every checkpoint, after the pairing rule."""

    key: UnitKey
    #: checkpoint id -> {seed: success}, restricted to seeds every checkpoint kept
    outcomes: dict[str, dict[int, bool]]
    seeds_kept: list[int]
    seeds_dropped: list[int]

    def rate(self, checkpoint_id: str) -> float:
        seeds = self.outcomes[checkpoint_id]
        return sum(seeds.values()) / len(seeds)

    def passed(self, checkpoint_id: str, rule: Dichotomy) -> bool:
        """Collapse several seeds into the one bit McNemar consumes.

        Five attempts at one pose separate "is this scenario hard" from "did the
        policy get lucky": 5/5 solved, 0/5 broken, 3/5 on the capability
        boundary. Which way the boundary is called is a choice, so it is
        explicit and configurable rather than assumed.
        """
        seeds = self.outcomes[checkpoint_id]
        if rule == "all":
            return all(seeds.values())
        return sum(seeds.values()) * 2 > len(seeds)


@dataclass
class Eligibility:
    """What was compared, what was not, and why. Printed above the result."""

    units: list[Unit] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    scenarios_in_all: int = 0
    only_in: dict[str, int] = field(default_factory=dict)
    units_dropped_below_floor: int = 0
    #: Two counters, because one number here is ambiguous and the ambiguity is
    #: the kind that survives review. A *seed slot* is one (unit, seed) pair --
    #: what the pairing rule actually removes. Removing one costs ``k`` episodes,
    #: one per checkpoint, so the two differ by a factor of ``len(checkpoints)``
    #: and a reader cannot tell which they are looking at from the number alone.
    seed_slots_dropped: int = 0
    episodes_dropped_to_preserve_pairing: int = 0
    episodes_excluded_infra: int = 0
    min_seeds: int = 0
    total_seed_slots: int = 0
    seed_counts: dict[int, int] = field(default_factory=dict)
    sessions: set[str] = field(default_factory=set)
    harness_versions: set[str] = field(default_factory=set)
    harness_surfaces: set[str] = field(default_factory=set)
    scene_hash_conflicts: dict[str, set[str]] = field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Scenarios in all {len(self.checkpoints)} checkpoints: {self.scenarios_in_all}"
        ]
        for checkpoint_id in self.checkpoints:
            count = self.only_in.get(checkpoint_id, 0)
            if count:
                lines.append(f"Only in {checkpoint_id}: {count}")
        if self.episodes_excluded_infra:
            lines.append(
                f"Episodes excluded as infra failures: {self.episodes_excluded_infra}"
            )
        if self.seed_slots_dropped:
            share = ""
            if self.total_seed_slots:
                share = f" ({self.seed_slots_dropped / self.total_seed_slots:.1%} of slots)"
            lines.append(
                f"Seed slots dropped to keep pairs matched: {self.seed_slots_dropped}"
                f"{share}, removing {self.episodes_dropped_to_preserve_pairing} episodes "
                f"across {len(self.checkpoints)} checkpoints"
            )
        if self.units_dropped_below_floor:
            lines.append(
                f"Scenarios refused for having fewer than {self.min_seeds} usable seeds: "
                f"{self.units_dropped_below_floor}"
            )
        if len(self.seed_counts) > 1:
            spread = ", ".join(f"{n} seeds: {c}" for n, c in sorted(self.seed_counts.items()))
            lines.append(f"Scenarios have unequal seed counts ({spread})")
        lines.append(f"Statistics computed on the {len(self.units)} scenarios in common.")
        return lines


def build_units(
    rows: Iterable[Mapping[str, Any]],
    *,
    checkpoints: Sequence[str] | None = None,
    min_seeds: int | None = None,
) -> Eligibility:
    """Group episode rows into paired units, applying the pairing rule.

    ``min_seeds`` defaults to half the planned seed count, rounded up, floored
    at one -- so a 3-seed run tolerates losing one seed and a 1-seed run
    tolerates nothing, which is the honest reading of "1 of 1 survived".
    """
    rows = list(rows)
    if checkpoints is None:
        checkpoints = sorted({r["checkpoint_id"] for r in rows})
    checkpoints = list(checkpoints)

    grouped: dict[tuple[str, str, str], dict[str, dict[int, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    keys: dict[tuple[str, str, str], UnitKey] = {}
    sessions: set[str] = set()
    harness_versions: set[str] = set()
    harness_surfaces: set[str] = set()
    scene_hashes: dict[str, set[str]] = defaultdict(set)
    infra = 0

    for row in rows:
        key = UnitKey(
            scene_id=row["scene_id"],
            task_id=row["task_id"],
            task_hash=row["task_hash"],
            scenario_hash=row["scenario_hash"],
        )
        keys.setdefault(key.as_tuple(), key)
        sessions.add(row["session_id"])
        if row.get("harness_version"):
            harness_versions.add(row["harness_version"])
        if row.get("harness_surface"):
            harness_surfaces.add(row["harness_surface"])
        scene_hashes[row["scene_id"]].add(row["scene_hash"])
        if row["is_infra_failure"]:
            infra += 1
        grouped[key.as_tuple()][row["checkpoint_id"]][row["seed"]] = row

    planned_seeds = max((len(s) for cps in grouped.values() for s in cps.values()), default=0)
    if min_seeds is None:
        min_seeds = max(1, math.ceil(planned_seeds / 2))

    eligibility = Eligibility(
        checkpoints=checkpoints,
        min_seeds=min_seeds,
        episodes_excluded_infra=infra,
        sessions=sessions,
        harness_versions=harness_versions,
        harness_surfaces=harness_surfaces,
        scene_hash_conflicts={
            scene: hashes for scene, hashes in scene_hashes.items() if len(hashes) > 1
        },
    )

    for tuple_key, per_checkpoint in grouped.items():
        present = [c for c in checkpoints if c in per_checkpoint]
        if len(present) < len(checkpoints):
            for checkpoint_id in present:
                eligibility.only_in[checkpoint_id] = (
                    eligibility.only_in.get(checkpoint_id, 0) + 1
                )
            continue
        eligibility.scenarios_in_all += 1

        all_seeds = sorted({s for seeds in per_checkpoint.values() for s in seeds})
        # THE RULE: a seed survives only if every checkpoint has a clean episode
        # for it. Otherwise the pair is incomplete and McNemar has nothing to
        # read, so it goes -- from every checkpoint, not just the failed one.
        kept = [
            seed
            for seed in all_seeds
            if all(
                seed in per_checkpoint[c] and not per_checkpoint[c][seed]["is_infra_failure"]
                for c in checkpoints
            )
        ]
        dropped = [s for s in all_seeds if s not in kept]
        eligibility.total_seed_slots += len(all_seeds)
        eligibility.seed_slots_dropped += len(dropped)
        eligibility.episodes_dropped_to_preserve_pairing += len(dropped) * len(checkpoints)

        if len(kept) < min_seeds:
            eligibility.units_dropped_below_floor += 1
            continue

        eligibility.units.append(
            Unit(
                key=keys[tuple_key],
                outcomes={
                    c: {seed: bool(per_checkpoint[c][seed]["success"]) for seed in kept}
                    for c in checkpoints
                },
                seeds_kept=kept,
                seeds_dropped=dropped,
            )
        )
        eligibility.seed_counts[len(kept)] = eligibility.seed_counts.get(len(kept), 0) + 1

    eligibility.units.sort(key=lambda u: (u.key.scene_id, u.key.task_id, u.key.scenario_hash))
    return eligibility


__all__ = ["Dichotomy", "Eligibility", "Unit", "UnitKey", "build_units"]
