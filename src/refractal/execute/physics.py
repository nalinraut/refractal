"""The physics surface: what the engine will actually do, recorded per episode.

``harness_surface`` exists because the harness sits between Refractal and the
run, and can change behaviour without changing anything Refractal hashes. So it
is digested at runtime, recorded on every row, and gates ``compare``.

robosuite is the same kind of layer, one level down. It sits between MuJoCo and
the benchmark and owns the robot model -- actuator gains, force ranges, friction
-- none of which appear in a BDDL file, a suite name, or an engine version. A
release that changes a gripper's ``forcerange`` changes the physics of every
episode and moves no identity in this system.

That is tolerable while nothing depends on the number. It stops being tolerable
the moment a torque-margin sweep exists, because the sweep *is* a curve over
that number: a silent change to the baseline does not perturb the results, it
rewrites the axis.

Why this does not go in ``plan_id``
-----------------------------------

The same reason ``harness_surface`` does not. Folding it into identity means
every robosuite bump orphans every result recorded before it, whether or not the
bump changed anything that matters. A precondition instead: recorded per
episode, and blocking a comparison whose rows disagree.

Facts, not a version string
---------------------------

The digest is over the values, not over ``robosuite.__version__``. A release
that bumps the version and leaves the model alone must not block a comparison,
and one that edits a ``forcerange`` in a patch release must. The version is
recorded beside it as provenance -- human-readable, never gating -- exactly as
``harness_version`` sits beside ``harness_surface``.

Duck-typed on purpose
---------------------

Nothing here imports MuJoCo, robosuite or numpy. It reads five attributes off
whatever it is handed, so it is testable with a dictionary and stays usable for
any engine that can describe its actuators the same way.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

__all__ = [
    "ABSENT",
    "actuator_facts",
    "compare_facts",
    "physics_digest",
    "physics_manifest",
]

#: What a row carries when nothing could be read. Distinct from a digest of an
#: empty model, which would collide with it and claim a fact nobody established.
ABSENT = "absent"


def actuator_facts(model: Any) -> dict[str, dict[str, Any]]:
    """Every actuator, and the limit that decides what a perturbation can do.

    ``force_limited`` is read from the engine's own flag, never inferred from
    the range. MuJoCo allows an actuator declared limited whose range happens to
    be ``[0, 0]``, and inferring would call it unlimited -- a second
    implementation of the engine's rule, which is how these things go wrong.
    """
    count = int(getattr(model, "nu", 0) or 0)
    if not count:
        return {}
    limited = model.actuator_forcelimited
    ranges = model.actuator_forcerange
    facts: dict[str, dict[str, Any]] = {}
    for index in range(count):
        name = model.actuator_id2name(index)
        if not name:
            continue
        low, high = (float(v) for v in ranges[index])
        facts[str(name)] = {
            "force_limited": bool(limited[index]),
            "forcerange": [low, high],
        }
    return facts


def physics_manifest(facts: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Per-actuator lines, so a changed digest can be *named*.

    Same reasoning as the harness manifest: the digest answers "did something
    move" and leaves whoever reads a comparison months later to find out what,
    by which time they no longer have both versions to hand.
    """
    return {
        name: f"{'limited' if fact['force_limited'] else 'unlimited'} "
        f"[{fact['forcerange'][0]:g}, {fact['forcerange'][1]:g}]"
        for name, fact in sorted(facts.items())
    }


def physics_digest(facts: Mapping[str, Mapping[str, Any]]) -> str:
    """Digest the values. Empty facts are ``ABSENT``, not a hash of nothing."""
    if not facts:
        return ABSENT
    hasher = hashlib.sha256()
    for name, line in sorted(physics_manifest(facts).items()):
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(line.encode())
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def compare_facts(
    before: Mapping[str, str], after: Mapping[str, str]
) -> dict[str, list[str]]:
    """Which actuators changed, appeared or vanished between two manifests."""
    return {
        "changed": sorted(
            f"{k}: {before[k]} -> {after[k]}"
            for k in before.keys() & after.keys()
            if before[k] != after[k]
        ),
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
    }
