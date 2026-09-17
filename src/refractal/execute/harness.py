"""What "the harness changed" should mean.

A harness version is too blunt to gate on. Measured over the 24 most recent
commits of ``allenai/vla-evaluation-harness``, a whole-tree digest changes on
**every single one** -- leaderboard data refreshes, docs edits, CI pin bumps, a
robodojo submodule pin. A precondition that fires on every dependency bump is a
flag people paste in without reading, and then it gates nothing.

So digest only what Refractal actually depends on: the subclass surface and the
protocol it speaks.

Same measurement, same 24 commits:

===================  =======================
digest               transitions where it changed
===================  =======================
whole tree           24 / 24  (100%)
wider surface        4 / 24   (17%)
**narrow surface**   **3 / 24  (12%)**
===================  =======================

And it fires on the right ones. Across the v0.5.0..HEAD range the narrow digest
is constant through ``9439ece`` (docs), ``5e22cf9`` (leaderboard data) and
``35f1200`` (robodojo pin), and changes on exactly the two commits that altered
behaviour Refractal cares about: ``dc2c4ba``, which changed how negotiated
observation parameters reach a benchmark constructor, and ``0865d42``, which
changed episode-to-shard assignment. Both touch ``orchestrator.py``.

Two columns, one of each kind (see the taxonomy in ``refractal/__init__.py``):

``harness_version``
    provenance. Human-readable, never gates. Answers "what was installed".

``harness_surface``
    precondition. Gates ``compare``. Answers "did anything I depend on move".
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

#: Modules Refractal subclasses, calls, or exchanges messages with. Everything
#: else in the harness -- 20 benchmark packages, 18 model servers, the
#: leaderboard, the CLI -- can change freely without affecting a comparison.
SURFACE = (
    "orchestrator.py",   # _build_recorder, observation-param merging, sharding
    "recording.py",      # EpisodeRecorder, NullEpisodeRecorder, StepRecorder
    "registry.py",       # resolve_import_string
    "types.py",          # Task, EpisodeResult, Observation, Action
    "runners",           # the EPISODE_START payload
    "protocol",          # the wire format
)

#: What the local backend records. Nothing talks to the harness there, and
#: saying so is more honest than recording a version that never ran.
LOCAL = "none/local-backend"


def surface_manifest(package_root: str | Path) -> dict[str, str]:
    """Per-file digests of the surface, so a change can be *named*.

    The digest alone answers "did something move" and forces whoever reads the
    block to go and find out what. That question is usually five minutes of
    ``git diff`` -- but only if you still have the two versions to hand, which
    whoever reads a comparison months later does not.

    A manifest costs a few hundred bytes per session and turns the answer into a
    file list. The gate stays the digest, because that is what belongs in a
    column; this is what makes the gate's verdict actionable rather than merely
    correct.
    """
    root = Path(package_root)
    entries: dict[str, str] = {}
    for name in sorted(SURFACE):
        target = root / name
        paths = (
            sorted(p for p in target.rglob("*.py") if p.is_file())
            if target.is_dir()
            else ([target] if target.is_file() else [])
        )
        if not paths:
            entries[name] = "absent"
            continue
        for path in paths:
            entries[str(path.relative_to(root).as_posix())] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()[:16]
    return entries


def compare_manifests(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, list[str]]:
    """Which surface files changed, appeared or vanished between two runs."""
    return {
        "changed": sorted(k for k in before.keys() & after.keys() if before[k] != after[k]),
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
    }


def surface_digest(package_root: str | Path) -> str:
    """Digest the files Refractal depends on, ignoring the rest of the harness."""
    root = Path(package_root)
    hasher = hashlib.sha256()
    for name in sorted(SURFACE):
        target = root / name
        paths = (
            sorted(p for p in target.rglob("*.py") if p.is_file())
            if target.is_dir()
            else ([target] if target.is_file() else [])
        )
        if not paths:
            # A surface module that has vanished is itself a change worth
            # noticing, so it is hashed as absent rather than skipped.
            hasher.update(f"{name}:absent".encode())
            continue
        for path in paths:
            hasher.update(str(path.relative_to(root).as_posix()).encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def describe_installed_harness() -> tuple[str, str, dict[str, str]]:
    """``(harness_version, harness_surface)`` for whatever is importable here.

    Called at container startup, where the harness exists. Falls back to the
    local-backend constant rather than guessing, because a fabricated version
    string in a provenance column is worse than an honest "nothing ran".
    """
    try:
        import vla_eval  # type: ignore
    except ImportError:
        return LOCAL, LOCAL, {}

    version = getattr(vla_eval, "__version__", "unknown")
    root = Path(vla_eval.__file__).parent
    return f"vla-eval {version}", surface_digest(root), surface_manifest(root)


__all__ = [
    "LOCAL",
    "SURFACE",
    "compare_manifests",
    "describe_installed_harness",
    "surface_digest",
    "surface_manifest",
]
