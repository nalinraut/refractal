"""Which checkpoint a path names, and whether that name is pinned.

A Hugging Face Hub id (``org/name``) resolves its default branch when a server
loads it, so two launches of one plan can load two different checkpoints under
one ``plan_id``. A hub id must therefore carry ``@<commit>``: the full 40-hex
commit, never a branch or tag, which can move. The revision is part of the
checkpoint's ``path``, so it enters ``plan_id`` with no further wiring.

A ``gs://`` checkpoint (openpi's assets) has no commit at all, and its bucket
object can be replaced in place. It is pinned by content instead:
``gs://bucket/path@sha256:<64-hex>``, the digest of the downloaded params
directory that the model server computes and reports.

Decided by the path's SHAPE, not by looking at the disk: whether a directory
exists differs between machines, and a plan has to mean the same thing on every
machine it is compiled on. A path starting with ``/``, ``.`` or ``~``, or
without exactly one ``/``, is local and is not a hub id.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CONTENT = re.compile(r"^sha256:[0-9a-f]{64}$")


def split_revision(path: str) -> tuple[str, str | None]:
    """``org/name@rev`` -> (``org/name``, ``rev``); no ``@`` -> (path, None)."""
    if "@" not in path:
        return path, None
    name, _, revision = path.rpartition("@")
    return name, revision


def is_hub_id(path: str) -> bool:
    return bool(_HUB_ID.match(split_revision(path)[0]))


def is_gcs(path: str) -> bool:
    return split_revision(path)[0].startswith("gs://")


def pinned_revision(path: str) -> str | None:
    """The pin a remote checkpoint carries, else None: a hub id's 40-hex commit,
    or a ``gs://`` path's ``sha256:`` params digest."""
    name, revision = split_revision(path)
    if not revision:
        return None
    if _HUB_ID.match(name) and _COMMIT.match(revision):
        return revision
    if name.startswith("gs://") and _CONTENT.match(revision):
        return revision
    return None


def unpinned(checkpoints: Iterable[Any]) -> list[tuple[str, str]]:
    """(id, path) of every remote checkpoint (hub id or gs://) without its pin."""
    return [(c.id, c.path) for c in checkpoints
            if (is_hub_id(c.path) or is_gcs(c.path)) and pinned_revision(c.path) is None]


__all__ = ["is_gcs", "is_hub_id", "pinned_revision", "split_revision", "unpinned"]
