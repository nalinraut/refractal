"""Which checkpoint a path names, and whether that name is pinned.

A Hugging Face Hub id (``org/name``) resolves its default branch when a server
loads it, so two launches of one plan can load two different checkpoints under
one ``plan_id``. A hub id must therefore carry ``@<commit>``: the full 40-hex
commit, never a branch or tag, which can move. The revision is part of the
checkpoint's ``path``, so it enters ``plan_id`` with no further wiring.

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


def split_revision(path: str) -> tuple[str, str | None]:
    """``org/name@rev`` -> (``org/name``, ``rev``); no ``@`` -> (path, None)."""
    if "@" not in path:
        return path, None
    name, _, revision = path.rpartition("@")
    return name, revision


def is_hub_id(path: str) -> bool:
    return bool(_HUB_ID.match(split_revision(path)[0]))


def pinned_revision(path: str) -> str | None:
    """The 40-hex commit a hub id is pinned to, else None."""
    name, revision = split_revision(path)
    if _HUB_ID.match(name) and revision and _COMMIT.match(revision):
        return revision
    return None


def unpinned(checkpoints: Iterable[Any]) -> list[tuple[str, str]]:
    """(id, path) of every hub-id checkpoint without a full commit pin."""
    return [(c.id, c.path) for c in checkpoints if is_hub_id(c.path) and pinned_revision(c.path) is None]


__all__ = ["is_hub_id", "pinned_revision", "split_revision", "unpinned"]
