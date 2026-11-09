"""Whether an installed distribution is one anybody else could install.

The identity chain is content-addressed everywhere except one link: an
external scene's hash is an assertion recorded in ``catalog/build.lock``, and
half of that assertion is a version string. Measured on a real lock:

    {"engine": "mujoco", "engine_version": "3.2.0",
     "libero_version": "0.1.0", "provider": "libero", "suite": "libero_object"}

Version labels, no digests. So an editable checkout of the provider whose
assets have been edited produces the same ``scene_hash`` as the published
release it was checked out from, and nothing in the results says so. That is
not a hypothetical upstream release changing physics under a kept version --
it is the case that exists on a developer's own machine, which is where it
will actually bite.

This records the condition rather than blocking on it. An editable install is
how the provider gets written in the first place, so refusing would refuse
the normal development loop; the problem is only that the condition is
invisible. Recorded in the lock and rendered on the verdict, it becomes a
stated one.

What RECORD can and cannot do
-----------------------------

pip writes a per-file sha256 into each distribution's RECORD, so one hash over
RECORD covers every installed file -- numpy's has 918 lines. That is a real
verification, and it is the strong version of this check for a normally
installed package.

It does nothing for an editable install, and measuring that is what settled
the design. An editable distribution's RECORD has 11 lines and they describe
a ``.pth`` file and the dist-info directory, not one line of the source the
interpreter will actually import. So a RECORD digest for an editable install
would be a hash that moves when the packaging moves and holds still when the
code changes -- worse than no claim, because it looks like one.

Hence: a digest where it means something, and an explicit statement that the
sources are unverifiable where it does not. The absent-or-meaningless RECORD
is itself the signal.

One measurement that stopped a false signal: the number of RECORD lines with
an empty hash field is 1 for an editable install AND 1 for numpy and pytest --
it is RECORD's own line, which carries no hash by construction. It is not a
discriminator, and it reads exactly like one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import Distribution, distributions
from typing import Iterable, Sequence


@dataclass(frozen=True)
class InstalledDist:
    """One distribution, and how much of it can be checked by a third party."""

    name: str
    version: str | None = None
    #: Installed from a local directory in editable mode. The sources on disk
    #: are what runs, and nothing recorded here constrains them.
    editable: bool = False
    #: Where it came from, when it did not come from an index: a local path or
    #: a VCS url. Present for any direct install, editable or not.
    source: str | None = None
    #: sha256 over the distribution's RECORD, which carries a sha256 per
    #: installed file. None when it would not mean anything -- see the module
    #: docstring.
    record_digest: str | None = None
    #: Not installed in the environment that was asked.
    missing: bool = False

    @property
    def verifiable(self) -> bool:
        """Whether someone else could check these bytes are the bytes used."""
        return self.record_digest is not None

    def document(self) -> dict[str, object]:
        """The lock's record. Keys are emitted only when set, so a normally
        installed package records nothing it does not need and locks written
        before this field keep their shape."""
        doc: dict[str, object] = {}
        if self.version is not None:
            doc["version"] = self.version
        if self.editable:
            doc["editable"] = True
        if self.source is not None:
            doc["source"] = self.source
        if self.record_digest is not None:
            doc["record_digest"] = self.record_digest
        if self.missing:
            doc["missing"] = True
        return doc

    def render(self) -> str:
        if self.missing:
            return f"{self.name} not installed"
        version = self.version or "unknown version"
        if self.editable:
            return (f"{self.name} {version} EDITABLE from {self.source} "
                    "(its sources are not verifiable from this record)")
        return f"{self.name} {version}"


def _of(dist: Distribution, name: str) -> InstalledDist:
    editable, source = False, None
    raw = None
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        raw = None
    if raw:
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = {}
        source = doc.get("url") or None
        editable = bool((doc.get("dir_info") or {}).get("editable"))

    digest = None
    if not editable:
        # Only where it constrains the sources. An editable install's RECORD
        # describes a .pth, so a digest over it would be a claim about nothing.
        try:
            record = dist.read_text("RECORD")
        except Exception:
            record = None
        if record:
            digest = "sha256:" + hashlib.sha256(record.encode()).hexdigest()

    return InstalledDist(
        name=name,
        version=dist.metadata["Version"] if dist.metadata else None,
        editable=editable,
        source=source,
        record_digest=digest,
    )


def installed_dists(
    names: Iterable[str], *, search_path: Sequence[str] | None = None
) -> dict[str, InstalledDist]:
    """Provenance for each named distribution, missing ones included.

    A missing distribution is recorded rather than skipped: "libero is not
    installed here" is a fact about the environment that wrote the lock, and
    dropping it would make a lock written without the provider look like one
    written with it.
    """
    wanted = list(names)
    found: dict[str, Distribution] = {}
    # `path=None` is NOT the default -- importlib.metadata iterates it, so
    # passing it explicitly raises TypeError rather than falling back to
    # sys.path. Caught by the existing build tests, which call this with no
    # search path; every test here had supplied one.
    search = {"path": list(search_path)} if search_path else {}
    for dist in distributions(**search):
        meta = dist.metadata
        if not meta:
            continue
        got = (meta["Name"] or "").strip()
        if not got:
            continue
        key = got.lower().replace("_", "-")
        if key not in found:
            found[key] = dist
    out: dict[str, InstalledDist] = {}
    for name in wanted:
        dist = found.get(name.lower().replace("_", "-"))
        out[name] = _of(dist, name) if dist is not None else InstalledDist(
            name=name, missing=True
        )
    return out


def editable_among(dists: dict[str, InstalledDist]) -> tuple[str, ...]:
    """Just the names, for the verdict line."""
    return tuple(sorted(n for n, d in dists.items() if d.editable))


__all__ = ["InstalledDist", "editable_among", "installed_dists"]
