"""Canonical serialization and content-addressed hashing.

Every identity in Refractal is a hash of a canonical byte string, so this module
is load-bearing for the entire project. It must guarantee two things:

1. **Stability.** The same logical value produces the same bytes on any machine,
   any CPython version, any platform, in any process.
2. **Insensitivity to differences nobody intended.** ``0.1`` and ``0.10``,
   ``0.0`` and ``-0.0``, and -- the one that actually bites -- floating-point
   noise in the last bit of a generated grid point must all collapse to the
   same bytes.

Guarantee 2 is understated in the spec, which frames the problem as ``0.1``
versus ``0.10``. Those already parse to the same double, so they are free. The
real hazard is arithmetic: ``0.08 + 2 * 0.02`` is ``0.12000000000000001`` while
``linspace(0.08, 0.16, 5)[2]`` may be ``0.12`` depending on the library, the
platform and the summation order. A generator that drifts by one ulp between
numpy versions silently forks scenario identity, and every comparison against
previously recorded results quietly becomes an empty join -- with no error
anywhere, because an empty join is a legal join.

The defence is to quantize: every number that enters an identity is rounded to
``QUANT_SIG_DIGITS`` significant digits *before* hashing, and the quantized
value -- not the raw one -- is what gets stored in ``plan.json`` and handed to
the simulator. Identity and execution therefore see the same number.

Twelve significant digits is far below double precision (~15-17) so real
distinctions survive, and far above any physical tolerance in a scene measured
in centimetres.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import CanonicalizationError

QUANT_SIG_DIGITS = 12

# Values that may appear in a canonical document.
_SCALARS = (str, int, float, bool, type(None))


def round_significant(x: float, digits: int = QUANT_SIG_DIGITS) -> float:
    """Round to ``digits`` significant figures, killing last-bit arithmetic noise."""
    if x == 0.0 or not math.isfinite(x):
        return x
    return float(f"{x:.{digits - 1}e}")


def normalize_number(x: int | float) -> float:
    """Quantize a float so arithmetic noise cannot fork identity.

    ``-0.0`` folds to ``0.0``: they compare equal but ``repr`` differs, which
    would be an identity fork that no test would ever catch by inspection.
    """
    if isinstance(x, bool):  # bool is an int subclass; do not let it through
        raise CanonicalizationError("booleans are not numbers in a scenario parameter")
    f = round_significant(float(x))
    if not math.isfinite(f):
        raise CanonicalizationError(f"non-finite number in identity: {x!r}")
    return 0.0 if f == 0.0 else f


def normalize_params(value: Any, *, _path: str = "$") -> Any:
    """Recursively normalize a scenario-parameter structure.

    Applied to generator output before hashing *and* before storage, so the
    number in ``plan.json`` is byte-for-byte the number that was hashed.
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        # Integers are already canonical. They have no 0.1-versus-0.10 problem
        # and no accumulated-error problem, so there is nothing to normalize --
        # and coercing them to float made `plan.json` say
        # `init_state_index: 3.0` for what is an index into a fixed array.
        #
        # That reads as wrong, and "reads as wrong but works" is the category
        # this project keeps finding real bugs in. The earlier justification was
        # that scenario parameters are physical quantities and so int and float
        # are the same kind of thing; true of a vial position, false of an index.
        #
        # Consequence, accepted deliberately: `friction: 1` and `friction: 1.0`
        # are now different scenarios. The difference is visible in the YAML
        # diff and in `catalog_hash`, which the previous silent coercion was not.
        return value
    if isinstance(value, float):
        return normalize_number(value)
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise CanonicalizationError(f"{_path}: parameter keys must be strings, got {k!r}")
            out[k] = normalize_params(v, _path=f"{_path}.{k}")
        return out
    if isinstance(value, (list, tuple)):
        return [normalize_params(v, _path=f"{_path}[{i}]") for i, v in enumerate(value)]
    raise CanonicalizationError(f"{_path}: {type(value).__name__} cannot appear in a scenario")


def _encode(value: Any, out: list[str], path: str) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        out.append(str(value))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"{path}: NaN and infinity cannot be hashed")
        q = round_significant(value)
        out.append(repr(0.0 if q == 0.0 else q))
    elif isinstance(value, str):
        # ensure_ascii so a UTF-8 locale difference cannot change the bytes.
        out.append(json.dumps(value, ensure_ascii=True))
    elif isinstance(value, Mapping):
        items = []
        for k in value:
            if not isinstance(k, str):
                raise CanonicalizationError(f"{path}: object keys must be strings, got {k!r}")
            items.append(k)
        out.append("{")
        for i, k in enumerate(sorted(items)):
            if i:
                out.append(",")
            out.append(json.dumps(k, ensure_ascii=True))
            out.append(":")
            _encode(value[k], out, f"{path}.{k}")
        out.append("}")
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for i, v in enumerate(value):
            if i:
                out.append(",")
            _encode(v, out, f"{path}[{i}]")
        out.append("]")
    else:
        raise CanonicalizationError(f"{path}: {type(value).__name__} is not canonicalizable")


def canonical_json(value: Any) -> bytes:
    """Serialize to the canonical byte form used for every hash in Refractal.

    Object keys sorted recursively; no insignificant whitespace; floats
    quantized and emitted via ``repr`` (shortest round-trip, stable across
    CPython versions); NaN and infinity rejected rather than encoded, because
    JSON has no representation for them and every library invents a different
    one.

    The int/float distinction is preserved, here and in
    :func:`normalize_params`. ``predicate_args: {slot: 4}`` is an index and
    ``4.0`` would read as a different thing; ``init_state_index: 3`` is an index
    too, and a generator that emitted ``3.0`` for it made ``plan.json`` --- the
    artifact whose entire job is to be read and checked --- say something that
    looks wrong.
    """
    out: list[str] = []
    _encode(value, out, "$")
    return "".join(out).encode("utf-8")


def sha256_of(data: bytes) -> str:
    """Prefixed digest, so a hash is self-describing when it turns up in a log."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_obj(value: Any) -> str:
    """Content address of a canonicalizable object."""
    return sha256_of(canonical_json(value))


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def hash_file_set(root: Path, paths: Iterable[Path]) -> str:
    """Hash a set of files by (relative path, content), order-independently.

    Names are included so that renaming a mesh changes the hash: a scene that
    references ``gripper_v2.stl`` is not the scene that referenced
    ``gripper.stl``, even if the bytes happen to match today.
    """
    entries = sorted(
        (str(Path(p).resolve().relative_to(root.resolve()).as_posix()), hash_file(Path(p)))
        for p in paths
    )
    return hash_obj([{"path": name, "sha256": digest} for name, digest in entries])
