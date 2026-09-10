"""Import strings: ``"module.path:symbol"``.

The spec says import strings are "resolved at load time" and that a symbol that
cannot be imported is a validation error raised during ``refractal plan``.
That cannot be true for all of them, and pretending otherwise breaks the
project's own first invariant.

``Task.predicate`` resolves into the user's adapter, which imports the
simulator. ``Checkpoint.server`` resolves into a model server, which imports
torch and probably touches a GPU. Resolving those at catalog-load time means
``refractal plan`` no longer runs on a bare laptop -- which is the property that
justified the compiler architecture in the first place.

So import strings are validated in two tiers:

* **Syntax, always, in ``schema``.** Costs nothing, catches the actual common
  typo (a dot where a colon belongs), never imports anything.
* **Resolution, by whoever needs the symbol.** ``resolve`` imports generators
  and filters, because it must call them to expand the scenario list.
  ``execute`` imports predicates and servers. ``schema`` imports neither.

The consequence for users is a documented contract rather than a check:
**generators and filters must be importable without a simulator**, because
``resolve`` runs where no simulator exists. A reachability filter that needs IK
belongs behind a pure kinematics module, or its output belongs baked into the
catalog by ``refractal build``.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

from .errors import ImportStringError

_IMPORT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")


def validate_import_string(value: str) -> str:
    """Check the shape only. Never imports. Safe to call from a pydantic validator."""
    if not isinstance(value, str) or not _IMPORT_RE.match(value):
        raise ImportStringError(
            f"{value!r} is not a valid import string; expected 'module.path:symbol' "
            "(a colon separates the module from the symbol, not a dot)"
        )
    return value


def resolve_import_string(value: str) -> Any:
    """Import and return the symbol. Only ``resolve`` and ``execute`` may call this."""
    validate_import_string(value)
    module_name, _, symbol = value.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportStringError(f"cannot import module {module_name!r} for {value!r}: {exc}") from exc
    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise ImportStringError(f"module {module_name!r} has no symbol {symbol!r}") from exc
