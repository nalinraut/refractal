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


#: How many positional arguments each kind of user callable takes. They are
#: different contracts and swapping them is easy, because both are just import
#: strings in YAML and neither names its shape.
ARITIES = {
    "filter": (1, "(scenario) -> bool"),
    "predicate": (2, "(state, args) -> bool"),
    "generator": (2, "(params, seed) -> list[dict]"),
}


def check_arity(fn: Any, kind: str, import_string: str) -> None:
    """Reject a callable whose shape says it is a different kind of thing.

    A comment at the call site protects the person who reads the comment, which
    is not the person who makes the mistake. This is the same check as a
    comment, made by the machine: a predicate handed to a `filter:` field has
    two positional parameters and is refused at build time, rather than raising
    a TypeError several hundred scenarios into an evaluation.

    ``*args`` is accepted -- a callable that takes anything cannot be shown to be
    wrong, and refusing it would break legitimate wrappers.
    """
    import inspect

    expected, shape = ARITIES[kind]
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return  # builtins and C callables have no introspectable signature

    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if any(p.kind is p.VAR_POSITIONAL for p in signature.parameters.values()):
        return

    required = [p for p in positional if p.default is inspect.Parameter.empty]
    if len(required) > expected or len(positional) < expected:
        other = next(
            (k for k, (n, _) in ARITIES.items() if n == len(positional) and k != kind),
            None,
        )
        hint = f" That is the shape of a {other}." if other else ""
        raise ImportStringError(
            f"{import_string!r} is declared as a {kind}, which must be {shape}, but it "
            f"takes {len(positional)} positional argument(s):"
            f" {inspect.signature(fn)}.{hint}"
        )


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
