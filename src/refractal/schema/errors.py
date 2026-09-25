"""Error types for the catalog layer.

One rule: an error raised here names the file, the field and the fix. The whole
point of validating a catalog up front is that a mistake costs a second rather
than three minutes into a run on someone else's GPU.
"""

from __future__ import annotations


class RefractalError(Exception):
    """Base for everything Refractal raises deliberately."""


class CanonicalizationError(RefractalError, ValueError):
    """A value cannot be canonicalized, so it cannot participate in identity."""


class ImportStringError(RefractalError, ValueError):
    """An import string is malformed, or its target cannot be resolved."""


class CatalogError(RefractalError, ValueError):
    """A catalog is invalid: bad reference, duplicate id, unknown key, missing file."""

    def __init__(self, message: str, *, file: str | None = None, path: str | None = None):
        self.file = file
        self.path = path
        where = " ".join(p for p in (file, path and f"at {path}") if p)
        super().__init__(f"{where}: {message}" if where else message)


class GeneratorError(RefractalError, ValueError):
    """A scenario generator was asked for something it cannot produce."""


class MissingExtraError(RefractalError, ImportError):
    """An optional dependency is needed for what was asked and is not installed.

    Subclasses ``ImportError`` as well, so ``except ImportError`` around a
    lazy import still catches it and the class can be raised from an ``except
    ImportError`` block without changing what callers can catch.

    Exists because the alternative is what a plain install used to do: ``pip
    install refractal`` followed by ``refractal run`` raised ``No module named
    'fsspec'`` from inside a module the user has never heard of, naming a
    package they did not choose. The install is correct and the command is
    reasonable; only the message was missing.
    """


class NotImplementedInV1(RefractalError):
    """A field exists in the schema, is validated, and is deliberately inert.

    Reserved fields are validated rather than ignored so that the identity
    format does not change when the feature arrives. See ``ScenarioSet.perturbations``.
    """
