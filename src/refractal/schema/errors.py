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


class NotImplementedInV1(RefractalError):
    """A field exists in the schema, is validated, and is deliberately inert.

    Reserved fields are validated rather than ignored so that the identity
    format does not change when the feature arrives. See ``ScenarioSet.faults``.
    """
