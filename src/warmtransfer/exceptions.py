"""Library exceptions."""

from __future__ import annotations


class ColdScoreError(Exception):
    """Base library exception."""


class SchemaError(ColdScoreError):
    """Input DataFrame schema violation (missing column, wrong dtype, duplicates, NaN)."""


class MissingInputError(ColdScoreError):
    """A required input was not provided to the method (see ``ColdStartMethod.requires``)."""


class NotFittedError(ColdScoreError):
    """``predict`` called before ``fit``."""


class RegistryError(ColdScoreError):
    """Component registry error (duplicate name, unknown name)."""


class LeakageError(ColdScoreError):
    """Cold items leaked into the training data (eval protocol violation)."""
