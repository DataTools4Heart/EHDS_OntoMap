"""Exceptions raised by :mod:`ehds_ontomap`."""

from __future__ import annotations

__all__ = ["InputError", "OntologyError", "OntomapError"]


class OntomapError(Exception):
    """Base class for every error raised by this library."""


class OntologyError(OntomapError, ValueError):
    """The ontology file or frame is missing, malformed or empty."""


class InputError(OntomapError, ValueError):
    """The list of terms to map cannot be interpreted."""
