"""ehds_ontomap: map free-text terms onto a ``code | term`` ontology.

Developed by the NLP4BIA group at the Barcelona Supercomputing Center within
the DataTools4Heart (DT4H) project, which has received funding from the
European Union's Horizon Europe research and innovation programme under grant
agreement No 101057849.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .errors import InputError, OntologyError, OntomapError
from .mapper import OntologyMapper, map_terms
from .methods import Matcher, available_methods
from .normalize import Normalizer, normalize_text
from .ontology import Ontology
from .result import Candidate, MappingParams, MappingResult, QueryResult
from .terms import TermList, load_terms

try:
    __version__ = version("ehds_ontomap")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled checkout
    __version__ = "0.0.0"

__all__ = [
    "Candidate",
    "InputError",
    "MappingParams",
    "MappingResult",
    "Matcher",
    "Normalizer",
    "Ontology",
    "OntologyError",
    "OntologyMapper",
    "OntomapError",
    "QueryResult",
    "TermList",
    "__version__",
    "available_methods",
    "load_terms",
    "map_terms",
    "normalize_text",
]
