"""Re-export of the built-in generators at their documented import path.

The API reference names them ``refractal.generators:linspace_grid``, so that
path has to exist. The implementations live in :mod:`refractal.schema.generators`
because the generator protocol is part of the schema contract.
"""

from .schema.generators import (
    MAX_SCENARIOS,
    Generator,
    ScenarioFilter,
    latin_hypercube,
    linspace_grid,
    random_sample,
)

__all__ = [
    "MAX_SCENARIOS",
    "Generator",
    "ScenarioFilter",
    "latin_hypercube",
    "linspace_grid",
    "random_sample",
]
