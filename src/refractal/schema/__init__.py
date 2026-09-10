"""``refractal.schema`` -- the catalog model, identity, and scenario generators.

Import rule, enforced by review and by ``tests/test_boundaries.py``: nothing in
this package may import Docker, a GPU library, a simulator, ``vla_eval``, or any
of ``refractal.execute`` / ``refractal.compare``. It must be importable and
fully testable on a bare laptop, because that is the property the whole
compiler architecture rests on.
"""

from .canonical import (
    QUANT_SIG_DIGITS,
    canonical_json,
    hash_obj,
    normalize_params,
    round_significant,
    sha256_of,
)
from .errors import (
    CanonicalizationError,
    CatalogError,
    GeneratorError,
    ImportStringError,
    NotImplementedInV1,
    RefractalError,
)
from .generators import Generator, ScenarioFilter, latin_hypercube, linspace_grid, random_sample
from .identity import (
    comparison_unit,
    episode_id,
    experiment_identity,
    plan_id,
    scenario_hash,
    scenario_identity,
    scene_hash,
    task_hash,
    task_identity,
)
from .importstr import check_arity, resolve_import_string, validate_import_string
from .loader import Catalog, load_catalog
from .models import (
    API_VERSION,
    Checkpoint,
    Device,
    FaultSpec,
    HardwareProfile,
    ParamSpec,
    Phase,
    ResourceShape,
    Run,
    Scene,
    ScenarioSet,
    Task,
)

__all__ = [
    "API_VERSION",
    "QUANT_SIG_DIGITS",
    "CanonicalizationError",
    "Catalog",
    "CatalogError",
    "Checkpoint",
    "Device",
    "FaultSpec",
    "Generator",
    "GeneratorError",
    "HardwareProfile",
    "ImportStringError",
    "NotImplementedInV1",
    "ParamSpec",
    "Phase",
    "RefractalError",
    "ResourceShape",
    "Run",
    "Scene",
    "ScenarioFilter",
    "ScenarioSet",
    "Task",
    "check_arity",
    "canonical_json",
    "comparison_unit",
    "episode_id",
    "experiment_identity",
    "hash_obj",
    "latin_hypercube",
    "linspace_grid",
    "load_catalog",
    "normalize_params",
    "plan_id",
    "random_sample",
    "resolve_import_string",
    "round_significant",
    "scenario_hash",
    "scenario_identity",
    "scene_hash",
    "sha256_of",
    "task_hash",
    "task_identity",
    "validate_import_string",
]
