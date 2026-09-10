"""``refractal.execute`` -- the only package that touches infrastructure.

Backends are strictly additive:

===========  ==========================================================
``local``    in-process, fake benchmark, no simulator. Also the
             permanent test fixture for ``compare``.
``compose``  Docker Compose, one host. The default, once it exists.
``k8s``      rendered from ``plan.json``, never hand-written. Not v1.
===========  ==========================================================

Nothing in ``schema`` or ``resolve`` may import anything from here.
"""

from .fake import FakeBenchmark
from .harness import LOCAL, SURFACE, describe_installed_harness, surface_digest
from .local import LOCAL_HARNESS_VERSION, RunSummary, run_local
from .results import EPISODES_SCHEMA, STEPS_SCHEMA, ResultWriter, comparison_prefix, read_episodes

__all__ = [
    "EPISODES_SCHEMA",
    "LOCAL_HARNESS_VERSION",
    "STEPS_SCHEMA",
    "LOCAL",
    "SURFACE",
    "FakeBenchmark",
    "describe_installed_harness",
    "surface_digest",
    "ResultWriter",
    "RunSummary",
    "comparison_prefix",
    "read_episodes",
    "run_local",
]
