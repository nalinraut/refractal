"""``refractal.compare`` -- the payoff.

Offline. Reads Parquet. No GPU, no Docker, no harness import. Works on any
results directory, including one Refractal did not produce.

It reports, in this order:

1. **Overlap first**, above everything else. Comparing an intersection while
   believing you compared everything is the failure this prevents.
2. **The full 2x2 per task**, not only the off-diagonal the test consumes.
3. **McNemar**, because the data is paired.
4. **A clustered bootstrap** for the rate difference, resampling scenarios.

No plots. Comparison and statistics ship; visualisation is DuckDB's job.
"""

from .pairing import Eligibility, Unit, UnitKey, build_units
from .verdict import (
    MIN_UNITS_FOR_CALIBRATED_CI,
    TaskVerdict,
    Verdict,
    evaluate,
    render,
)
from .stats import (
    BootstrapResult,
    Discordance,
    TestResult,
    clustered_bootstrap,
    contingency,
    mcnemar_exact,
    mcnemar_unclustered,
    two_proportion_z,
)

__all__ = [
    "MIN_UNITS_FOR_CALIBRATED_CI",
    "BootstrapResult",
    "Discordance",
    "Eligibility",
    "TestResult",
    "Unit",
    "UnitKey",
    "build_units",
    "clustered_bootstrap",
    "contingency",
    "mcnemar_exact",
    "mcnemar_unclustered",
    "two_proportion_z",
    "TaskVerdict",
    "Verdict",
    "evaluate",
    "render",
]
