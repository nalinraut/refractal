"""Success predicates, and the passthrough for benchmarks you did not write.

``Task.predicate`` is required rather than optional, deliberately: a missing
predicate should be an error, and an explicit passthrough is a decision recorded
in the catalog rather than a field someone forgot.
"""

from __future__ import annotations

from typing import Any, Mapping


def from_benchmark(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    """Defer to the benchmark's own verdict.

    The correct predicate for a wrapped third-party benchmark, not a workaround.
    LIBERO decides whether a LIBERO task succeeded; re-deriving that from
    observations would be inventing a second, disagreeing definition of success
    for a task whose definition is not ours.

    The adapter puts the benchmark's verdict in ``state`` under
    ``benchmark_success``; ``args`` is unused and accepted so the signature
    matches every other predicate.
    """
    if "benchmark_success" not in state:
        raise KeyError(
            "from_benchmark expects the adapter to supply 'benchmark_success' in the "
            "episode state. It is a passthrough for the benchmark's own verdict, so "
            "there is nothing to fall back to."
        )
    return bool(state["benchmark_success"])


__all__ = ["from_benchmark"]
