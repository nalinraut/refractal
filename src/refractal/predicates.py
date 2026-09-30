"""Success predicates, and the passthrough for providers you did not write.

``Task.predicate`` is required rather than optional, deliberately: a missing
predicate should be an error, and an explicit passthrough is a decision recorded
in the catalog rather than a field someone forgot.
"""

from __future__ import annotations

from typing import Any, Mapping


def from_provider(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    """Defer to the provider's own verdict.

    The correct predicate for a wrapped third-party suite, not a workaround.
    LIBERO decides whether a LIBERO task succeeded; re-deriving that from
    observations would be inventing a second, disagreeing definition of success
    for a task whose definition is not ours.

    The adapter puts that verdict in ``state`` under ``provider_success``;
    ``args`` is unused and accepted so the signature matches every other
    predicate.

    Deferring is not the same as not deciding. The catalog still owns the
    definition: it records that this task's verdict comes from elsewhere, and
    ``refractal build`` folds the provider's own goal facts into ``task_hash``
    through ``task_facts``, so a provider release that moves the goal without
    editing the instruction moves your identity rather than your results.
    """
    for key in ("provider_success", "benchmark_success"):
        if key in state:
            return bool(state[key])
    raise KeyError(
        "from_provider expects the adapter to supply 'provider_success' in the "
        "episode state. It is a passthrough for the provider's own verdict, so "
        "there is nothing to fall back to. ('benchmark_success' is also accepted, "
        "for adapters written against the older name.)"
    )


#: The current name is ``from_provider``. This alias exists for compatibility
#: and is not a second function: it is the same object under the name 61 task
#: declarations across five catalogs already use.
#:
#: Not renamed outright because ``predicate`` is inside ``task_hash``. Renaming
#: it would move the hash of every one of those tasks, orphaning the 15,490
#: episodes recorded under them and making future runs refuse to join past
#: ones. That is a real cost, paid for a word.
#:
#: Why the word changed: ``Benchmark`` is vla-eval's name for its own
#: interface, and it arrived with the bridge that implements it. Refractal's
#: vocabulary is scene, task, scenario, seed, episode, checkpoint. None of those
#: says benchmark, and someone reading "write a Benchmark" concludes they need a
#: published suite rather than something that steps a simulator.
#:
#: Use ``from_provider`` in new catalogs. Leave existing ones alone: changing
#: the string changes the experiment, which is exactly the property that makes
#: the rename expensive.
from_benchmark = from_provider


__all__ = ["from_benchmark", "from_provider"]
