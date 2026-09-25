"""Scenario generators: the protocol, the contract, and the three built-ins.

A generator turns a ``params`` block into a concrete list of scenarios. The
plan stores that list, not the generator spec, because a generator whose
implementation drifts would make the provenance a lie -- the same catalog would
expand to different scenarios and nothing would say so.

But storing the output only helps *after* a plan exists. What makes a plan
reproducible from a catalog at all is the contract:

    Given identical ``params`` and ``seed``, a generator MUST return the
    identical list, in the identical order, on any machine and any Python
    version.

Which means:

* no ``random`` or ``numpy.random`` without the supplied seed;
* no set iteration, no reliance on dict ordering beyond insertion order;
* no filesystem, network or clock access;
* **quantized output.** Every returned value passes through
  ``normalize_params``, so a value that differs in the last bit between
  platforms cannot fork scenario identity. This is the clause most likely to be
  forgotten by a third-party generator, and the one whose absence is silent.

The built-ins take no numpy dependency on purpose. It is not about install
weight -- it is that ``numpy.linspace`` has changed its summation strategy
between releases, and a scenario identity that depends on which numpy is
installed is not an identity.
"""

from __future__ import annotations

import itertools
import random
import statistics
from typing import Any, Callable, Mapping, Protocol, Sequence

from .canonical import normalize_params
from .errors import GeneratorError
from .models import ParamSpec

#: Guard against a grid that was not meant. Five params at ten steps each is
#: 100,000 scenarios; a sixth is a million. Catching that at plan time costs a
#: comparison, and the alternative is discovering it as a stalled process.
MAX_SCENARIOS = 1_000_000


class Generator(Protocol):
    """The callable an import string in ``ScenarioSet.generator`` must resolve to."""

    def __call__(self, params: Mapping[str, ParamSpec], seed: int) -> list[dict[str, Any]]:
        ...


ScenarioFilter = Callable[[Mapping[str, Any]], bool]


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _linspace(lo: float, hi: float, steps: int) -> list[Any]:
    """Inclusive linear spacing, preserving the authored numeric type.

    Written as ``lo + i * (hi - lo) / (steps - 1)`` rather than by accumulation
    so error does not compound along the axis and the endpoint is exact rather
    than approached.

    **Integer preservation.** ``range: [0, 49], steps: 50`` describes indices, not
    quantities, and ``plan.json`` should say ``init_state_index: 3`` rather than
    ``3.0``. So when the bounds are integers *and* the spacing divides evenly,
    the axis is computed with integer arithmetic and stays integral.

    The condition is ``span % (steps - 1) == 0``, checked on integers, not a
    tolerance on the floats afterwards. Structural rather than numerical: asking
    whether ``2.9999999999999996`` is "close enough to an integer" would
    eventually call a genuinely fractional grid integral, and the whole point is
    that identity must not depend on a judgement call.

    **Uniform across the parameter, never per-value.** ``range: [0, 1]`` with
    ``steps: 3`` has integer bounds but yields ``0.5``, so the entire axis
    becomes float. Mixing ``0`` and ``0.5`` in one parameter would make the
    canonical form depend on which grid point you landed on.
    """
    if steps == 1:
        return [lo]
    span = hi - lo
    if _is_int(lo) and _is_int(hi) and span % (steps - 1) == 0:
        step = span // (steps - 1)
        return [lo + i * step for i in range(steps)]
    return [lo + i * span / (steps - 1) for i in range(steps)]


def _uniform_numeric(values: list[Any]) -> list[Any]:
    """Widen a mixed int/float choice list to all-float.

    Same rule as the range axis: the type is a property of the parameter, not of
    the individual value. ``choices: [1, 2.5]`` becomes ``[1.0, 2.5]`` so the
    canonical form does not depend on which choice was drawn.
    """
    numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric and any(isinstance(v, float) for v in numeric):
        return [
            float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for v in values
        ]
    return values


def _axis_values(spec: ParamSpec, rng: random.Random) -> list[Any]:
    """The values one parameter contributes to a cross product."""
    form = spec.form
    if form == "range":
        return _linspace(spec.range[0], spec.range[1], int(spec.steps))
    if form == "choices":
        return _uniform_numeric(list(spec.choices))
    if form == "constant":
        return [spec.value]
    if form == "random":
        return [_draw(spec, rng.random()) for _ in range(int(spec.samples))]
    # unreachable: ParamSpec.form returns one of exactly four strings or raises,
    # and all four are handled above. Kept so that adding a fifth form without
    # extending this function fails loudly instead of returning None.
    raise GeneratorError(f"unhandled parameter form {form!r}")


def _draw(spec: ParamSpec, u: float) -> Any:
    """Map a uniform ``u`` in [0, 1) onto a parameter's domain.

    Inverse-CDF for the normal case rather than ``rng.gauss``, so that random
    sampling and Latin hypercube sampling share one mapping and consume exactly
    one uniform per value. That keeps the RNG stream position predictable,
    which is what makes the output reproducible.
    """
    form = spec.form
    if form == "constant":
        return spec.value
    if form == "choices":
        values = _uniform_numeric(list(spec.choices))
        return values[min(int(u * len(values)), len(values) - 1)]
    if form == "range":
        values = _linspace(spec.range[0], spec.range[1], int(spec.steps))
        return values[min(int(u * len(values)), len(values) - 1)]
    if form == "random":
        lo, hi = spec.range
        if spec.distribution == "uniform":
            return lo + u * (hi - lo)
        # Clamp rather than reject: a normal draw outside the declared range is
        # a pose outside the scene, and truncating is the intended reading of
        # supplying both a distribution and a range.
        x = statistics.NormalDist(spec.mean, spec.std).inv_cdf(min(max(u, 1e-12), 1 - 1e-12))
        return min(max(x, lo), hi)
    # unreachable: reached only for form == "random", and _check_form has
    # already refused a random spec whose distribution is neither uniform nor
    # normal. Kept so a third distribution cannot be added silently.
    raise GeneratorError(f"unhandled parameter form {form!r}")


def _sample_count(params: Mapping[str, ParamSpec]) -> int:
    counts = {name: int(s.samples) for name, s in params.items() if s.form == "random"}
    if not counts:
        raise GeneratorError(
            "a sampling generator needs a sample count: give at least one parameter "
            "the random form (range + samples + distribution)"
        )
    distinct = set(counts.values())
    if len(distinct) != 1:
        raise GeneratorError(
            f"sampling generators draw joint samples, so every 'samples' must agree; got {counts}"
        )
    return distinct.pop()


def _check_size(n: int, generator: str) -> None:
    if n > MAX_SCENARIOS:
        raise GeneratorError(
            f"{generator} would expand to {n:,} scenarios, above the {MAX_SCENARIOS:,} guard. "
            "Reduce steps, or split the set."
        )


def _finish(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_params(dict(r)) for r in rows]


# --------------------------------------------------------------------------
# built-ins
# --------------------------------------------------------------------------


def linspace_grid(params: Mapping[str, ParamSpec], seed: int) -> list[dict[str, Any]]:
    """Full cross product of every axis, in the parameter declaration order.

    Order comes from the YAML mapping's insertion order, which pydantic
    preserves, so the same catalog produces the same ordering everywhere. That
    matters less than it looks -- identity is by hash, not by position -- but a
    stable order makes ``plan.json`` diffable, and diffable is the whole point
    of writing a plan to a file.
    """
    rng = random.Random(seed)
    axes = [(name, _axis_values(spec, rng)) for name, spec in params.items()]
    total = 1
    for _, values in axes:
        total *= len(values)
    _check_size(total, "linspace_grid")
    names = [name for name, _ in axes]
    return _finish(
        [dict(zip(names, combo)) for combo in itertools.product(*[v for _, v in axes])]
    )


def random_sample(params: Mapping[str, ParamSpec], seed: int) -> list[dict[str, Any]]:
    """``n`` independent draws from the joint space.

    Every parameter contributes one value per sample rather than one axis of a
    cross product: constants stay constant, choices are drawn uniformly, a
    ``range``/``steps`` parameter is drawn uniformly from its grid points.
    """
    n = _sample_count(params)
    _check_size(n, "random_sample")
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        rows.append({name: _draw(spec, rng.random()) for name, spec in params.items()})
    return _finish(rows)


def latin_hypercube(params: Mapping[str, ParamSpec], seed: int) -> list[dict[str, Any]]:
    """Stratified coverage: each axis split into ``n`` bins, one sample per bin.

    Better than independent random draws at low sample counts, which is the
    regime that matters here -- 24 scenarios is a normal set size, and with
    independent draws 24 points routinely leave a quarter of an axis untouched.
    """
    n = _sample_count(params)
    _check_size(n, "latin_hypercube")
    rng = random.Random(seed)
    columns: dict[str, list[Any]] = {}
    for name, spec in params.items():
        strata = list(range(n))
        rng.shuffle(strata)
        columns[name] = [_draw(spec, (strata[i] + rng.random()) / n) for i in range(n)]
    return _finish([{name: columns[name][i] for name in params} for i in range(n)])


__all__ = [
    "MAX_SCENARIOS",
    "Generator",
    "ScenarioFilter",
    "latin_hypercube",
    "linspace_grid",
    "random_sample",
]
