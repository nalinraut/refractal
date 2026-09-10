"""Symbols the ``refractal init`` example points at.

Real, so the example's import strings resolve rather than dangling. Trivial, so
nobody mistakes them for something to build on: the predicates are what a
success check *looks* like, and ``EchoServer`` exists so ``run.yaml`` names a
server that can be imported.

Under ``--backend local`` none of these are called -- the fake benchmark
produces outcomes directly. They matter the moment a real adapter arrives, and
they are here now so the example does not ship a promise it cannot keep.
"""

from __future__ import annotations

from typing import Any, Mapping


def cube_in_bowl(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    """Success when the cube is within ``tolerance`` of the bowl centre."""
    tolerance = float(args.get("tolerance", 0.05))
    cube = state.get("cube_pos", (0.0, 0.0, 0.0))
    bowl = state.get("bowl_pos", (0.0, 0.0, 0.0))
    return sum((c - b) ** 2 for c, b in zip(cube[:2], bowl[:2])) <= tolerance**2


def cube_left_of(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    """Success when the cube ends up left of ``x``."""
    return float(state.get("cube_pos", (0.0,))[0]) < float(args["x"])


class EchoServer:
    """A model server that does nothing, so the example names something importable."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def predict(self, observation: Mapping[str, Any]) -> list[float]:
        return [0.0] * 7


__all__ = ["EchoServer", "cube_in_bowl", "cube_left_of"]
