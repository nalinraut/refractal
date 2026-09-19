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


def object_in_slot(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    """A success check with arguments, which is the shape that matters.

    ``predicate_args`` is hashed into ``task_hash``, so slot 4 and slot 7 are
    different tasks rather than one task run twice. That is the whole reason the
    args are immutable and in the identity.
    """
    return state.get("slot") == args.get("slot")


def ee_near(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    """A phase predicate. Phases are in ``task_hash`` too."""
    return bool(state.get("near"))


def lifted(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    return bool(state.get("lifted"))


def over(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    return bool(state.get("over"))


def seated(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    return bool(state.get("seated"))


def object_in_container(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    return state.get("container") == args.get("container")


def cube_in_bowl(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    """Success when the cube is within ``tolerance`` of the bowl centre."""
    tolerance = float(args.get("tolerance", 0.05))
    cube = state.get("cube_pos", (0.0, 0.0, 0.0))
    bowl = state.get("bowl_pos", (0.0, 0.0, 0.0))
    return sum((c - b) ** 2 for c, b in zip(cube[:2], bowl[:2])) <= tolerance**2


def cube_left_of(state: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    """Success when the cube ends up left of ``x``."""
    return float(state.get("cube_pos", (0.0,))[0]) < float(args["x"])


def within_reach(scenario: Mapping[str, Any]) -> bool:
    """A scenario filter: drop cube poses outside a plausible workspace.

    Note the signature. A **filter** takes one argument, the scenario dict, and
    is called by ``refractal build``. A **predicate** takes ``(state, args)`` and
    is called by the adapter during an episode. They are different contracts and
    swapping them is an easy mistake -- the CI job that exercises the filter path
    was written against ``cube_left_of`` first, which takes two.
    """
    x = float(scenario.get("cube_x", 0.0))
    y = float(scenario.get("cube_y", 0.0))
    return 0.06 <= x <= 0.19 and abs(y) <= 0.08


class GR00TServer:
    """A stand-in for a model server, so ``run.yaml`` names one that imports.

    ``server`` is a render-time field -- nothing resolves it during ``plan`` --
    but a shipped example whose import strings do not resolve teaches the format
    wrongly, and a reader cannot tell which dangling name is deliberate.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class EchoServer:
    """A model server that does nothing, so the example names something importable."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def predict(self, observation: Mapping[str, Any]) -> list[float]:
        return [0.0] * 7


__all__ = ["EchoServer", "cube_in_bowl", "cube_left_of", "within_reach"]
