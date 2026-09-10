"""Refractal -- scene-coherent placement and paired comparison for policy evaluation.

The harness answers *"what did this checkpoint score."* Refractal answers *"did
my change help."*

Refractal is a compiler, not a runtime: a catalog goes in, a ``plan.json`` comes
out, and the plan executes. The schedule is decided once, up front, and written
to a file you can read, diff and commit -- which for evaluation is better than a
runtime scheduler, because placement becomes part of the provenance.

Packages, ordered by what they are allowed to touch:

===============  ====================================  =====================
Package          Does                                  Touches infra?
===============  ====================================  =====================
``schema``       catalog, validation, identity          no
``resolve``      catalog -> ``plan.json``               no
``execute``      runs a plan; backends                  yes, only this one
``compare``      Parquet -> verdict                     no
``cli``          wires them together                    --
===============  ====================================  =====================

Plan time versus render time
----------------------------

One rule decides which side of the ``plan.json`` boundary a fact belongs on:

    **Anything that differs between two people running the same experiment is
    render-time, not plan-time.**

A plan is a portable description of an experiment. If two people can execute the
same plan and get artifacts that are not interchangeable, the plan is carrying
something it should not.

Plan-time, and inside ``plan_id``: scenes, tasks, scenario sets, checkpoint ids,
seeds, seed base, tier. Everything that decides *which episodes exist*.

Render-time, and outside ``plan_id``: ``results_uri``, the uid and gid the
containers run as, ``execution_mode``, hardware profile, and every placement
decision derived from it -- device assignment, cpuset, worker count.

Placement is the edge case worth stating explicitly, because it *is* written
into ``plan.json`` and still must not enter ``plan_id``. The plan records it so
the schedule is diffable and reviewable before anything is spent, which is the
point of compiling rather than reconciling. But placement is hardware-dependent
and therefore not portable, so it is a rendering of the experiment rather than
part of its identity. Two people who plan the same catalog on different machines
get different worker layouts, the same ``plan_id``, and results that join.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
