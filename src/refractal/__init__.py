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

Provenance fields
-----------------

A **provenance field** records *where a fact came from*. It never enters an
identity hash and it never gates a decision. Its only job is to let a reader
tell an assertion from a measurement, and to let ``compare`` warn.

This shape recurs, and naming it here beats rediscovering it each time:

===========================  ==========================================
``Plan.catalog_hash``        which bytes were read, next to ``plan_id``,
                             which is what the experiment *is*
``ResourceShape.measured_at``  set when a prober measured the shape, unset
                             when a human declared it
``FilterEntry.source_sha``   the filter body that produced the recorded
                             survivors; verified best-effort, deliberately
                             outside the staleness key
``episodes.session_id``      which run produced the row, so a resumed
                             ``harness_version``           comparison can be flagged for latency
===========================  ==========================================

Two rules follow, and both have already been broken once by accident:

* **A provenance field must not be promoted into an identity.** Folding
  ``plan_schema`` into ``plan_id`` was exactly this mistake -- a format version
  is provenance, and using it as identity orphans every prior result on a
  cosmetic bump.
* **An identity must not be demoted into provenance.** If a fact changes what
  the numbers mean, it belongs in a hash or a gate. ``scene_hash`` disagreeing
  across checkpoints blocks the comparison; it is not a note.

The test for which one you have: *would two runs differing only in this field
still be comparable?* Yes means provenance. No means identity.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
