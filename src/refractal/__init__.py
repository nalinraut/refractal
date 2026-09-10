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

Identity, precondition, provenance
----------------------------------

Three kinds of field, not two. The middle one is easy to collapse into either
neighbour and is where the interesting mistakes live.

=============  =============  ==================  ===================================
kind           in ``plan_id``  gates?              instances
=============  =============  ==================  ===================================
identity       yes            by construction     ``scenario_hash``, ``task_hash``,
                                                  the checkpoint set, ``seeds``
precondition   no             yes, loudly         ``scene_hash``, ``harness_version``
provenance     no             never               ``catalog_hash``, ``measured_at``,
                                                  ``source_sha``, ``session_id``
=============  =============  ==================  ===================================

**Identity** decides which episodes exist. Change one and you have a different
experiment, so results recorded before and after must not join -- and by
construction they cannot, because the ``plan_id`` differs.

**Precondition** decides whether a comparison is *meaningful*. It is kept out of
the key deliberately: putting ``scene_hash`` in the join key would make a mesh
edit produce an empty join, and an empty join is a legal result that raises
nothing. Kept out and checked separately, the same edit produces a sentence
somebody has to read. ``harness_version`` is the same shape -- vla-eval's own
paper reports a harness-side integration parameter moving a success rate by 55
points -- and gets the same treatment: out of ``plan_id``, because pinning it
there would invalidate every historical comparison on a dependency bump, but
gating ``compare``, because two runs from different harnesses may not be
comparable at all.

**Provenance** records where a fact came from. Its only job is to let a reader
tell an assertion from a measurement, and to let ``compare`` annotate.

Two questions separate them, and you need both:

1. *Would two runs differing only in this field be the same experiment?*
   No -> identity.
2. *Should ``compare`` still produce a number?*
   No -> precondition. Yes -> provenance.

``session_id`` answers yes to the second, with a note about warmup and thermal
state. ``scene_hash`` answers no. ``harness_version`` answers no, and that is a
decision taken explicitly rather than by omission -- with an override, because
most harness commits change no behaviour at all.

Both directions have been got wrong here already, which is why this is written
down: folding ``plan_schema`` into ``plan_id`` promoted provenance to identity,
and leaving a ``scene_hash`` mismatch as a note would have demoted a
precondition to provenance.
"""

try:  # single source of truth is pyproject; this mirrors it at runtime
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("refractal")
except (ImportError, PackageNotFoundError):  # running from a source tree
    __version__ = "0.0.0+source"

__all__ = ["__version__"]
