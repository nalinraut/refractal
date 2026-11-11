"""What a comparison reports, assembled in one place.

The point of the type is that every number arrives with what it needs to be
read: a p-value with its family size and its grain, a restricted mean with
its tau and penalty, a median with whether it was defined. A bare number in
this output is a bug.

The ordering is deliberate. The median leads among the scalars because it is
the only one needing no convention defended -- no tau, no penalty, nothing to
justify -- and RMST follows with its parameters stated inline rather than as
a footnote.

And an irony worth stating where it will be read. Three phases of work moved
the measurement from a bit to a distribution, and the McNemar path still
reduces every scenario's seeds to one bit through the dichotomy rule. It
stays as the convention-free headline, but it is now the LOSSIEST instrument
here, and this report must not present it as the strongest. The grain column
is what makes that visible without editorialising: McNemar's statistic sees
one bit per scenario, the permutation's sees every episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Scalar:
    """One number, with everything needed to read it."""

    name: str
    value: float | None
    #: Stated inline, never as a footnote. "rmst" alone is not a number
    #: anyone can act on; "rmst(tau=300, ghost_penalty=300.0)" is.
    parameters: tuple[tuple[str, str], ...] = ()
    #: Clustered interval where one was computed.
    low: float | None = None
    high: float | None = None
    #: Why there is no value, when there is none. The median of a cell where
    #: fewer than half succeeded is undefined, not zero and not the budget.
    undefined_reason: str | None = None

    def render(self) -> str:
        head = self.name
        if self.parameters:
            head += "(" + ", ".join(f"{k}={v}" for k, v in self.parameters) + ")"
        if self.value is None:
            return f"{head}: undefined ({self.undefined_reason or 'no value'})"
        body = f"{self.value:.4g}"
        if self.low is not None and self.high is not None:
            body += f" [{self.low:.4g}, {self.high:.4g}]"
        return f"{head}: {body}"


@dataclass(frozen=True)
class TestLine:
    """One p-value, with the grain it was computed at.

    GRAIN IS NOT OPTIONAL. McNemar collapses each scenario's seeds to one bit
    before looking; the permutation test keeps every episode inside its
    functional. Both p-values are correct for what they test and they are NOT
    commensurable as evidence -- one discarded five sixths of its input before
    looking. A reader seeing McNemar p=1.00 beside permutation p=0.008 will
    otherwise conclude the second is simply more powerful, when part of the
    difference is that it was given more data.
    """

    name: str
    p_value: float
    #: e.g. "scenario, seeds collapsed via dichotomy=majority" or
    #: "cell, seeds preserved inside the functional"
    grain: str
    adjusted_p: float | None = None
    #: Holm family size. An adjusted p cannot be read without it.
    m: int | None = None
    detail: tuple[tuple[str, str], ...] = ()

    def render(self) -> str:
        out = f"{self.name}: p={self.p_value:.4g}"
        if self.adjusted_p is not None:
            out += f"  holm={self.adjusted_p:.4g} (m={self.m})"
        out += f"  [grain: {self.grain}]"
        if self.detail:
            out += "  " + ", ".join(f"{k}={v}" for k, v in self.detail)
        return out


@dataclass(frozen=True)
class Authorship:
    """Who wrote the apparatus, per part.

    Named ``authorship`` rather than ``provenance`` -- that word already names
    one of the three column categories here and would mean two things -- and
    rather than folding into ``tier``, which is smoke/regression/full.

    THREE parts, because any one can be customer-supplied independently:

    ``adapter``  who drives the simulator
    ``scene``    who supplied the assets
    ``task``     who wrote the success predicate

    All three ``catalog`` means a verdict comparable against the catalog. Any
    one ``customer`` means it is not, and the field names which.

    **A customer-apparatus verdict is still internally valid.** Both arms ran
    through the same adapter, so an adapter bias cancels in the pairing --
    that is what pairing is for. What it cannot support is a claim against
    the catalog, or across customers, because the thing held constant was
    constant only within this comparison.
    """

    adapter: str = "catalog"
    scene: str = "catalog"
    task: str = "catalog"

    @property
    def all_catalog(self) -> bool:
        return all(v == "catalog" for v in (self.adapter, self.scene, self.task))

    def render(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in
                          (("adapter", self.adapter), ("scene", self.scene),
                           ("task", self.task)))
        if self.all_catalog:
            return f"{parts}  (comparable against the catalog)"
        customer = sorted(k for k, v in
                          (("adapter", self.adapter), ("scene", self.scene),
                           ("task", self.task)) if v != "catalog")
        return (f"{parts}  (internally valid -- both arms ran through the same "
                f"apparatus -- but not comparable against the catalog: "
                f"{', '.join(customer)} customer-supplied)")


@dataclass(frozen=True)
class Reproducible:
    """Whether everything needed to re-run was captured.

    Orthogonal to ``Authorship`` and failing for different reasons: a catalog
    run missing ``engine_version`` is unreproducible though every line of it
    was written here, and a customer adapter with complete provenance
    re-runs exactly while remaining conditional on code nobody here read.
    One enum over both would lose the distinction a reader needs.
    """

    complete: bool = True
    #: What is absent, named. "incomplete" with no reason is unactionable.
    missing: tuple[str, ...] = ()
    #: Distributions installed in editable mode when the lock was written,
    #: from ``build``. An external scene's hash is built from version strings,
    #: so edits to an editable checkout move no identity -- the run is not
    #: reproducible from any published release, however complete its
    #: provenance fields are.
    editable: tuple[str, ...] = ()
    #: Whether the question was asked at all. False where no lock was read, so
    #: an empty ``editable`` means "not checked" rather than "none".
    installs_checked: bool = False
    #: Whether anything checked that the servers were configured the way the
    #: arms declare. One line, because the answer is currently always the same
    #: and a reader must not have to infer it from silence.
    #:
    #: `server_args` is in `plan_id` and in the arm descriptor, and Refractal's
    #: execute path NEVER READS IT -- `build_eval_config` sends a server only
    #: its URL. So it describes behaviour without causing it, which no other
    #: field in the identity chain does: `scene_hash` covers files that
    #: actually load and `task_hash` covers the predicate actually evaluated,
    #: while `server_args` covers a label. `plan_id` can therefore be
    #: confidently wrong about how a policy was driven.
    #:
    #: Keeping it in `plan_id` is still right -- two runs declaring different
    #: arguments ARE different experiments, and the declaration is the only
    #: record of intent. What is not right is leaving the record silent about
    #: whether anyone checked it.
    server_args_verified: str | None = None
    #: Non-native fields on the rows, as ``field -> how many rows took which
    #: route``, from each row's ``field_provenance``.
    #:
    #: FLAGS, never refuses, for the reason the whole taxonomy turns on: where
    #: a derivation is exact the rows genuinely are comparable, so refusing
    #: would fragment the result space and add no information. What it stops is
    #: a reader taking a reconstructed field for an observed one -- and a row
    #: can hold three classes at once, which is why it is per field rather
    #: than a single flag.
    provenance: tuple[tuple[str, str], ...] = ()

    def render(self) -> str:
        base = "yes" if self.complete else "no (" + "; ".join(self.missing) + ")"
        # Always stated, including when there is nothing to say. An absent
        # line reads as a check that ran and found nothing, which is the one
        # thing it cannot distinguish itself from -- the same reason
        # `declined` prints `none`.
        if self.server_args_verified:
            base = f"{base}; server_args {self.server_args_verified}"
        if self.provenance:
            routes = "; ".join(f"{field} {how}" for field, how in self.provenance)
            base = f"{base}; DERIVED FIELDS: {routes}"
        if not self.installs_checked:
            return f"{base}; installs not checked"
        if self.editable:
            return (f"{base}; EDITABLE installs: {', '.join(self.editable)} "
                    "-- edits to these move no identity")
        return f"{base}; all installs from a published release"


@dataclass(frozen=True)
class Finding:
    """One statement, and what it is a statement ABOUT.

    Some findings are about an arm and some are about the catalog the arms
    were measured against, and a reader acts on those differently: the first
    is a result, the second means the measurement itself needs fixing before
    any result from it is worth reading. So the subject is a field rather
    than something inferred from the wording.
    """

    #: "arm" or "catalog".
    about: str
    text: str

    def render(self) -> str:
        if self.about == "catalog":
            return f"CATALOG: {self.text}"
        return self.text


@dataclass(frozen=True)
class ArmReport:
    """One arm, as declared and as it behaved."""

    arm: str
    #: The declared descriptor, including chunk_length when set.
    descriptor: Mapping[str, object]
    scalars: tuple[Scalar, ...]
    #: Reported as a rate, never folded into a distribution.
    apparatus: int = 0
    episodes: int = 0

    @property
    def apparatus_rate(self) -> float:
        return self.apparatus / self.episodes if self.episodes else 0.0


@dataclass(frozen=True)
class ComparisonReport:
    """Everything a comparison concluded, and under what choices.

    ``plan_id`` says what ran; ``analysis_hash`` says how it was read. Two
    verdicts are comparable only under the same pair, and a difference in
    either is a refusal rather than a footnote.
    """

    plan_id: str
    analysis_hash: str
    #: smoke | regression | full, from the plan. NOT the "auditable or
    #: conditional" distinction -- no such vocabulary exists in this codebase
    #: today, and inventing one here would put a second meaning on a word the
    #: catalog already uses.
    tier: str
    #: Holm family size, after any pruning for declared-identical arms.
    m: int
    #: Orthogonal to tier, and to each other. See their own docstrings.
    authorship: "Authorship" = field(default_factory=lambda: Authorship())
    reproducible: "Reproducible" = field(default_factory=lambda: Reproducible())
    arms: tuple[ArmReport, ...] = ()
    pairwise: tuple[tuple[tuple[str, str], tuple[TestLine, ...]], ...] = ()
    #: One statement. Carries no p-value, and spans many analysis hashes by
    #: construction, so it records the range swept rather than pretending a
    #: single identifier covers it.
    stability: str | None = None
    stability_swept: tuple[tuple[str, str], ...] = ()
    #: Checks that did not run, and why. A verdict with no mention of a check
    #: reads as one where it ran and found nothing.
    #: Statements that are not a number on an arm: per-phase readings, and
    #: the catalog findings that fall out of them.
    findings: tuple["Finding", ...] = ()
    #: The order each arm attempted its phases in, when the task declares a
    #: prerequisite graph that admits more than one.
    #:
    #: Reported whether or not anything was refused. Choosing a good order is
    #: part of what is being evaluated, so two arms picking different ones is
    #: a finding about the policies rather than only a reason the chained
    #: comparison cannot run -- and if it only ever appeared inside a
    #: refusal, the comparisons that proceeded would hide it.
    traversals: tuple[str, ...] = ()
    declined: tuple[str, ...] = ()
    blocking: tuple[str, ...] = ()

    def apparatus_differs(self, threshold: float = 0.05) -> bool:
        """Whether the arms' apparatus rates are far enough apart to matter.

        Dropping apparatus rows is safe arithmetically and not when the drop
        rate is arm-dependent: the arms are then compared over different
        subsets of the scenarios they were assigned.
        """
        rates = [a.apparatus_rate for a in self.arms]
        return bool(rates) and (max(rates) - min(rates)) > threshold

    def render(self) -> str:
        out = [f"plan_id       {self.plan_id}",
               f"analysis_hash {self.analysis_hash}",
               f"tier          {self.tier}",
               f"authorship    {self.authorship.render()}",
               f"reproducible  {self.reproducible.render()}",
               f"holm family   m = {self.m}"]
        for arm in self.arms:
            desc = ", ".join(f"{k}={v!r}" for k, v in sorted(arm.descriptor.items()))
            out.append(f"\n  {arm.arm}  ({desc})")
            for s in arm.scalars:
                out.append(f"    {s.render()}")
            if arm.episodes:
                out.append(f"    apparatus: {arm.apparatus}/{arm.episodes} "
                           f"({arm.apparatus_rate:.1%})")
        if self.apparatus_differs():
            out.append("\n  WARNING: apparatus rates differ materially across arms; "
                       "the arms were compared over different subsets.")
        for f in self.findings:
            if f.about == "catalog":
                out.append(f"\n  {f.render()}")
        for (a, b), lines in self.pairwise:
            out.append(f"\n  {a} vs {b}")
            for line in lines:
                out.append(f"    {line.render()}")
        if self.traversals:
            out.append("\n  phase order taken")
            for line in self.traversals:
                out.append(f"    {line}")
        arm_findings = [f for f in self.findings if f.about != "catalog"]
        if arm_findings:
            out.append("")
            for f in arm_findings:
                out.append(f"  {f.render()}")
        if self.stability:
            swept = ", ".join(f"{k}={v}" for k, v in self.stability_swept)
            out.append(f"\n  stability: {self.stability}")
            if swept:
                out.append(f"    swept: {swept}  (spans many analysis hashes "
                           "by construction)")
        # ALWAYS printed, including when empty. A section that vanishes when
        # there is nothing in it leaves a reader unable to tell a check that
        # ran and passed from one that never ran -- which is the whole reason
        # the variance detector's declines were made visible in the first
        # place, and it would be undone here by an `if`.
        if self.declined:
            for d in self.declined:
                out.append(f"\n  declined: {d}")
        else:
            out.append("\n  declined: none")
        for b in self.blocking:
            out.append(f"\n  BLOCKED: {b}")
        return "\n".join(out)


__all__ = [
    "ArmReport",
    "Finding",
    "Authorship",
    "ComparisonReport",
    "Reproducible",
    "Scalar",
    "TestLine",
]
