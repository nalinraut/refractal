# Checks that cannot fail

Six times in this project a check ran, reported success, and never exercised
the thing it names. Each was found by accident, by a different route, and each
one had been passing for some time. They are one shape, and the shape is worth
naming because the next instance will not look like any of these.

> A check whose pass and whose "nothing to check" look identical is not a
> check.

It is the same defect the rest of this codebase is built to prevent, one level
up. A verdict that prints nothing when a test declines cannot be told from one
where the test passed, which is why `declined` prints `none`. A check that
cannot fail cannot be told from one that holds — and it is worse, because
somebody wrote it down and stopped worrying.

## The six

**A test fixture asserted against a column that did not exist.** A Phase 6
test built rows carrying `termination_map_digest` before that column was in
the schema. A hand-built dict never passes through the Arrow schema, so the
key rode along, the test passed, and it concealed a refusal that could never
fire on a real result. Nothing in the production path would ever have read
that key.

**A fixture whose arithmetic divided evenly.** The declared-versus-realized
chunk check compared `steps / decisions` against the declaration. In the
fixture every episode's step count was a multiple of the chunk length, so the
ratio came out exactly 10 and the check looked correct. On recorded data the
last chunk of an episode is partial: 104 steps over 11 calls is 9.455, and the
check reported 67 distinct "chunk lengths" for an arm that had run at one.

**A regression test whose two values were tied.** The test asserting that an
adjusted p-value is never below the p-value it adjusts was written against a
fixture where both were 1.0. `adjusted >= p` holds for tied numbers whichever
way they are paired, so it passed with the bug still in place.

**An optional argument every test supplied.** `permutation_test` and
`budget_curve` were given `ghost_penalty` at all 17 calls between them, so the
None branch — a refusal, raised per unit inside a resampling loop — had never
run. `build_units` was given `checkpoints` at all 238 calls, so the branch
that discovers the arms from the data had never run. The default is what
production uses.

**A guard that stopped recognising its subject.** The fixture-column guard
matches row fixtures by their keys. A refactor renaming `checkpoint_id` would
make it match nothing, find no violations, and pass. Guards that enumerate
have this everywhere: the set they walk can become empty.

**A success signal that could not distinguish two outcomes.** Three of these.
`git check-ignore` exits 0 when any rule matches, *including a negation*, so
exit 0 cannot tell "ignored" from "explicitly not ignored". The count of
RECORD lines with an empty hash field is 1 for an editable install and also 1
for numpy and pytest — it is RECORD's own line, which carries no hash by
construction, so a check built on it fires on everything.
`distributions(path=None)` is not `distributions()`: the first raises, because
importlib.metadata iterates what it is given rather than falling back.

## A sibling: checks that match a proxy for the thing

Adjacent to the six, and worth keeping separate because the rule is different.
Above, a check's pass and its nothing-to-check look alike. Here, the check
examines a **stand-in** for what it names -- and the failure runs both ways.

Twice, the stand-in was source text and the thing was structure.

**A verdict's wording depended on nothing reading `server_args`, so a test
asserted the string was absent from the execute path.** It fired on a comment
that explained why nothing reads it. So the test was asserting that nobody
MENTIONS the field, which is a different claim: it goes off on any passing
remark, and a real read could hide behind a rewording -- `getattr(c,
"server" + "_args")`, a loop over a dict of declarations, an alias. Replaced by
an AST walk for `Attribute(attr="server_args")` and for a string constant
outside a docstring.

**The harness's `_call_seed` had to agree with ours, and comparing the source
would have been the obvious check.** It is the same trap from the other side:
two spellings of one arithmetic are equal while a changed constant inside an
unchanged-looking line is not. So that check extracts the function by AST,
executes it, and compares 35 (seed, step) pairs -- the values, not the text.

The pattern: **when a claim is about behaviour, check behaviour.** Source text
is a proxy that both over-matches and under-matches, and it does so silently in
both directions. Structure is closer; running the thing is closest.

## What stopped each one

These are cheap, and each came out of a specific failure above.

**Reintroduce the bug and watch the test fail.** Then restore, and confirm the
source is byte-identical afterwards. A test never seen failing is not yet
written. This is the only one of these that catches an unknown shape rather
than a known one, and it caught the tied-fixture case: the test was correct
and the fixture could not exercise it.

**Put a count floor on any guard that enumerates.** The fixture-column guard
asserts it still matches at least ten fixtures. A guard that finds nothing to
check must fail, not pass.

**Pin the legal near-misses on any guard that refuses.** The phase-cycle guard
refuses a cycle; it also tests a four-long chain and a diamond, which are the
legal cases it most resembles. A guard with no near-miss tests can be
over-refusing and look correct.

**Choose fixture values that are not tied, zero, uniform, or on a boundary.**
Tied values satisfy inequalities in both directions. Evenly dividing values
hide a partial remainder. A 3-of-6 majority is the knife edge of a
greater-than rule. Pick values whose wrongness would be visible.

**Test the absence when a document's wording depends on one.** The verdict says
nothing verifies `server_args`, which is true only while nothing in the execute
path reads it. A test asserts that, so the day something does, the wording gets
revisited instead of going quietly stale. This is the count floor inverted, and
it applies anywhere prose asserts that something does not happen.

**Never let every test supply an optional argument.** Measure it rather than
reading for it: wrap each function, record `optional - set(bound.arguments)`,
run the suite. **And count the calls** — a wrapper that was never reached
reports every default as untested, which is itself an instance of this
document. Patch the package re-export as well as the defining module, because
tests usually import from the package.

**Verify against a second artefact where one exists.** The chunk length was
recovered twice, from unrelated records: the gap between consecutive inference
calls in the server's log (exactly 10 on 13,205 of 13,205 consecutive calls)
and `decisions` per episode via `decisions == ceil(steps / k)`. Agreement
between two derivations over different data is worth more than any single
check, and disagreement is a finding either way.

## Where this leaves the taxonomy

`reading-a-comparison.md` says to refuse on a substantive difference and flag
on unverifiability, and warns that the converse error — saying nothing — turns
unverifiability into apparent verification. This document is the same sentence
about tests rather than results. A check that cannot fail is a check that says
nothing while appearing to say something, and the cost is higher than silence
because nobody looks again.

## Two footnotes

A `_`-prefixed helper in a `unittest.TestCase` subclass shares a namespace
with the framework. `_outcome` is taken -- TestCase keeps an `_Outcome`
instance there -- so a helper by that name is shadowed at run time and the
call fails with "object is not callable" from a line that looks fine.

A guard belongs in the layer the thing it guards lives in.
`chunk_lengths_consistent_with` started in `compare/pairing.py` and the
execute path needed it, which would have made execute import from compare --
backwards, since compare reads what execute writes. It moved to
`schema/outcome.py`, where the other `(steps, decisions)` invariants already
are: the right home rather than a convenient one.
