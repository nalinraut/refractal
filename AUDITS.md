# Audit notes

Not documentation. This records how *verification itself* has failed in this
repo, because that failure mode has now cost more than any single bug.

Everything checkable lives in a check: `scripts/lint_guards.py` for guards,
`scripts/verify_harness_claims.py` for assumptions about vla-eval,
`scripts/lint_provenance_claims.py` for "measured" comments,
`tests/test_docs_match_code.py` for claims the docs make about the code. This
file holds only what no script can hold: the shape of a mistake.

## The adjacent-question pattern

**Verify a question next to the one being asked, and accept the answer.**

The evidence is real and the reasoning is sound. What goes wrong is the gap
between the question answered and the question asked, and the gap closes
silently because the answer looks like an answer.

Confirmed instances:

| what was checked | what was being asked |
|---|---|
| `PlannedScene.model_fields` had `metrics` | the *plan* carried a metrics value |
| Cloudflare returned 200 | the project existed |
| `pgrep -f X` matched | a process other than the matching shell was running |
| the tail of `compare` output | the whole output, which had two tasks |
| `api_version` appeared in the docs | `apiVersion`, the YAML spelling, appeared |

The last two are from the documentation audit of 2026-10-01 — a pass whose
stated job was finding exactly this, in which it was committed twice in one
sitting. Both were caught before the findings were acted on, and both were
reported as withdrawn rather than quietly dropped, which is the only reason
they are countable.

### The positive form

The habit that catches it, stated as an instruction rather than a warning:

> **Point at the row this check would reject. If you can't, the fixture cannot
> exercise it.**

Generalised beyond fixtures: name the observation that would make this
conclusion false, then confirm the check can produce that observation.

### Two polarities, and the second is newer

The familiar polarity is **"I expected it to work, so I believed it did."** The
evidence fits more than one story and the comforting one gets picked.

The inverse showed up on 2026-10-01 while re-verifying vla-eval across two
minor releases. `episode_idx = ep % max_ep` in 0.8.0's `orchestrator.py`
looked like a change to the counter that selects init states — precisely what
`check_index_contract` exists to guard. It had been in 0.6.0 all along.

Nothing was wrong with the reading. What was wrong is that it *was* a reading:
a single version inspected while looking for changes, where only a diff can
show one. Had it been reported, the result would have been a false alarm and
plausibly a defensive change to code that was working correctly.

So: **"I was looking for changes, so I saw one."** The expected story was the
alarming one, and alarm is not a safeguard against the pattern — it is another
prior for the evidence to satisfy. A surface digest firing is an instruction to
*diff*, not to *read*, and the two feel identical while you are doing them.
