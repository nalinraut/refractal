# Open question: is a smoke run the same experiment as a full run?

A release blocker, and smaller than
[the worker-unit question](planner-question-worker-unit.md) — but not a tweak,
because it changes what "the same experiment" means.

## Where it stands

`tier` **is** in `plan_id` today, via `experiment_identity`. The reasoning, from
that function's docstring:

> `tier` *is* included, because it changes which scenarios exist. That has a cost
> worth stating: a smoke run and a full run of the same catalog get different ids,
> so the smoke results cannot be reused as a head start on the full run even
> though the smoke set is a strict subset of it.

Both halves of that are true, which is why it is a question rather than a bug.

## What `tier` actually does

```python
TIER_FRACTION = {"smoke": 0.10, "regression": 0.50, "full": 1.0}
```

It subsamples scenarios by a **deterministic hash fraction** — `_hash_fraction`
maps a `scenario_hash` onto `[0, 1)` — stratified, floor of one per set. So the
subsetting is stable and nested: a scenario in `smoke` is in `regression` is in
`full`, for the same catalog, forever.

That nesting is what makes the question live. If the tiers were independent
samples there would be nothing to discuss.

## The case for keeping it in

A tier changes **which scenarios exist**, and that is the same kind of fact as
which tasks exist. Two runs over different scenario sets are not the same
experiment, and joining them would mean averaging a 10% sample with a 100% one —
weighting the smoke scenarios twice and silently overrepresenting them.

## The case for taking it out

The subset is *nested and deterministic*, so the overlap is exactly known rather
than statistical. `compare` already works per comparison unit and already reports
how many scenarios are in common; a smoke run and a full run could join on their
shared scenarios with the extra full-run scenarios simply having one arm's worth
of data. That is the "tier promotion" case: run smoke in CI on every commit, run
full nightly, and let the nightly reuse what the smoke run already paid for.

The cost of keeping it in is therefore real compute, repeated, for a subset whose
membership is a pure function of the hashes.

## Why it is not obvious

The two positions disagree about what `plan_id` is *for*.

* If `plan_id` answers **"may these rows be averaged together"**, tier belongs
  out: the rows are individually fine, and the weighting problem is `compare`'s
  to handle, which it already does by reporting units in common.
* If `plan_id` answers **"was this the same experiment"**, tier belongs in: a
  10% run was a different experiment, whatever its rows individually support.

Everything else in `experiment_identity` was decided under the second reading —
`results_uri` and `execution_mode` are out because they do not change which
episodes exist, `tier` is in because it does. Taking `tier` out would be adopting
the first reading for one field, which is the thing to avoid: a rule that holds
for four fields and not the fifth is not a rule.

## What would settle it

**Whether tier promotion is wanted in practice.** Not a design argument — a usage
one. If nobody ever wants a nightly full run to reuse that morning's smoke
results, the cost is zero and the current answer is right on the stricter
reading.

Worth deciding before publishing, because it is exactly the kind of field whose
answer becomes load-bearing the moment somebody has results under it.

### The third option, which may be the real answer

Keep `tier` in `plan_id` and give `compare` an explicit, opt-in way to pool two
plan ids whose scenario sets are known to nest — stated as a flag, recorded in the
verdict, and refused when the nesting does not actually hold. That keeps
`experiment_identity` reading one way, and makes promotion a thing somebody asks
for out loud rather than a thing that happens because two ids collided.
