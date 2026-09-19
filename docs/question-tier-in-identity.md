# Closed: a smoke run is a different experiment, and may be reused explicitly

**Answered: `tier` stays in `plan_id`, and `compare --promote-from` makes reuse
an explicit request that is refused when the runs do not nest.**

Kept as a document because the reasoning is the point. The answer preserves the
reading every other field in `experiment_identity` was decided under — "was this
the same experiment" rather than "may these rows be averaged" — and pays the cost
that reading implies, rather than making one field an exception.

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

## The answer, implemented

`compare RESULTS PLAN_ID --promote-from OTHER_PLAN_ID`. The two runs keep their
different ids; nothing is merged. Their rows are pooled for one comparison, on
request, and the verdict records that it happened and from where — a rendered
verdict that did not say so would read as an ordinary one, which is the failure
promotion exists to avoid.

### The nesting check is exact, and that is the design

`episode_id` is derived from `(scene_hash, task_hash, scenario_hash, seed,
checkpoint_id)`. **Tier is not in it.** Subsetting changes which episodes exist,
not what any of them is — so a smoke episode and the full run's counterpart
already carry the same id.

Which makes the condition exact rather than a heuristic: promotion is legal when
every promoted episode id is one the target plan contains. No tolerance, nothing
to tune. If anything other than the tier moved, the hashes moved, the ids do not
match, and it refuses.

Verified against the two real runs, which differ in `max_steps` rather than tier:

```
error: 600 of 600 promoted episode(s) are not in this plan ... The runs do not nest.
exit=2
```

`max_steps` is in `task_hash`, so every id moved. That is not a tier difference
and pooling them would average two experiments.

### What the tests check that the fixtures could not

The unit fixtures build subsets by hand, which would pass even if `tier` did not
nest at all. So there is a test against the real mechanism: plan the same catalog
at `smoke` and at `full`, and assert the smoke episode ids are a strict subset.

That assumption — `TIER_FRACTION` subsampling by a deterministic hash fraction,
so a smoke scenario is a regression scenario is a full scenario — was load-bearing
for this whole design and nothing had checked it. It holds.

### Direction matters

A superset does not nest inside a subset: promoting a full run into a smoke run
is refused. Tested, because the asymmetry is easy to lose in an implementation
that thinks of the check as "do these overlap".
