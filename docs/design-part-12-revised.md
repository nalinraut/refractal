# Part 12 — Why build it when a model can generate it

*Revised 2026-09-09, after watching the loop close upstream. Drop-in replacement
for Part 12 of `refractal-design.md`.*

GPT-6 Astra shipped Sept 3, 2026. A frontier model can one-shot a plausible
Refractal: a planner, a catalog schema, Helm templates. Code generation is not the
moat and pretending otherwise would be silly.

What one-shot generation produces is the **generic** version. What it does not
produce are the decisions that came from contact with the actual system:

- topology family is the compiled `MjModel`, not the task
- vla-eval's shard assignment degenerates when
  `gcd(num_shards, episodes_per_task) > 1`, and their documented default hit the
  worst case exactly
- `max_batch_size` defaults to 1, so the GPU is probably idle
- batch composition changing mid-run makes identical scenarios give different
  numerics
- Docker `--gpus` cannot partition VRAM
- their success-rate denominator includes crashed workers

None of those are in a prompt. They came from cloning repos and measuring.

## The shard finding is now fixed upstream, and that is the interesting part

The second bullet is history. It was filed, and six days later a maintainer
merged `0865d42`:

```python
_SHARD_SHUFFLE_SEED = 42

def _shard_work_items(work_items, num_shards, shard_id):
    """Fixed-seed shuffle so a shard never collects one episode index across
    every task (gcd(num_shards, episodes_per_task) > 1 does that); re-sort by
    task to keep env rebuilds rare."""
```

The PR carries `Co-Authored-By: Claude Fable 5.1`. Three of the five commits
since the v0.5.0 pin do.

Taken alone that is easy to under-read as a maintainer's convenience — a faster
way to close issues that someone else specified. The paper closes off that
reading. §IV-A of arXiv 2603.13966v2 states that *"an AI agent (Claude Code with
Opus 4.6) reviewed 1,704 papers via MCP tool integrations"* to build the
657-result leaderboard across 17 benchmarks. That is not an agent helping with a
task the team was doing anyway; it is the method by which one of the paper's
four contributions exists at all.

**So agent-driven work is load-bearing here, not incidental.** The six-day
turnaround from filed issue to merged fix is a symptom of that, not the evidence
for it — and a turnaround time is the weaker fact anyway, since it can be
explained by an attentive maintainer. Published methodology cannot.

The obvious reading is still that the moat shrank: a finding that took a week of
measurement was closed by an agent in an afternoon. That reading is half right,
and it inverts if you look at what the agent actually consumed.

**The agent did not find the bug. It implemented a report.** What it was handed
was a specification: the mechanism (`gcd(num_shards, episodes_per_task) > 1`),
the consequence (a shard collects one episode index across every task), and
enough context to know that re-sorting by task afterwards matters because
otherwise you pay an environment rebuild per item. Given that, the fix is
twelve lines and largely mechanical. Without it, the symptom is "some shards
seem slower" and nothing in the codebase looks wrong.

So the cost of a finding decomposes, and the two halves are moving in opposite
directions:

| | Cost trend | Why |
|---|---|---|
| **Implementing** a specified defect | → 0 | Agents on the issue tracker, on both sides |
| **Specifying** one | unchanged | Requires suspecting the right thing and measuring it |

Code generation collapsed the first column. It did nothing to the second,
because the second is not a generation problem — it is knowing which measurement
settles a question, and being willing to run it.

## What is actually durable

Three things, in increasing order of scarcity:

1. **Knowing which measurement settles a question.** Sharding looked fine until
   someone computed max-worker-time over mean-worker-time and got 1.025, which
   is what said "a smarter scheduler can win 2.5%, do not build one." The same
   instinct is what noticed the gcd collision, and it is why `refractal plan`
   prints a cost estimate before spending anything.
2. **Spotting that generated infrastructure is subtly wrong in a way that
   quietly corrupts numbers.** A denominator that includes crashed workers.
   A join on `scenario_hash` that silently merges two tasks. A `plan_id` that
   folds in the format version and orphans every prior result on a cosmetic
   bump. Each of those produces output that looks correct.
3. **Specifying a defect precisely enough that fixing it becomes mechanical.**
   This is the one the upstream PRs demonstrate, and the one worth getting
   deliberate about, because it is now the step that converts a finding into a
   fix someone else pays for.

> A one-shot eval platform that counts infra failures as policy failures looks
> perfect and produces garbage. Someone has to notice.

That sentence survives the revision unchanged. Astra with tool access could
clone vla-eval and find the shard collision — probably faster than we did. What
it will not do unprompted is doubt a number that looks fine.

## The honest version of this argument

It applies to Refractal too. A well-specified Refractal is generatable, and this
document is most of that specification. Nothing here is protected by being hard
to type.

What is not in the document is the contact that produced it: the run that
measured 1.025, the afternoon spent working out that batch composition changing
mid-run perturbs numerics, the read through `_build_task_result` that noticed
what the denominator was. Those are what the document is a *compression of*, and
the compression is lossy in the direction that matters — you can regenerate the
text without being able to regenerate the judgment that would tell you when the
text is wrong.

## Practical implications

**Build it with the agents.** The implementation is the cheap part now. Use
Astra or Claude Code and spend the time on the design decisions and the
measurements, which are the parts that do not compress.

**File the findings upstream, and file more of them.** This is a change from the
original conclusion, which treated findings as private advantage. Filed
findings are now close to merged fixes, at a cost of one well-written issue with
file and line numbers. Every one that lands moves work off Refractal's side of
the boundary permanently — an upstream fix needs no adapter, no wrapper, no
route-around, and no maintenance when the pin moves. Route around a defect and
you own the workaround forever; specify it well and someone else's agent
removes it in a week.

Standing candidates from this work, in the order worth filing:

1. **`_build_recorder` has no config hook**, despite `NullEpisodeRecorder`
   proving the seam already exists. A config hook removes the subclass from
   Refractal entirely, and it is a smaller ask than the others because it
   changes no semantics — only where the recorder class comes from. Note the
   `self._store` gate has to move too, or the hook is inert.
2. **`db_path` is plumbed to every model server and read by none.** Either it
   grows a caller or it is an affordance that should be documented as one. And
   "a path to one SQLite file" does not generalise to a URI plus a partition
   key, which is worth saying before anyone builds on it.
3. **`_ALL_RECORD_FIELDS` accepts whatever each benchmark declares**, so a
   validation list that every implementation redefines is not validating much.
   Largest ask of the three, because it changes what the field means.

**For the pitch:** do not sell a company code they can generate, and do not sell
them findings that a maintainer's agent will fix for free next month. Sell them
the judgment about what their eval numbers actually mean. That is the part with
no generation path.
