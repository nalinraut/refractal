# Harness integration contract

What Refractal relies on in `allenai/vla-evaluation-harness`, verified against
the source rather than the README. Anything here that stops being true is a
broken build, so re-check it when moving the pin.

## The pin — not as simple as "take HEAD"

`v0.5.0` **is** `4aeb436`; the tag points at it exactly. So PyPI's
`vla-eval==0.5.0` contains none of the five later commits, and both fixes that
concern Refractal are unreleased.

| Option | Gets | Costs |
|---|---|---|
| `vla-eval==0.5.0` | resolvable from PyPI, cacheable, publishable | no `dc2c4ba`, no `0865d42` |
| `vla-eval @ git+https://github.com/allenai/vla-evaluation-harness@35f1200` | both fixes | a PEP 508 direct reference |

**The direct reference is not free.** PyPI rejects uploads whose metadata
contains a direct URL dependency. If `pip install refractal` is the seconds
test — and it is, it is the first line of the Part 10 demo — then Refractal has
to stay publishable, and a git pin in `install_requires` forecloses that.

Which of the two fixes actually justifies the cost:

* **`dc2c4ba` (MRO kwargs) does.** It changes the adapter contract (§4). Under
  `0.5.0`, negotiated observation params stop at a wrapper benchmark instead of
  reaching its parent, and unrecognised keys vanish silently rather than
  warning — so the gate proposed in §4 cannot be built against `0.5.0` at all.
* **`0865d42` (the shard fix) probably does not** — see below. It fixes a code
  path Refractal may never execute.

**Recommendation:** keep `vla-eval>=0.5.0` in `pyproject.toml` so Refractal
stays publishable, and pin the git ref in the *container image* the `execute`
backend builds, where a direct reference costs nothing. The pin then lives in a
Dockerfile, which is also where the engine version and the rest of the
environment already get fixed. Revisit when `0.6.0` ships, which folds the
question away entirely.

Every surface Refractal subclasses is byte-identical across the range —
`git diff 4aeb436..35f1200` over `recording.py`, `runners/`,
`model_servers/base.py`, `registry.py` and `types.py` is empty — so moving
between them breaks nothing in §1 either way.

### The harness version is a render-time fact, so it must be recorded per episode

The pin that actually runs episodes lives in the Dockerfile, not in
`pyproject.toml`. That makes the harness version an environment fact, and
`dc2c4ba` proves it is a *behavioural* one: it changes which observation
parameters reach a benchmark, so two runs against different harness versions are
not strictly comparable.

It cannot go in `plan_id` — `refractal plan` runs on a bare laptop and has no
way to know what the container will contain. So it gets the same treatment as
`session_id`: **resolved at container startup, written to every episode row, and
flagged by `compare` when a comparison spans two versions.**

Note the asymmetry with `engine_version`, which *is* in `scene_hash`. That one is
declared in the catalog by `refractal build` and so is knowable at plan time; the
harness version is not knowable until render time and therefore has to be caught
at read time instead. Same class of fact, two different enforcement points,
because of when each becomes known.

### Does the shard fix apply to us at all? An unresolved design tension

`0865d42` fixes vla-eval's round-robin `--shard-id` / `--num-shards` split.
Refractal may never use it, and the design currently says both things:

* `plan.json` gives each worker an **explicit episode list**
  (`PlannedWorker.episodes: []string`). That is finer-grained than a shard index
  and is the whole basis of resumability, since resume subtracts completed
  `episode_id`s from an assignment.
* Part 6 of the design doc says k8s Indexed Jobs map straight onto
  `--shard-id` / `--num-shards` and that "vla-eval's existing model needs no
  rework."

**Those are incompatible.** If Refractal assigns episodes explicitly, it must
drive each worker with its own task set and vla-eval's sharding is dead code
from our point of view — in which case `0865d42` is irrelevant to us and
`JOB_COMPLETION_INDEX` maps onto *our* `(scene_id, shard_index)`, not onto
theirs. If instead we delegate to their sharding, we give up explicit episode
assignment and resume gets much weaker.

The explicit list is the right choice and the rest of the design already assumes
it. Part 6's "Indexed Jobs are static sharding" observation still holds — the
index selects which `PlannedWorker` a pod is — but the claim that it maps onto
vla-eval's `--shard-id` should go. Needs settling before `execute`, not before
`resolve`.

---

## 1. The three hooks, and why no fork is needed

| Hook | Site | Note |
|---|---|---|
| Import-string benchmark loading | `orchestrator.py:232` → `registry.py:14` | `resolve_import_string(cfg.benchmark)`. `"so101_eval.benchmark:SO101Benchmark"` works identically. No registration, no plugin API, no upstream PR. |
| `Task` is a plain dict | `types.py:46` | `Task: TypeAlias = Dict[str, Any]`. A scenario dict drops straight in. |
| Recorder injection by subclass | `orchestrator.py:471` | `_build_recorder` is a method, not a config hook. `EpisodeRecorder` (`recording.py:363`) is a plain class, not a frozen ABC, and `NullEpisodeRecorder(EpisodeRecorder)` (`recording.py:538`) is the precedent. |

### Overriding the recorder takes two methods, not one

`_build_recorder` short-circuits before it ever reaches the recorder class:

```python
if self._store is None or rec_cfg is None:
    return NullEpisodeRecorder()
```

`self._store` is a `RecordingStore` built at `orchestrator.py:181` from
`db_path_for_eval(...)` — a SQLite path. Override `_build_recorder` alone and a
Parquet recorder never runs, because `_store` is still `None`. Whatever sets up
`_store` has to be overridden too, or `_build_recorder` has to stop consulting it.

### The recorder surface has seven members, not six

`record_step(**fields)`, `record_video(frame)`, `close(...)`, and the properties
`sid`, `eid`, `eval_id`, `db_path` — plus **`is_active`**, which the spec's list
omits.

`db_path` returns `""` from `ParquetEpisodeRecorder`. It is plumbed
(`runners/{sync,live}_runner.py` → `serve.py:141` → `model_servers/base.py:55`,
exposed as a property at `:72`) and **read by nothing**: no model server module
touches `recording_db_path` or opens a `StepRecorder`. It is an affordance, not
a feature, and returning empty costs nothing today.

If server-side recording is ever wanted — action-chunk variance, inference
latency, which chunk index a step came from — this is the right mechanism, but
note that "a path to one SQLite file" does not generalise to a URI plus a
partition key.

---

## 2. Route around the aggregate, do not fix it

`results/collector.py:68`, `_build_task_result`, now says it outright:

> All episodes count toward metrics equally — no exclusions. Episodes with
> `failure_reason` are included as failures (success=False) and their count is
> reported separately via `num_errors` for visibility.

`num_errors` was promoted to the top level of `BenchmarkResult` and the summary
line in `0865d42`, so a crashed worker is now *visible* rather than invisible —
but it is still in the denominator. `refractal compare` reads raw episode rows
and computes its own numbers; their aggregate sits unused.

This is why `is_infra_failure` is its own column in `episodes.parquet` rather
than something inferred from a `failure_reason` string.

---

## 3. Spec edit — the compose backend must set `user`

Benchmark containers run as **root by default** (`DockerConfig.user` is `None` →
no `--user` flag → image default). Everything written to the mounted output
directory is then root-owned, and the host-side merge fails with
`Permission denied`. Upstream #130 / #136 documents the escape hatch:
`docker.user: host` → `$(id -u):$(id -g)`, or an explicit `"<uid>:<gid>"`
(`config.py:80`, applied at `cli/main.py:259-268`).

**That escape hatch is not available to Refractal**, because `refractal run
--backend compose` renders its own compose file rather than going through
`vla-eval`'s `docker run` path. The equivalent has to be emitted:

```yaml
services:
  vial-rack-v1-0:
    user: "1000:1000"       # literal, resolved at render time
```

Three details that are easy to get wrong:

* **Resolve at render time, in `execute` — never at plan time.** A uid is a
  property of whoever is running, not of the experiment. This is the general
  rule stated in `refractal/__init__.py`: anything that differs between two
  people running the same experiment is render-time. Same boundary that keeps
  `results_uri` out of `plan_id`.
* **Emit literal values, not `${UID}:${GID}`.** Neither is exported by default
  in a POSIX shell, so the interpolation silently yields `:` and Compose either
  errors or falls back to root depending on version. The plan is meant to be a
  file you can read; a rendered compose file should be too.
* **Pre-create the results directory on the host** before invoking Compose.
  Docker creates a missing bind-mount target as root regardless of `user:`, and
  a non-root container then cannot write into it.

  **This line needs a comment in the code saying why it is there**, because it
  reads as removable defensive `mkdir` and it is not. It also passes locally
  under exactly the conditions that hide it: any previous root-owned run already
  left the directory in place, so deleting the pre-create is green on the
  machine you delete it from and broken on a clean checkout.

Without this the failure is not loud: the run completes, the Parquet files land,
and `refractal compare` fails at read time on someone else's machine.

---

## 4. Spec edit — MRO kwargs behaviour belongs in the adapter contract

`dc2c4ba` (#134, fixes #132) changed how model-server `observation_params` reach
a benchmark constructor. `_accepted_init_params` (`orchestrator.py:53`) walks the
**MRO**, collecting named parameters and stopping at the first class that does
not take `**kwargs`. `_merge_observation_params` then forwards each negotiated
param the chain accepts, and:

```python
logger.warning(
    "Model-server observation parameter %r cannot be forwarded to %s; "
    "the benchmark constructor does not accept it", key, benchmark_cls.__qualname__
)
```

Two consequences for the SO-101 adapter:

* **Forwarding `**kwargs` to a parent now works.** Negotiated params reach a
  wrapper benchmark's parent rather than stopping at the wrapper. This is what
  LIBERO-Plus/Pro/Mem needed and what any layered adapter gets for free.
* **An unrecognised key warns instead of vanishing, and that warning is a
  signal.** It means the model server is trying to configure something — an
  image size, a camera set, a control mode — that the adapter's constructor does
  not name. Silently dropped, that is a policy evaluated under observation
  settings nobody chose, which is exactly the class of quiet corruption
  Refractal exists to catch.

**Make it a gate, not a log line.** Step 4's acceptance test already runs one
scenario standalone and the identical scenario through the harness and asserts
the outcomes match. Add: assert no `cannot be forwarded` warning is emitted. It
costs one log handler and it catches a constructor drifting out of sync with the
server's negotiated params — a difference that would otherwise show up as a
small unexplained success-rate gap between the two paths.

---

## 5. `max_batch_size` is not a universal knob

It defaults to `1` (`model_servers/predict.py:138`), so the GPU is probably idle
and sweeping 1 / 8 / 32 is worth doing before concluding anything about
throughput. But `max_batch_size > 1` raises `NotImplementedError` unless the
server overrides `predict_batch` (`predict.py:202`).

`groot.py` implements it, so the SO-101 path is fine. Only five server modules
do, so `server_args: {max_batch_size: 8}` is not portable advice — it belongs
next to the specific server, not in a template.

---

## 6. `_ALL_RECORD_FIELDS`

`orchestrator.py:490` validates configured `step_fields` against
`getattr(benchmark, "_ALL_RECORD_FIELDS", None)`. Widening it is established
practice: `robodojo` adds `score`, `duobench` adds `stage`, and `maniskill2`,
`simpler`, `behavior1k` and `robomme` add `terminated`/`truncated`.

The adapter declares its own set, so this constrains nothing. Still worth raising
upstream: a field list every benchmark redefines to whatever it needs is not
validating much.

---

## 7. What changed under the pin, and what it costs the design doc

`4aeb436..35f1200` is five commits. Two matter:

**`0865d42` — the gcd shard collision is fixed.** Round-robin sharding used to
pin one episode index per shard whenever
`gcd(num_shards, episodes_per_task) > 1`. It now shuffles with a fixed seed
before slicing and re-sorts by task to keep env rebuilds rare:

```python
_SHARD_SHUFFLE_SEED = 42
```

`refractal-design.md` Part 12 lists this as a finding a prompt could not have
produced. Revised text in
[design-part-12-revised.md](design-part-12-revised.md) — the example survives,
reframed around the distinction it actually demonstrates.

**`dc2c4ba`** — §4 above.

The remaining three are a leaderboard data refresh, a robodojo pin bump, and the
#136 docs change behind §3.

### Worth noticing about the authorship

Three of those five commits carry `Co-Authored-By: Claude Fable 5.1`. The
maintainers are running an agent over the issue tracker, which is why two issues
filed here went from filed to merged in six days.

A well-specified issue with file and line numbers is now close to a merged fix,
which makes filing them cheap — and every one that lands moves work off
Refractal's side of the boundary permanently. Standing candidates, in the order
worth filing:

1. **`_build_recorder` has no config hook** (§1), despite `NullEpisodeRecorder`
   proving the seam exists. File this one first: it would remove the subclass
   from Refractal entirely, and it is a smaller ask than the others because it
   changes no semantics, only where the recorder class comes from. Flag that the
   `self._store` gate has to move with it or the hook is inert.
2. **`db_path` plumbed to every model server with no caller** (§1).
3. **`_ALL_RECORD_FIELDS` looseness** (§6) — largest ask, since it changes what
   the field means.

The wider reading of the agent-authored PRs is in
[design-part-12-revised.md](design-part-12-revised.md).

---

## 8. The paper — prior art, and the 55-point number read carefully

**arXiv 2603.13966v2**, *"vla-eval: A Unified Evaluation Harness for
Vision-Language-Action Models"* — Choi, Lee, Park (MAUM.AI), Kim, Krishna, Fox
(AI2), Yu (SNU). Submitted 2026-03-14, v2 2026-04-17. Verified against the arXiv
API and the HTML full text, not from memory.

Read §III-B and §IV-A before writing `resolve`. §IV-A is prior art for part of
the catalog.

### The numbers, and where they actually come from

| Claim | Status |
|---|---|
| Up to 47× wall-clock speedup, 2,000 LIBERO episodes in ~18 min (14h → 18min) | In the abstract. Confirmed. |
| Reproduces published scores across six VLA codebases and three benchmarks | In the abstract. Confirmed. |
| A single undocumented parameter can shift success rates by up to 55 pp | **Not** in the abstract; it is §III-B, and the specifics matter. |
| 14 benchmarks, six model servers | Paper, as of April. The repo at `35f1200` has **20 benchmark packages and 18 model servers** — `refractal-design.md`'s "18 benchmarks, 13 model servers" is stale in the direction of understating. |

§III-B, in full, is a list of integration failures:

- **X-VLA on LIBERO, wrong proprioceptive state source: 97.8% → 42%.** This is
  the 55 pp.
- OpenVLA-OFT quaternion→axis-angle without antipodal normalization:
  LIBERO-Goal 97% → 83%, LIBERO-Long 95% → 56%.
- OpenVLA applies an undocumented center crop (scale 0.9) at eval time; omitting
  it costs ~3 pp.
- GR00T expects end-effector pose as proprioceptive input, a field that exists
  only in an internal simulator fork; without it, 30–55% → **0%**.
- Absolute versus delta action modes — both valid 7D vectors, indistinguishable
  from the data alone — produces 0% as positions accumulate.

### Do not claim this as evidence for scenario identity

Every one of those is a **model-side integration parameter**: which observation
is fed to the policy, how an action is interpreted, how an image is cropped.
None of them is a scenario parameter. Pitching 55 pp as evidence for
content-addressed *scenario* identity would be an overclaim, and an audience who
has read the paper will know it.

Worse, the paper's own fix already covers that layer. The conclusion:

> vla-eval records the full evaluation configuration alongside every result,
> making any run reproducible from a single config file.

That is provenance, and it is prior art for a real slice of what Refractal
claims. Saying otherwise in a pitch is how you lose the room.

### What the number does support — lead with the demonstration

**Lead with this.** A proprioceptive state source is precisely a
`Checkpoint.server_args` entry, and `experiment_identity` hashes `server_args`
into `plan_id`. So two runs that disagree on it get different `plan_id`s and
land in different comparison directories. **The X-VLA failure is not detectable
in Refractal — it is unrepresentable as a comparison.**

That takes ten seconds at a whiteboard, and it uses their own strongest example
against the gap they left. It is a test, not an argument:
`test_identity.py::test_plan_id_tracks_server_args`.

**Then generalise, in one sentence.** If a single undocumented parameter can
move a success rate 55 points, a 4-point difference between two runs is
uninterpretable unless you can prove both runs shared every parameter. Their fix
proves a single run is *re-runnable*; it does not prove two runs are
*comparable*, because a config file per run says nothing about whether the two
expansions were identical, and the episode is still `(task_name, index)`, so
there is no per-episode key to check it with. Reproducibility and comparability
are different properties and the paper delivers the first.

That ordering matters. The general form is the more important claim but the
weaker opening, because it asks the room to accept a distinction before seeing
it pay off. The specific one earns the distinction first.

Same problem, one layer deeper: they standardised the protocol and recorded the
config; Refractal puts the configuration into the identity of every episode, so
a mismatch cannot survive into a comparison.

### §IV-A is prior art for `tasks.yaml`

> Evaluation protocols vary across papers: SimplerEnv spans three incomparable
> robot configurations; CALVIN ABC→D and ABCD→D splits are not comparable;
> LIBERO papers report 4 or 5 suites. We established canonical protocol
> definitions for each benchmark, standardizing task subsets, metrics, splits,
> and comparability constraints.

"Comparability constraints" is the catalog's job, arrived at from the other
direction. The difference worth being clear about: theirs are **human-curated,
per-benchmark, and enumerated** — a fixed definition of what LIBERO-Goal means.
Refractal's are **generated and hashed**, so comparability is a property the
data carries rather than a convention a maintainer upholds.

So it is prior art for the `tasks.yaml` layer and not for scenario expansion.
Cite it as a shared premise rather than pretending to have found the problem
first.

### One more data point for Part 12

§IV-A: *"An AI agent (Claude Code with Opus 4.6) reviewed 1,704 papers via MCP
tool integrations"* to build the leaderboard.

This is a bigger fact than the agent-authored PRs. Those could be read as a
maintainer's convenience — a faster way to close issues someone else specified.
An agent reviewing 1,704 papers *as published methodology* means agent-driven
work is load-bearing in how the leaderboard exists at all: the 657-result
artifact is not something the team did by hand and mentioned an agent helped
with. See [design-part-12-revised.md](design-part-12-revised.md).
