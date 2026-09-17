# Draft issue: `_store` is unreachable, so `_build_recorder` cannot be used

Ready to file against `allenai/vla-evaluation-harness`. Verified at v0.6.0
(`c233542`).

This supersedes the "`_build_recorder` has no config hook" entry on the standing
candidates list in [../harness-integration.md](../harness-integration.md). That
version named a design gap; this one names the mechanism, has a working
workaround to point at, and can therefore be answered with a small change rather
than a discussion.

---

## What happens

`Orchestrator._build_recorder` is a method rather than a config hook, which reads
as a deliberate seam — `NullEpisodeRecorder` subclasses `EpisodeRecorder` and is
returned through it, so a replacement recorder looks like the intended extension
point.

It cannot be used, because of the gate:

```python
# orchestrator.py
if self._store is None or rec_cfg is None:
    return NullEpisodeRecorder()
```

Both conditions are controlled by the same flag, in opposite directions:

```python
def _effective_recording_config(raw, *, no_save: bool):
    if no_save:
        return None                      # no_save=True  -> rec_cfg is None

async def run(self):
    if not self.no_save:
        self._store = RecordingStore(...)  # no_save=False -> a SQLite store
```

So:

| `no_save` | `rec_cfg` | `self._store` | `_build_recorder` returns |
|---|---|---|---|
| `True` | `None` | `None` | `NullEpisodeRecorder` |
| `False` | set | a `RecordingStore` | `EpisodeRecorder` |

There is no combination where `_build_recorder` is consulted *and* the store is
something the subclass chose. Overriding the method alone produces a run that
completes, reports success and records nothing — which is indistinguishable from a
correct run until somebody reads the results.

`self._store` is also assigned **inside `run()`**, so there is no method to
override to change it.

## Why anyone would want this

A backend that records to something other than one local SQLite file. Ours writes
Parquet to an `fsspec` URI, partitioned so several workers can write concurrently
and a later run can resume by listing what already exists. That needs the recorder
replaced and the store bypassed; it does not need any change to how episodes run.

## The workaround, offered as evidence the seam is wanted rather than as a fix

```python
class ParquetOrchestrator(Orchestrator):
    def __init__(self, *args, **kwargs):
        self._parquet_store = NullRecordingStore()   # null object, not a sentinel
        super().__init__(*args, **kwargs)

    @property
    def _store(self):
        return self._parquet_store

    @_store.setter
    def _store(self, value):
        return None        # swallow run()'s assignment and the finally's None

    def _build_recorder(self, rec_cfg, task, bench_eval_id, safe, task_idx, ep, benchmark):
        ...
```

It works and we are shipping it, with two things worth saying plainly because they
are the argument:

- **It silently discards an assignment the base class makes deliberately.** That is
  not a thing a downstream package should be doing to yours.
- **It is fragile in a specific way.** If a future version builds the store and
  passes it to a helper rather than assigning to `self._store`, the setter never
  fires; the property still returns the null store, the bridge still appears to
  work, and anything reading `self._store` expecting what it just built gets the
  wrong type. We gate on a digest of `orchestrator.py` and print this assumption
  when it moves, which is a fair amount of machinery to carry for a five-line
  seam.

`NullRecordingStore` also has to implement `upsert_eval_metadata` and `close`,
because the harness calls both on the store outside the recorder path. A bare
sentinel raises on the first benchmark.

## Suggested change

Either would remove the need for the above entirely.

**A recorder factory in the config**, resolved the way benchmarks and servers
already are:

```yaml
recording:
  recorder: "my_package.recording:ParquetEpisodeRecorder"
```

**Or decouple the store from `no_save`** — a `recorder_factory` argument to
`Orchestrator.__init__`, defaulting to today's behaviour, with the gate testing
the factory rather than the store. `no_save` keeps meaning "record nothing", and
"record somewhere else" stops being the same question as "record nothing".

Happy to send a PR for whichever shape you prefer.

## What we are not asking for

Not a Parquet backend in the harness, and not a storage abstraction. One
injection point. Everything else — layout, partitioning, atomic writes, resume —
belongs on our side of the line and is already there.
