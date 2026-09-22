# Execution modes

`execution_mode` in `run.yaml` decides the order episodes run in. Two values.

```yaml
run:
  execution_mode: serial
```

| mode | what happens | pick it when |
|---|---|---|
| `serial` | every task for checkpoint A, then every task for checkpoint B | one policy at a time fits in VRAM, or you want durations comparable across checkpoints |
| `concurrent` | both checkpoints at once, each against its own server | you want the wall clock halved and do not need to compare durations |

It is recorded on every row and is **not** part of the experiment's identity, so
a run in either mode joins a comparison with the other.

## `concurrent`

Both checkpoints run at the same time against separate servers. Requires both servers
resident, so check VRAM before choosing it.

**It also multiplies what a worker occupies.** One thread per checkpoint means
one environment per checkpoint, so a two-checkpoint comparison puts two
simulators in every worker. The resource shape in `scenes.yaml` is written per
environment and the planner multiplies by the checkpoint count, which is what
makes the Compose `cpuset` two cores wide rather than one.

This is worth knowing when you read a plan and see more cores reserved than the
scene appears to ask for. It is also worth knowing because it was wrong: the
planner used to spend the per-environment figure once per worker, and a
container pinned to one core ran two simulators inside it at 1.8x the per-step
cost, with nothing raised and both runs completing.

Rows produced under it carry `concurrent_with`, naming the other checkpoints that
were running:

```
  task-0  baseline   ok=True  mode=concurrent  concurrent_with='candidate'  5.3s
  task-0  candidate  ok=True  mode=concurrent  concurrent_with='baseline'   5.2s
```

**Success rates measured under contention are fine. Durations are not.** A
5.3-second episode that shared a GPU is not comparable with a 5.3-second episode
that did not, and `elapsed_sec` sits in the same row. `concurrent_with` is how
you tell them apart afterwards.

## Scratch directories

Each invocation gets its own subdirectory under `--harness-output`. Two
orchestrators sharing one scratch directory race on the same temp file and one
dies with `FileNotFoundError` on a `.tmp` path. This is handled for you; it
matters if you are pointing several runs at one directory by hand.

## Ordering and identity

Changing the mode changes nothing about which episodes exist or what any of them
is, so it does not move `plan_id`. Two runs of one experiment in different modes
belong in one comparison. Success rates are unaffected; durations are not.

A session is one `refractal run`; `run --session-id ID` fixes it, which is what
makes a test's output reproducible.

`compare` warns when the rows it is pooling span more than one **session**,
because warmup and thermal conditions differ across them. It does **not** check
that they agree on mode -- `execution_mode` is recorded per row so you can tell
them apart yourself, and `concurrent_with` is what you filter on.

Nor does it check placement. Two runs of one experiment can be pinned
differently -- cpusets are excluded from `plan_id` along with everything else
about where work runs -- and nothing in a result row records the pin. Since the
pin can move per-step cost by most of a factor of two, a latency claim pooled
across runs needs you to know they were placed alike. Success rates do not care.
