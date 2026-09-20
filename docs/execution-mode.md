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
belong in one comparison, and `compare` warns when the rows it is pooling do not
agree on mode or session. Success rates are unaffected; durations are not.
