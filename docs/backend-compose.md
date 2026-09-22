# Running in containers

`--backend compose` runs one container per worker, each invoking the same
`--worker` entrypoint a single process would.

Model servers stay outside. The containers hold benchmark workers only.

## Render first

```console
$ refractal render plan.json \
    --server baseline=ws://host.docker.internal:8000 \
    --server candidate=ws://host.docker.internal:8001 \
    --user "$(id -u):$(id -g)" \
    --catalog ./catalog --results ./results \
    -o docker-compose.yml
  wrote docker-compose.yml  (10 service(s))
  next:  mkdir -p ./results && docker compose -f docker-compose.yml up
```

`render` needs no Docker and touches nothing. Read the file before running it.

```yaml
services:
  bench-v1-0:
    image: refractal-bench:local
    user: "1000:1000"
    cpuset: "0"
    mem_limit: 2048m
    restart: "no"
    extra_hosts: ["host.docker.internal:host-gateway"]
    command: [refractal, run, /plan/plan.json,
              --worker, bench-v1/0, --backend, vla-eval,
              -o, /results, --harness-output, /scratch,
              --server, "baseline=ws://host.docker.internal:8000",
              --server, "candidate=ws://host.docker.internal:8001",
              --catalog, /catalog]
    volumes:
      - ./plan.json:/plan/plan.json:ro
      - ./catalog:/catalog:ro
      - ./results:/results
```

## Then run

```console
$ mkdir -p ./results
$ refractal run plan.json --backend compose \
    --server baseline=ws://host.docker.internal:8000 \
    --server candidate=ws://host.docker.internal:8001 \
    --catalog ./catalog -o ./results
```

This renders and then calls `docker compose up`. The rendered file is written
either way, so you can always run it by hand.

## Three things that will bite

**Create the results directory before starting.** Docker creates a missing
bind-mount target as `root` whatever `user:` says, and a non-root container then
cannot write into it. `--backend compose` does this for you; `docker compose up`
by hand does not.

**Pass `--user` as a literal.** `"$(id -u):$(id -g)"` expands in your shell.
`"${UID}:${GID}"` does not: neither is exported by default, so it interpolates
to `":"` and Compose either errors or silently runs as root.

**Address the servers on the host.** `host.docker.internal` with the
`extra_hosts` mapping the renderer emits, or your host's LAN address. Not a
service name; the servers are not in this file.

## Flags

`render` and `run --backend compose` take the same rendering flags, because the
second calls the first. These are they:

| | |
|---|---|
| `--server CKPT=URL` | one per checkpoint, pointing at the host |
| `--user UID:GID` | emitted literally; defaults to the current user |
| `--catalog DIR` | mounted read-only as provenance |
| `--results DIR` | the only writable mount |
| `--image ENGINE=IMAGE` | override the image for an engine |
| `--mount HOST:CONTAINER` | extra read-only mount, repeatable |
| `--no-host-gateway` | omit the `host.docker.internal` mapping |
| `--gpus N` | GPUs to reserve per worker |
| `--video` | emit `--video` in each service's command |
| `--frame-every N` | keep every Nth frame, default 10 |

`render` additionally takes `--target compose` (the only target; k8s is out of
scope), `-o/--output` for where to write, and `--plan-file` for the host path to
the plan when it differs from the path you passed.

`run --backend compose` additionally takes:

| | |
|---|---|
| `--compose-file PATH` | where to write, default `docker-compose.yml` |
| `--detach` | `up -d` instead of waiting |

## Video

`--video` records frames and writes one WebP sprite strip per episode into
`frames/` beside the results, laid left to right at `--frame-every`.

A worker asked for frames that produces none fails rather than finishing quietly,
because a silent empty `frames/` looks identical to a run nobody asked to record
and is only noticed after the GPU time is spent.

Recording is off unless asked for. It is not free, and whether to spend that is
not a property of the experiment -- which is why it is a flag here and not a
field in the catalog. Two runs of one plan, one recorded and one not, are the
same experiment and join.

## When render refuses

**A plan whose workers cannot run.** The index contract is checkable from the
plan alone, so it is checked before any container starts:

```
error: 540 of 600 episode group(s) in this plan cannot run on the vla-eval
backend, starting with bench-v1/1/cup-on-shelf/baseline/seed0. The harness
counts episodes from zero within a task ...
Declare `partition_unit: task` on the scene's resource shape and re-plan.
```

**A checkpoint with no server**, named.

**An engine with no image.** Map one with `--image mujoco=my-image:tag`.

## Building a worker image

The image needs the simulator, the harness, and Refractal in one interpreter.

```dockerfile
FROM <benchmark-base-image>

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp

RUN uv venv --python 3.11 "$VIRTUAL_ENV" && chmod -R a+rX /opt/uv-python
RUN uv pip install --no-cache-dir refractal[vla-eval] <your-simulator>

RUN mkdir -p /scratch && chmod 1777 /scratch
```

Four lines there exist because the container runs as an arbitrary non-root user:

- `UV_PYTHON_INSTALL_DIR`: the default is under `/root`, mode 700, and the venv
  symlinks into it. Without this: `Could not find platform independent libraries`
  before any of your code runs.
- `chmod -R a+rX /opt/uv-python`: same reason.
- `HOME=/tmp`: a non-root user needs somewhere writable. If your simulator reads
  config from `$HOME`, copy it there too.
- `/scratch` at `1777`: `--harness-output` names it, and a non-root process
  cannot create a directory at `/`.

Also `chmod -R a+rX` anything your base image copied in as mode 600. Editable
installs read their source at import time, and a root-owned source tree fails for
a non-root user.

## Parallelism

One container per worker, and the planner decides how many workers a scene gets.
A scene whose resource shape says `partition_unit: task` gets at most one worker
per task; a scene that vectorises usually gets one worker total, because the
batch already holds every episode.

Several scenes give several workers regardless.

### Why every service is pinned

Compose has no scheduler. Without `cpuset` the containers contend for cores, and
contention does not raise -- it makes every timing measurement noise, differently
on each run, which for a tool whose output is a claim about a measured difference
is the worst available failure.

The pin is as wide as the worker: `cpu_cores` from the resource shape, which is
per environment, times the environments one worker holds. Under `execution_mode:
concurrent` that is the checkpoint count, so the `cpuset: "0"` in the example
above becomes `"0-1"` when two checkpoints are compared.

A pin narrower than the worker's threads is the same failure moved inside the
container, where a worker's own threads contend instead of its neighbours', and
it reads as the cost of containerisation rather than as a misconfiguration.

### What containers do and do not buy

They parallelise the simulator. They do not parallelise inference, which on a VLA
evaluation is most of the per-step cost: the workers hold environments, and the
model servers are outside, shared, and answering whatever arrives. Adding workers
fills those servers' queues; whether the queue becomes a batch is a property of
the server, not of this file.

Size the fleet to the servers, not to the cores. Ten containers against two
servers on one card killed a server outright, mid-run and silently.
