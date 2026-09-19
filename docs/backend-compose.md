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

| | |
|---|---|
| `--server CKPT=URL` | one per checkpoint, pointing at the host |
| `--user UID:GID` | emitted literally; defaults to the current user |
| `--catalog DIR` | mounted read-only as provenance |
| `--results DIR` | the only writable mount |
| `--image ENGINE=IMAGE` | override the image for an engine |
| `--mount HOST:CONTAINER` | extra read-only mount |
| `--no-host-gateway` | omit the `host.docker.internal` mapping |
| `--compose-file PATH` | where to write, default `docker-compose.yml` |
| `--detach` | `up -d` instead of waiting |

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
