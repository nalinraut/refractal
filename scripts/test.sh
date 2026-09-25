#!/usr/bin/env bash
# The test command. The ONLY test command.
#
# Run it as `./scripts/test.sh`. It exits non-zero when anything fails, so
# `./scripts/test.sh && git push` is the habit worth having: a green local run
# that CI cannot reproduce is not a green run, and the way that keeps happening
# is two commands rather than one.
#
# Local runs used to be `PYTHONPATH=src:tests python -m unittest discover -s tests`,
# which puts tests/ on sys.path and makes `from test_bridge import ...` resolve.
# CI runs `-t .`, where the modules are `tests.test_bridge` and the bare import
# fails. Five consecutive pushes were green locally and red in CI, and nobody
# looked, because "393 tests pass" was true of a command CI does not run.
#
# So there is one command and both use it. A local result that CI cannot
# reproduce is not a result.
set -euo pipefail
cd "$(dirname "$0")/.."

# `python` exists on CI runners and not on every developer machine. The first
# version of this script hardcoded it and failed locally with "exec: python: not
# found" -- the script written to stop local and CI diverging, diverging.
PY=${PYTHON:-}
if [ -z "$PY" ]; then
    PY=$(command -v python3 || command -v python) || {
        echo "no python3 or python on PATH; set PYTHON=/path/to/python" >&2
        exit 1
    }
fi
# `src` on the path, because CI runs `pip install -e .` and an editable install
# puts exactly this directory there. So the two run the same code, which is the
# property this script exists to hold -- and the command now works from a fresh
# clone with the dependencies present and no install step.
#
# Only `src`, never `tests`. That was the original bug: PYTHONPATH=src:tests made
# `from test_bridge import ...` resolve locally and fail in CI, where `-t .` makes
# them `tests.test_bridge`. Adding tests/ back would reintroduce it silently.
#
# What this cannot catch is a packaging error -- a module that imports fine from
# src and is missing from the built wheel. CI's install is what catches that.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# A filtered run produces partial coverage, and partial coverage makes every
# guard the filter skipped look newly unreached. So arguments mean "just the
# tests", and the guard check belongs to the full run only.
if [ "$#" -gt 0 ]; then
    exec "$PY" -m unittest discover -s tests -t . "$@"
fi

# Guards, under coverage. A guard nobody has seen fire is a guard nobody knows
# works, and this project has already shipped one: an error message asserting a
# guarantee about `max_envs` that the code did not provide, found by accident
# because nothing had ever run the line.
#
# Skipped rather than failed when coverage is absent, because the extra is not
# required to develop here. Loudly, because a check that silently does not run
# is indistinguishable from one that passes.
if "$PY" -c "import coverage" >/dev/null 2>&1; then
    "$PY" -m coverage run --source=src/refractal -m unittest discover -s tests -t .
    "$PY" -m coverage json -o .coverage.json -q
    "$PY" scripts/lint_guards.py .coverage.json
else
    "$PY" -m unittest discover -s tests -t .
    echo
    echo "note: coverage is not installed, so guards were NOT checked."
    echo "      pip install coverage, or run: pip install -e '.[test]'"
fi
