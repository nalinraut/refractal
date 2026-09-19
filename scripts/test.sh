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
exec "$PY" -m unittest discover -s tests -t . "$@"
