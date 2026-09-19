#!/usr/bin/env bash
# The test command. The ONLY test command.
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
exec python -m unittest discover -s tests -t . "$@"
