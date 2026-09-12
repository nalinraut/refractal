# Releasing

A PyPI upload is permanent. You can yank or delete a release, but the
name-plus-version is burned forever — `refractal 0.1.0a1` can never be uploaded
twice. So everything here is about getting it right once rather than recovering.

## Checklist

```console
# 1. The tree is what you think it is.
git status --short                       # must be empty
PYTHONPATH=src python -m unittest discover -s tests -t .

# 2. DELETE dist/ AND REBUILD. Do not reuse it. See below.
rm -rf dist && uv build

# 3. Verify the artifacts, not the source tree.
python - <<'PY'
import zipfile
names = zipfile.ZipFile("dist/refractal-<version>-py3-none-any.whl").namelist()
pkgs = sorted({n.split("/")[1] for n in names if n.startswith("refractal/") and "/" in n[11:]})
assert pkgs == ["build", "compare", "execute", "resolve", "schema"], pkgs
PY
twine check dist/*

# 4. Install the wheel into a clean venv and run the seconds test from it.
#    Without the execute extra first: `plan` and `build` must work with no
#    pyarrow, no fsspec and no simulator.

# 5. Rehearse. TestPyPI is free and catches README rendering.
twine upload --repository testpypi dist/*

# 6. Upload.
twine upload dist/*
```

## Why step 2 says delete rather than rebuild

**`dist/` gets deleted and rebuilt, never reused.** This is not hygiene, it is the
one failure in this process that is invisible.

The version-permanence argument is about not burning a version number on a wrong
rule. The near-miss came from the direction nobody was watching: not uploading
too early, but having built the artifacts *before* two fixes landed and then not
rebuilding. `dist/refractal-0.1.0a1-py3-none-any.whl` carried a stale
`scenario_hash` rule and a missing scene-freshness check, and the filename was
byte-identical to the correct one. Nothing in `twine check`, `git status` or the
test suite looks at `dist/`.

Uploading that would have burned `0.1.0a1` on a `scenario_hash` rule we had
already decided was wrong — which is exactly the outcome the permanence argument
exists to prevent, arriving through a door the argument did not mention.

## Why the wheel is checked rather than the source tree

A `.gitignore` pattern of `build/` — unanchored — also matches
`src/refractal/build/`, and hatchling honours VCS ignores. The entire
`refractal.build` package was silently absent from both git and the wheel while
`git status` reported clean and the test suite passed against the working tree.

Step 3 exists because of that, and the CI `seconds-test` job installs the built
wheel rather than an editable checkout for the same reason. Two bugs have now
been caught by looking at the artifact instead of the repo.

## Version numbers

- Pre-release (`0.1.0a1`) means `pip install refractal` reports no matching
  distribution until a final release exists. Correct while there is no simulator
  adapter, but note the README's install line assumes a final version.
- If a placeholder release was ever published to hold the name, **yank it** in
  the same sitting as the first real release. A final `0.0.0` outranks a
  pre-release `0.1.0a1` in pip's resolution, so the stub is what users get.

## Licence and author

`Apache-2.0`, and `authors` carries a name with no email. Both are effectively
permanent once published. Confirm them before step 6 rather than after.
