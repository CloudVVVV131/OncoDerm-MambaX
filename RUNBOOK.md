# Supported local release operations

This is a source and analysis archive, not an end-to-end reproduction claim.
The supported packaging CLI is `tools/release_check.py`. It uses only the Python
standard library (Python 3.9 or newer), never imports anything in `archive/`, and
does not start research scripts. No installation or GPU is needed for these commands.

## Verify the public package

From the extracted release root:

```text
python -B tools/release_check.py verify --release-root .
```

This checks every released file against `RELEASE_MANIFEST.csv` (path, size in bytes,
SHA-256), rejects unlisted/missing files, and excludes only the manifest itself from
self-hashing. A PASS establishes package integrity only, not scientific validity,
third-party ownership or reproduction. Symlinks/junctions are unsupported.

## Check explicit path bindings without loading research data

1. Keep the release tree read-only. Put private bindings and a new workspace outside it.
2. Copy `examples/bindings.example.json` to a private location. Replace the examples
   with absolute paths valid on your host. No author drive letter or private mapping
   is required. Example paths intentionally do not exist until you supply inputs.
3. Use `logical_paths.example.json`: its meaningful input names map to three actual
   archived `R52_PATH_*` tokens. This is a small demonstration, not a universal mapping
   for all historical drivers. A different template requires an explicit schema for
   every token it contains and bindings with exactly those keys. Preserve actual
   input identity requirements; do not invent a new split or replace a missing asset.

```text
python -B tools/release_check.py check-plan --release-root . --bindings /private/bindings.json --template examples/path_binding_demo.json
python -B tools/release_check.py materialize --release-root . --bindings /private/bindings.json --template examples/path_binding_demo.json --workspace /private/new-dry-run-workspace
```

On Windows use your own absolute drive paths for the two private locations. The
workspace's parent must already exist and the workspace itself must not exist.
The workspace cannot be inside/above the release or overlap an input directory.
Bindings are existence/type checked by `stat`; their underlying image, weight or
clinical contents are not read. `check-plan` writes nothing. `materialize` writes
one text copy and `MATERIALIZATION_RECEIPT.json` only in the new workspace. It never
overwrites an existing file, source config or historical output.

Only JSON/YAML/TOML text is supported. Only `${R52_PATH_*}` tokens are substituted;
all other source bytes, including scientific settings, old deadlines and checksums,
are preserved. JSON syntax is checked. YAML/TOML are text-only and not semantically
validated. Unresolved tokens, unknown/missing binding keys, unsafe paths and quoting
that requires changing the template are rejected. Use a simple filesystem alias
when a path contains unsupported quoting characters. UTF-8 and UTF-8 BOM are supported.

Exit code `0`: requested packaging operation passed. Exit code `2`: invalid input,
missing file, integrity mismatch or refused write. A successful materialization is
**not** permission to run an experiment and **does not** verify any scientific input's
content hash. The historical weight/data/source-identity checks remain required and
must not be disabled to make a materialized config run.

## Historical research entry points

Everything under `archive/` is an execution/source snapshot: **archive-only / manual
setup required**. Do not run a historical `--help` as a safety test; a module may have
top-level execution. No historical training, inference, statistics, bootstrap,
calibration or figure driver was run during release preparation. A path-bound text
copy is not a migrated driver. Runtime constraints, scientific hashes, deadlines,
input availability and environmental differences require a separately reviewed
work plan before future authorized research. G01/G05/G07 remain limitations.

The excluded DermaMamba port is not silently replaced. Follow the release's license
scope and optional-registry note; its method/configuration/results remain a record,
not evidence that an omitted implementation can execute from this package.

## Limited adapter tests

```text
python -B tests/test_release_check.py --report /private/adapter-test-receipt.json
```

The tests create temporary synthetic directories and empty input placeholders,
exercise only the packaging wrapper, and prohibit network/process/tensor reads
and writes to the synthetic release. No model, patient data, research module or
historical driver is imported. The receipt explicitly limits what passed.

## Optional registry source staging
See OPTIONAL_DEPENDENCIES.md for `python tools/stage_optional_registry.py --destination <absolute-new-workspace> --dry-run`. It only checks/stages source. No model import/runtime has been validated. The archived DermaMamba-specific test is not part of these supported checks. Old catalog paths resolve under archive/.
