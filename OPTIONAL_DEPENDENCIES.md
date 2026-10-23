# Optional comparator and manual model setup

The only adapted registry is `adapters/august_registry/registry.py`. Its parent source and both SHA identities are in OVERLAY_IDENTITY.json. Historical `archive/.../registry.py` remains unchanged. No model was constructed to test this overlay: only the dispatcher AST was exercised with inert stubs.

From the extracted release root, check the source-only staging operation:
```text
python tools/stage_optional_registry.py --destination <absolute-new-workspace> --dry-run
```
The command imports no project package, writes nothing and checks the archived registry/overlay identities. Without `--dry-run`, it copies only the archived August `lesionmamba` Python package into a new workspace and applies the overlay there; it refuses existing destinations, symlinks and the release tree. This is source staging, not environment installation or experiment execution.

Any later model use requires the recorded dependencies and legitimate inputs, plus manual stage-specific setup. The historical test `archive/original_framework/august_snapshot/tests/test_bioengineering_supplement.py` imports the excluded comparator at top level; it is archive-only and is not part of the supported release checks. The full historical test suite is NOT claimed to pass in this release. Other historical optional-backend fallbacks remain their recorded code, not newly validated formal comparators.

No scientific parameter, model definition, data split, score, freeze guard or historical deadline is changed. Never use the overlay to claim a reproduced DermaMamba result.
