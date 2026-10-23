# Interpreting archive records

Every retained old candidate member is copied byte-for-byte under archive/; EXCLUSIONS.csv and RELEASE_LINEAGE.csv are the current file boundary. Old FILE_MANIFEST.csv validates the old collected candidate, including its now-excluded port; it is historical, not the new release manifest. Current integrity is RELEASE_MANIFEST.csv.

Old readmes, code-availability statements, copyright-review flags, closed execution windows, private path tokens and completion fields are evidence of that earlier stage only. Current decisions are the root README/FINAL_RELEASE_STATUS/LICENSE_SCOPE. Paths in an old catalog are relative to archive/, not the root. A catalog mention of an excluded implementation does not mean the file is distributed.

Only the release wrapper commands in RUNBOOK.md are supported/tested. All research drivers, including historical materialize helpers, figure renderers and training/evaluation --help invocations, are archive-only/manual setup required. No universal launch wrapper or arbitrary deadline/hash bypass has been added. Historical model and result identities are retained.
