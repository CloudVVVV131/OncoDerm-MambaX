# Derm7pt Download and Integrity Report

Status: COMPLETE
Completed: 2026-10-15T23:42:30+08:00

## Provenance and Acquisition

- Official instructions: https://derm.cs.sfu.ca/Download.html
- Official archive endpoint: https://derm.cs.sfu.ca/restricted/release_v0.zip
- Author processing repository: https://github.com/jeremykawahara/derm7pt
- The user completed the official application and reported downloading the archive through the local browser. The assistant subsequently inspected the local archive. No browser credentials, cookies, tokens, or application responses were read or recorded.
- The processing repository was not substituted for the image dataset. No unofficial source was used in this workflow.
- The archive is a readable ZIP (PK signature), not an HTML login/error page.

## Files and Space

- Original archive: `${R52_PATH_7B057E80D5C3}`
- Original archive size: 159,112,080 bytes (151.74 MiB).
- Local SHA-256: `89a4749e7e43d1c2e73876aaeb867af5b2624e929ee3d1ae7202268588566b54`
- No official checksum was found on the public page inspected. This SHA-256 is a local fingerprint, NOT an official hash validation or independent source-authenticity proof.
- ZIP CRC: PASS for every member. Every extracted file also matches its corresponding archive member by byte count and SHA-256.
- Archive size, SHA-256, and modification time remained unchanged during this operation.
- Free space before extraction: 11,864,305,664 bytes (11.049 GiB).
- Extracted file bytes: 163,238,049 (155.68 MiB). Space was checked with an additional 1 GiB reserve.
- New extraction directory: `${R52_PATH_096589CED5EE}`
- Preserved original data root: `${R52_PATH_AA3C2EDA8C25}`
- Total extracted files: 2,020. No existing data directory was overwritten.
- The archive has distinct `images/FCL/` and `images/FCl/` directories. The new `images` directory was checked for case-sensitive support before extraction so these names would remain distinct. No original paths were normalized or merged. No global filesystem setting was changed by this script.

## Inventory and Linkage

- Physical images: 2,013 JPEG files, retained without changes.
- Metadata: `${R52_PATH_CB55B27527EC}`; 1,011 rows.
- Dermoscopic images are identified by the original `derm` column: 1,011 references to 1,011 distinct paths; 0 missing paths.
- Clinical photographs are identified by the original `clinic` column: 1,011 references to 1,011 distinct paths; 0 missing paths.
- 9 exact image paths are referenced by BOTH modality columns. Thus 1,011 + 1,011 references correspond to 2,013 physical files. These original mappings were preserved; no relabeling, deduplication, case selection, or visual reclassification was performed.
- Unreferenced image files: 0. Blank modality references: 0.
- The public download page's thumbnail caption mentions 2,045 images, whereas this archive contains 2,013 physical image files. This source-count discrepancy is recorded without guessing its cause. All image paths referenced by this archive's metadata are present; the page caption alone is not evidence of specific missing cases.
- Original suggested split files: `meta/train_indexes.csv` (413 rows), `meta/valid_indexes.csv` (203 rows), `meta/test_indexes.csv` (395 rows). These were inventoried, not adopted as a new experimental protocol.
- Documentation: `README.txt`, `clinic.html`, and `derm.html` are present. HTML and other archive contents were not executed.
- Missing required files according to the archive README and metadata linkage: 0.

## Usage Notices and Boundaries

The included README identifies academic research use, CC BY-NC-ND 4.0, a notice against redistributing images, and retained rights of Dr. Giuseppe Argenziano. The original README is preserved unchanged; this report does not expand those permissions. No dataset images were uploaded or redistributed.

This operation only inspected, hashed, extracted, and inventoried the existing archive. No labels, images, manuscript, model, or previous results were modified. No training, inference, performance analysis, P4 evaluation, or P1/P2/P3 rerun was performed.

## Verification Artifacts

An independent final read-only check confirmed all 2,020 file paths and 38 directory paths match the ZIP exactly, including case; all 2,020 extracted file fingerprints were rechecked. All 2,013 image files have JPEG signatures. `FCL` contains 75 files, `FCl` contains 1, and they are distinct filesystem directories. The only directory-attribute change was enabling case sensitivity on the new empty extraction `images` directory through a successful Windows elevation request; no global setting or existing data directory was changed. The original archive SHA-256 still matches the initial local fingerprint.

- Machine-readable verification: `${R52_PATH_F6F7B32FF209}`
- Extracted file inventory and local fingerprints: `${R52_PATH_B2B3858B8852}`
- Local verification/extraction script: `${R52_PATH_4080A10B223A}`
