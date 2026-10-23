# New auxiliary set selection

This is a newly declared 100-pair training set, not recovered historical membership.
Read localization200_restored.csv only as original image/mask path navigation.
Check candidate pair existence, dimensions, nonempty binary masks, file SHA256 and
unresized PIL RGB pixel SHA256. Check original HAM split IDs/lesion IDs and available
patient relationships and same-definition hashes against HAM val/test and SIIM.
Existing perceptual candidates are screening results, not identity declarations.
Unknown lesion/patient IDs remain unknown: no comprehensive independence claim.

Eligible rows are sorted stably by image_id. Select numpy Generator(PCG64(20261012))
choice(n, size=100, replace=False); preserve selection order. Both conditions share
this set. All selected images are within the old localization200 navigation set;
old localization measurements are not independent localization validation for these
new models. No new localization test was performed.

Private full paths/IDs are in fim_mask_train100.csv locally. The delivery contains
only hashed image identifiers and original image/mask hashes, without clinical rows.
