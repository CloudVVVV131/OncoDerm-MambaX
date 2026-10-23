# Inputs, identities and lawful access

No individual/member rows are distributed in this release. Precise recorded input/initialization hashes remain in their stage configs/records under archive/; they identify originals and are not replacement data. Obtain permitted assets and exact split records separately; do not regenerate random splits, infer auxiliary100, or reuse another stage's controlled auxiliary set.

| Input | Recorded route / requirement |
|---|---|
| HAM10000 | https://doi.org/10.7910/DVN/DBW86T ; historical package route: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 |
| SIIM–ISIC2020 | https://doi.org/10.34970/2020-ds01 ; historical JPEG packaging: https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-512x512 ; comply with provider terms including recorded noncommercial restrictions. |
| ISIC2018 masks | https://challenge.isic-archive.com/data/ ; exact original auxiliary100 is not recovered. Localization200 and later controlled memberships are distinct. |
| Derm7pt | https://derm.cs.sfu.ca/Download.html ; existing record says academic access, no image redistribution. `derm` and `clinic` are not interchangeable. |
| Pretrained initialization/checkpoints | Stage-specific timm/torchvision or author checkpoint identity where recorded; weights not distributed, versions missing in the historical record remain unknown. |
| Exact splits, linkage, individual outputs | Separate authorized access to the recorded originals is needed. Input schemas and per-stage requirements are documented under archive/LICENSES_AND_DATA_ACCESS.md and per-unit readmes; those old permission-status paragraphs are historical. |

These links are inherited from collected records, not newly visited. This release makes no promise that a download is presently available or that dataset terms grant redistribution. Diagnosis and linkage IDs are never clinical model features. Original 22-dimensional encoding and structured missingness remain stage-specific; no replacement encoding is introduced.

Historical local_release/docs/DATA.md mentions six released ID manifests. **They are not present in this scoped candidate.** Do not interpret that old text as current inclusion or generate replacement splits. Recorded packaging identities (not newly downloaded): HAM archive 5,582,914,511 bytes, MD5 fd84b842e863f3c82f2f4231da626014; SIIM JPEG archive 2,825,694,558 bytes, MD5 cb02bdebe2901b7ae8c847250b277ea8. Existing ISIC identities remain in archive/original_framework/local_release/scripts/prepare_data.py, read as source only. Derm7pt release_v0.zip: 159,112,080 bytes, local SHA256 89a4749e7e43d1c2e73876aaeb867af5b2624e929ee3d1ae7202268588566b54 (not an official provider checksum).
