# Bioengineering supplement implementation traceability

This matrix binds every experiment requested by
`Bioengineering_投稿前差距清单与修改方案_OncoDerm-MambaX_2026-08-14.md`
to an executable configuration, analysis artifact, and completion check. It is
an implementation record, not a claim that new numerical experiments have
already run on the local Windows machine.

| Priority | Requirement | Executable implementation | Required evidence |
|---|---|---|---|
| P0 | Same-split Swin-Tiny, seed 42 | `configs/bioengineering/swin_tiny_seed42.yaml`; `run_bioengineering_supplement.sh` | Table 3 row, Table S5 row, prediction hashes, fixed split/run seed audit |
| P0 | Same-split DermaMamba, seed 42 | `configs/bioengineering/dermamamba_architecture_faithful_seed42.yaml`; `lesionmamba/models/dermamamba_reference.py` | Table 3 row, Table S5 row, locked source revision, model metadata, frozen run-config audit |
| P0 | Reuse B1/B2/B4/B6 under split 42 | Existing `configs/submission_v1` runs; interrupted runs are not reusable | Exact direct-table membership and prediction/source alignment audit |
| P0 | Optional EfficientNet-B0 | `configs/bioengineering/efficientnet_b0_seed42.yaml`; controlled by `INCLUDE_EFFICIENTNET` | Table S5 row and architecture provenance when enabled |
| P0 | Architecture-intended direct comparison | `scripts/20_bioengineering_direct_summary.py` | `tables/Table3_direct_recent_baselines.csv` |
| P0 | Seven-class MEL-score fairness | `scripts/20_bioengineering_direct_summary.py` | `tables/TableS5_architecture_fairness.csv`; per-image score table |
| P1 | B6-M true-Mamba F4 gate | `configs/bioengineering/b6m_true_mamba_seed42.yaml`; strict official `mamba-ssm` backend | CUDA smoke test, seed-42 metrics, Params/FLOPs/latency gate JSON |
| P1 | Conditional B6-M seeds 2024/3407 | Seed-specific configs; launched only after the frozen seed-42 gate passes | Stability run table contains all three seeds only after a passed gate |
| P1 | B6 versus DermaMamba paired bootstrap | Image-ID alignment and paired resampling in `bioengineering_supplement.py` | ROC-AUC and PR-AUC deltas, 10,000 requested resamples, finite-sample-corrected p-values |
| P1 | Expand localization audit to 200 masks | `scripts/07_eval_mask_explainability.py` | Real unique-pair result at n=200, or auditable `not_run_insufficient_pairs`; no duplication/synthesis |
| P2 | B6 independent split-seed 2024 | `configs/bioengineering/b6_crosssplit2024_seed42.yaml` | Two-row descriptive sensitivity table; split integrity and nonidentity audit |

## Frozen boundaries

- Direct comparisons use HAM10000 lesion-grouped split seed 42, run seed 42,
  224 x 224 inputs, and the locked SIIM-ISIC 2020 external set.
- External labels are used only for final evaluation. Operating thresholds are
  derived from internal validation predictions and transferred unchanged.
- The DermaMamba comparator is a resource-matched clean-room architecture port,
  not an official checkpoint reproduction. The repository revision and all
  adaptations are recorded in `configs/bioengineering/source_lock.json` and
  run metadata.
- B6-M does not replace the primary B6 model. It is an implementation
  sensitivity experiment and expands to additional run seeds only after its
  predeclared gate passes.
- Cross-split results are descriptive because the internal test images differ;
  they are never pooled with or substituted for the fixed-split direct table.

## Completion authority

`scripts/22_bioengineering_completion_audit.py` is the terminal quality gate.
The one-click runner exits successfully only when it writes
`results/bioengineering_direct_baselines/completion_audit.json` with
`status: passed`. Missing, stale, misaligned, protocol-inconsistent, or
under-resampled artifacts fail the run rather than being silently accepted.
