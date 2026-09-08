# Reproduction Scope

## Configurations and checkpoints

`configs/experiments.csv` links each public configuration to a retained run ID.
These are resolved run configurations with portable paths, explicit split seeds
and input-presence checks. B0-B6 at seed 42,
B2/B4/B6 at seeds 2024 and 3407, EfficientNet-B0, Swin-Tiny, B6-M seed 42 and
B6 cross-partition seed 42 are included.

Training creates new run directories using the released model and loss
definitions. Numerical reproducibility depends on the configuration, data,
package versions and hardware. Test and external evaluation require the best
validation checkpoint. Use trusted checkpoint files with the supplied restricted
loader.

## Endpoint definitions

- Seven-class order: `akiec, bcc, bkl, df, mel, nv, vasc`.
- `prob_mel`: seven-class softmax MEL probability.
- `prob_mel_endpoint`: dedicated binary-head sigmoid where that head exists,
  otherwise the seven-class MEL probability. Select the score stream that matches
  the reported endpoint.
- Multiclass `mel_auc` is calculated from the seven-class MEL probability.
- The endpoint command selects validation thresholds and applies them unchanged
  to test predictions and external evaluation.
- Temperature scaling is fit on validation logits.

Active auxiliary and consistency losses depend on both coefficients and enabled
output heads. B0-B6 compare implemented configurations; component effects are
interpreted within each configuration's enabled heads and losses.

## Additional analyses

`scripts/17_submission_v1_evidence.py` implements equal-weight score integration,
validation-threshold transport and image-level paired bootstrap from saved
predictions. It requires complete matched predictions for the configured members.
It is distinct from the later patient-clustered Figure 2 analysis. Outputs are
generated under the configured result root. Use a separate directory for new outputs.

`scripts/10_robustness_lite.py` characterizes controlled perturbations using
`configs/eval/robustness_submission_v1.yaml`. `scripts/04_measure_efficiency.py`
measures selected models under the current runtime. Measurements from different
GPU/software environments should be reported separately. Figure S4's all6 burden
is an arithmetic sequential estimate obtained by summing member profiles.

```bash
python scripts/07_eval_mask_explainability.py --run-ids outputs/runs/<B4_run> outputs/runs/<B6_run> --min-masks 200 --max-masks 200
```

For B4/B6 the localization command uses returned internal attention maps; the
Grad-CAM branch serves models without that output. S7 presents aggregate
quantitative localization metrics.

## Rendering

`figures/render_figures.py` renders Figure 3 and Figures S2-S5 from aggregate data.
`figures/render_s7.py` renders the final quantitative-only S7 from its table,
completion metadata and ID manifest. Outputs go to the Git-ignored directory
`build/figures/`. The manifest provides IDs and stored map hashes, while map pixels
remain separate assets. The renderer uses the aggregate tables. Source CSV values
remain unchanged.

The released renderers cover Figure 3 and Figures S2-S5/S7. Other paper figures
and the original result archives are maintained separately.
