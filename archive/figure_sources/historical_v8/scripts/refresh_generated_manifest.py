from build_bioengineering_data_figures import build_generated_manifest

report = build_generated_manifest()
print(f"manifest_rows={len(report)}")
