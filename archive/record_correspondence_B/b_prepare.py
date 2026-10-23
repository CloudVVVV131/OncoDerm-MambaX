"""Bind only accepted sources and fixed original models; no A1 reprofiling."""
from b_common import *
import shutil
import sys
import numpy as np
import pandas as pd
import yaml

def main():
    assert not (OUT / "B_RUN_PROTOCOL.json").exists(), "Already frozen"
    PRIVATE.mkdir(exist_ok=True)
    source = []
    def observe(path, role, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected:
            assert digest == expected, f"Source SHA mismatch: {path}"
        source.append(dict(path=str(path), role=role, bytes=path.stat().st_size, sha256=digest))
        return digest

    for line in (START / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            observe(START / name.lstrip("* "), "task package", digest)
    for name in ["B_CLINICAL_MATCHING_PROTOCOL.md", "B_PROTOCOL_PARAMETERS.json",
                 "clinical_matching_reference.py", "test_clinical_matching_reference.py"]:
        shutil.copy2(START / name, OUT / name)
    source_manifest = rows(OC / "MODEL_SOURCE_MANIFEST.csv")
    for r in source_manifest:
        p = Path(r["actual_import_copy"])
        observe(p, "executed original model source", r["sha256"])
        dest = OUT / "model_code" / p.relative_to(OC / "model_code")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        assert sha(dest) == r["sha256"]
    sys.path.insert(0, str(OUT / "model_code"))
    from lesionmamba.data.datasets import encode_metadata
    ids = read(OC / "FIXED_MODEL_IDENTITIES.json")
    assert [e["tag"] for e in ids] == [f"ORIG_{f}_s{s}" for f in ["B2","B6"] for s in [42,2024,3407]]
    observe(OC / "FIXED_MODEL_IDENTITIES.json", "authoritative model identity")
    for e in ids:
        observe(e["checkpoint_path"], "fixed original best", e["sha256"])
        assert Path(e["checkpoint_path"]).stat().st_size == e["bytes"]
        observe(e["config_path"], "original config", e["config_sha256"])
        observe(e["config_copy"], "executed config", e["config_sha256"])
        dest = OUT / "original_configs" / f'{e["tag"]}.yaml'
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(e["config_copy"], dest)
    write(OUT / "FIXED_MODEL_IDENTITIES.json", ids)
    cfgs = [yaml.safe_load(Path(e["config_copy"]).read_text(encoding="utf-8-sig")) for e in ids]
    mc = cfgs[0]["data"]["metadata"]
    assert all(c["data"]["metadata"] == mc for c in cfgs)
    null = encode_metadata(pd.Series(dict(age=np.nan, sex="unknown", localization="unknown")), mc).numpy()
    assert np.array_equal(null, read(OC / "CLINICAL_MAPPING.json")["E_NULL"])
    acceptance = {r["absolute_path"]:r for r in rows(A1 / "SOURCE_MANIFEST.csv")}
    metadata_path = ROOT.parent / "_figure_source_freeze_20260819/ISIC_2020_Training_GroundTruth_v2.csv"
    observe(metadata_path, "accepted SIIM metadata", acceptance[str(metadata_path)]["sha256"])
    siim = pd.read_csv(metadata_path, dtype={"image_name":str,"patient_id":str}).set_index("image_name")
    alignment = pd.read_csv(B2 / "local_only/metadata/SIIM_alignment.csv", dtype=str)
    assert alignment.image_name.is_unique
    siim = siim.loc[alignment.image_name].reset_index()
    assert siim.patient_id.tolist() == alignment.patient_id.tolist()
    assert np.array_equal(siim.target.astype(int), alignment.target.astype(int))
    mapped = pd.DataFrame(dict(age=pd.to_numeric(siim.age_approx, errors="coerce"),
                              sex=siim.sex, localization=siim.anatom_site_general_challenge.replace({"torso":"trunk"})))
    encoded = np.stack([encode_metadata(r,mc).numpy() for _,r in mapped.iterrows()])
    old_real = np.load(B2 / "local_only/metadata/SIIM_E_REAL.npy", allow_pickle=False)
    assert np.array_equal(encoded, old_real)
    image_rows = rows(B2 / "local_only/metadata/SIIM_images.csv")
    assert [r["image_id"] for r in image_rows] == siim.image_name.tolist()
    order = siim.sort_values(["patient_id","image_name"],kind="stable").index.to_numpy()
    old_to_new = np.argsort(order)
    s = siim.iloc[order].reset_index(drop=True)
    s_records = [dict(path=image_rows[i]["path"],sha256=image_rows[i]["sha256_historical"],
                      image_id=image_rows[i]["image_id"]) for i in order]
    groups = [g.index.to_numpy() for _,g in s.groupby("patient_id",sort=False)]
    specs = []
    import clinical_matching_reference as ref
    def save_binding(name, clinical, labels, groups, records, old_order):
        clinical = np.asarray(clinical, dtype=np.float32)
        unique, types = np.unique(clinical,axis=0,return_inverse=True)
        hit = np.flatnonzero(np.all(unique==null,axis=1))
        null_type = int(hit[0]) if len(hit) else len(unique)
        if not len(hit):
            unique = np.concatenate([unique,null[None]])
        strata = ref.make_strata(groups,len(labels))
        group_ids = np.empty(len(labels),dtype=np.int64)
        for i,g in enumerate(groups):
            group_ids[g]=i
        p = PRIVATE / name
        p.mkdir(exist_ok=True)
        for key,value in dict(vectors=unique,types=types,labels=labels,group_ids=group_ids,
                              old_order=old_order).items():
            np.save(p/f"{key}.npy", value, allow_pickle=False)
        np.savez(p/"strata.npz", **{str(i):a for i,a in enumerate(strata)})
        csvout(p/"images.csv",records)
        single = [a for a in strata if len(a)==1]
        detail=dict(cohort=name,n=len(labels),positive=int(np.sum(labels)),groups=len(groups),
                    observed_types=len(np.unique(types)),grid_types=len(unique),null_type=null_type,
                    singleton_size_groups=len(single),singleton_images=sum(a.size for a in single),
                    canonical_order="patient_id/image_name exact lexical" if name=="SIIM" else "cluster_case_num/case_num numeric and exact derm path",
                    grouping_unit="patient" if name=="SIIM" else "case/same-image cluster, not verified patient",
                    private_binding_hashes={f.name:sha(f) for f in p.iterdir() if f.is_file()})
        write(p/"binding.json",detail)
        specs.append(detail)
    save_binding("SIIM", encoded[order], s.target.to_numpy(np.int8), groups,s_records,old_to_new)
    binding = read(OC / "COHORT_BINDING.json")
    for filename,key in [("COHORT_ALIGNMENT.csv","alignment_sha256"),("E_AVAILABLE.npy","clinical_sha256"),
                         ("labels.npy","labels_sha256"),("clusters.npy","clusters_sha256")]:
        observe(OC/"private_predictions"/filename,"accepted Derm binding",binding[key])
    d = pd.read_csv(OC/"private_predictions/COHORT_ALIGNMENT.csv")
    de = np.load(OC/"private_predictions/E_AVAILABLE.npy",allow_pickle=False)
    dy = np.load(OC/"private_predictions/labels.npy",allow_pickle=False)
    dc = np.load(OC/"private_predictions/clusters.npy",allow_pickle=False)
    assert np.array_equal(d.target,dy)
    cluster_check=pd.DataFrame(dict(case_cluster=d.cluster_case_num,saved_cluster=dc))
    assert cluster_check.groupby("case_cluster").saved_cluster.nunique().max()==1
    assert cluster_check.groupby("saved_cluster").case_cluster.nunique().max()==1
    write(OUT/"DERM_CLUSTER_REPRESENTATION_QC.json",dict(
        status="PASS",same_partition=True,clusters=int(d.cluster_case_num.nunique()),
        note="Historical clusters.npy stores encoded cluster indices, not numeric case_num representatives. Verified bijection; canonical order uses original cluster_case_num as prescribed."))
    mapping = read(OC/"CLINICAL_MAPPING.json")
    dre = []
    for r in d.to_dict("records"):
        loc=r["location"] if r["location"] in mapping["unsupported_nonempty"] else mapping["location_mapping"][r["location"]]
        dre.append(encode_metadata(pd.Series(dict(age=np.nan,sex=r["sex"],localization=loc)),mc).numpy())
    assert np.array_equal(de,np.stack(dre))
    do = d.sort_values(["cluster_case_num","case_num","derm"],kind="stable").index.to_numpy()
    ds = d.iloc[do].reset_index(drop=True)
    dg = [g.index.to_numpy() for _,g in ds.groupby("cluster_case_num",sort=False)]
    dr = [dict(path=str(DATA/"images"/r["derm"]),sha256=r["file_sha256"],
               image_id=str(r["case_num"])) for r in ds.to_dict("records")]
    save_binding("Derm7pt",de[do],dy[do],dg,dr,np.argsort(do))
    assert [(s["n"],s["positive"],s["groups"],s["observed_types"],s["singleton_images"]) for s in specs] == [(33126,584,2056,167,1488),(1011,252,1011,16,0)]
    assert specs[0]["singleton_size_groups"] == 19
    th = B2 / "DEVELOPMENT_THRESHOLDS.csv"
    observe(th,"original six HAM validation fixed thresholds")
    shutil.copy2(th,OUT/"DEVELOPMENT_THRESHOLDS.csv")
    csvout(OUT/"SOURCE_MANIFEST.csv",source)
    freeze=dict(protocol_id="FIM_B_ORIGINAL_RECORD_MATCHING_v1.0_20261020",
                first_execution="2026-10-20T21:41:01+08:00",deadline=DEADLINE.isoformat(),frozen_at=now(),
                K=50,B=2000,seed=20261020,phase=[1,2,3],models=[e["tag"] for e in ids],
                cohorts=specs,precision="FP32",TF32=False,training_runs=0,workers_max=2,
                model_sources={str(p.relative_to(OUT)):sha(p) for p in (OUT/"model_code").rglob("*.py")},
                code_hashes={p.name:sha(p) for p in OUT.glob("*.py")},
                privacy="All individual records, vectors, grids, maps and predictions stay private",
                scientific_protocol_sha256=sha(OUT/"B_CLINICAL_MATCHING_PROTOCOL.md"))
    write(OUT/"B_RUN_PROTOCOL.json",freeze)
    log("Input binding and protocol freeze passed; no A1 formal statistics rerun.")

if __name__=="__main__":
    main()
