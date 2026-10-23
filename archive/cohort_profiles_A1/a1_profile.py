"""CPU metadata-only A1. No model, prediction, image or checkpoint imports.

Run with -B. Inputs are an explicit allowlist derived from accepted Stage0.
All row-level material stays in memory. Writes use exclusive creation.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ZONE = timezone(timedelta(hours=8))
START = datetime.now(ZONE).isoformat(timespec="seconds")
NA_TOKENS = {"nan", "na", "n/a", "null", "none"}
UNKNOWN_TOKENS = {"unknown", "unk", "unspecified", "not specified"}
REASONS = ["column_absent", "blank", "NA_token", "explicit_unknown", "invalid_or_nonfinite_numeric", "valid"]
CACHE = {}
MANIFEST = {}
QC = []
BLOCKERS = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(name, actual, expected):
    ok = actual == expected
    QC.append(dict(check=name, observed=actual, expected=expected, passed=bool(ok)))
    if not ok:
        raise ValueError("QC failed: " + name + "; aggregates=" + repr((actual, expected)))


def read_source(path, role, expected=None, reference="current fingerprint only"):
    path = Path(path).resolve()
    key = str(path)
    if key not in CACHE:
        a = path.stat()
        data = path.read_bytes()
        b = path.stat()
        check("stable read: " + path.name, (a.st_size, a.st_mtime_ns), (b.st_size, b.st_mtime_ns))
        CACHE[key] = data
        MANIFEST[key] = dict(role=role, absolute_path=key,
            relative_to_project=os.path.relpath(path, ROOT), bytes=a.st_size,
            modified_at_Asia_Shanghai=datetime.fromtimestamp(a.st_mtime, ZONE).isoformat(),
            mtime_ns=a.st_mtime_ns, sha256=digest(data), expected_sha256=expected or "",
            expected_hash_source=reference, hash_comparison="MATCH" if expected else "NO_PRIOR_HASH",
            source_content_packaged=False, post_read_unchanged=None)
    if expected:
        check("source SHA: " + path.name, digest(CACHE[key]), expected)
        MANIFEST[key].update(expected_sha256=expected, expected_hash_source=reference, hash_comparison="MATCH")
    return CACHE[key]


def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def obj(data):
    return json.loads(data.decode("utf-8-sig"))


def write_csv(name, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (OUT / name).open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: ("NA" if v is None else v) for k, v in r.items()})


def write_json(name, value):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_md(name, text):
    with (OUT / name).open("x", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def raw_state(row, field, numeric=False):
    if field is None or field not in row:
        return "column_absent", None
    s = row[field].strip()
    if not s:
        return "blank", None
    if s.lower() in NA_TOKENS:
        return "NA_token", None
    if s.lower() in UNKNOWN_TOKENS:
        return "explicit_unknown", None
    if numeric:
        try:
            value = float(s)
        except ValueError:
            return "invalid_or_nonfinite_numeric", None
        if not math.isfinite(value):
            return "invalid_or_nonfinite_numeric", None
        return "valid", value
    return "valid", s


def quantiles(values):
    if not values:
        return dict(n=0, min=None, q1=None, median=None, q3=None, max=None)
    a = np.asarray(values, dtype=np.float64)
    q = np.quantile(a, [0.25, 0.5, 0.75], method="linear")
    return dict(n=len(a), min=float(a.min()), q1=float(q[0]), median=float(q[1]), q3=float(q[2]), max=float(a.max()))


def independent_linear(values, q):
    if not values:
        return None
    a = sorted(values)
    position = (len(a) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return a[lo] + (a[hi] - a[lo]) * (position - lo)


def pct(n, den):
    return 100.0 * n / den if den else None


class TensorAllocationOnly(ast.NodeTransformer):
    """Retain original pure encoder body; replace only tensor allocation.

    No torch import or GPU dependency is introduced. Return dtype remains f32.
    """
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "torch":
            if node.attr in ("tensor", "empty", "float32"):
                attr = {"tensor": "asarray", "empty": "empty", "float32": "float32"}[node.attr]
                return ast.copy_location(ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=attr, ctx=ast.Load()), node)
        return node


def pure_encoder(source):
    tree = ast.parse(source)
    wanted = {"DEFAULT_SEX_VALUES", "DEFAULT_LOCALIZATION_VALUES", "metadata_dim", "_norm_text", "encode_metadata"}
    selected, locations = [], {}
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in wanted:
            selected.append(node)
            locations[name] = [node.lineno, node.end_lineno]
    check("pure encoder symbol whitelist", set(locations), wanted)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected, type_ignores=[])
    module = ast.fix_missing_locations(TensorAllocationOnly().visit(module))
    scope = {"pd": pd, "np": np, "Any": Any}
    exec(compile(module, "selected_metadata_functions_no_model_import", "exec"), scope)
    return scope, locations


def manual_encoder(age, sex, site, sex_vocab, site_vocab):
    age_missing = bool(pd.isna(age))
    try:
        a = 0.0 if age_missing else float(age) / 100.0
    except (TypeError, ValueError):
        a, age_missing = 0.0, True
    def norm(v):
        if pd.isna(v):
            return "unknown"
        return str(v).strip().lower() or "unknown"
    sx, st = norm(sex), norm(site)
    sx_token = sx if sx in sex_vocab else "unknown"
    st_token = st if st in site_vocab else "unknown"
    return np.asarray([a] + [float(sx_token == x) for x in sex_vocab] +
        [float(st_token == x) for x in site_vocab] + [float(age_missing), float(sx == "unknown"), float(st == "unknown")], dtype=np.float32)


def main():
    global ROOT
    cfg = json.loads((OUT / "a1_inputs.json").read_text(encoding="utf-8"))
    ROOT = Path(cfg["project_root"]).resolve()
    task, base = Path(cfg["task_dir"]), Path(cfg["accepted_stage0"])
    b2, oc = Path(cfg["original_b2_root"]), Path(cfg["originalcore_root"])
    read_source(OUT / "a1_inputs.json", "new reproduction configuration")
    read_source(Path(__file__), "new metadata-only implementation")
    package = read_source(cfg["task_zip"], "user-specified A1 start package")
    with zipfile.ZipFile(io.BytesIO(package)) as z:
        check("start ZIP CRC", z.testzip(), None)
        checksum = read_source(task / "SHA256SUMS.txt", "start package checksum index").decode("utf-8-sig")
        for line in checksum.splitlines():
            h, name = line.split(None, 1)
            name = name.strip()
            content = read_source(task / name, "current A1 task/scope", h, "SHA256SUMS.txt")
            check("extracted ZIP member " + name, content, z.read(name))
    fields = rows(read_source(base / "COHORT_FIELDS_AVAILABILITY.csv", "accepted metadata paths and field definitions"))
    registered = rows(read_source(base / "RESULTS_SOURCE_REGISTER.csv", "accepted index; no traversal/re-hash of indexed assets"))
    manuscripts = rows(read_source(base / "MANUSCRIPT_BASELINE_MANIFEST.csv", "accepted manuscript identities only; manuscripts not opened"))
    read_source(base / "BRANCH_SEPARATION_REGISTER.md", "historical scope register superseded by v1.2")
    read_source(cfg["prior_missing_rule_script"], "static Stage0 missing-token specification; not imported")
    registry = {str(Path(r["absolute_path"]).resolve()): r["sha256"] for r in registered}
    def permitted(path, role):
        key = str(Path(path).resolve())
        return read_source(path, role, registry.get(key), "accepted RESULTS_SOURCE_REGISTER.csv" if key in registry else "explicit task dependency; current fingerprint")
    encoder_path = b2 / "model_code/lesionmamba/data/datasets.py"
    encoder_text = permitted(encoder_path, "static pure metadata functions").decode("utf-8-sig")
    scope, lines = pure_encoder(encoder_text)
    token_map = rows(permitted(b2 / "METADATA_TOKEN_MAPPING.csv", "historical SIIM mapping/count check"))
    b2_protocol = obj(permitted(b2 / "B2_FROZEN_PROTOCOL.json", "historical original-model scoring identity; no statistics executed"))
    siim_mapping = obj(permitted(b2 / "metadata_mapping.json", "explicit mapping dependency of original evaluation"))
    derm_mapping = obj(permitted(oc / "CLINICAL_MAPPING.json", "frozen Derm7pt clinical mapping"))
    binding = obj(permitted(oc / "COHORT_BINDING.json", "original-cohort binding summary"))
    identities = obj(permitted(oc / "FIXED_MODEL_IDENTITIES.json", "six model identities only; referenced files not opened"))
    oc_protocol = obj(permitted(oc / "ORIGINALCORE_PROTOCOL.json", "explicit label mapping dependency; only labels used"))
    for name in ["core_common.py", "core_prepare.py", "core_data.py"]:
        permitted(oc / name, "static binding/encoding dependency; not imported or executed")
    alignment_path = oc / "private_predictions/COHORT_ALIGNMENT.csv"
    alignment = rows(read_source(alignment_path, "case/same-image binding metadata, NOT predictions", binding["alignment_sha256"], "COHORT_BINDING.json#alignment_sha256"))
    sex_vocab, site_vocab = scope["DEFAULT_SEX_VALUES"], scope["DEFAULT_LOCALIZATION_VALUES"]
    check("original vocab sex", sex_vocab, derm_mapping["sex_vocabulary"])
    check("original vocab site", site_vocab, derm_mapping["site_vocabulary"])
    check("SIIM token order", siim_mapping["token_order"], derm_mapping["token_order"])
    check("SIIM age scale", siim_mapping["age"]["scale"], 100)
    check("six original model identities", sorted((x["model"], x["training_seed"]) for x in identities), sorted((m,s) for m in ["ORIG_B2", "ORIG_B6"] for s in [42,2024,3407]))
    encoder_cfg = dict(enabled=True, age_scale=100.0, sex_values=sex_vocab, localization_values=site_vocab, include_missing_indicators=True)
    check("original metadata dimension", scope["metadata_dim"](encoder_cfg), 22)
    check("structured NULL coverage", scope["encode_metadata"]({"age":np.nan,"sex":"unknown","localization":"unknown"}, encoder_cfg).tolist(), derm_mapping["E_NULL"])
    pos_labels = set(oc_protocol["labels"]["positive"])
    neg_labels = set(oc_protocol["labels"]["negative"])
    check("Derm positive label category count", len(pos_labels), 6)
    characteristics, distributions, coverage, coverage_details = [], [], [], []
    cohort_data, summaries = {}, {}
    protected_ids = set()

    for spec in cfg["cohorts"]:
        name = spec["name"]
        matches = {}
        for f in fields:
            if f["cohort"] == spec["stage0_cohort"]:
                for p,h in zip(f["source_absolute_paths"].split(" | "), f["source_sha256"].split(" | ")):
                    if Path(p).name == spec["filename"]:
                        matches[p] = h
        check(name + " exact registered source", len(matches), 1)
        path, prior_hash = next(iter(matches.items()))
        content = read_source(path, name + " source metadata", prior_hash, "COHORT_FIELDS_AVAILABILITY.csv")
        data = rows(content)
        # Same cached bytes supply the independent pandas parser; no second data source.
        frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, na_filter=False)
        model_frame = pd.read_csv(io.BytesIO(content))
        n = len(data)
        check(name + " row count", n, spec["expected_rows"])
        check(name + " independent CSV parse", frame.to_dict("records"), data)
        if name.startswith("HAM"):
            check(name + " labels", sorted({r["dx"] for r in data}), ["akiec","bcc","bkl","df","mel","nv","vasc"])
            positive = sum(r["dx"] == "mel" for r in data)
        elif name == "SIIM":
            check("SIIM target vocabulary", sorted({r["target"] for r in data}), ["0", "1"])
            positive = sum(r["target"] == "1" for r in data)
        else:
            check("Derm original 20 labels", sorted({r["diagnosis"] for r in data}), sorted(pos_labels | neg_labels))
            positive = sum(r["diagnosis"] in pos_labels for r in data)
        check(name + " positive label count", positive, spec["expected_positive"])
        def add(field, statistic, count=None, den=None, value=None, value_unit=None, note=""):
            characteristics.append(dict(cohort=name, field=field, statistic=statistic, n=count,
                denominator=den, percent=pct(count,den) if count is not None and den else None,
                value=value, value_unit=value_unit or spec["unit"], statistical_unit=spec["unit"],
                source_path=path, source_field=spec.get(field,field), note=note))
        add("cohort", "rows", n, n)
        add("label", "melanoma_positive", positive, n, note="non-melanoma is not synonymous with benign")
        add("label", "non_melanoma", n-positive, n)
        ids = {}
        for typ in ["image", "lesion", "patient", "case"]:
            col = spec[typ]
            if col and col in data[0]:
                valid_ids = [r[col] for r in data if raw_state(r,col)[0] == "valid"]
                ids[typ] = len(set(valid_ids))
                add(typ, "distinct_identifier", ids[typ], value_unit=typ + " identifier", note="Distinct identifiers, not percent of rows")
                add(typ, "unavailable_identifier_rows", n-len(valid_ids), n)
                # Only IDs, not clinical values, are retained transiently for package leakage checks.
                protected_ids.update(v for v in valid_ids if len(v) >= 8 and not v.isdigit())
            else:
                ids[typ] = None
                add(typ, "distinct_identifier", note="NA: no reliable " + typ + " identifier in this source")
        states = {}
        for field in ["age", "sex", "site"]:
            col = spec[field]
            state = [raw_state(r,col,field=="age") for r in data]
            states[field] = state
            counts = Counter(x[0] for x in state)
            check(name + ": " + field + " missing partition", sum(counts.values()), n)
            add(field, "valid", counts["valid"], n)
            add(field, "unavailable", n-counts["valid"], n)
            for reason in REASONS[:-1]:
                add(field, reason, counts[reason], n)
                distributions.append(dict(cohort=name, field=field, row_type="missing_reason", category=reason,
                    n=counts[reason], denominator=n, percent=pct(counts[reason],n), valid_denominator=counts["valid"],
                    percent_of_valid=None, statistical_unit=spec["unit"], source_path=path, source_field=col))
        ages = [v for s,v in states["age"] if s == "valid"]
        age_summary = quantiles(ages)
        for stat in ["min", "q1", "median", "q3", "max"]:
            add("age", stat, len(ages), len(ages) or None, age_summary[stat], "years", "Finite valid ages only; linear quantile; no imputation/clipping")
        add("age", "observed_zero_not_missing", sum(v == 0 for v in ages), n)
        add("age", "observed_negative_not_excluded", sum(v < 0 for v in ages), n)
        # Independently derive valid ages and quantiles using a separately parsed DataFrame.
        if spec["age"] in frame:
            s = frame[spec["age"]].str.strip()
            excluded = s.str.lower().isin(NA_TOKENS | UNKNOWN_TOKENS | {""})
            numbers = pd.to_numeric(s.mask(excluded), errors="coerce")
            ages_second = numbers[np.isfinite(numbers)].astype(float).tolist()
        else:
            ages_second = []
        check(name + " independently parsed valid ages", sorted(ages), sorted(ages_second))
        for stat,q in [("q1",.25),("median",.5),("q3",.75)]:
            check(name + " independent linear " + stat, age_summary[stat], independent_linear(ages_second,q))
        cats = {}
        for field,col in [("sex",spec["sex"]),("site",spec["site"]),("diagnosis",spec["label"])] + ([("diagnosis_method","dx_type")] if name.startswith("HAM") else []):
            counts = Counter(r[col].strip() for r in data)
            independent_counts = {str(k):int(v) for k,v in frame[col].str.strip().value_counts(dropna=False).items()}
            check(name + ": independent full " + field + " distribution", dict(counts), independent_counts)
            cats[field] = dict(counts)
            for cat,count in sorted(counts.items()):
                status = raw_state({col:cat},col)[0] if field in ["sex","site"] else "label_or_method"
                vden = sum(s=="valid" for s,_ in states[field]) if field in states else n
                distributions.append(dict(cohort=name, field=field, row_type="source_category", category=cat or "[blank]",
                    missing_reason=status, n=count, denominator=n, percent=pct(count,n), valid_denominator=vden,
                    percent_of_valid=pct(count,vden) if status in ["valid","label_or_method"] else None,
                    statistical_unit=spec["unit"], source_path=path, source_field=col))
        if name == "SIIM":
            check("SIIM diagnosis unknown does not exclude target", sum(r["diagnosis"]=="unknown" for r in data), 27124)
        mapped = []
        vectors = []
        for idx,r in enumerate(data):
            pr = model_frame.iloc[idx]
            if name == "SIIM":
                def norm(v):
                    return "unknown" if pd.isna(v) else (" ".join(str(v).strip().lower().split()) or "unknown")
                age, sex, site = pr[spec["age"]], norm(pr[spec["sex"]]), norm(pr[spec["site"]])
                site = siim_mapping["site"]["synonyms"].get(site,site)
            elif name == "Derm7pt":
                raw_site = r[spec["site"]]
                if raw_site not in derm_mapping["location_mapping"]:
                    raise ValueError("Unexpected Derm7pt site category")
                age, sex = np.nan, r[spec["sex"]]
                site = raw_site if raw_site in derm_mapping["unsupported_nonempty"] else derm_mapping["location_mapping"][raw_site]
            else:
                age, sex, site = pr[spec["age"]], pr[spec["sex"]], pr[spec["site"]]
            vector = scope["encode_metadata"]({"age":age,"sex":sex,"localization":site}, encoder_cfg)
            reference = manual_encoder(age,sex,site,sex_vocab,site_vocab)
            if not np.array_equal(vector,reference, equal_nan=True):
                raise ValueError("Independent metadata encoder mismatch in " + name)
            if not np.isfinite(vector).all():
                raise ValueError("Nonfinite original encoded vector in " + name + "; no encoder repair authorized")
            vectors.append(tuple(float(x) for x in vector))
            mapped.append(dict(age=age,sex=sex,site=site))
        check(name + " all-row independent encoding", len(vectors), n)
        for field,vocab,lo,hi,flag in [("age",None,0,1,19),("sex",sex_vocab,1,4,20),("site",site_vocab,4,19,21)]:
            buckets = Counter()
            details = Counter()
            discrepancies = Counter()
            for i,r in enumerate(data):
                state = states[field][i][0]
                raw_valid = state == "valid"
                v = vectors[i]
                token = None if field == "age" else vocab[int(np.argmax(v[lo:hi]))]
                flag_missing = int(v[flag])
                unsupported = raw_valid and token == "unknown" if field != "age" else False
                bucket = "truly_missing" if not raw_valid else ("nonempty_unsupported" if unsupported else "available_supported")
                buckets[bucket] += 1
                buckets["encoded_unknown"] += int(token == "unknown")
                buckets["missing_flag_1"] += flag_missing
                if bool(not raw_valid) != bool(flag_missing):
                    discrepancies[(state, flag_missing)] += 1
                # No exact ages or linked clinical combinations are exported here.
                category = "finite_value" if field == "age" and raw_valid else ("["+state+"]" if not raw_valid else r[spec[field]].strip())
                passed = "age/100" if field == "age" and raw_valid else ("missing" if field == "age" else str(mapped[i][field]))
                details[(category,state,passed,str(token) if token is not None else "NA",flag_missing,bucket)] += 1
            check(name + " " + field + " coverage partition", sum(buckets[x] for x in ["truly_missing","nonempty_unsupported","available_supported"]), n)
            coverage.append(dict(cohort=name,field=field,denominator=n,statistical_unit=spec["unit"],
                available_supported=buckets["available_supported"], nonempty_unsupported=buckets["nonempty_unsupported"],
                truly_missing=buckets["truly_missing"],encoded_unknown=buckets["encoded_unknown"] if vocab else None,
                missing_flag_1=buckets["missing_flag_1"],raw_vs_flag_disagreement=sum(discrepancies.values()),
                disagreement_reasons=json.dumps([{ "raw_reason":k[0],"flag":k[1],"n":v} for k,v in discrepancies.items()]),
                source_path=path,source_field=spec[field],encoder_path=str(encoder_path),encoder_lines="61-101",
                mapping_source=str(oc/"CLINICAL_MAPPING.json") if name=="Derm7pt" else (str(b2/"metadata_mapping.json") if name=="SIIM" else "Original encoder defaults")))
            for (cat,state,passed,token,flag_missing,bucket),count in sorted(details.items()):
                coverage_details.append(dict(cohort=name,field=field,source_category=cat,raw_state=state,
                    value_passed_to_encoder=passed,encoded_token=token,missing_flag=flag_missing,coverage_state=bucket,
                    n=count,denominator=n,percent=pct(count,n),source_path=path,source_field=spec[field]))
        cohort_data[name] = dict(spec=spec,rows=data,frame=frame,states=states,vectors=vectors,source=path)
        summaries[name] = dict(n=n,positive=positive,percent_positive=pct(positive,n),unit=spec["unit"],age=age_summary,
            missing={f: n-sum(s=="valid" for s,v in states[f]) for f in states}, ids=ids,categories=cats)
    check("HAM split total", sum(summaries[c]["n"] for c in summaries if c.startswith("HAM")), 10015)
    # Recheck the accepted missingness totals without re-freezing source assets.
    for prior_cohort in ["HAM10000","SIIM-ISIC2020","Derm7pt"]:
        names = [s["name"] for s in cfg["cohorts"] if s["stage0_cohort"]==prior_cohort]
        for concept,f in [("age","age"),("sex","sex"),("anatomical_site","site")]:
            p = next(x for x in fields if x["cohort"]==prior_cohort and x["concept"]==concept)
            check(prior_cohort+" prior "+f+" unavailable", sum(summaries[x]["missing"][f] for x in names), int(p["unavailable_rows"]))
    for r in token_map:
        field = "sex" if r["field"]=="sex" else "site"
        found = sum(x["n"] for x in coverage_details if x["cohort"]=="SIIM" and x["field"]==field and
            x["value_passed_to_encoder"]==r["normalized_value_passed_to_encoder"] and x["encoded_token"]==r["encoder_token"] and x["missing_flag"]==int(r["missing_flag"]))
        check("SIIM frozen mapping category " + r["source_value_normalized"], found, int(r["count"]))

    patient_rows, structure = patient_structure(cohort_data["SIIM"], site_vocab)
    verify_patient_structure(cohort_data["SIIM"], structure, site_vocab)
    derm = derm_structure(cohort_data["Derm7pt"], alignment, binding, pos_labels, oc)
    identities_clean = [{k:r[k] for k in ["model","training_seed","run_id","sha256"]} for r in identities]
    design = dict(status="DESIGN_INPUTS_COMPLETE_PROTOCOL_NOT_FROZEN",computed_at=START,
        inference_executed=False,donor_mapping_created=False,metadata_read_once_and_shared=True,
        original_models=identities_clean,checkpoint_hash_status="Historical identity references only; weights not opened",
        encoder=dict(path=str(encoder_path),functions_and_lines=lines,dimension=22,token_order=derm_mapping["token_order"],
            null_vector=derm_mapping["E_NULL"],age_scale=100,vector_equality="Exact float32 component equality; includes missing flags"),
        SIIM=structure,Derm7pt=derm,
        raw_vs_encoder_discrepancies=[r for r in coverage if r["raw_vs_flag_disagreement"]],
        pending_protocol_parameters=["Exchange unit and algorithm", "Within-patient multi-lesion ordering and correspondence", "Unequal block sizes and singleton size strata", "Self-donor/fixed-point handling", "Repeated vector handling and actual-change reporting", "Missingness-mode movement/stratification", "Fixed K and RNG", "Mapping sharing across six models and heads", "Primary family/cohort estimands and multiplicity", "Uncertainty intervals and donor dependence", "Resource budget and checkpoints"],
        scope="Metadata-only inputs; no performance, donor mapping, random sampling or inference",blockers=BLOCKERS)

    # All outputs are generated only after numerical and source checks succeed.
    for key,entry in MANIFEST.items():
        p=Path(key); st=p.stat()
        check("source unchanged at completion: "+p.name, (digest(p.read_bytes()),st.st_size,st.st_mtime_ns), (entry["sha256"],entry["bytes"],entry["mtime_ns"]))
        entry["post_read_unchanged"]=True
    write_csv("COHORT_CLINICAL_CHARACTERISTICS.csv",characteristics)
    write_csv("COHORT_FIELD_DISTRIBUTIONS.csv",distributions)
    write_csv("CLINICAL_ENCODING_COVERAGE.csv",coverage)
    write_csv("CLINICAL_ENCODING_CATEGORY_DETAIL.csv",coverage_details)
    write_csv("SIIM_PATIENT_STRUCTURE.csv",patient_rows)
    write_csv("SOURCE_MANIFEST.csv",list(MANIFEST.values()))
    write_json("CLINICAL_MATCHING_DESIGN_INPUTS.json",design)
    write_md("COHORT_CLINICAL_CHARACTERISTICS_TABLE.md",table_md(summaries))
    write_md("COHORT_SUMMARY_METHOD.md",method_md(cfg,lines,structure,derm))
    write_md("CLINICAL_MATCHING_PROTOCOL_DRAFT.md",draft_md(design))
    write_md("SCOPE_DECISION_v1_2.md",scope_md(task,manuscripts))
    write_md("A1_PROFILE_REPORT.md",report_md(summaries,coverage,structure,derm))
    # QC serializes only aggregates/hashes, never compared raw lists or row values.
    sanitized=[]
    for q in QC:
        if isinstance(q["observed"],(list,dict,set,bytes,tuple)):
            sanitized.append({"check":q["check"],"passed":q["passed"],"comparison":"In-memory equality; values withheld; see aggregate tables"})
        else:
            sanitized.append(q)
    write_json("A1_PROFILE_QC.json",dict(status="PASS",checks=sanitized,failed_checks=0,
        metadata_sources=5,metadata_rows=sum(s["n"] for s in summaries.values()),
        source_files_hashed=len(MANIFEST),full_526_source_refreeze=False,
        independent_checks="csv.DictReader vs pandas parser; Counter vs pandas value_counts; numpy linear vs sorted interpolation; AST pure encoder vs independent numpy implementation; independently vectorized pandas patient consistency and vector-sharing checks",
        versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},
        no_row_level_output=True, package_validation_receipt="DELIVERY_RECEIPT.json (external to ZIP)"))
    write_json("A1_PROFILE_PROGRESS.json",dict(status="COMPLETE",first_A1_launch=True,started_at=START,
        completed_at=datetime.now(ZONE).isoformat(timespec="seconds"),A1="COMPLETE",B_DESIGN_INPUTS="COMPLETE",
        B_PROTOCOL="DRAFT_NOT_EXECUTED_NOT_FINAL_FROZEN",B_INFERENCE="NOT_STARTED",NEW_TRAINING=0,
        model_weights_opened=0,model_predictions_read=0,model_performance_computed=False,
        input_metadata_files=5,cohorts={k:v["n"] for k,v in summaries.items()},blockers=BLOCKERS,
        next_action="STOP; await ChatGPT protocol review",output_directory=str(OUT),
        zip="R5_2_CLINICAL_PROFILE_RESULTS.zip",zip_validation_receipt="DELIVERY_RECEIPT.json (external)"))
    package_outputs(protected_ids)
    print(json.dumps(dict(status="COMPLETE",output=str(OUT),cohorts={k:{"n":v["n"],"positive":v["positive"],"age":v["age"],"missing":v["missing"]} for k,v in summaries.items()},SIIM=structure["overview"],Derm7pt=derm["overview"],qc_checks=len(QC)),ensure_ascii=False,indent=2))


def patient_structure(c, site_vocab):
    groups = defaultdict(list)
    for i,r in enumerate(c["rows"]):
        check_value = raw_state(r,"patient_id")[0]
        if check_value != "valid":
            raise ValueError("SIIM patient_id missing; grouping requires decision")
        groups[r["patient_id"]].append(i)
    n, g = len(c["rows"]), len(groups)
    check("SIIM patient groups",g,2056)
    sizes = [len(ix) for ix in groups.values()]
    size_counts = Counter(sizes)
    check("SIIM weighted patient-size sum",sum(k*v for k,v in size_counts.items()),n)
    independent_sizes = c["frame"].groupby("patient_id",sort=False).size().tolist()
    check("SIIM independent patient sizes",sorted(sizes),sorted(independent_sizes))
    table=[]
    for k,count in sorted(size_counts.items()):
        table.append(dict(record_type="group_size_frequency",field="patient_block_size",
            n_images_per_patient=k,n_patients=count,n_images_in_bin=k*count,
            denominator_patients=g,percent_patients=pct(count,g),denominator_images=n,
            percent_images=pct(k*count,n),candidate_same_size_exchange_stratum=count>=2))
    summary=quantiles(sizes)
    for key in ["min","q1","median","q3","max"]:
        table.append(dict(record_type="group_size_summary",field="images_per_patient",statistic=key,value=summary[key],denominator_patients=g))
    eligible_patients=sum(v for v in size_counts.values() if v>=2)
    eligible_images=sum(k*v for k,v in size_counts.items() if v>=2)
    singleton_strata=sum(v==1 for v in size_counts.values())
    singleton_images=sum(k for k,v in size_counts.items() if v==1)
    overview=dict(n_images=n,n_patients=g,images_per_patient=summary,n_size_strata=len(size_counts),
        candidate_equal_size_exchange_patients=eligible_patients,candidate_equal_size_exchange_patient_percent=pct(eligible_patients,g),
        candidate_equal_size_exchange_images=eligible_images,candidate_equal_size_exchange_image_percent=pct(eligible_images,n),
        singleton_size_strata=singleton_strata,singleton_stratum_patients=singleton_strata,singleton_stratum_images=singleton_images,
        singleton_stratum_image_percent=pct(singleton_images,n),coverage_only_no_exchange_executed=True)
    check("SIIM stratum coverage patients",eligible_patients+singleton_strata,g)
    check("SIIM stratum coverage images",eligible_images+singleton_images,n)
    consistency={}
    uniquely_known={}
    for field in ["age","sex","site_raw","site_encoded"]:
        counts=Counter(); image_counts=Counter(); unique_values=[]
        for indices in groups.values():
            if field=="site_encoded":
                values=[site_vocab[int(np.argmax(c["vectors"][i][4:19]))] for i in indices if c["vectors"][i][21]==0]
            else:
                original="site" if field=="site_raw" else field
                values=[c["states"][original][i][1] for i in indices if c["states"][original][i][0]=="valid"]
            unique=set(values); n_unique=len(unique)
            kind="0" if n_unique==0 else ("1" if n_unique==1 else ">1")
            mode="all_missing" if not values else ("values_no_missing" if len(values)==len(indices) else "values_plus_missing")
            counts[(kind,mode)]+=1
            image_counts[(kind,mode)]+=len(indices)
            # Strict eligibility: all rows observed and exactly one value, never first/mode/mean.
            if n_unique==1 and mode=="values_no_missing" and field in ["age","sex"]:
                unique_values.append(next(iter(unique)))
        check("SIIM "+field+" consistency patient partition",sum(counts.values()),g)
        check("SIIM "+field+" consistency image partition",sum(image_counts.values()),n)
        records=[]
        for kind in ["0","1",">1"]:
            for mode in ["all_missing","values_no_missing","values_plus_missing"]:
                r=dict(unique_valid_values=kind,missing_pattern=mode,n_patients=counts[(kind,mode)],n_images=image_counts[(kind,mode)],
                    denominator_patients=g,percent_patients=pct(counts[(kind,mode)],g))
                records.append(r)
                table.append(dict(record_type="within_patient_consistency",field=field,**r,
                    interpretation="encoded unknown with site_missing=0 remains an observed category" if field=="site_encoded" else "Source missingness rules"))
        consistency[field]=records
        if field in ["age","sex"]:
            u=dict(eligible_patients=len(unique_values),all_patients=g,percent_eligible=pct(len(unique_values),g),
                definition="Exactly one valid value AND no missing rows within patient; mixed missingness excluded, not resolved")
            if field=="age":
                u["age_years"]=quantiles(unique_values)
                for stat,value in u["age_years"].items():
                    table.append(dict(record_type="uniquely_determined_patient_summary",field=field,statistic=stat,value=value,
                        denominator_patients=len(unique_values),all_patients=g,interpretation=u["definition"]))
            else:
                u["sex_counts"]=dict(sorted(Counter(unique_values).items()))
                for category,count in u["sex_counts"].items():
                    table.append(dict(record_type="uniquely_determined_patient_distribution",field=field,category=category,n_patients=count,
                        denominator_patients=len(unique_values),percent_patients=pct(count,len(unique_values)),all_patients=g,interpretation=u["definition"]))
            uniquely_known[field]=u
    patient_vectors=Counter(); patient_vector_images=Counter()
    vector_patients=defaultdict(set); vector_image_count=Counter()
    for patient,indices in groups.items():
        num=len({c["vectors"][i] for i in indices})
        patient_vectors[num]+=1; patient_vector_images[num]+=len(indices)
        for i in indices:
            vector_patients[c["vectors"][i]].add(patient)
            vector_image_count[c["vectors"][i]]+=1
    frequency=[]
    for k,count in sorted(patient_vectors.items()):
        r=dict(n_unique_full_vectors_per_patient=k,n_patients=count,n_images=patient_vector_images[k],denominator_patients=g,percent_patients=pct(count,g))
        frequency.append(r);table.append(dict(record_type="within_patient_full_vector_frequency",field="full_22dim_vector",**r))
    shared={v for v,patients in vector_patients.items() if len(patients)>=2}
    shared_patients=set().union(*(vector_patients[v] for v in shared)) if shared else set()
    shared_images=sum(vector_image_count[v] for v in shared)
    vector_links=sum(len(patients) for patients in vector_patients.values())
    shared_links=sum(len(vector_patients[v]) for v in shared)
    cross=dict(n_distinct_full_vectors=len(vector_patients),n_vectors_shared_across_patients=len(shared),
        percent_vectors_shared=pct(len(shared),len(vector_patients)),n_images_with_shared_vectors=shared_images,
        percent_images_with_shared_vectors=pct(shared_images,n),n_patients_with_any_shared_vector=len(shared_patients),
        percent_patients_with_any_shared_vector=pct(len(shared_patients),g),n_unique_patient_vector_links=vector_links,
        n_links_to_shared_vectors=shared_links,percent_links_to_shared_vectors=pct(shared_links,vector_links),
        interpretation="Shared vector values can make a changed donor yield unchanged input; no donor mapping created")
    histogram=Counter((len(p),vector_image_count[v]) for v,p in vector_patients.items())
    cross_hist=[]
    for (patients,images),count in sorted(histogram.items()):
        r=dict(n_patients_per_vector=patients,n_images_per_vector=images,n_distinct_vectors=count,
            n_patient_vector_links=patients*count,n_images_in_bin=images*count)
        cross_hist.append(r);table.append(dict(record_type="cross_patient_vector_frequency",field="full_22dim_vector",**r))
    check("SIIM vector frequency weighted rows",sum(x["n_images_in_bin"] for x in cross_hist),n)
    return table,dict(overview=overview,group_size_frequency=[r for r in table if r["record_type"]=="group_size_frequency"],
        within_patient_consistency=consistency,uniquely_determined_patient_fields=uniquely_known,
        within_patient_full_vector_frequency=frequency,cross_patient_repeated_vectors=cross,
        cross_patient_vector_frequency=cross_hist,patient_ids_exported=False,sites_unified=False)


def verify_patient_structure(c, structure, site_vocab):
    frame=c["frame"]
    group=frame["patient_id"]
    size=frame.groupby("patient_id").size()
    computed={}
    for field,col in [("age","age_approx"),("sex","sex"),("site_raw","anatom_site_general_challenge")]:
        s=frame[col].str.strip()
        missing=s.str.lower().isin(NA_TOKENS | UNKNOWN_TOKENS | {""})
        clean=s.mask(missing)
        if field=="age":
            clean=pd.to_numeric(clean,errors="coerce")
            clean=clean.where(np.isfinite(clean))
        computed[field]=clean
    site=frame["anatom_site_general_challenge"].str.strip().str.lower().str.replace(r"\s+"," ",regex=True)
    missing=site.isin(["","unknown"])
    site=site.replace({"torso":"trunk"})
    computed["site_encoded"]=site.where(site.isin(site_vocab),"unknown").mask(missing)
    for field,s in computed.items():
        unique=s.groupby(group).nunique(dropna=True)
        observed=s.groupby(group).count()
        table=pd.DataFrame({"unique":unique,"observed":observed,"size":size})
        table["kind"]=np.where(table["unique"]==0,"0",np.where(table["unique"]==1,"1",">1"))
        table["mode"]=np.where(table["observed"]==0,"all_missing",np.where(table["observed"]==table["size"],"values_no_missing","values_plus_missing"))
        aggregate=table.groupby(["kind","mode"]).agg(n_patients=("size","size"),n_images=("size","sum"))
        for record in structure["within_patient_consistency"][field]:
            key=(record["unique_valid_values"],record["missing_pattern"])
            second=tuple(int(v) for v in aggregate.loc[key]) if key in aggregate.index else (0,0)
            check("independent patient consistency "+field+"/"+"/".join(key),second,(record["n_patients"],record["n_images"]))
    # Hex keeps embedded/trailing NUL bytes out of pandas membership hashing.
    # These in-memory keys are never exported and preserve all f32 components.
    vector_table=pd.DataFrame({"patient":group,"vector":[np.asarray(v,dtype=np.float32).tobytes().hex() for v in c["vectors"]]})
    per_patient=vector_table.groupby("patient")["vector"].nunique()
    frequency={int(k):int(v) for k,v in per_patient.value_counts().items()}
    primary={r["n_unique_full_vectors_per_patient"]:r["n_patients"] for r in structure["within_patient_full_vector_frequency"]}
    check("independent within-patient vector variety",frequency,primary)
    n_patients=vector_table.drop_duplicates(["patient","vector"]).groupby("vector").size()
    n_images=vector_table.groupby("vector").size()
    vectors=set(n_patients[n_patients>=2].index)
    flag=vector_table["vector"].isin(vectors)
    cross=structure["cross_patient_repeated_vectors"]
    check("independent unique vectors",int(len(n_images)),cross["n_distinct_full_vectors"])
    check("independent shared vectors",len(vectors),cross["n_vectors_shared_across_patients"])
    check("independent rows with shared vectors",int(flag.sum()),cross["n_images_with_shared_vectors"])
    check("independent patients with shared vectors",int(vector_table.loc[flag,"patient"].nunique()),cross["n_patients_with_any_shared_vector"])
    check("independent patient-vector links",int(n_patients.sum()),cross["n_unique_patient_vector_links"])
    check("independent shared vector links",int(n_patients[n_patients>=2].sum()),cross["n_links_to_shared_vectors"])


def derm_structure(c, alignment, binding, positive_labels, oc):
    source={r["case_num"]:i for i,r in enumerate(c["rows"])}
    check("Derm unique case_num",len(source),len(c["rows"]))
    check("Derm binding row count",len(alignment),binding["included_rows"])
    check("Derm binding case membership",set(source),{r["case_num"] for r in alignment})
    check("Derm exclusions",binding["excluded_rows"],0)
    groups=defaultdict(list)
    for r in alignment:
        i=source[r["case_num"]]; raw=c["rows"][i]
        if any(raw[k]!=r[k] for k in ["derm","sex","location"]):
            raise ValueError("Derm original binding/source correspondence mismatch")
        if int(r["target"]) != int(raw["diagnosis"] in positive_labels):
            raise ValueError("Derm frozen target/source diagnosis mismatch")
        groups[r["cluster_case_num"]].append(i)
    check("Derm frozen row/label correspondence",len(alignment),len(c["rows"]))
    check("Derm frozen cluster count",len(groups),binding["clusters"])
    check("Derm same clinic/derm references",sum(r["clinic"]==r["derm"] for r in c["rows"]),binding["clinic_derm_same_reference_count"])
    size_counts=Counter(len(indices) for indices in groups.values())
    variety=Counter(len({c["vectors"][i] for i in indices}) for indices in groups.values())
    vector_counts=Counter(c["vectors"])
    repetition=Counter(vector_counts.values())
    return dict(overview=dict(n_rows=len(c["rows"]),n_clusters=len(groups),n_distinct_full_vectors=len(vector_counts),
            reliable_patient_grouping=False,statistical_unit="case_num / original confirmed same-image clusters; not verified patients",
            clinic_derm_same_reference_count=binding["clinic_derm_same_reference_count"]),
        cluster_size_frequency=[dict(n_rows_per_cluster=k,n_clusters=v,n_rows_in_bin=k*v) for k,v in sorted(size_counts.items())],
        within_cluster_full_vector_frequency=[dict(n_unique_full_vectors_per_cluster=k,n_clusters=v) for k,v in sorted(variety.items())],
        full_vector_multiplicity_frequency=[dict(n_rows_per_vector=k,n_distinct_vectors=v,n_rows_in_bin=k*v) for k,v in sorted(repetition.items())],
        metadata_sha256=MANIFEST[str(Path(c["source"]).resolve())]["sha256"],
        binding_summary_sha256=MANIFEST[str((oc/"COHORT_BINDING.json").resolve())]["sha256"],
        alignment_sha256=binding["alignment_sha256"],
        binding_source="OriginalCore COHORT_BINDING.json + private_predictions/COHORT_ALIGNMENT.csv (metadata only)",
        alignment_rows_verified=True,repeat_image_audit_performed=False,patient_identity_inferred=False)


def table_md(summaries):
    names=list(summaries)
    output=["# Cohort Clinical Characteristics", "", "Candidate table for manuscript integration; no manuscript was edited.", "",
        "| Characteristic | " + " | ".join(names) + " |", "|---|"+"---:|"*5]
    def row(label,values):
        output.append("| "+label+" | "+" | ".join(values)+" |")
    def number(v):
        return "NA" if v is None else f"{v:g}"
    row("Statistical unit",[s["unit"] for s in summaries.values()])
    row("Total rows, n",[f'{s["n"]:,}' for s in summaries.values()])
    row("Melanoma positive, n (%)",[f'{s["positive"]:,} ({s["percent_positive"]:.2f})' for s in summaries.values()])
    for key in ["image","lesion","patient","case"]:
        row("Distinct "+key+" identifiers, n",["NA (not supplied/reliably defined)" if s["ids"][key] is None else f'{s["ids"][key]:,}' for s in summaries.values()])
    row("Valid age, n",[str(s["age"]["n"]) for s in summaries.values()])
    row("Age, median [Q1, Q3], years",["NA (source column absent)" if s["age"]["n"]==0 else f'{number(s["age"]["median"])} [{number(s["age"]["q1"])}, {number(s["age"]["q3"])}]' for s in summaries.values()])
    row("Age, min-max, years",["NA (source column absent)" if s["age"]["n"]==0 else f'{number(s["age"]["min"])}-{number(s["age"]["max"])}' for s in summaries.values()])
    for field in ["age","sex","site"]:
        row(field.title()+" unavailable, n (%)",[f'{s["missing"][field]} ({pct(s["missing"][field],s["n"]):.2f})' for s in summaries.values()])
    for field in ["sex","site","diagnosis_method"]:
        output.extend(["", "## Complete Source "+field.replace("_"," ").title()+" Distribution", "",
            "| Source category | " + " | ".join(names) + " |", "|---|"+"---:|"*5])
        categories=sorted(set().union(*(set(s["categories"].get(field,{})) for s in summaries.values())))
        for category in categories:
            values=[]
            for s in summaries.values():
                if field not in s["categories"]:
                    values.append("NA (source column absent)")
                else:
                    count=s["categories"][field].get(category,0)
                    values.append(f"{count:,} ({pct(count,s['n']):.2f})")
            row(category or "[blank]",values)
    output.extend(["", "## Notes", "",
        "Counts and percentages use all image/case rows in each column; age quantiles use finite observed values only (linear method).",
        "Sex/site source categories are not harmonized in this table. In particular, explicit unknown and blank are separate observed missingness categories.",
        "A zero means an observed category count of zero; NA means the identifier/field is absent or cannot reliably support that identity.",
        "Derm7pt age is absent, not observed zero. Each derm reference is one case; clinic photos are not counted as additional cases.",
        "HAM lesion identifiers and Derm7pt case identifiers are not patient identifiers. SIIM patient-level summaries are separate in SIIM_PATIENT_STRUCTURE.csv.",
        "Non-melanoma is not synonymous with benign. SIIM unknown diagnosis does not override its original binary target.",
        "Percentages may not sum to exactly 100 after rounding. Full precision, missing-reason partitions and diagnosis distributions are provided in the CSVs."])
    return "\n".join(output)


def method_md(cfg,lines,siim,derm):
    sources="\n".join(f'- {s["name"]}: {s["filename"]}; age={s["age"]}, sex={s["sex"]}, site={s["site"]}, label={s["label"]}.' for s in cfg["cohorts"])
    return f"""# Cohort Summary Method

## Scope and sources
First A1 execution, metadata-only, local CPU. Five tables were read into a bounded in-memory cache and shared between A1 and B design summaries. SHA-256, size and modification time were checked before/after processing. Accepted Stage0 source registration was reused, not rehashed in full. R5.1 documents were not opened or edited.

{sources}

All source rows are retained. HAM split42 partitions are reported separately; their union is only an integrity check. Derm7pt uses all 1,011 original derm cases and the original case/same-image binding, not clinic photographs as additional cases. There is no new overlap or image audit. Labels are dx=mel (HAM), target=1 (SIIM), and the six positive diagnoses in OriginalCore ORIGINALCORE_PROTOCOL.json#labels (Derm7pt). Original diagnosis/method distributions are exported as aggregates. No labels or clinical values are inferred from images, notes or IDs.

## Missingness and denominators
Mutually exclusive priority: source column absent; stripped blank; case-insensitive NA token; explicit unknown token; numeric parse/nonfinite failure (age only); valid. NA tokens are {sorted(NA_TOKENS)}. Explicit unknown tokens are {sorted(UNKNOWN_TOKENS)}. This reproduces the accepted Stage0 rule in freeze_baseline.py:196-213. Whitespace is stripped for descriptive categories, with no category collapsing beyond that; exact source spellings remain in distributions.

Categorical percentages and unavailable proportions use all image/case rows. percent_of_valid is separate and only applies to valid categories. Missing-reason and source-category rows in COHORT_FIELD_DISTRIBUTIONS.csv are two alternative views, not additive. Source categories marked unknown are counted once in their source-category distribution and once in the independently labelled missing-reason view.

Finite ages only enter median/Q1/Q3 and range. Quantiles use numpy method=linear: index=(n-1)*q with linear interpolation. Age zero remains observed; no imputation, clipping, arbitrary upper-age exclusion or outlier deletion. Derm7pt age quantiles are NA, never computed from encoded zero. NA count fields distinguish undefined quantities from observed zero. Age-summary rows carry valid n and denominator, while value is the age statistic in years.

## Original clinical encoding
Original source: {cfg['original_b2_root']}/model_code/lesionmamba/data/datasets.py. Symbol/line map: {json.dumps(lines)}.
Only metadata constants and pure metadata functions were AST-extracted; torch tensor allocation was changed to an equivalent float32 numpy allocation. No project imports, torch, GPU, model, checkpoint or prediction files were used. Every metadata vector was independently checked against a second numpy implementation in memory. No per-case vectors are exported.

The 22 dimensions retain age/100, 3 sex and 15 site one-hots, then age_missing/sex_missing/site_missing. Exact vocabulary/order and E_NULL are included in CLINICAL_MATCHING_DESIGN_INPUTS.json. HAM follows the original pandas CSV handling and encoder defaults. SIIM uses metadata_mapping.json and METADATA_TOKEN_MAPPING.csv (strip/lower/collapse whitespace, torso to trunk). Derm7pt uses CLINICAL_MAPPING.json and core_data.py:21-30: unsupported head neck/buttocks are passed as nonempty unsupported strings, so the one-hot becomes unknown but site_missing remains zero. Missing age is passed as NaN, then age=0 and age_missing=1.

Raw truly_missing, nonempty_unsupported and available_supported form a partition; encoded_unknown and missing_flag_1 are overlapping encoder outcomes and are not additional partition categories. encoded_unknown is NA for scalar age. Unknown-category membership is not equivalent to true missingness. Category-level aggregates and any observed raw-vs-flag differences are explicit. No encoder has been repaired. The original age encoder does not explicitly reject infinity after float conversion; no nonfinite encoded ages occurred in these inputs. Future datasets must not silently inherit an assumption of equivalence.

## Patient and case structure
SIIM groups only by its complete source patient_id, with {siim['overview']['n_patients']} groups. Image-weighted and patient-weighted quantities are kept separate. Equal-size-block coverage is merely the proportion in group-size strata containing at least two patients; it is not a donor map, a chosen exchange policy, an exclusion rule or proof of value change.

For each patient and field, count distinct valid values (0,1,>1) crossed with all_missing / values_no_missing / values_plus_missing. Encoded-site validity uses site_missing=0; nonempty unsupported values mapped to unknown remain valid in this encoded representation. Collapsing categories in the encoder does not mean the original sites agree. Sites are never unified within patient.

Patient-level age/sex summaries use only patients with exactly one distinct valid value AND no missing rows for that field. Other patients stay in explicit consistency/missingness states. No first/mode/mean is used to assign a patient vector. Full-vector counts use exact float32 equality across all 22 dimensions including missing flags. Shared-vector proportions separately use distinct vectors, all image rows, all patients and unique patient-vector links as denominators. Patient counts across different vectors are not additive.

Derm7pt reuses OriginalCore COHORT_BINDING.json and its SHA-bound COHORT_ALIGNMENT.csv (case-binding metadata despite the private_predictions directory name, not model predictions). Only membership, source derm/sex/location correspondence and cluster labels are read. There are {derm['overview']['n_clusters']} existing clusters, not verified patients. No case_id is promoted to patient_id, and no raw image is reread. Row-level identities, vector values and clinical combinations remain private in memory.

## QC, privacy and reproducibility
Independent checks: csv.DictReader versus pandas; Counter versus pandas value_counts for all categories; numpy quantiles versus sorted-value interpolation; AST encoder versus independent numpy encoder; prior missing totals and SIIM mapping counts; cluster/size summations; source hashes before/after.
Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}. No RNG, statistical tests, model performance, bootstrap, donor mapping or inference is executed. a1_profile.py and a1_inputs.json reproduce these aggregate outputs in a new output folder with the same explicitly named local inputs. Exclusive creation prevents overwriting previous output files.

ZIP uses an explicit whitelist of aggregates/methods/code/QC only. CRC and member SHA-256 are checked and recorded in an external DELIVERY_RECEIPT.json; the ZIP does not contain its own final hash. Input raw metadata, manuscript files, weights, images, predictions, credentials and individual identifiers are not packaged.
"""


def draft_md(design):
    siim=design["SIIM"]["overview"]; derm=design["Derm7pt"]["overview"]
    model_table="\n".join(f'| {r["model"]} | {r["training_seed"]} | `{r["run_id"]}` | `{r["sha256"]}` |' for r in design["original_models"])
    pending="\n".join('- '+s+': NOT_FINAL_FROZEN.' for s in design["pending_protocol_parameters"])
    return f"""# Clinical Matching Protocol Draft

**Status: DRAFT_NOT_EXECUTED_NOT_FINAL_FROZEN**

## Given research question
For the same images and six fixed original checkpoints, quantify ranking/operating differences between image-matched available clinical records and a prospectively specified reassignment reference. This concerns dependence of these models on clinical-record correspondence; it does not by itself establish interaction superiority, clinical causality or improved original model scores. No performance was generated in A1.

## Fixed original model scope
| Family | Training seed | Original run ID | Historical checkpoint SHA-256 (not reopened) |
|---|---|---|---|
{model_table}

Original B2/B6 are not M0/FIM FULL, B2S, clean models or regional/TKEEP candidates. Historical identity JSONs are references; A1 did not validate weights or load checkpoints.

## Given inputs and scoring
- E_MATCHED: actual available complete clinical vector matched to each image.
- E_NULL: original structured missing vector, not an all-zero vector.
- E_SHUFFLED: jointly reassigned complete clinical records under a protocol still to be finalized.
- Primary score: original seven-class softmax MEL (class index 4); primary metric AP. The dedicated melanoma head and additional outcomes remain separately labelled in the eventual protocol. No AP or other performance is computed here.
- SIIM and Derm7pt remain separate cohorts. The score unit is image/case; exchange units and uncertainty units are not interchangeable.
- Complete clinical records include age, sex, site and all missingness flags. Do not shuffle columns independently or force all lesions of a patient to share a site.

## Observed structure for design
SIIM: {siim['n_images']} images, {siim['n_patients']} explicit patients, {siim['n_size_strata']} group-size strata. Strata with >=2 patients cover {siim['candidate_equal_size_exchange_patients']} patients and {siim['candidate_equal_size_exchange_images']} images ({siim['candidate_equal_size_exchange_image_percent']:.4f}% of rows). Singleton size strata: {siim['singleton_size_strata']}, containing {siim['singleton_stratum_images']} images. This is candidate coverage only, not an exchange decision. Exact consistency and vector repetition summaries are in CLINICAL_MATCHING_DESIGN_INPUTS.json and SIIM_PATIENT_STRUCTURE.csv.

Derm7pt: {derm['n_rows']} cases, {derm['n_clusters']} original case/same-image clusters, {derm['n_distinct_full_vectors']} distinct observed full clinical vectors. No reliable patient linkage is established by these fields.

A changed donor can yield an unchanged vector. Unequal blocks cannot be duplicated/cropped without explicitly changing distribution-preservation claims. Patient-level uncertainty grouping does not determine a valid reassignment algorithm.

## Pending ChatGPT finalization
{pending}

No K, donor assignment, RNG, interval algorithm or significance family is fixed by this draft. Historical bootstrap seeds/counts are not imported as the new protocol. No trial mappings or model results were generated to select a design. A1 stops here.
"""


def scope_md(task,manuscripts):
    plan=task/"PROJECT_MASTER_PLAN_OncoDerm_Frontiers_v1_2.md"
    current=digest(CACHE[str(plan.resolve())])
    documents="\n".join(f'- {r["role"]} ({r["format"]}): `{r["absolute_path"]}`; accepted SHA-256 `{r["sha256"]}` (inherited, not rehashed).' for r in manuscripts if r["version"]=="R5.1")
    return f"""# Scope Decision v1.2

Current-state authority: `{task / '00_START_HERE.md'}`. This is the FIRST A1 launch; no earlier A1 result is presumed.

Current scope authority: `{plan}`; SHA-256 `{current}`. The accepted Stage0 directory is delivery_20261019_120756. Its 526-entry index is reused; no all-source audit or manuscript reread was performed.

Original OncoDerm B0-B6 remain the manuscript core. Approved FULL/ZERO, regional readout and P5/P6/TKEEP retain their existing R5.1 roles, without rerunning, deleting or reclassifying them as original B2/B6. M0 is new FIM FULL, not original B6; B2S is not original B2.

The separate revision_v2/A24/T1/T2/Q/Existing_Research direction and canceled C1 reuse remain excluded. No experimental contents from those branches were accessed. v1.0/v1.1 and historical reports remain unchanged; the resolved scope question is not reopened.

This task reads five registered metadata tables and the bounded original encoder/mapping/binding dependencies only. It computes descriptive fields and design aggregates, not model outcomes. Route B inference and reassignment are NOT_STARTED.

## Accepted R5.1 identities (index reuse only)
{documents}
"""


def report_md(summaries,coverage,siim,derm):
    overview=siim["overview"]
    rows_text="\n".join(f'| {name} | {s["n"]} | {s["positive"]} ({s["percent_positive"]:.2f}%) | {s["age"]["n"]} | '+
        ("NA: source column absent" if not s["age"]["n"] else f'{s["age"]["median"]:g} [{s["age"]["q1"]:g}, {s["age"]["q3"]:g}]')+
        f' | {s["missing"]["age"]}/{s["missing"]["sex"]}/{s["missing"]["site"]} |' for name,s in summaries.items())
    enc="\n".join(f'- {r["cohort"]} {r["field"]}: supported {r["available_supported"]}; nonempty unsupported {r["nonempty_unsupported"]}; raw missing {r["truly_missing"]}; flag=1 {r["missing_flag_1"]}.' for r in coverage if r["field"]=="site")
    cons=siim["within_patient_consistency"]
    multiline="\n".join(f'- {f}: '+str(sum(r["n_patients"] for r in states if r["unique_valid_values"]==">1"))+" patients have >1 observed value/category; "+str(sum(r["n_patients"] for r in states if r["missing_pattern"]=="values_plus_missing"))+" have observed/missing mixed rows." for f,states in cons.items())
    return f"""# A1_PROFILE_REPORT

## Execution status
**COMPLETE: A1 formal profiles and B_DESIGN_INPUTS. B inference NOT_STARTED; new training 0.**
First A1 launch under 00_START_HERE and master v1.2. Local CPU only. No manuscript edits, environment changes, weights, images, model predictions or performance computations. No bootstrap, donor mappings, threshold selection or calibration. Outputs are new files under `{OUT}`.

## Five-group descriptive summary
| Group | Rows | Melanoma n (%) | Valid age n | Age median [Q1, Q3], years | Unavailable age/sex/site n |
|---|---:|---:|---:|---|---|
{rows_text}

Full source sex/site categories and missing-reason distributions are included; no top-k truncation. HAM labels are dx=mel; SIIM uses target, not diagnosis unknown; Derm7pt uses the frozen six positive diagnoses. Non-melanoma is not equivalent to benign. Quantiles use finite valid ages and linear interpolation; encoded missing age zero is not an observed age.

## Actual original encoding coverage
{enc}
Observed raw-missing versus encoder-flag disagreements: {sum(r['raw_vs_flag_disagreement'] for r in coverage)} across field/row comparisons. Unknown one-hot and missing flag are reported separately. SIIM torso to trunk and Derm7pt unsupported head neck/buttocks rules are unchanged. All 44,152 in-memory encoded records were cross-checked by two metadata-only implementations; no tensors or case vectors were exported.

## SIIM patient structure
{overview['n_patients']} explicit patient groups; images per patient: median {overview['images_per_patient']['median']:g}, Q1 {overview['images_per_patient']['q1']:g}, Q3 {overview['images_per_patient']['q3']:g}, range {overview['images_per_patient']['min']:g}-{overview['images_per_patient']['max']:g}.

{multiline}

Patient age summaries use {siim['uniquely_determined_patient_fields']['age']['eligible_patients']} fully observed, single-valued patients; sex summaries use {siim['uniquely_determined_patient_fields']['sex']['eligible_patients']}. Mixed missingness stays explicit. No lesion sites are unified and no first/mode/mean is used to create a patient vector.

Equal-size strata containing at least two patients cover {overview['candidate_equal_size_exchange_patients']} patients and {overview['candidate_equal_size_exchange_images']} images ({overview['candidate_equal_size_exchange_image_percent']:.4f}%). {overview['singleton_size_strata']} singleton-size strata account for {overview['singleton_stratum_images']} images. These are structural candidate-coverage facts only; no cases were excluded or mapped to donors.

Full-vector repetition: {siim['cross_patient_repeated_vectors']['n_distinct_full_vectors']} distinct vectors; {siim['cross_patient_repeated_vectors']['n_vectors_shared_across_patients']} occur in multiple patients. Exact frequency/denominator summaries are supplied for later design.

## Derm7pt structure
OriginalCore binding is reused: {derm['overview']['n_rows']} cases, {derm['overview']['n_clusters']} case/same-image clusters, {derm['overview']['n_distinct_full_vectors']} full-vector types. Nine clinic=derm references remain one case each. No age field or verified patient identity is inferred. No repeated image audit was run.

## Delivery and QC
Formal table: COHORT_CLINICAL_CHARACTERISTICS.csv and its readable five-column TABLE.md; full categories: COHORT_FIELD_DISTRIBUTIONS.csv; encoding: CLINICAL_ENCODING_COVERAGE.csv and CATEGORY_DETAIL.csv; definitions: COHORT_SUMMARY_METHOD.md.
Route B inputs: SIIM_PATIENT_STRUCTURE.csv, CLINICAL_MATCHING_DESIGN_INPUTS.json and CLINICAL_MATCHING_PROTOCOL_DRAFT.md. Draft status is DRAFT_NOT_EXECUTED_NOT_FINAL_FROZEN. ChatGPT still determines exchange algorithm, K/RNG, unequal-block policy, donor/fixed-point rules, uncertainty and comparison family.

Limited QC verifies source identities/unchanged bytes, rows/labels, denominators, missing partitions, accepted counts, independent quantiles/categories/encoding, and group totals. SOURCE_MANIFEST.csv covers only inputs actually read, not a new freeze of all 526 sources. No manuscript/source was modified. Package CRC and member fingerprints are verified in the external DELIVERY_RECEIPT.json.

## Actual blockers and stop
Asset/identity/computation blockers: {len(BLOCKERS)}. Pending Route B design decisions are expected next-stage work, not A1 failures. ZIP includes reports, final progress, aggregates, methods and minimal reproducible scripts only. No raw metadata, per-case identifiers/clinical rows, vectors, predictions, images, weights or credentials. Stop here for ChatGPT review; do not start B or manuscript editing.
"""


def package_outputs(protected_ids):
    names=["a1_profile.py","a1_inputs.json","COHORT_CLINICAL_CHARACTERISTICS.csv","COHORT_CLINICAL_CHARACTERISTICS_TABLE.md",
        "COHORT_FIELD_DISTRIBUTIONS.csv","CLINICAL_ENCODING_COVERAGE.csv","CLINICAL_ENCODING_CATEGORY_DETAIL.csv",
        "COHORT_SUMMARY_METHOD.md","SIIM_PATIENT_STRUCTURE.csv","CLINICAL_MATCHING_DESIGN_INPUTS.json",
        "CLINICAL_MATCHING_PROTOCOL_DRAFT.md","SCOPE_DECISION_v1_2.md","SOURCE_MANIFEST.csv",
        "A1_PROFILE_REPORT.md","A1_PROFILE_PROGRESS.json","A1_PROFILE_QC.json"]
    # Source IDs are not placed in the products; this additionally checks exact long identifiers.
    contents={name:(OUT/name).read_bytes() for name in names}
    import re
    id_pattern=re.compile("|".join(re.escape(x) for x in sorted(protected_ids,key=len,reverse=True))) if protected_ids else None
    if id_pattern:
        for name,content in contents.items():
            if id_pattern.search(content.decode("utf-8-sig")):
                raise ValueError("Identifier found in delivery product: "+name)
    path=OUT/"R5_2_CLINICAL_PROFILE_RESULTS.zip"
    with zipfile.ZipFile(path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for name in names:
            z.write(OUT/name,name)
    members=[]
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or sorted(z.namelist())!=sorted(names):
            raise ValueError("ZIP CRC/member whitelist failure")
        for name in names:
            actual=digest(z.read(name)); expected=digest(contents[name])
            if actual!=expected:
                raise ValueError("ZIP member SHA failure: "+name)
            info=z.getinfo(name)
            members.append(dict(name=name,bytes=info.file_size,sha256=actual,crc32=f"{info.CRC:08x}",matches_local=True))
    write_json("DELIVERY_RECEIPT.json",dict(status="ZIP_VERIFIED",verified_at=datetime.now(ZONE).isoformat(timespec="seconds"),
        zip_path=str(path),zip_bytes=path.stat().st_size,zip_sha256=digest(path.read_bytes()),crc_pass=True,
        exact_whitelist_pass=True,all_members_sha_match=True,members=members,
        long_identifier_leak_scan_pass=True,raw_row_export=False,receipt_external_to_zip=True))


if __name__ == "__main__":
    main()
