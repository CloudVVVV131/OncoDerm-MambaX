"""Synthetic unit tests only; no model/dataset access and no efficacy validation."""
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import clinical_matching_reference as ref


def main():
    checks = []
    def check(name, condition):
        passed = bool(condition)
        checks.append({"check": name, "passed": passed})
        if not passed:
            raise AssertionError(name)
    rng = np.random.default_rng(73120)
    # includes one singleton-size stratum and several exchangeable strata
    sizes = [2, 2, 3, 3, 3, 5]
    blocks = []
    off = 0
    for s in sizes:
        blocks.append(np.arange(off, off+s))
        off += s
    n = off
    strata = ref.make_strata(blocks, n)
    group = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    size_per_row = np.concatenate([np.full(s, s) for s in sizes])
    frozen = np.concatenate([b for b in blocks if sizes.count(len(b)) == 1])
    active = np.setdiff1d(np.arange(n), frozen)
    types = rng.integers(0, 6, size=n)
    for k in range(20):
        p = ref.donor_slots(strata, n, ref.stream(1, 1, repeat=k))
        check(f"permutation bijection {k}", np.array_equal(np.sort(p), np.arange(n)))
        check(f"singleton native {k}", np.array_equal(p[frozen], frozen))
        check(f"cross-patient donors {k}", np.all(group[active] != group[p[active]]))
        check(f"equal-size donors {k}", np.array_equal(size_per_row, size_per_row[p]))
        check(f"complete vector multiplicities {k}", np.array_equal(np.bincount(types, minlength=6), np.bincount(types[p], minlength=6)))
        for b in blocks:
            check(f"one donor block {k}/{b[0]}", len(np.unique(group[p[b]])) == 1)
        check(f"deterministic map {k}", np.array_equal(p, ref.donor_slots(strata, n, ref.stream(1, 1, repeat=k))))
    for i in range(25):
        y = rng.integers(0, 2, 100)
        s = rng.random(100) if i % 2 else rng.integers(0, 7, 100).astype(float)
        got = ref.ap_auc(y, s)
        expected = [average_precision_score(y, s), roc_auc_score(y, s)]
        check(f"sklearn AP/AUC and ties {i}", np.allclose(got, expected, atol=1e-14, rtol=0))
    y = np.array([0, 1, 0, 1])
    check("constant scores", np.allclose(ref.ap_auc(y, np.zeros(4)), [0.5, 0.5]))
    # A synthetic score table. Score differences have no biological interpretation.
    y = rng.integers(0, 2, n)
    table = rng.uniform(0.01, .99, (6, n, 7)).astype(np.float32)
    point = ref.observed_reassignments(table, types, 6, y, strata, 1, repeats=5)
    check("point shape", point["shuffled_per_repeat_per_run"].shape == (5, 6, 2))
    check("difference identity", np.allclose(
        point["matched_minus_shuffled_per_run"] + point["shuffled_minus_null_per_run"],
        point["matched_per_run"] - point["null_per_run"], atol=1e-14, rtol=0))
    out = ref.bootstrap_one(table, types, 6, y, strata, 1, 0, repeats=5)
    out2 = ref.bootstrap_one(table, types, 6, y, strata, 1, 0, repeats=5)
    check("bootstrap restart deterministic", np.array_equal(out["matched_minus_shuffled"], out2["matched_minus_shuffled"]))
    # Independent explicit construction confirms resampling reaches donors as well.
    rows = ref.resample_rows(strata, n, ref.stream(1, 2, draw=0, attempt=int(out["rejected_one_class_samples"])))
    check("resampling preserves row count", len(rows) == n)
    check("resampling fixed singletons", np.array_equal(rows[frozen], frozen))
    for a in strata:
        for b in a:
            check(f"whole block resampling {b[0]}", len(np.unique(group[rows[b]])) == 1)
    mm = np.zeros((6,2)); ss = np.zeros((6,2))
    for r in range(6):
        yy=y[rows]
        v=table[r, rows, types[rows]]
        mm[r]=[average_precision_score(yy,v), roc_auc_score(yy,v)]
        for k in range(5):
            p=ref.donor_slots(strata,n,ref.stream(1,3,draw=0,repeat=k,attempt=int(out["rejected_one_class_samples"])))
            v=table[r, rows, types[rows[p]]]
            ss[r]+=np.array([average_precision_score(yy,v),roc_auc_score(yy,v)])/5
    check("nested bootstrap versus explicit sklearn", np.allclose(out["matched_minus_shuffled"], ref.family_mean(mm-ss), atol=1e-14,rtol=0))
    # Distinguish mean metric from implicit prediction ensembling.
    preds=[table[0,np.arange(n),types[ref.donor_slots(strata,n,ref.stream(1,1,repeat=k))]] for k in range(5)]
    each=np.mean([average_precision_score(y,p) for p in preds])
    check("metric-then-average definition", abs(each-point["shuffled_per_repeat_per_run"][:,0,0].mean())<1e-14)
    result={"status":"PASS","checks":checks,"n_checks":len(checks),
            "data":"Synthetic arrays only; no project images, labels, metadata rows, models or predictions used",
            "scope":"Numerical/indexing/reproducibility tests; NOT empirical coverage validation, clinical findings or model performance",
            "numpy":np.__version__}
    target=Path(__file__).with_name('SYNTHETIC_TEST_REPORT.json')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"PASS: {len(checks)} synthetic checks")

if __name__ == '__main__':
    main()
