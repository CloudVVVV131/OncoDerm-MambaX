"""Protocol-defined point and nested bootstrap estimates from actual fixed score grids."""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[k]="1"
from b_common import *
import sys
import time
import argparse
import concurrent.futures
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
import clinical_matching_reference as ref

HEADS=["prob_mel","prob_mel_endpoint"]
METRICS=["AP","ROC_AUC","Brier","binary_NLL","ECE15"]
SEEDS=[42,2024,3407]
FAMILIES=["ORIG_B2","ORIG_B6"]

def bind(cohort):
    p=PRIVATE/cohort
    b=read(p/"binding.json")
    z=np.load(p/"strata.npz",allow_pickle=False)
    st=tuple(z[str(i)] for i in range(len(z.files)))
    tags=read(OUT/"B_RUN_PROTOCOL.json")["models"]
    for tag in tags:
        receipt=read(OUT/f"{tag}_{cohort}_COMPLETE.json")
        assert receipt["status"]=="COMPLETE"
        assert sha(p/f"{tag}_score_grid.npy")==receipt["grid_sha256"]
    tables=[np.load(p/f"{tag}_score_grid.npy",mmap_mode="r",allow_pickle=False) for tag in tags]
    return dict(cohort=cohort,p=p,b=b,strata=st,tables=tables,
                y=np.load(p/"labels.npy"),types=np.load(p/"types.npy"),
                groups=np.load(p/"group_ids.npy"),code=ref.COHORT_CODES[cohort])

def fast_ap_auc(y,score):
    # Validity checks are once at grid admission. Tie endpoints and float64 sums
    # are identical to the supplied reference, without repeated full-array guards.
    order=np.argsort(-score,kind="stable")
    s=score[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(y)-1]
    tp=np.cumsum(y[order],dtype=np.float64)[ends]
    fp=ends+1-tp
    pt=np.r_[0.,tp[:-1]]
    pf=np.r_[0.,fp[:-1]]
    positive=float(np.sum(y,dtype=np.float64))
    negative=len(y)-positive
    return np.array([np.sum((tp-pt)*tp/(ends+1))/positive,
                     np.sum((fp-pf)*(tp+pt))/(2*positive*negative)])

def binary(y,s):
    p=np.asarray(s,dtype=np.float64)
    ap,auc=fast_ap_auc(y,p)
    clip=np.clip(p,1e-7,1-1e-7)
    bins=np.minimum((p*15).astype(int),14)
    ece=0.
    for k in range(15):
        take=bins==k
        if take.any():
            ece+=take.mean()*abs(p[take].mean()-y[take].mean())
    return dict(AP=float(ap),ROC_AUC=float(auc),Brier=float(np.mean((p-y)**2)),
                binary_NLL=float(np.mean(-y*np.log(clip)-(1-y)*np.log1p(-clip))),ECE15=float(ece))

def thresholds_for(ident):
    values=rows(OUT/"DEVELOPMENT_THRESHOLDS.csv")
    a=[r for r in values if "ORIG_"+r["model"]==ident["model"] and int(r["training_seed"])==ident["training_seed"]]
    assert len(a)==4
    return a

def points(cohort):
    if (OUT/f"{cohort}_POINTS_COMPLETE.json").exists():
        return
    a=bind(cohort)
    y,types,tables,b=a["y"],a["types"],a["tables"],a["b"]
    n=len(y)
    ix=np.arange(n)
    identities=read(OUT/"FIXED_MODEL_IDENTITIES.json")
    metrics=[];counts=[];diagnostics=[]
    fixed=np.concatenate([v.ravel() for v in a["strata"] if len(v)==1]) if b["singleton_images"] else np.array([],int)
    active=np.ones(n,bool);active[fixed]=False
    vectors=np.load(a["p"]/"vectors.npy",allow_pickle=False)
    def score(condition,repeat,donor_types):
        for mi,e in enumerate(identities):
            scores=tables[mi][ix,donor_types]
            for hi,head in enumerate(HEADS):
                key=dict(cohort=cohort,family=e["model"],training_seed=e["training_seed"],head=head,
                         condition=condition,repeat=repeat,images=n,positive=int(y.sum()),prevalence=float(y.mean()))
                p=scores[:,hi].astype(np.float64)
                met=binary(y,p)
                metrics.append(dict(**key,**met))
                for r in thresholds_for(e):
                    if r["score"]!=head:continue
                    t=float(r["threshold"])
                    pred=p>=t
                    tp=int(np.sum(pred&(y==1)));fn=int(np.sum(~pred&(y==1)))
                    fp=int(np.sum(pred&(y==0)));tn=int(np.sum(~pred&(y==0)))
                    counts.append(dict(**key,target_sensitivity=r["target_sensitivity"],
                        threshold_original_text=r["threshold"],TP=tp,FN=fn,FP=fp,TN=tn,
                        sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
                        PPV=tp/(tp+fp) if tp+fp else None,NPV=tn/(tn+fn) if tn+fn else None,
                        undefined_reason=";".join(x for x in [
                            "PPV:no_predicted_positive" if not tp+fp else "",
                            "NPV:no_predicted_negative" if not tn+fn else ""] if x),
                        threshold_source="original B2 HAM_val_NATIVE; unchanged"))
    score("E_MATCHED",-1,types)
    score("E_NULL",-1,np.full(n,b["null_type"],dtype=int))
    for k in range(50):
        if expired():raise TimeoutError("Window reached during point estimates")
        donor=ref.donor_slots(a["strata"],n,ref.stream(a["code"],1,repeat=k))
        assert np.array_equal(np.sort(donor),ix)
        assert np.array_equal(donor[fixed],fixed)
        assert np.all(a["groups"][active]!=a["groups"][donor[active]])
        assert np.array_equal(np.bincount(types),np.bincount(types[donor]))
        for st in a["strata"]:
            assert np.all(np.ptp(a["groups"][donor[st]],axis=1)==0)
        native=vectors[types];other=vectors[types[donor]]
        diag=dict(cohort=cohort,repeat=k,images=n,groups=b["groups"],
                  donor_changed_fraction=float(np.mean(a["groups"]!=a["groups"][donor])),
                  vector_changed_fraction=float(np.mean(types!=types[donor])),
                  fixed_singleton_groups=b["singleton_size_groups"],fixed_singleton_images=len(fixed),
                  permutation_bijective=True,complete_record_frequencies_preserved=True)
        for field,sl in [("age",slice(0,1)),("sex",slice(1,4)),("site",slice(4,19)),
                         ("age_missing",slice(19,20)),("sex_missing",slice(20,21)),("site_missing",slice(21,22))]:
            diag[field+"_changed_fraction"]=float(np.any(native[:,sl]!=other[:,sl],axis=1).mean())
        diagnostics.append(diag)
        score("E_BLOCK_SHUFFLED",k,types[donor])
    csvout(OUT/f"{cohort}_ALL_METRICS.csv",metrics)
    csvout(OUT/f"{cohort}_FROZEN_THRESHOLD_RESULTS.csv",counts)
    csvout(OUT/f"{cohort}_REASSIGNMENT_DIAGNOSTICS.csv",diagnostics)
    write(OUT/f"{cohort}_POINTS_COMPLETE.json",dict(status="COMPLETE",K=50,metric_rows=len(metrics),threshold_rows=len(counts),completed_at=now()))
    log(f"{cohort} 50 original-cohort mappings and all descriptive endpoints COMPLETE")

WORK=None
def init_worker(cohort):
    global WORK
    WORK=bind(cohort)

def boot_one(draw):
    a=WORK
    if expired():return None
    t=time.monotonic()
    y,types,tables=a["y"],a["types"],a["tables"]
    n=len(y)
    for attempt in range(10000):
        rr=ref.resample_rows(a["strata"],n,ref.stream(a["code"],2,draw=draw,attempt=attempt))
        yy=y[rr]
        if 0<yy.sum()<n:break
    else:raise RuntimeError("No valid two-class bootstrap sample")
    matched=np.stack([fast_ap_auc(yy,g[rr,types[rr],0]) for g in tables])
    null=np.stack([fast_ap_auc(yy,g[rr,a["b"]["null_type"],0]) for g in tables])
    shuffled=np.zeros((6,2),np.float64)
    same_original_count=0
    fixed=a["b"]["singleton_images"]
    for k in range(50):
        if expired():return None
        donor_slots=ref.donor_slots(a["strata"],n,ref.stream(a["code"],3,draw=draw,repeat=k,attempt=attempt))
        donor=rr[donor_slots]
        shuffled+=np.stack([fast_ap_auc(yy,g[rr,types[donor],0]) for g in tables])/50
        same_original_count+=int(np.sum(a["groups"][rr]==a["groups"][donor]))-fixed
    value=np.stack([ref.family_mean(matched-shuffled),ref.family_mean(shuffled-null)])
    return dict(draw=draw,values=value.tolist(),attempt=attempt,seconds=time.monotonic()-t,
                duplicate_original_donor_fraction_exchangeable=same_original_count/(50*(n-fixed)))

def numerical_tests():
    rng=np.random.default_rng(719)
    errs=[]
    for i in range(30):
        y=rng.integers(0,2,300)
        s=rng.random(300).astype(np.float32) if i%2 else rng.integers(0,8,300).astype(np.float32)
        expected=np.array([average_precision_score(y,s),roc_auc_score(y,s)])
        got=fast_ap_auc(y,s)
        errs.append(float(np.max(np.abs(got-expected))))
        assert np.allclose(got,expected,atol=1e-14,rtol=0)
    # Execute the optimized bootstrap wrapper against the supplied reference on synthetic data.
    global WORK
    groups=[np.arange(0,2),np.arange(2,4),np.arange(4,7),np.arange(7,10),np.arange(10,15)]
    n=15
    st=ref.make_strata(groups,n)
    table=rng.uniform(.01,.99,(6,n,5,2)).astype(np.float32)
    types=rng.integers(0,4,n)
    y=np.array([0,1,0,0,1,1,0,0,1,0,1,1,0,0,1])
    ids=np.concatenate([np.full(len(g),i) for i,g in enumerate(groups)])
    WORK=dict(y=y,types=types,tables=list(table),strata=st,code=1,
              b=dict(null_type=4,singleton_images=5),groups=ids)
    x=boot_one(0)
    z=ref.bootstrap_one(table[:,:,:,0],types,4,y,st,1,0)
    expected=np.stack([z["matched_minus_shuffled"],z["shuffled_minus_null"]])
    assert np.array_equal(np.asarray(x["values"]),expected)
    WORK=None
    write(OUT/"OPTIMIZED_KERNEL_QC.json",dict(status="PASS",sklearn_max_error=max(errs),
          nested_reference_exact_match=True,K_tested=50,synthetic_only=True))

def bootstrap(cohort):
    p=PRIVATE/cohort/"bootstrap"
    p.mkdir(exist_ok=True)
    completed={int(f.stem.split("_")[1]) for f in p.glob("draw_*.json")}
    if len(completed)==2000:return
    start=time.monotonic()
    done=len(completed)
    numerical_tests()
    iterator=iter(i for i in range(2000) if i not in completed)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init_worker,initargs=(cohort,)) as pool:
        pending={}
        def submit():
            if expired():return
            try:i=next(iterator)
            except StopIteration:return
            pending[pool.submit(boot_one,i)]=i
        submit();submit()
        while pending:
            finished,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                i=pending.pop(future)
                result=future.result()
                if result is not None:
                    write(p/f"draw_{i:04d}.json",result)
                    done+=1
                    if done in [5,10] or done%100==0 or done==2000:
                        elapsed=time.monotonic()-start
                        rate=(done-len(completed))/elapsed
                        eta=(2000-done)/rate if rate else None
                        write(OUT/f"{cohort}_BOOTSTRAP_PROGRESS.json",dict(status="COMPLETE" if done==2000 else "RUNNING",
                            completed_valid=done,required_valid=2000,inner_K=50,workers=2,
                            elapsed_seconds=elapsed,estimated_remaining_seconds=eta,updated_at=now()))
                        log(f"{cohort} nested bootstrap {done}/2000; inner K=50; ETA {eta/60:.1f} min")
                submit()
    write(OUT/f"{cohort}_BOOTSTRAP_PROGRESS.json",dict(status="COMPLETE" if done==2000 else "PARTIAL",
         completed_valid=done,required_valid=2000,inner_K=50,updated_at=now()))

def combine():
    all_metrics=[];all_counts=[];all_diag=[];intervals=[];bootdiag={}
    contrasts=[];summary=[];perrun=[]
    for cohort in ["SIIM","Derm7pt"]:
        if not (OUT/f"{cohort}_POINTS_COMPLETE.json").exists():continue
        frame=pd.read_csv(OUT/f"{cohort}_ALL_METRICS.csv")
        all_metrics.extend(frame.to_dict("records"))
        all_counts.extend(rows(OUT/f"{cohort}_FROZEN_THRESHOLD_RESULTS.csv"))
        all_diag.extend(rows(OUT/f"{cohort}_REASSIGNMENT_DIAGNOSTICS.csv"))
        for (family,head),g in frame.groupby(["family","head"],sort=False):
            for metric in METRICS:
                mat=g[g.condition=="E_MATCHED"].sort_values("training_seed")
                nul=g[g.condition=="E_NULL"].sort_values("training_seed")
                sh=g[g.condition=="E_BLOCK_SHUFFLED"].pivot(index="repeat",columns="training_seed",values=metric).sort_index(axis=1)
                assert sh.shape==(50,3)
                mv=mat[metric].to_numpy();nv=nul[metric].to_numpy();sv=sh.mean().to_numpy()
                assert np.allclose(mv-sv+sv-nv,mv-nv,atol=1e-14,rtol=0)
                for cond,values in [("E_MATCHED",mv),("E_NULL",nv),("E_BLOCK_SHUFFLED",sv)]:
                    summary.append(dict(cohort=cohort,family=family,head=head,metric=metric,condition=cond,
                                        mean=float(values.mean()),sample_SD=float(values.std(ddof=1)),training_runs=3,K=50))
                for contrast,delta,repdelta in [
                    ("matched_minus_shuffled",mv-sv,mv.mean()-sh.mean(axis=1).to_numpy()),
                    ("shuffled_minus_null",sv-nv,sh.mean(axis=1).to_numpy()-nv.mean()),
                    ("matched_minus_null",mv-nv,None)]:
                    contrasts.append(dict(cohort=cohort,family=family,head=head,metric=metric,contrast=contrast,
                        mean=float(delta.mean()),training_seed_sample_SD=float(delta.std(ddof=1)),
                        mapping_SD=float(repdelta.std(ddof=1)) if repdelta is not None else None,
                        mapping_MCSE=float(repdelta.std(ddof=1)/np.sqrt(50)) if repdelta is not None else None,
                        K=50,training_runs=3))
                    for seed,delta_s in zip(sorted(SEEDS),delta):
                        perrun.append(dict(cohort=cohort,family=family,head=head,metric=metric,contrast=contrast,
                                          training_seed=seed,value=float(delta_s),K=50))
        draws=[read(f) for f in sorted((PRIVATE/cohort/"bootstrap").glob("draw_*.json"))]
        bootdiag[cohort]=dict(valid_completed=len(draws),required=2000,inner_K=50,
            single_class_rejections=sum(r["attempt"] for r in draws),
            duplicate_original_donor_fraction_exchangeable_mean=float(np.mean([r["duplicate_original_donor_fraction_exchangeable"] for r in draws])) if draws else None,
            unit="patient" if cohort=="SIIM" else "case/same-image cluster, not verified patient",
            conditional_on="six fixed models, observed block-size strata and singleton-size blocks",
            bootstrap_copy_slots_are_distinct=True,donor_pool_rebuilt_every_draw=True,
            no_old_matched_minus_null_interval_replaced=True)
        if len(draws)==2000:
            assert [r["draw"] for r in draws]==list(range(2000))
            a=np.asarray([r["values"] for r in draws])
            for ci,contrast in enumerate(["matched_minus_shuffled","shuffled_minus_null"]):
                for fi,family in enumerate(FAMILIES):
                    for mi,metric in enumerate(["AP","ROC_AUC"]):
                        levels=[.95,.975] if cohort=="SIIM" and contrast=="matched_minus_shuffled" and metric=="AP" else [.95]
                        for level in levels:
                            lo,hi=np.quantile(a[:,ci,fi,mi],[(1-level)/2,1-(1-level)/2],method="linear")
                            intervals.append(dict(cohort=cohort,family=family,head=HEADS[0],metric=metric,contrast=contrast,
                                confidence_level=level,lower=float(lo),upper=float(hi),B=2000,inner_K=50,
                                role="primary" if cohort=="SIIM" and contrast=="matched_minus_shuffled" and metric=="AP" else "supporting",
                                method="nested stratified whole-block percentile bootstrap; linear quantile",
                                multiplicity="approximate Bonferroni reference for two primary AP comparisons" if level==.975 else "nominal"))
    if all_metrics:
        csvout(OUT/"PER_RUN_NATIVE_NULL_METRICS.csv",[r for r in all_metrics if r["condition"]!="E_BLOCK_SHUFFLED"])
        csvout(OUT/"REASSIGNMENT_METRICS.csv",[r for r in all_metrics if r["condition"]=="E_BLOCK_SHUFFLED"])
        csvout(OUT/"THREE_SEED_SUMMARY.csv",summary)
        csvout(OUT/"PER_RUN_CONTRASTS.csv",perrun)
        csvout(OUT/"MATCHING_CONTRASTS.csv",contrasts)
        csvout(OUT/"FROZEN_THRESHOLD_RESULTS.csv",all_counts)
        csvout(OUT/"REASSIGNMENT_DIAGNOSTICS.csv",all_diag)
        csvout(OUT/"MATCHING_INTERVALS.csv",intervals)
        write(OUT/"BOOTSTRAP_DIAGNOSTICS.json",bootdiag)
        f=pd.DataFrame(all_counts)
        valuecols=["TP","FN","FP","TN","sensitivity","specificity","PPV","NPV"]
        for c in valuecols:f[c]=pd.to_numeric(f[c],errors="coerce")
        # First average mappings within seed; then summarize three fixed runs.
        keys=["cohort","family","head","condition","target_sensitivity"]
        means=f.groupby(keys+["training_seed"],sort=False)[valuecols].agg(lambda a:a.mean() if a.notna().all() else np.nan).reset_index()
        agg=[]
        for key,g in means.groupby(keys,sort=False):
            assert len(g)==3
            for c in valuecols:
                v=g[c].to_numpy()
                agg.append(dict(**dict(zip(keys,key)),metric=c,mean=float(v.mean()) if np.isfinite(v).all() else None,
                    sample_SD=float(v.std(ddof=1)) if np.isfinite(v).all() else None,
                    defined_runs=int(np.isfinite(v).sum()),undefined_reason="" if np.isfinite(v).all() else "at_least_one_run_or_repeat_undefined",
                    K=50,training_runs=3))
        csvout(OUT/"FROZEN_THRESHOLD_SUMMARY.csv",agg)
    from b_verify import verify
    verify()
    return bootdiag

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["points","bootstrap","combine","test"])
    p.add_argument("--cohort",choices=["SIIM","Derm7pt"])
    args=p.parse_args()
    if args.action=="points":points(args.cohort)
    elif args.action=="bootstrap":bootstrap(args.cohort)
    elif args.action=="combine":combine()
    else:numerical_tests()
