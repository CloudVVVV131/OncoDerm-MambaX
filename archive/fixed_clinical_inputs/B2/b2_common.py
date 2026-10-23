"""OncoDerm B2, 2026-10-12: output-only IO and fixed local source navigation."""
import os
import sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PACKAGE = OUT / 'start_package/OncoDerm_B2_Inference_Only_Start'
NAV = PACKAGE / 'B1_Navigation'
LOCAL = OUT / 'local_only'
for key in ['TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME', 'NUMBA_CACHE_DIR']:
    os.environ[key] = str(LOCAL / 'cache' / key.lower())
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '4'
sys.dont_write_bytecode = True
import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_json(name, obj):
    p = OUT / name
    assert p.resolve().is_relative_to(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def save_csv(name, records):
    p=OUT/name
    assert p.resolve().is_relative_to(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def log(message):
    line=f'[{now()}] {message}'
    print(line,flush=True)
    with (OUT/'EXECUTION.log').open('a',encoding='utf-8') as f:
        f.write(line+'\n')


def entries():
    return csv_rows(NAV/'B2_CHECKPOINT_NAVIGATION.csv')


def model_tag(entry):
    return entry['model']+'_seed'+str(entry['training_seed'])


def guard_sources(check_weights=False):
    for row in csv_rows(OUT/'INPUT_MANIFEST.csv'):
        p=Path(row['path'])
        if row['role']=='checkpoint' and not check_weights:
            assert p.stat().st_size==int(row['bytes']) and p.stat().st_mtime_ns==int(row['mtime_ns'])
        else:
            assert sha(p)==row['sha256'], 'Source changed: '+str(p)
