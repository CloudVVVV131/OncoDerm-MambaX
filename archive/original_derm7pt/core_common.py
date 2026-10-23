"""Isolated OriginalCore evaluation utilities; no historical driver imports."""
import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
B2 = ROOT / 'frontiers_enhancement_B2_20261012_043342_488463'
P4 = OUT.parent / 'P4_Derm7pt_20261016_052736'
DATA = OUT.parent / 'External_Data/Derm7pt/extracted_20261015_233237_183679/release_v0'
PRIVATE = OUT / 'private_predictions'
ZONE = timezone(timedelta(hours=8))
START = datetime.fromisoformat('2026-10-17T18:04:50+08:00')
DEADLINE = (START + timedelta(hours=2)).timestamp()
CONDITIONS = ['E_AVAILABLE', 'E_NULL']
HEADS = ['prob_mel', 'prob_mel_endpoint']
FAMILIES = ['ORIG_B2', 'ORIG_B6']
SEEDS = [42, 2024, 3407]
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

def now():
    return datetime.now(ZONE).isoformat(timespec='seconds')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def csv_save(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def progress(**changes):
    path = OUT / 'ORIGINALCORE_PROGRESS.json'
    value = read(path) if path.exists() else dict(started_at=START.isoformat(), deadline=datetime.fromtimestamp(DEADLINE,ZONE).isoformat(), new_training_runs=0, identity='PENDING', inference='NOT_STARTED', statistics='NOT_STARTED', thresholds='PENDING', packaging='NOT_STARTED', completed_models=0, valid_bootstrap=0)
    value.update(changes)
    value['updated_at'] = now()
    dump(path, value)

def log(message):
    line = now() + ' ' + message
    print(line, flush=True)
    with (OUT / 'EXECUTION.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')
