"""P5 isolated output paths and two-hour execution window, standard library only."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
P3=OUT.parent/'P3_20261015_134650'
P4=OUT.parent/'P4_Derm7pt_20261016_052736'
SOURCE=OUT/'source'
PRIVATE=OUT/'private_predictions'
ZONE=timezone(timedelta(hours=8))
START=datetime.fromisoformat('2026-10-16T14:06:55+08:00')
DEADLINE=(START+timedelta(hours=2)).timestamp()
SEEDS=[42,2024,3407]
MODELS=['M0','R_IMG','R_CLIN','XATTN','B2S']
CONDITIONS=['NATIVE','SHIFT_HALF','PERMUTE_FIXED']

def now(): return datetime.now(ZONE).isoformat(timespec='seconds')
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
def dump(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)
def stopped(): return time.time()>=DEADLINE or (OUT/'STOP_REQUESTED').exists()
def progress(**fields):
    p=OUT/'P5_PROGRESS.json'
    state=read(p) if p.exists() else {'new_training_runs':0,'task_A':'NOT_STARTED','task_B':'NOT_STARTED'}
    state.update(fields); state.update(updated_at=now(),started_at=START.isoformat(),deadline=datetime.fromtimestamp(DEADLINE,ZONE).isoformat())
    dump(p,state)
def log(message):
    line=f'{now()} {message}'; print(line,flush=True)
    with (OUT/'P5_EXECUTION.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
