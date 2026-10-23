"""P4 isolated paths and resumable four-hour execution window; stdlib only."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
P3 = OUT.parent / 'P3_20261015_134650'
DATA_PARENT = OUT.parent / 'External_Data/Derm7pt'
DATA = DATA_PARENT / 'extracted_20261015_233237_183679/release_v0'
FIRST = Path(__import__('os').environ['R52_PATH_F74ED8905ECC'])
SOURCE = OUT / 'source'
PRIVATE = OUT / 'private_predictions'
SEEDS = [42, 2024, 3407]
MODELS = ['M0', 'R_IMG', 'R_CLIN', 'XATTN', 'B2S']
ZONE = timezone(timedelta(hours=8))
START = datetime.fromisoformat('2026-10-16T05:27:36+08:00')
DEADLINE = (START + timedelta(hours=4)).timestamp()

def now():
    return datetime.now(ZONE).isoformat(timespec='seconds')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def stopped():
    return time.time() >= DEADLINE or (OUT / 'STOP_REQUESTED').exists()

def update_progress(**fields):
    path = OUT / 'P4_PROGRESS.json'
    state = read(path) if path.exists() else {'new_training_runs': 0, 'expected_models': 15}
    state.update(fields)
    state.update(updated_at=now(), deadline=datetime.fromtimestamp(DEADLINE, ZONE).isoformat())
    dump(path, state)

def log(message):
    line = f'{now()} {message}'
    print(line, flush=True)
    with (OUT / 'p4_execution.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')
