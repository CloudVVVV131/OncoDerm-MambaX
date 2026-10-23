"""P3 local paths, atomic records and authorization window (stdlib only)."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import time

OUT = Path(__file__).resolve().parent
WINDOW = json.loads((OUT / 'WINDOW.json').read_text(encoding='utf-8-sig'))
DEADLINE = datetime.fromisoformat(WINDOW['deadline']).timestamp()
ROOT = OUT.parent.parent
ARCHIVE = ROOT / 'oncoderm_bioengineering_full_results_20260817_110459/project'
REV = ROOT / 'project/revision'
SOURCE = OUT / 'source'
PRIVATE = OUT / 'private_predictions'
SEEDS = [42, 2024, 3407]
MODELS = ['M0', 'R_IMG', 'R_CLIN', 'XATTN', 'B2S']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stopped():
    return time.time() >= DEADLINE or (OUT / 'STOP_REQUESTED').exists()


def update_progress(**fields):
    path = OUT / 'P3_PROGRESS.json'
    state = read(path) if path.exists() else {}
    state.update(fields)
    state['updated_at'] = datetime.now().astimezone().isoformat()
    dump(path, state)


def log(message):
    line = f'{datetime.now().astimezone().isoformat()} {message}'
    print(line, flush=True)
    with (OUT / 'evaluation_statistics.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')
