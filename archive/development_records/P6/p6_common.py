"""Local P6 paths and pure file helpers; safe for spawned main imports."""
from pathlib import Path
import csv
import hashlib
import json
import os

OUT = Path(__file__).resolve().parent
PKG = OUT / 'FIM_RegionClinical_Dev_Start'
WINDOW = json.loads((OUT / 'WINDOW.json').read_text(encoding='utf-8-sig'))
OLD = Path(WINDOW['source_seed42'])
P1 = Path(WINDOW['source_P1'])
P2 = Path(WINDOW['source_P2'])
P3 = Path(WINDOW['source_P3'])
VARIANTS = ['FIM_TKEEP_REGION', 'FIM_TKEEP_GLOBAL']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
