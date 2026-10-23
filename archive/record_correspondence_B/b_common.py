"""Route B isolated IO. Historical inputs are read-only."""
from pathlib import Path
import csv
import hashlib
import json
import os
from datetime import datetime

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OC = ROOT / "OriginalCore_Derm7pt_20261017_181919"
B2 = ROOT.parent / "frontiers_enhancement_B2_20261012_043342_488463"
START = OUT.parent / "B_Start_20261020_214101"
A1 = OUT.parent / "A1_CohortProfile_20261020_054154"
DATA = ROOT / "External_Data/Derm7pt/extracted_20261015_233237_183679/release_v0"
PRIVATE = OUT / "private"
DEADLINE = datetime.fromisoformat("2026-10-21T01:41:01+08:00")

def now():
    return datetime.now().astimezone().isoformat()

def expired():
    return datetime.now().astimezone() >= DEADLINE

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def rows(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def csvout(p, values):
    values = list(values)
    if not values:
        return
    fields = list(dict.fromkeys(k for row in values for k in row))
    with Path(p).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(values)

def log(message):
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with (OUT / "EXECUTION.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
