from __future__ import annotations
from pathlib import Path
import csv, os, traceback

def ensure_parent(path): Path(path).parent.mkdir(parents=True,exist_ok=True)

def append_csv(path, row, fieldnames):
    ensure_parent(path); p=Path(path); new=not p.exists()
    with p.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        if new: w.writeheader()
        w.writerow(row)

def completed_keys(path, cols):
    p=Path(path); out=set()
    if not p.exists(): return out
    with p.open(encoding='utf-8') as f:
        for r in csv.DictReader(f): out.add(tuple(r[c] for c in cols))
    return out

def log_failure(path, task, exc):
    append_csv(path,{'task':task,'error_type':type(exc).__name__,'message':str(exc).replace('\n',' ')[:1000]},['task','error_type','message'])
