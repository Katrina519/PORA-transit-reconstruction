"""既有固定生成器接口和独立目标核验。"""
import sys
import time
import json
import hashlib
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
PREVIOUS=WORK/'nea_task_collision_repair_20261110_v1'
IMPORT_START=time.perf_counter()
spec=importlib.util.spec_from_file_location('early_feedback_frozen_driver',PREVIOUS/'run.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
IMPORT_SECONDS=time.perf_counter()-IMPORT_START
b,q,r,torch=d.b,d.q,d.r,d.torch
audit=q.load('early_feedback_cost_verifier',WORK/'nea_spo_v2_joint_20261109_v1/audit.py')
sys.path.insert(0,str(ROOT))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(routes):return tuple(sorted(min(tuple(rr),tuple(rr[::-1])) for rr in routes))


def check_sources():
    previous=read(PREVIOUS/'交付封存.json')
    for path,value in previous['files'].items():assert sha(PREVIOUS/path)==value,path
    hashes=read(PREVIOUS/'independent/COMPLETE.json')['input_hashes']
    assert all(sha(p)==v for p,v in hashes.items())
    return hashes
