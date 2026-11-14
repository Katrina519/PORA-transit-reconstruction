"""冻结单步A/B对照的独立输入与既有运行接口。"""
import sys
import time
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
JOINT = WORK / 'nea_spo_v2_joint_20261109_v1'
MUM = WORK / 'nea_spo_v2_mumford0_20261109_v1'
IMPORT_START = time.perf_counter()
sys.path.insert(0, str(JOINT))
import bridge as b
IMPORT_SECONDS = time.perf_counter() - IMPORT_START


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def canonical(routes):
    return tuple(sorted(min(tuple(rr), tuple(rr[::-1])) for rr in routes))


def models():
    a = read(WORK / 'nea_spo_v2_matched_ablation_20261108_v1/模型冻结.json')['models']['v2_spo']
    assert a == read(MUM / '模型冻结.json')['model']
    return {'A_spo': {'model': a, 'encoding': 'raw'},
            'B_spo': read(JOINT / '模型冻结.json')['document']}
