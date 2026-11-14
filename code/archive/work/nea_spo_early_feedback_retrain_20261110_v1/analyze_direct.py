"""复用前轮独立成本及逐条审计，输出全部作者配对。"""
import sys
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PREVIOUS=ROOT.parent/'nea_task_collision_repair_20261110_v1'
sys.path.insert(0,str(PREVIOUS))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main(phase):
    a=load('early_feedback_result_analysis',PREVIOUS/'analyze.py');a.ROOT=ROOT;a.analyze(phase)
    b=load('early_feedback_raw_audit',PREVIOUS/'audit_runs.py');b.ROOT=ROOT;b.audit(phase)


if __name__=='__main__':main(sys.argv[1])
