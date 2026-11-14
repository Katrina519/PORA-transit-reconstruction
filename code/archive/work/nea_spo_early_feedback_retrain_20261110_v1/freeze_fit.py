"""拟合前来源冻结和明确的新增训练/确认状态隔离。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e


def main():
    request=e.read(e.ROOT/'请求冻结.json');old=e.read(request['old_tasks_source'])
    registered=e.read(e.WORK/'nea_spo_v2_joint_20261109_v1/输入冻结.json')['contexts']
    old_mum_states={t['state'] for t in old if t['city']=='mumford0'}
    old_routes={c['state']:c['routes'] for c in registered if c['city']=='mumford0'}
    assert set(old_routes)==old_mum_states
    old_keys={e.canonical(v) for v in old_routes.values()}
    added=[e.canonical(c['routes']) for c in request['training_contexts']]
    assert len(added)==len(set(added)) and not old_keys.intersection(added)
    held=[e.canonical(c['routes']) for c in request['fresh_contexts']]
    assert len(held)==4 and len(set(held))==4 and not set(held).intersection(old_keys|set(added))
    tasks=e.read(e.ROOT/'训练任务.json')
    assert tasks[:211]==old and len(tasks)==211+12*len(added)
    files=[p for p in e.ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json']]
    hashes=e.check_sources();hashes.update({str(p):e.sha(p) for p in files})
    e.dump(e.ROOT/'拟合输入冻结.json',{'input_hashes':hashes,'old_tasks':211,'added_tasks':len(tasks)-211,
        'old_mum_states':len(old_mum_states),'added_training_states':len(added),'held_states':4,
        'state_overlap':False,'capture_time':'实际反馈采集完成后、拟合之前；不冒称首次采集前SHA冻结',
        'no_candidate_fitting_yet':True})
    print('FIT_INPUT_FROZEN',len(hashes),len(tasks),'tasks, state disjointness PASS',flush=True)


if __name__=='__main__':main()
