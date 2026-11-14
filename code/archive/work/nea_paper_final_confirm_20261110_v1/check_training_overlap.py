"""显式覆盖两个冻结主模型的全部训练状态。"""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def canon(R):return tuple(sorted(min(tuple(r),tuple(r[::-1])) for r in R))
def main():
    paths=[WORK/'restoration_native_transfer_20261105_v1/父网与特征.json',WORK/'nea_spo_balanced_tasks_20261108_v1/诊断完成.json',WORK/'nea_spo_v2_joint_20261109_v1/输入冻结.json',WORK/'nea_spo_v2_joint_20261109_v1/训练任务.json']
    mapping={('mandl',s):r for s,r in read(paths[0])['routes'].items()}
    mapping.update({('mandl',r['parent']):r['routes'] for r in read(paths[1])['records']})
    mapping.update({(r['city'],r['state']):r['routes'] for r in read(paths[2])['contexts']})
    tasks=read(paths[3]);states={(t['city'],t['state']) for t in tasks}
    assert all(k in mapping for k in states)
    identities={(city,canon(mapping[city,state])) for city,state in states}
    request=read(ROOT/'请求冻结.json')
    assert all((c['city'],canon(c['routes'])) not in identities for c in request['contexts'])
    out={'training_states_checked':len(states),'new_parents':len(request['contexts']),'no_overlap':True,
      'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    with (ROOT/'训练状态查重补核.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(out['training_states_checked'],'training states /',out['new_parents'],'new parents: PASS')
if __name__=='__main__':main()
